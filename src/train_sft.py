import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = Path("data/processed/train.jsonl")
VALID_FILE = Path("data/processed/valid.jsonl")

SEED = 42

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03

MAX_LENGTH = 256

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ==================================================
# Arguments
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="debug",
    )

    return parser.parse_args()


# ==================================================
# Device / precision
# ==================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_precision(device):
    """
    Returns:
        model_dtype
        use_amp
        amp_dtype
        use_grad_scaler
    """

    if device.type == "cuda":

        # Ampere or newer generally supports BF16 well.
        if torch.cuda.is_bf16_supported():
            return (
                torch.bfloat16,
                True,
                torch.bfloat16,
                False,
            )

        # Older CUDA GPUs: use FP16 + GradScaler.
        return (
            torch.float16,
            True,
            torch.float16,
            True,
        )

    # For MPS/CPU debugging prioritize stability.
    return (
        torch.float32,
        False,
        None,
        False,
    )


# ==================================================
# Dataset
# ==================================================

class DetoxDataset(Dataset):

    def __init__(
        self,
        examples,
        tokenizer,
    ):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        example = self.examples[idx]

        source = example["input"]
        target = example["target"]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a text rewriting assistant. "
                    "Your task is to remove toxic, rude, or "
                    "offensive language while preserving the "
                    "original meaning as much as possible."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite the following sentence in a neutral "
                    "and non-toxic style. Preserve the original "
                    "meaning and output only the rewritten "
                    "sentence.\n\n"
                    f"Sentence: {source}"
                ),
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": target,
            }
        ]

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].squeeze(0)

        attention_mask = encoded[
            "attention_mask"
        ].squeeze(0)

        labels = input_ids.clone()

        prompt_length = min(
            len(prompt_ids),
            MAX_LENGTH,
        )

        # Assistant-only loss
        labels[:prompt_length] = -100

        # Ignore padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==================================================
# Load examples
# ==================================================

def load_examples(path, limit=None):

    examples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            examples.append(
                json.loads(line)
            )

            if (
                limit is not None
                and len(examples) >= limit
            ):
                break

    return examples


# ==================================================
# Forward helper
# ==================================================

def forward_model(
    model,
    batch,
    device,
    use_amp,
    amp_dtype,
):

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    if use_amp:

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):
            outputs = model(**batch)

    else:
        outputs = model(**batch)

    return outputs


# ==================================================
# Validation
# ==================================================

def evaluate_validation_loss(
    model,
    dataloader,
    device,
    use_amp,
    amp_dtype,
):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            outputs = forward_model(
                model,
                batch,
                device,
                use_amp,
                amp_dtype,
            )

            total_loss += (
                outputs.loss.item()
            )

            num_batches += 1

    average_loss = (
        total_loss / num_batches
    )

    model.train()

    return average_loss


# ==================================================
# Main
# ==================================================

def main():

    args = parse_args()

    start_time = time.perf_counter()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = get_device()

    # --------------------------------------------------
    # Mode
    # --------------------------------------------------

    if args.mode == "debug":

        num_train_examples = 100
        num_valid_examples = 100
        num_epochs = 1

        output_dir = Path(
            "models/"
            "qwen2.5-0.5b-paradetox-lora-debug"
        )

    else:

        num_train_examples = None
        num_valid_examples = None
        num_epochs = 3

        output_dir = Path(
            "models/"
            "qwen2.5-0.5b-paradetox-lora-full"
        )

    best_model_dir = (
        output_dir / "best"
    )

    # --------------------------------------------------
    # Precision
    # --------------------------------------------------

    (
        model_dtype,
        use_amp,
        amp_dtype,
        use_grad_scaler,
    ) = get_precision(device)

    print("=" * 60)
    print("SFT CONFIGURATION")
    print("=" * 60)

    print(f"Mode:              {args.mode}")
    print(f"Device:            {device}")
    print(f"Model dtype:       {model_dtype}")
    print(f"Mixed precision:   {use_amp}")
    print(f"Seed:              {SEED}")
    print(f"Epochs:            {num_epochs}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Warmup ratio:      {WARMUP_RATIO}")
    print(f"Max length:        {MAX_LENGTH}")

    if device.type == "cuda":
        print(
            "GPU:               "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 60)

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print(
        f"\nLoading model: {MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------
    # Base model
    # --------------------------------------------------

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=model_dtype,
        )
    )

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.to(device)

    print()
    model.print_trainable_parameters()
    print()

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_examples = load_examples(
        TRAIN_FILE,
        num_train_examples,
    )

    valid_examples = load_examples(
        VALID_FILE,
        num_valid_examples,
    )

    print(
        f"Training examples: "
        f"{len(train_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(valid_examples)}"
    )

    train_dataset = DetoxDataset(
        train_examples,
        tokenizer,
    )

    valid_dataset = DetoxDataset(
        valid_examples,
        tokenizer,
    )

    # Explicit generator makes training shuffle
    # reproducible.
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Number of optimizer steps
    # --------------------------------------------------

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_dataloader)
            / GRADIENT_ACCUMULATION_STEPS
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * num_epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        * WARMUP_RATIO
    )

    # --------------------------------------------------
    # Cosine scheduler
    # --------------------------------------------------

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                total_optimizer_steps
            ),
        )
    )

    # --------------------------------------------------
    # GradScaler
    #
    # BF16 does not need scaling.
    # FP16 does.
    # --------------------------------------------------

    scaler = None

    if use_grad_scaler:
        scaler = torch.amp.GradScaler(
            "cuda"
        )

    print()
    print(
        f"Optimizer steps / epoch: "
        f"{optimizer_steps_per_epoch}"
    )

    print(
        f"Total optimizer steps: "
        f"{total_optimizer_steps}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    print()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    optimizer.zero_grad()

    model.train()

    optimizer_step = 0

    best_valid_loss = float("inf")

    for epoch in range(num_epochs):

        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )
        print("=" * 60)

        epoch_loss = 0.0

        running_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in enumerate(
            train_dataloader
        ):

            outputs = forward_model(
                model,
                batch,
                device,
                use_amp,
                amp_dtype,
            )

            loss = outputs.loss

            epoch_loss += loss.item()

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            # ------------------------------------------
            # Backward
            # ------------------------------------------

            if scaler is not None:

                scaler.scale(
                    scaled_loss
                ).backward()

            else:

                scaled_loss.backward()

            running_loss += loss.item()
            accumulated_batches += 1

            should_step = (
                (batch_idx + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            is_last_batch = (
                batch_idx + 1
                == len(train_dataloader)
            )

            if should_step or is_last_batch:

                # --------------------------------------
                # Optimizer step
                # --------------------------------------

                if scaler is not None:

                    scaler.step(optimizer)
                    scaler.update()

                else:

                    optimizer.step()

                scheduler.step()

                optimizer.zero_grad()

                optimizer_step += 1

                avg_loss = (
                    running_loss
                    / accumulated_batches
                )

                current_lr = (
                    scheduler.get_last_lr()[0]
                )

                # Avoid printing thousands of lines.
                if (
                    optimizer_step % 100 == 0
                    or optimizer_step == 1
                    or is_last_batch
                ):
                    print(
                        f"Step "
                        f"{optimizer_step}/"
                        f"{total_optimizer_steps} | "
                        f"Loss: "
                        f"{avg_loss:.4f} | "
                        f"LR: "
                        f"{current_lr:.2e}"
                    )

                running_loss = 0.0
                accumulated_batches = 0

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        train_loss = (
            epoch_loss
            / len(train_dataloader)
        )

        print(
            "\nRunning validation..."
        )

        valid_loss = (
            evaluate_validation_loss(
                model,
                valid_dataloader,
                device,
                use_amp,
                amp_dtype,
            )
        )

        print(
            f"\nEpoch {epoch + 1} complete | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Validation Loss: "
            f"{valid_loss:.4f}"
        )

        # --------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            best_model_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                best_model_dir
            )

            tokenizer.save_pretrained(
                best_model_dir
            )

            print(
                "New best validation loss: "
                f"{best_valid_loss:.4f}"
            )

            print(
                "Best adapter saved to: "
                f"{best_model_dir}"
            )

        print()

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    hours = int(
        elapsed // 3600
    )

    minutes = int(
        (elapsed % 3600) // 60
    )

    seconds = (
        elapsed % 60
    )

    print("=" * 60)
    print("Training complete.")
    print(
        f"Best validation loss: "
        f"{best_valid_loss:.4f}"
    )
    print(
        f"Best adapter: "
        f"{best_model_dir}"
    )
    print(
        f"Total runtime: "
        f"{hours}h "
        f"{minutes}m "
        f"{seconds:.1f}s"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()