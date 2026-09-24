import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = Path("data/processed/train.jsonl")
VALID_FILE = Path("data/processed/valid.jsonl")

OUTPUT_DIR = Path(
    "models/qwen2.5-0.5b-paradetox-lora-smoke"
)

BEST_MODEL_DIR = OUTPUT_DIR / "best"

# Smoke-test settings
NUM_EXAMPLES = 100
NUM_VALID_EXAMPLES = 100

NUM_EPOCHS = 1

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4
MAX_LENGTH = 256

SEED = 42

# LoRA settings
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ==================================================
# Device
# ==================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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

        # --------------------------------------------------
        # Prompt: system + user
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Prompt text
        #
        # add_generation_prompt=True adds the assistant
        # generation marker. Everything up to this point
        # will be masked from the training loss.
        # --------------------------------------------------

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --------------------------------------------------
        # Full conversation: prompt + target
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Tokenize prompt
        # --------------------------------------------------

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        # --------------------------------------------------
        # Tokenize full conversation
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Assistant-only labels
        # --------------------------------------------------

        labels = input_ids.clone()

        prompt_length = min(
            len(prompt_ids),
            MAX_LENGTH,
        )

        # Ignore:
        # system + user + assistant generation marker
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

def load_examples(
    path,
    limit=None,
):

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
# Validation
# ==================================================

def evaluate_validation_loss(
    model,
    dataloader,
    device,
):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    average_loss = (
        total_loss / num_batches
    )

    # Switch back to training mode
    model.train()

    return average_loss


# ==================================================
# Main
# ==================================================

def main():

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = get_device()

    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print(
        f"Loading model: {MODEL_NAME}"
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
    # Load base model
    #
    # float32 is intentionally used for the first
    # MPS training tests for stability.
    # --------------------------------------------------

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
        )
    )

    # --------------------------------------------------
    # LoRA configuration
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

    # ==================================================
    # Training data
    # ==================================================

    train_examples = load_examples(
        TRAIN_FILE,
        NUM_EXAMPLES,
    )

    print(
        f"Training examples: "
        f"{len(train_examples)}"
    )

    train_dataset = DetoxDataset(
        train_examples,
        tokenizer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ==================================================
    # Validation data
    # ==================================================

    valid_examples = load_examples(
        VALID_FILE,
        NUM_VALID_EXAMPLES,
    )

    print(
        f"Validation examples: "
        f"{len(valid_examples)}"
    )

    valid_dataset = DetoxDataset(
        valid_examples,
        tokenizer,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    optimizer.zero_grad()

    # ==================================================
    # Training setup
    # ==================================================

    model.train()

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_dataloader)
            / GRADIENT_ACCUMULATION_STEPS
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * NUM_EPOCHS
    )

    optimizer_step = 0

    best_valid_loss = float("inf")

    print()
    print(
        "Starting LoRA smoke-test training..."
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        "Optimizer steps per epoch: "
        f"{optimizer_steps_per_epoch}"
    )

    print(
        "Total optimizer steps: "
        f"{total_optimizer_steps}"
    )

    print()

    # ==================================================
    # Training loop
    # ==================================================

    for epoch in range(NUM_EPOCHS):

        epoch_loss = 0.0
        running_loss = 0.0
        accumulated_batches = 0

        for batch_idx, batch in enumerate(
            train_dataloader
        ):

            # ------------------------------------------
            # Move batch to device
            # ------------------------------------------

            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            # ------------------------------------------
            # Forward
            # ------------------------------------------

            outputs = model(**batch)

            loss = outputs.loss

            epoch_loss += loss.item()

            # ------------------------------------------
            # Gradient accumulation
            # ------------------------------------------

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            running_loss += loss.item()
            accumulated_batches += 1

            # ------------------------------------------
            # Decide whether to update parameters
            # ------------------------------------------

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

                optimizer.step()
                optimizer.zero_grad()

                optimizer_step += 1

                # Use the actual number of accumulated
                # batches. This also handles a final
                # incomplete accumulation group.
                avg_loss = (
                    running_loss
                    / accumulated_batches
                )

                print(
                    f"Epoch {epoch + 1} | "
                    f"Step {optimizer_step}/"
                    f"{total_optimizer_steps} | "
                    f"Loss: {avg_loss:.4f}"
                )

                running_loss = 0.0
                accumulated_batches = 0

        # ==================================================
        # End-of-epoch training loss
        # ==================================================

        train_loss = (
            epoch_loss
            / len(train_dataloader)
        )

        # ==================================================
        # Validation
        # ==================================================

        print()
        print(
            f"Running validation after "
            f"epoch {epoch + 1}..."
        )

        valid_loss = (
            evaluate_validation_loss(
                model,
                valid_dataloader,
                device,
            )
        )

        print()
        print(
            f"Epoch {epoch + 1} complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {valid_loss:.4f}"
        )

        # ==================================================
        # Save best validation checkpoint
        # ==================================================

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            BEST_MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                BEST_MODEL_DIR
            )

            tokenizer.save_pretrained(
                BEST_MODEL_DIR
            )

            print(
                "New best validation loss: "
                f"{best_valid_loss:.4f}"
            )

            print(
                "Best LoRA adapter saved to: "
                f"{BEST_MODEL_DIR}"
            )

        print()

    # ==================================================
    # Finished
    # ==================================================

    print("=" * 60)
    print("Training complete.")
    print(
        f"Best validation loss: "
        f"{best_valid_loss:.4f}"
    )
    print(
        f"Best LoRA adapter: "
        f"{BEST_MODEL_DIR}"
    )
    print("=" * 60)


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    main()