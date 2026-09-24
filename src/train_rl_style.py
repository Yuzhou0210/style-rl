import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import GRPOConfig, GRPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_ADAPTER_PATH = (
    "models/"
    "qwen2.5-0.5b-paradetox-lora-full/"
    "best"
)

TRAIN_FILE = Path(
    "data/processed/train.jsonl"
)

REWARD_MODEL_NAME = (
    "s-nlp/roberta_toxicity_classifier"
)

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="debug",
    )

    return parser.parse_args()


def build_prompt(source):
    return [
        {
            "role": "system",
            "content": (
                "You are a text rewriting assistant. "
                "Your task is to remove toxic, rude, "
                "or offensive language while preserving "
                "the original meaning as much as possible."
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


def load_unique_sources(path):
    sources = []
    seen = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            example = json.loads(line)
            source = example["input"]

            if source not in seen:
                seen.add(source)
                sources.append(source)

    return sources


def completion_to_text(completion):
    # For conversational prompts, TRL may return
    # the completion as a list of message dicts.
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts = []

        for item in completion:
            if isinstance(item, dict):
                parts.append(
                    item.get("content", "")
                )
            else:
                parts.append(str(item))

        return " ".join(parts)

    return str(completion)


def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "RL training should be run on CUDA."
        )

    device = "cuda"

    print("=" * 60)
    print("RL STYLE-ONLY CONFIGURATION")
    print("=" * 60)
    print(f"Mode:             {args.mode}")
    print(f"Device:           {device}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Base model:       {MODEL_NAME}")
    print(f"SFT adapter:      {SFT_ADAPTER_PATH}")
    print(f"Reward model:     {REWARD_MODEL_NAME}")
    print("=" * 60)

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Load SFT policy
    # --------------------------------------------------

    print("\nLoading base model...")

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
        )
    )

    print("Loading trainable SFT LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Load reward model
    # --------------------------------------------------

    print("\nLoading reward model...")

    reward_tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL_NAME
    )

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL_NAME,
            dtype=torch.float32,
        )
        .to(device)
    )

    reward_model.eval()

    for parameter in reward_model.parameters():
        parameter.requires_grad = False

    print(
        "Reward label mapping:",
        reward_model.config.id2label,
    )

    # --------------------------------------------------
    # Reward function
    # --------------------------------------------------

    def style_reward(
        completions,
        **kwargs,
    ):
        texts = [
            completion_to_text(completion)
            for completion in completions
        ]

        encoded = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = reward_model(
                **encoded
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            toxicity = probabilities[:, 1]

            rewards = 1.0 - toxicity

        return rewards.float().cpu().tolist()

    # --------------------------------------------------
    # Prepare unique-source dataset
    # --------------------------------------------------

    sources = load_unique_sources(
        TRAIN_FILE
    )

    print(
        f"\nUnique training sources: "
        f"{len(sources)}"
    )

    random.Random(SEED).shuffle(sources)

    if args.mode == "debug":
        sources = sources[:100]

    train_dataset = Dataset.from_dict(
        {
            "prompt": [
                build_prompt(source)
                for source in sources
            ],
        }
    )

    print(
        f"RL training prompts: "
        f"{len(train_dataset)}"
    )

    # --------------------------------------------------
    # GRPO configuration
    # --------------------------------------------------

    if args.mode == "debug":
        output_dir = (
            "models/"
            "qwen2.5-0.5b-paradetox-rl-style-debug"
        )

        max_steps = 10

    else:
        output_dir = (
            "models/"
            "qwen2.5-0.5b-paradetox-rl-style-full"
        )

        max_steps = -1

    training_args = GRPOConfig(
        output_dir=output_dir,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        num_train_epochs=1,
        max_steps=max_steps,

        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        num_generations=4,
        max_completion_length=64,

        temperature=1.0,
        top_p=1.0,

        use_vllm=False,

        beta=0.0,

        logging_steps=1,
        logging_first_step=True,

        save_strategy="no",

        report_to="none",

        seed=SEED,

        log_completions=True,
        num_completions_to_print=4,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=style_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")

    trainer.train()

    # --------------------------------------------------
    # Save final adapter
    # --------------------------------------------------

    final_dir = Path(
        output_dir
    ) / "final"

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.model.save_pretrained(
        final_dir
    )

    tokenizer.save_pretrained(
        final_dir
    )

    print()
    print("=" * 60)
    print("RL STYLE-ONLY TRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Final adapter saved to: "
        f"{final_dir}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()