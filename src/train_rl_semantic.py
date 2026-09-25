import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_ADAPTER_PATH = Path(
    "models/qwen2.5-0.5b-paradetox-lora-full/best"
)

TRAIN_FILE = Path("data/processed/train.jsonl")

# Style reward model.
# IMPORTANT:
# This is different from the final toxicity evaluator
# (unitary/toxic-bert).
STYLE_REWARD_MODEL_NAME = (
    "s-nlp/roberta_toxicity_classifier"
)

# Semantic reward model.
# IMPORTANT:
# This is different from the final semantic evaluator
# (sentence-transformers/all-MiniLM-L6-v2).
SEMANTIC_REWARD_MODEL_NAME = (
    "sentence-transformers/all-mpnet-base-v2"
)

SEED = 42

# Keep the two reward components equally weighted.
#
# GRPO will normalize rewards within each group when
# scale_rewards="group", so we do not manually rescale the
# raw style logit margin or semantic cosine similarity here.
STYLE_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = (
    "You are a text rewriting assistant. "
    "Your task is to remove toxic, rude, or offensive language "
    "while preserving the original meaning as much as possible."
)


def build_prompt(sentence: str):
    """
    Build exactly the same frozen prompt used in
    Zero-shot, SFT, and Style-only RL.
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Rewrite the following sentence in a neutral "
                "and non-toxic style. Preserve the original "
                "meaning and output only the rewritten sentence."
                f"\n\nSentence: {sentence}"
            ),
        },
    ]


# ============================================================
# Data
# ============================================================

def load_unique_sources(path: Path):
    """
    Load training examples and keep each toxic source only once.

    ParaDetox contains multiple neutral references for some
    toxic inputs. RL operates on unique source sentences.
    """
    sources = []
    seen = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            source = example["input"]

            if source not in seen:
                seen.add(source)
                sources.append(source)

    return sources


# ============================================================
# Completion helper
# ============================================================

def completion_to_text(completion):
    """
    TRL may return a completion either as a plain string
    or as a conversational list of message dictionaries.
    """

    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, list):
        parts = []

        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    parts.append(content)
            else:
                parts.append(str(item))

        return " ".join(parts).strip()

    return str(completion).strip()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="debug",
    )

    args = parser.parse_args()

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for RL training."
        )

    device = "cuda"

    print("=" * 70)
    print("RL Style + Semantic Training")
    print("=" * 70)
    print(f"Mode:   {args.mode}")
    print(f"Device: {device}")
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # --------------------------------------------------------
    # Load base language model
    # --------------------------------------------------------

    print(f"Loading base model: {MODEL_NAME}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )

    # --------------------------------------------------------
    # Load SFT adapter
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Style+Semantic RL starts from the SAME SFT checkpoint
    # as Style-only RL.
    #
    # Do NOT initialize from the Style-only RL adapter.
    # --------------------------------------------------------

    print(
        f"Loading trainable SFT adapter: "
        f"{SFT_ADAPTER_PATH}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Load style reward model
    # --------------------------------------------------------

    print()
    print(
        f"Loading style reward model: "
        f"{STYLE_REWARD_MODEL_NAME}"
    )

    style_tokenizer = AutoTokenizer.from_pretrained(
        STYLE_REWARD_MODEL_NAME
    )

    style_model = (
        AutoModelForSequenceClassification.from_pretrained(
            STYLE_REWARD_MODEL_NAME,
            dtype=torch.float32,
        )
    )

    style_model.to(device)
    style_model.eval()

    for parameter in style_model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Load semantic reward model
    # --------------------------------------------------------

    print()
    print(
        f"Loading semantic reward model: "
        f"{SEMANTIC_REWARD_MODEL_NAME}"
    )

    semantic_model = SentenceTransformer(
        SEMANTIC_REWARD_MODEL_NAME,
        device=device,
    )

    semantic_model.eval()

    # --------------------------------------------------------
    # Reward function 1: Style
    # --------------------------------------------------------

    def style_reward(
        completions,
        **kwargs,
    ):
        """
        Style reward:

            R_style =
                neutral_logit - toxic_logit

        Higher reward means the reward classifier considers
        the generated sentence more neutral / less toxic.

        We intentionally use the raw logit margin rather than
        1 - P(toxic), because the probability reward saturated
        during the earlier Style-only GRPO pilot.
        """

        texts = [
            completion_to_text(completion)
            for completion in completions
        ]

        encoded = style_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            logits = style_model(
                **encoded
            ).logits

            # For s-nlp/roberta_toxicity_classifier:
            #
            # index 0 = neutral
            # index 1 = toxic

            neutral_logits = logits[:, 0]
            toxic_logits = logits[:, 1]

            rewards = (
                neutral_logits
                - toxic_logits
            )

        return (
            rewards
            .float()
            .cpu()
            .tolist()
        )

    # --------------------------------------------------------
    # Reward function 2: Semantic preservation
    # --------------------------------------------------------

    def semantic_reward(
        completions,
        source=None,
        **kwargs,
    ):
        """
        Semantic reward:

            R_semantic =
                cosine(
                    embedding(source),
                    embedding(completion)
                )

        Embeddings are L2-normalized, so their dot product
        equals cosine similarity.

        The source is the original toxic sentence, NOT the
        neutral reference. This encourages preservation of the
        original meaning without requiring a gold rewrite.
        """

        if source is None:
            raise ValueError(
                "The 'source' column is required "
                "for semantic reward."
            )

        texts = [
            completion_to_text(completion)
            for completion in completions
        ]

        # TRL should replicate dataset columns so that source
        # aligns with each generated completion.
        #
        # Convert to list explicitly for SentenceTransformer.
        sources = list(source)

        with torch.no_grad():
            source_embeddings = semantic_model.encode(
                sources,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            completion_embeddings = (
                semantic_model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

            rewards = (
                source_embeddings
                * completion_embeddings
            ).sum(dim=-1)

        return (
            rewards
            .float()
            .cpu()
            .tolist()
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print(f"Loading training data: {TRAIN_FILE}")

    sources = load_unique_sources(
        TRAIN_FILE
    )

    print(
        f"Unique training sources: "
        f"{len(sources)}"
    )

    # Keep exactly the same deterministic shuffle as the
    # Style-only experiment.
    rng = random.Random(SEED)
    rng.shuffle(sources)

    if args.mode == "debug":
        sources = sources[:100]

        output_dir = Path(
            "models/"
            "qwen2.5-0.5b-paradetox-rl-semantic-debug"
        )

        max_steps = 100

    else:
        output_dir = Path(
            "models/"
            "qwen2.5-0.5b-paradetox-rl-semantic-full"
        )

        max_steps = -1

    print(
        f"Training sources used: "
        f"{len(sources)}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    #
    # Keep "source" as an additional dataset column.
    # TRL passes dataset columns to reward functions.
    # --------------------------------------------------------

    train_dataset = Dataset.from_dict(
        {
            "prompt": [
                build_prompt(source)
                for source in sources
            ],
            "source": sources,
        }
    )

    print()
    print("Example source:")
    print(train_dataset[0]["source"])

    print()
    print("Example prompt:")
    print(train_dataset[0]["prompt"])

    # --------------------------------------------------------
    # GRPO configuration
    # --------------------------------------------------------

    print()
    print(
        f"Reward weights: "
        f"style={STYLE_WEIGHT}, "
        f"semantic={SEMANTIC_WEIGHT}"
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),

        # ----------------------------
        # Training
        # ----------------------------

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        num_train_epochs=1,
        max_steps=max_steps,

        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0,

        # ----------------------------
        # Precision
        # ----------------------------

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        # ----------------------------
        # GRPO generation
        # ----------------------------

        num_generations=4,
        max_completion_length=64,

        temperature=1.0,
        top_p=1.0,

        use_vllm=False,

        # No KL penalty, matching Style-only.
        beta=0.0,

        # ----------------------------------------------------
        # Reward combination
        # ----------------------------------------------------
        #
        # TRL combines:
        #
        # 0.5 * style_reward
        # +
        # 0.5 * semantic_reward
        #
        # scale_rewards="group" keeps the same GRPO reward
        # normalization strategy as the Style-only experiment.
        # ----------------------------------------------------

        reward_weights=[
            STYLE_WEIGHT,
            SEMANTIC_WEIGHT,
        ],

        scale_rewards="group",

        # ----------------------------
        # Logging
        # ----------------------------

        logging_steps=10,
        logging_first_step=True,

        report_to="none",

        log_completions=(
            args.mode == "debug"
        ),

        num_completions_to_print=(
            4
            if args.mode == "debug"
            else None
        ),

        # ----------------------------
        # Checkpoints
        # ----------------------------

        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,

        # ----------------------------
        # Reproducibility
        # ----------------------------

        seed=SEED,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = GRPOTrainer(
        model=model,

        reward_funcs=[
            style_reward,
            semantic_reward,
        ],

        args=training_args,

        train_dataset=train_dataset,

        processing_class=tokenizer,
    )

    # --------------------------------------------------------
    # Resume handling
    # --------------------------------------------------------

    checkpoint_exists = (
        output_dir.exists()
        and any(
            output_dir.glob(
                "checkpoint-*"
            )
        )
    )

    print()
    print("=" * 70)

    if (
        args.mode == "full"
        and checkpoint_exists
    ):
        print(
            "Checkpoint found. "
            "Resuming full training..."
        )

        trainer.train(
            resume_from_checkpoint=True
        )

    else:
        print(
            "Starting training from "
            "the SFT policy..."
        )

        trainer.train()

    # --------------------------------------------------------
    # Save final adapter
    # --------------------------------------------------------

    final_dir = (
        output_dir
        / "final"
    )

    print()
    print(
        f"Saving final adapter to: "
        f"{final_dir}"
    )

    trainer.save_model(
        str(final_dir)
    )

    tokenizer.save_pretrained(
        str(final_dir)
    )

    print()
    print("=" * 70)
    print("Training complete.")
    print(
        f"Final adapter: "
        f"{final_dir}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()