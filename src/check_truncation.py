import json
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = Path("data/processed/train.jsonl")
VALID_FILE = Path("data/processed/valid.jsonl")

MAX_LENGTH = 256


def load_examples(path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def get_lengths(example, tokenizer):

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

    prompt_text = tokenizer.apply_chat_template(
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

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # IMPORTANT:
    # No truncation here. We want the real token lengths.
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    return len(prompt_ids), len(full_ids)


def check_split(name, path, tokenizer):

    examples = load_examples(path)

    total = len(examples)

    full_truncated = 0
    prompt_truncated = 0
    target_part_truncated = 0

    max_prompt_length = 0
    max_full_length = 0

    full_lengths = []

    worst_examples = []

    for example in examples:

        prompt_length, full_length = get_lengths(
            example,
            tokenizer,
        )

        full_lengths.append(full_length)

        max_prompt_length = max(
            max_prompt_length,
            prompt_length,
        )

        max_full_length = max(
            max_full_length,
            full_length,
        )

        # Full sequence exceeds MAX_LENGTH.
        if full_length > MAX_LENGTH:
            full_truncated += 1

        # Prompt alone already fills/exceeds context.
        if prompt_length >= MAX_LENGTH:
            prompt_truncated += 1

        # Some assistant target tokens would be cut.
        if (
            prompt_length < MAX_LENGTH
            and full_length > MAX_LENGTH
        ):
            target_part_truncated += 1

        worst_examples.append(
            (
                full_length,
                prompt_length,
                example["input"],
                example["target"],
            )
        )

    full_lengths.sort()

    def percentile(p):
        index = int(
            (len(full_lengths) - 1) * p
        )
        return full_lengths[index]

    print()
    print("=" * 60)
    print(f"{name} TRUNCATION CHECK")
    print("=" * 60)

    print(f"Examples:               {total}")
    print(f"MAX_LENGTH:             {MAX_LENGTH}")

    print()
    print(
        f"Max prompt length:      "
        f"{max_prompt_length}"
    )
    print(
        f"Max full length:        "
        f"{max_full_length}"
    )

    print()
    print(
        f"Full length p50:        "
        f"{percentile(0.50)}"
    )
    print(
        f"Full length p90:        "
        f"{percentile(0.90)}"
    )
    print(
        f"Full length p95:        "
        f"{percentile(0.95)}"
    )
    print(
        f"Full length p99:        "
        f"{percentile(0.99)}"
    )

    print()
    print(
        f"Full sequences > 256:   "
        f"{full_truncated} "
        f"({full_truncated / total * 100:.2f}%)"
    )

    print(
        f"Prompts >= 256:         "
        f"{prompt_truncated} "
        f"({prompt_truncated / total * 100:.2f}%)"
    )

    print(
        f"Targets partly cut:     "
        f"{target_part_truncated} "
        f"({target_part_truncated / total * 100:.2f}%)"
    )

    print()
    print("Top 5 longest examples:")
    print("-" * 60)

    worst_examples.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for i, (
        full_length,
        prompt_length,
        source,
        target,
    ) in enumerate(
        worst_examples[:5],
        start=1,
    ):

        print()
        print(f"[{i}]")
        print(
            f"Prompt tokens: {prompt_length}"
        )
        print(
            f"Full tokens:   {full_length}"
        )
        print(f"Input:  {source}")
        print(f"Target: {target}")


def main():

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    check_split(
        "TRAIN",
        TRAIN_FILE,
        tokenizer,
    )

    check_split(
        "VALIDATION",
        VALID_FILE,
        tokenizer,
    )


if __name__ == "__main__":
    main()