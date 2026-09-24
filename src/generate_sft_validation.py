import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_ADAPTER_PATH = (
    "models/"
    "qwen2.5-0.5b-paradetox-lora-full/"
    "best"
)

VALID_FILE = Path(
    "data/processed/valid.jsonl"
)

OUTPUT_FILE = Path(
    "outputs/sft_validation_predictions.jsonl"
)

SEED = 42
NUM_EXAMPLES = 200
MAX_NEW_TOKENS = 64


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_messages(source):
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


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print("=" * 60)
    print("SFT VALIDATION GENERATION")
    print("=" * 60)
    print(f"Device:       {device}")
    print(f"Base model:   {MODEL_NAME}")
    print(f"SFT adapter:  {SFT_ADAPTER_PATH}")
    print(f"Validation:   {VALID_FILE}")
    print(f"Output:       {OUTPUT_FILE}")
    print("=" * 60)

    # --------------------------------------------------
    # Load validation sources
    # --------------------------------------------------

    sources = load_unique_sources(
        VALID_FILE
    )

    print(
        f"\nUnique validation sources: "
        f"{len(sources)}"
    )

    # Use exactly the same fixed validation subset
    # as generate_rl_validation.py.
    random.Random(SEED).shuffle(sources)

    sources = sources[:NUM_EXAMPLES]

    print(
        f"Selected validation sources: "
        f"{len(sources)}"
    )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print("\nLoading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    # --------------------------------------------------
    # Load SFT adapter
    # --------------------------------------------------

    print("Loading SFT adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_PATH,
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting generation...\n")

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, source in enumerate(
            sources,
            start=1,
        ):
            messages = build_messages(
                source
            )

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            input_length = (
                inputs["input_ids"].shape[1]
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_tokens = outputs[
                0,
                input_length:
            ]

            prediction = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            result = {
                "input": source,
                "prediction": prediction,
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Print first 10 examples and every 50th example
            if i <= 10 or i % 50 == 0:
                print(
                    f"[{i}/{len(sources)}]"
                )
                print(
                    f"Input:      {source}"
                )
                print(
                    f"Prediction: {prediction}"
                )
                print()

    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(
        f"Predictions saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()