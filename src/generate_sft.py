import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = (
    "models/"
    "qwen2.5-0.5b-paradetox-lora-full/"
    "best"
)

INPUT_FILE = Path(
    "data/processed/test.jsonl"
)

OUTPUT_FILE = Path(
    "outputs/sft_predictions.jsonl"
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_unique_inputs(path):
    unique_inputs = []
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
                unique_inputs.append(source)

    return unique_inputs


def build_prompt(sentence, tokenizer):
    messages = [
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
                f"Sentence: {sentence}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Adapter: {ADAPTER_PATH}")

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Model dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
        )
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model = model.to(device)
    model.eval()

    inputs = load_unique_inputs(
        INPUT_FILE
    )

    print(
        f"Unique test inputs: {len(inputs)}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for i, sentence in enumerate(
            inputs,
            start=1,
        ):
            prompt = build_prompt(
                sentence,
                tokenizer,
            )

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            input_length = (
                encoded["input_ids"].shape[1]
            )

            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.eos_token_id
                    ),
                )

            generated_ids = output_ids[
                0,
                input_length:,
            ]

            prediction = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            result = {
                "input": sentence,
                "prediction": prediction,
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (
                i <= 10
                or i % 100 == 0
                or i == len(inputs)
            ):
                print(
                    f"[{i}/{len(inputs)}]"
                )

                if i <= 10:
                    print(
                        f"Input:      {sentence}"
                    )
                    print(
                        f"Prediction: {prediction}"
                    )
                    print()

    print()
    print(
        f"Predictions saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()