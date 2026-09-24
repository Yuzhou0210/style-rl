import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "models/Qwen2.5-0.5B-Instruct"

INPUT_FILE = Path("data/toy.jsonl")
OUTPUT_FILE = Path("outputs/toy_predictions.jsonl")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    results = []

    for i, example in enumerate(examples):
        sentence = example["input"]
        target_style = example["target_style"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a writing assistant. "
                    "Rewrite text in the requested style while preserving "
                    "its original meaning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Rewrite the following sentence in a {target_style} style. "
                    f"Preserve the original meaning and output only the rewritten sentence.\n\n"
                    f"Sentence: {sentence}"
                ),
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Remove the original prompt tokens.
        output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]

        prediction = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()

        result = {
            "input": sentence,
            "target_style": target_style,
            "prediction": prediction,
        }

        results.append(result)

        print(f"\n[{i + 1}/{len(examples)}]")
        print(f"Input:      {sentence}")
        print(f"Prediction: {prediction}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDone! Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()