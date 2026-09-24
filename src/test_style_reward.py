import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


REWARD_MODEL = (
    "s-nlp/roberta_toxicity_classifier"
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Reward model: {REWARD_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL
        )
        .to(device)
    )

    model.eval()

    print()
    print("Label mapping:")
    print(model.config.id2label)

    texts = [
        "You are a fucking idiot.",
        "You are stupid.",
        "I disagree with you.",
        "Thank you for your help.",
    ]

    encoded = tokenizer(
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
        logits = model(**encoded).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    # Model card:
    # index 0 = neutral
    # index 1 = toxic
    toxicity_scores = probabilities[:, 1]

    style_rewards = 1.0 - toxicity_scores

    print()
    print("=" * 70)
    print("STYLE REWARD TEST")
    print("=" * 70)

    for text, toxicity, reward in zip(
        texts,
        toxicity_scores.tolist(),
        style_rewards.tolist(),
    ):
        print()
        print(f"Text:       {text}")
        print(f"Toxicity:   {toxicity:.4f}")
        print(f"Reward:     {reward:.4f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()