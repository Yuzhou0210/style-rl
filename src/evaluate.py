import json
from pathlib import Path

import argparse

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline



SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOXICITY_MODEL = "unitary/toxic-bert"

TOXICITY_THRESHOLD = 0.5


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to prediction JSONL file",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name shown in evaluation output",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()

    prediction_file = args.predictions
    experiment_name = args.name

    # --------------------------------------------------
    # Load predictions
    # --------------------------------------------------

    with prediction_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        examples = [json.loads(line) for line in f]

    inputs = [x["input"] for x in examples]
    predictions = [x["prediction"] for x in examples]

    print(f"Evaluating {len(examples)} examples.")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # 1. Semantic similarity
    # --------------------------------------------------

    print(f"\nLoading semantic model: {SEMANTIC_MODEL}")

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL,
        device=device,
    )

    input_embeddings = semantic_model.encode(
        inputs,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    prediction_embeddings = semantic_model.encode(
        predictions,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarities = (
        input_embeddings * prediction_embeddings
    ).sum(dim=1)

    semantic_scores = similarities.cpu().tolist()

    # --------------------------------------------------
    # 2. Toxicity
    # --------------------------------------------------

    print(f"Loading toxicity model: {TOXICITY_MODEL}")

    # pipeline expects:
    #   -1 for CPU
    #    0 for CUDA/MPS
    pipeline_device = 0 if device in ["cuda", "mps"] else -1

    toxicity_classifier = pipeline(
        "text-classification",
        model=TOXICITY_MODEL,
        tokenizer=TOXICITY_MODEL,
        device=pipeline_device,
    )

    input_toxicity_results = toxicity_classifier(
        inputs,
        truncation=True,
        batch_size=8,
    )

    output_toxicity_results = toxicity_classifier(
        predictions,
        truncation=True,
        batch_size=8,
    )

    input_toxicity = [
        result["score"]
        for result in input_toxicity_results
    ]

    output_toxicity = [
        result["score"]
        for result in output_toxicity_results
    ]

    # --------------------------------------------------
    # Per-example results
    # --------------------------------------------------

    print("\nPer-example results:\n")

    detox_successes = 0

    for i, (
        example,
        semantic,
        tox_in,
        tox_out,
    ) in enumerate(
        zip(
            examples,
            semantic_scores,
            input_toxicity,
            output_toxicity,
        ),
        start=1,
    ):
        is_non_toxic = tox_out < TOXICITY_THRESHOLD

        if is_non_toxic:
            detox_successes += 1

        print(f"[{i}/{len(examples)}]")
        print(f"Input:      {example['input']}")
        print(f"Prediction: {example['prediction']}")
        print(f"Semantic:   {semantic:.4f}")
        print(f"Tox input:  {tox_in:.4f}")
        print(f"Tox output: {tox_out:.4f}")
        print(f"Non-toxic:  {is_non_toxic}")
        print()

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    mean_semantic = sum(semantic_scores) / len(semantic_scores)

    mean_input_toxicity = (
        sum(input_toxicity) / len(input_toxicity)
    )

    mean_output_toxicity = (
        sum(output_toxicity) / len(output_toxicity)
    )

    detox_success_rate = (
        detox_successes / len(examples)
    )

    print("=" * 60)
    print(f"{experiment_name} EVALUATION")
    print("=" * 60)

    print(f"Examples:                {len(examples)}")
    print(f"Semantic similarity:     {mean_semantic:.4f}")
    print(f"Input toxicity:          {mean_input_toxicity:.4f}")
    print(f"Output toxicity:         {mean_output_toxicity:.4f}")
    print(f"Detox success rate:      {detox_success_rate:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()