#!/usr/bin/env python3

"""
Re-evaluate all four ParaDetox systems with the frozen final evaluators.

Semantic evaluator:
    sentence-transformers/all-MiniLM-L6-v2

Toxicity evaluator:
    unitary/toxic-bert

Semantic metric:
    cosine similarity between normalized embeddings of
    source input and generated prediction

Detox success:
    toxicity_score < 0.5

Outputs:
    scored/zero_shot_scored.jsonl
    scored/sft_scored.jsonl
    scored/rl_style_scored.jsonl
    scored/rl_semantic_scored.jsonl
    scored/aggregate_results.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "scored"

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOXICITY_MODEL_NAME = "unitary/toxic-bert"

TOXICITY_THRESHOLD = 0.5
SEMANTIC_BATCH_SIZE = 64
TOXICITY_BATCH_SIZE = 32

FILES = {
    "Zero-shot": BASE_DIR / "zero_shot_predictions.jsonl",
    "SFT-LoRA": BASE_DIR / "sft_predictions.jsonl",
    "Style-only GRPO": BASE_DIR / "rl_style_predictions.jsonl",
    "Style+Semantic GRPO": BASE_DIR / "rl_semantic_predictions.jsonl",
}

OUTPUT_FILES = {
    "Zero-shot": OUTPUT_DIR / "zero_shot_scored.jsonl",
    "SFT-LoRA": OUTPUT_DIR / "sft_scored.jsonl",
    "Style-only GRPO": OUTPUT_DIR / "rl_style_scored.jsonl",
    "Style+Semantic GRPO": OUTPUT_DIR / "rl_semantic_scored.jsonl",
}

# Original reported results.
# Used only for reproduction checking.
ORIGINAL_RESULTS = {
    "Zero-shot": {
        "semantic_similarity": 0.7268,
        "toxicity_score": 0.3980,
        "detox_success": 0.5879,
    },
    "SFT-LoRA": {
        "semantic_similarity": 0.7983,
        "toxicity_score": 0.0685,
        "detox_success": 0.9514,
    },
    "Style-only GRPO": {
        "semantic_similarity": 0.4153,
        "toxicity_score": 0.0033,
        "detox_success": 1.0000,
    },
    "Style+Semantic GRPO": {
        "semantic_similarity": 0.8657,
        "toxicity_score": 0.0774,
        "detox_success": 0.9430,
    },
}


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Prefer Apple Silicon MPS on Mac.
    Fall back to CUDA or CPU when unavailable.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# JSONL utilities
# ============================================================

def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "input" not in row:
                raise ValueError(
                    f"{path}: line {line_number} has no 'input' field."
                )

            if "prediction" not in row:
                raise ValueError(
                    f"{path}: line {line_number} has no 'prediction' field."
                )

            rows.append(row)

    return rows


def save_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Input validation
# ============================================================

def load_and_validate_files():
    datasets = {}

    print("\nLoading prediction files...")

    for method, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"\nMissing file:\n{path}\n"
                f"Please place all four prediction JSONL files "
                f"in {BASE_DIR}"
            )

        rows = load_jsonl(path)
        datasets[method] = rows

        unique_inputs = len({row["input"] for row in rows})

        print(
            f"{method:<22} "
            f"N={len(rows):4d}  "
            f"unique_inputs={unique_inputs:4d}"
        )

    # --------------------------------------------------------
    # Check equal lengths
    # --------------------------------------------------------

    lengths = {method: len(rows) for method, rows in datasets.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Prediction files have different lengths: {lengths}"
        )

    n = next(iter(lengths.values()))

    # --------------------------------------------------------
    # Check exact paired alignment
    # --------------------------------------------------------

    methods = list(datasets.keys())
    reference_method = methods[0]

    reference_inputs = [
        row["input"] for row in datasets[reference_method]
    ]

    for method in methods[1:]:
        current_inputs = [
            row["input"] for row in datasets[method]
        ]

        if current_inputs != reference_inputs:

            for i, (a, b) in enumerate(
                zip(reference_inputs, current_inputs)
            ):
                if a != b:
                    raise ValueError(
                        "\nInput alignment failed.\n"
                        f"First mismatch at index {i}:\n\n"
                        f"{reference_method}:\n{a}\n\n"
                        f"{method}:\n{b}\n"
                    )

            raise ValueError(
                f"Input alignment failed for {method}."
            )

    if len(set(reference_inputs)) != n:
        raise ValueError(
            "Test inputs are not unique. "
            "Expected one row per unique held-out source."
        )

    print(
        f"\n✓ All four systems contain {n} paired unique test inputs."
    )

    return datasets


# ============================================================
# Semantic similarity
# ============================================================

def load_semantic_model():
    print(
        f"\nLoading semantic evaluator:\n"
        f"  {SEMANTIC_MODEL_NAME}"
    )

    model = SentenceTransformer(
        SEMANTIC_MODEL_NAME,
        device=str(DEVICE),
    )

    return model


def compute_semantic_similarity(
    model,
    sources,
    predictions,
):
    """
    SentenceTransformer embeddings are L2-normalized.
    Therefore dot product == cosine similarity.
    """

    source_embeddings = model.encode(
        sources,
        batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    prediction_embeddings = model.encode(
        predictions,
        batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarities = np.sum(
        source_embeddings * prediction_embeddings,
        axis=1,
    )

    return similarities.astype(float)


# ============================================================
# Toxicity evaluation
# ============================================================

def load_toxicity_model():
    print(
        f"\nLoading toxicity evaluator:\n"
        f"  {TOXICITY_MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOXICITY_MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        TOXICITY_MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("\nToxicity model label mapping:")
    print(model.config.id2label)
    print(model.config.label2id)

    return tokenizer, model


def find_toxic_label_index(model):
    """
    unitary/toxic-bert is expected to expose a toxic label.
    We inspect the config rather than blindly assuming an index.
    """

    id2label = model.config.id2label

    for idx, label in id2label.items():
        if str(label).lower() == "toxic":
            return int(idx)

    # Some configs may use LABEL_0/LABEL_1 rather than semantic names.
    # For unitary/toxic-bert, the toxicity output is normally index 0.
    print(
        "\nWARNING: Could not find a label literally named 'toxic'."
    )
    print(
        "For unitary/toxic-bert, falling back to output index 0."
    )

    return 0


@torch.inference_mode()
def compute_toxicity_scores(
    tokenizer,
    model,
    texts,
):
    toxic_index = find_toxic_label_index(model)

    scores = []

    for start in tqdm(
        range(0, len(texts), TOXICITY_BATCH_SIZE),
        desc="Toxicity",
    ):
        batch = texts[
            start:start + TOXICITY_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        logits = outputs.logits

        # ----------------------------------------------------
        # Toxic-BERT is a multi-label classifier.
        # Therefore sigmoid is required, NOT softmax.
        # ----------------------------------------------------

        probabilities = torch.sigmoid(logits)

        toxic_probs = probabilities[:, toxic_index]

        scores.extend(
            toxic_probs.detach().cpu().float().numpy().tolist()
        )

    return np.asarray(scores, dtype=float)


# ============================================================
# Aggregate metrics
# ============================================================

def calculate_aggregates(rows):
    semantic = np.asarray(
        [row["semantic_similarity"] for row in rows],
        dtype=float,
    )

    toxicity = np.asarray(
        [row["toxicity_score"] for row in rows],
        dtype=float,
    )

    detox = np.asarray(
        [row["detox_success"] for row in rows],
        dtype=float,
    )

    return {
        "n": len(rows),
        "semantic_similarity": float(semantic.mean()),
        "toxicity_score": float(toxicity.mean()),
        "detox_success": float(detox.mean()),
    }


# ============================================================
# Reproduction check
# ============================================================

def print_reproduction_check(results):
    print("\n")
    print("=" * 94)
    print("AGGREGATE RESULTS")
    print("=" * 94)

    header = (
        f"{'Method':<23}"
        f"{'Semantic':>12}"
        f"{'Toxicity':>12}"
        f"{'Detox':>12}"
        f"{'Original Sem.':>15}"
        f"{'Original Tox.':>15}"
        f"{'Original Detox':>15}"
    )

    print(header)
    print("-" * 94)

    for method, metrics in results.items():
        original = ORIGINAL_RESULTS[method]

        print(
            f"{method:<23}"
            f"{metrics['semantic_similarity']:>12.4f}"
            f"{metrics['toxicity_score']:>12.4f}"
            f"{100 * metrics['detox_success']:>11.2f}%"
            f"{original['semantic_similarity']:>15.4f}"
            f"{original['toxicity_score']:>15.4f}"
            f"{100 * original['detox_success']:>14.2f}%"
        )

    print("=" * 94)

    print("\nAbsolute differences from original reported values:")

    max_difference = 0.0

    for method, metrics in results.items():
        original = ORIGINAL_RESULTS[method]

        sem_diff = abs(
            metrics["semantic_similarity"]
            - original["semantic_similarity"]
        )

        tox_diff = abs(
            metrics["toxicity_score"]
            - original["toxicity_score"]
        )

        detox_diff = abs(
            metrics["detox_success"]
            - original["detox_success"]
        )

        max_difference = max(
            max_difference,
            sem_diff,
            tox_diff,
            detox_diff,
        )

        print(
            f"{method:<23} "
            f"semantic={sem_diff:.6f}  "
            f"toxicity={tox_diff:.6f}  "
            f"detox={detox_diff:.6f}"
        )

    print()

    # This is not a statistical test.
    # It is only a practical reproduction diagnostic.
    if max_difference <= 0.001:
        print(
            "✓ Reproduction check PASSED within tolerance 0.001."
        )
    else:
        print(
            "⚠ Reproduction check differs by more than 0.001."
        )
        print(
            "Do NOT run significance tests yet. "
            "First investigate evaluator/configuration differences."
        )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("ParaDetox unified evaluation")
    print("=" * 70)

    print(f"\nWorking directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load and validate predictions
    # --------------------------------------------------------

    datasets = load_and_validate_files()

    # All systems have exactly the same input list.
    first_method = next(iter(datasets))

    sources = [
        row["input"]
        for row in datasets[first_method]
    ]

    # --------------------------------------------------------
    # 2. Load evaluators once
    # --------------------------------------------------------

    semantic_model = load_semantic_model()

    toxicity_tokenizer, toxicity_model = (
        load_toxicity_model()
    )

    # --------------------------------------------------------
    # 3. Encode source sentences once
    # --------------------------------------------------------

    print("\nEncoding test sources once...")

    source_embeddings = semantic_model.encode(
        sources,
        batch_size=SEMANTIC_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    aggregate_results = {}

    # --------------------------------------------------------
    # 4. Evaluate each system
    # --------------------------------------------------------

    for method, rows in datasets.items():

        print("\n")
        print("=" * 70)
        print(f"Evaluating: {method}")
        print("=" * 70)

        predictions = [
            row["prediction"]
            for row in rows
        ]

        # ----------------------------------------------------
        # Semantic similarity
        # ----------------------------------------------------

        print("\nEncoding predictions for semantic similarity...")

        prediction_embeddings = semantic_model.encode(
            predictions,
            batch_size=SEMANTIC_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        semantic_scores = np.sum(
            source_embeddings * prediction_embeddings,
            axis=1,
        )

        # ----------------------------------------------------
        # Toxicity
        # ----------------------------------------------------

        print("\nComputing toxicity scores...")

        toxicity_scores = compute_toxicity_scores(
            toxicity_tokenizer,
            toxicity_model,
            predictions,
        )

        detox_success = (
            toxicity_scores < TOXICITY_THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # Construct scored rows
        # ----------------------------------------------------

        scored_rows = []

        for i, row in enumerate(rows):

            scored_row = {
                "input": row["input"],
                "prediction": row["prediction"],
                "semantic_similarity": float(
                    semantic_scores[i]
                ),
                "toxicity_score": float(
                    toxicity_scores[i]
                ),
                "detox_success": int(
                    detox_success[i]
                ),
            }

            scored_rows.append(scored_row)

        # ----------------------------------------------------
        # Save per-example scores
        # ----------------------------------------------------

        output_path = OUTPUT_FILES[method]

        save_jsonl(
            scored_rows,
            output_path,
        )

        print(
            f"\nSaved per-example scores:\n"
            f"  {output_path}"
        )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        metrics = calculate_aggregates(
            scored_rows
        )

        aggregate_results[method] = metrics

        print(
            f"\n{method}\n"
            f"  N                   = {metrics['n']}\n"
            f"  Semantic similarity = "
            f"{metrics['semantic_similarity']:.4f}\n"
            f"  Output toxicity     = "
            f"{metrics['toxicity_score']:.4f}\n"
            f"  Detox success       = "
            f"{100 * metrics['detox_success']:.2f}%"
        )

    # --------------------------------------------------------
    # 5. Save aggregate results
    # --------------------------------------------------------

    aggregate_path = (
        OUTPUT_DIR / "aggregate_results.json"
    )

    with aggregate_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            aggregate_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 6. Reproduction check
    # --------------------------------------------------------

    print_reproduction_check(
        aggregate_results
    )

    print(
        f"\nAggregate results saved to:\n"
        f"  {aggregate_path}"
    )

    print(
        "\nEvaluation finished."
    )


if __name__ == "__main__":
    main()

