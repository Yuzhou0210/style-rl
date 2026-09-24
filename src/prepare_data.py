import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


SEED = 42

OUTPUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def save_jsonl(data, path):
    with path.open("w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    print("Loading ParaDetox...")
    dataset = load_dataset("s-nlp/paradetox")["train"]

    # --------------------------------------------------
    # 1. Group all neutral references by toxic source
    # --------------------------------------------------

    grouped = defaultdict(list)

    for row in dataset:
        source = row["en_toxic_comment"].strip()
        target = row["en_neutral_comment"].strip()

        if not source or not target:
            continue

        grouped[source].append(target)

    sources = list(grouped.keys())

    print(f"Unique toxic sources: {len(sources)}")

    # --------------------------------------------------
    # 2. Shuffle SOURCES instead of individual pairs
    # --------------------------------------------------

    random.shuffle(sources)

    n = len(sources)

    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    train_sources = sources[:train_end]
    valid_sources = sources[train_end:valid_end]
    test_sources = sources[valid_end:]

    # --------------------------------------------------
    # 3. Restore source-reference pairs
    # --------------------------------------------------

    def build_examples(source_list):
        examples = []

        for source in source_list:
            for target in grouped[source]:
                examples.append(
                    {
                        "input": source,
                        "target": target,
                        "target_style": "neutral",
                    }
                )

        return examples

    train_data = build_examples(train_sources)
    valid_data = build_examples(valid_sources)
    test_data = build_examples(test_sources)

    # --------------------------------------------------
    # 4. Save
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_data, OUTPUT_DIR / "valid.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    print()
    print("Unique source split:")
    print(f"Train sources:      {len(train_sources)}")
    print(f"Validation sources: {len(valid_sources)}")
    print(f"Test sources:       {len(test_sources)}")

    print()
    print("Source-reference pairs:")
    print(f"Train pairs:      {len(train_data)}")
    print(f"Validation pairs: {len(valid_data)}")
    print(f"Test pairs:       {len(test_data)}")
    print(
        f"Total pairs:      "
        f"{len(train_data) + len(valid_data) + len(test_data)}"
    )

    # --------------------------------------------------
    # 5. Sanity check: no source leakage
    # --------------------------------------------------

    train_set = set(train_sources)
    valid_set = set(valid_sources)
    test_set = set(test_sources)

    assert train_set.isdisjoint(valid_set)
    assert train_set.isdisjoint(test_set)
    assert valid_set.isdisjoint(test_set)

    print()
    print("Source overlap check: PASSED")
    print("No toxic source appears in more than one split.")

    print()
    print("Saved to:")
    print(OUTPUT_DIR / "train.jsonl")
    print(OUTPUT_DIR / "valid.jsonl")
    print(OUTPUT_DIR / "test.jsonl")


if __name__ == "__main__":
    main()