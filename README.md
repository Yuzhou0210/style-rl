# Reinforcement Learning for Text Detoxification

This repository contains the code and experimental results for a course
project investigating reinforcement learning for controllable text
generation.

The project studies the following research question:

> **Can reinforcement learning improve detoxification in controllable
> language generation beyond supervised fine-tuning without sacrificing
> semantic preservation?**

The experiments use
[ParaDetox](https://aclanthology.org/2022.acl-long.469/) and
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
A supervised LoRA model is first trained on toxic--neutral sentence pairs.
Two GRPO variants are then initialized independently from the same SFT
checkpoint:

1. **Style-only GRPO** — optimizes a toxicity-classifier reward.
2. **Style+Semantic GRPO** — combines toxicity reduction with a semantic
   similarity reward.

The results show that optimizing toxicity alone can achieve very strong
detoxification while substantially damaging semantic preservation. Adding
an explicit semantic reward largely prevents this failure mode and improves
semantic similarity beyond both SFT and style-only GRPO, although it does
not improve detoxification success over the SFT baseline.

---

## Repository Structure

```text
style-rl/
├── README.md
├── requirements.txt
├── .gitignore
│
├── prepare_data.py
├── baseline.py
├── train_sft.py
├── generate_sft.py
├── train_rl_style.py
├── generate_rl_style.py
├── train_rl_semantic.py
├── generate_rl_semantic.py
├── evaluate.py
│
├── results/
│   ├── aggregate_results.json
│   ├── zero_shot_scored.jsonl
│   ├── sft_scored.jsonl
│   ├── rl_style_scored.jsonl
│   └── rl_semantic_scored.jsonl
│
└── report/
    └── main.pdf
```

Model checkpoints are not included in the repository because of their size.
All RL experiments can be reproduced starting from the base model and the
training scripts provided here.

---

## Experimental Setup

### Dataset

The experiments use **ParaDetox**, which contains 19,744 toxic--neutral
source--reference pairs.

Some toxic source sentences have multiple neutral references. To prevent
the same toxic source from appearing in different partitions, the dataset
is split at the **toxic-source level** rather than at the individual-pair
level.

A fixed random seed of `42` is used for an 80/10/10 split.

| Split | Unique Sources | Source--Reference Pairs |
|---|---:|---:|
| Train | 9,541 | 15,765 |
| Validation | 1,192 | 1,932 |
| Test | 1,194 | 2,047 |
| **Total** | **11,927** | **19,744** |

The different stages use the data as follows:

- **SFT:** all 15,765 training source--reference pairs.
- **GRPO:** 9,541 unique toxic training sources.
- **Final evaluation:** 1,194 unique held-out toxic sources, each evaluated
  exactly once.

Human neutral references are used as targets during supervised training but
are not used in the final automatic semantic metric.

---

## Models

### Base Model

All experiments use:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

All systems use the same rewriting instruction:

> You are a text rewriting assistant. Your task is to remove toxic, rude,
> or offensive language while preserving the original meaning as much as
> possible.

The user prompt asks the model to rewrite the input sentence in a neutral
and non-toxic style while preserving its meaning.

---

### Supervised Fine-Tuning

The SFT baseline uses LoRA and is trained on all 15,765 training
source--reference pairs.

Main hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Epochs | 3 |
| Learning rate | `2e-4` |
| Scheduler | Cosine |
| Warmup ratio | `0.03` |
| Maximum sequence length | 256 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |

LoRA is applied to the `q`, `k`, `v`, and `o` projection layers.

Only assistant tokens contribute to the supervised training loss.

---

### Style-only GRPO

Style-only GRPO starts from the best SFT checkpoint and optimizes a reward
derived from:

```text
s-nlp/roberta_toxicity_classifier
```

The reward is the classifier logit margin

```text
R_style = z_neutral - z_toxic
```

so outputs classified as more neutral receive larger rewards.

Main GRPO settings:

| Hyperparameter | Value |
|---|---:|
| Epochs | 1 |
| Learning rate | `5e-6` |
| Scheduler | Cosine |
| Number of generations | 4 |
| Maximum completion length | 64 |
| Temperature | 1.0 |
| Top-p | 1.0 |
| KL coefficient β | 0 |

The KL coefficient is set to zero so that no additional reference-policy
penalty is applied. This keeps the comparison focused on the effect of the
reward formulation.

---

### Style+Semantic GRPO

The Style+Semantic model is initialized **independently from the same SFT
checkpoint** rather than from the Style-only GRPO model.

Its reward combines toxicity reduction with source--output semantic
similarity.

The style reward is normalized as

```text
R_style_norm = tanh((z_neutral - z_toxic) / 5)
```

and semantic preservation is measured using cosine similarity from:

```text
sentence-transformers/all-mpnet-base-v2
```

The final reward is

```text
R = 0.5 * R_style_norm + 0.5 * R_semantic
```

where `R_semantic` is the cosine similarity between the source sentence and
the generated rewrite.

The remaining GRPO settings are kept consistent with the style-only
condition.

---

## Installation

Python 3.10+ is recommended.

Clone the repository:

```bash
git clone https://github.com/Yuzhou0210/style-rl.git
cd style-rl
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
torch
transformers
datasets
peft
trl
sentence-transformers
scikit-learn
numpy
```

GPU training is strongly recommended for SFT and GRPO.

---

## Data Preparation

Prepare ParaDetox and construct the source-level train/validation/test
splits with:

```bash
python prepare_data.py
```

The split uses seed `42`.

All neutral references belonging to the same toxic source are assigned to
the same partition to avoid source-level leakage.

---

## Running the Experiments

### 1. Zero-shot Baseline

Run the prompted base model without additional training:

```bash
python baseline.py
```

---

### 2. Supervised Fine-Tuning

Train the LoRA SFT model:

```bash
python train_sft.py
```

Generate predictions using the trained SFT adapter:

```bash
python generate_sft.py
```

---

### 3. Style-only GRPO

Train the style-only GRPO model:

```bash
python train_rl_style.py
```

Generate predictions:

```bash
python generate_rl_style.py
```

---

### 4. Style+Semantic GRPO

Train the combined-reward GRPO model:

```bash
python train_rl_semantic.py
```

Generate predictions:

```bash
python generate_rl_semantic.py
```

Both GRPO experiments start independently from the same SFT checkpoint.

---

## Evaluation

Final evaluation uses models that are separate from the toxicity and
semantic models used during RL training.

### Semantic Preservation

Semantic similarity is computed using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

For each test example, normalized sentence embeddings are computed for the
original toxic source and the generated rewrite:

```text
Semantic Similarity = cosine(E(source), E(prediction))
```

Higher values indicate greater semantic preservation.

### Toxicity

Output toxicity is evaluated using:

```text
unitary/toxic-bert
```

Two toxicity metrics are reported:

- **Output Toxicity:** mean predicted toxicity probability; lower is better.
- **Detox Success:** percentage of outputs with toxicity below `0.5`;
  higher is better.

Run the evaluation script with:

```bash
python evaluate.py
```

---

## Results

All systems are evaluated on the same **1,194 unique held-out ParaDetox
sources**.

| Method | Semantic Similarity ↑ | Output Toxicity ↓ | Detox Success ↑ |
|---|---:|---:|---:|
| Zero-shot | 0.7268 | 0.3979 | 58.88% |
| SFT-LoRA | 0.7983 | 0.0685 | 95.14% |
| Style-only GRPO | 0.4153 | **0.0033** | **100.00%** |
| Style+Semantic GRPO | **0.8657** | 0.0774 | 94.30% |

Style-only GRPO strongly optimizes the toxicity objective but causes a
large decrease in semantic similarity. Qualitative inspection shows cases
where fluent and non-toxic outputs change facts or reverse the meaning of
the source sentence.

Adding the semantic reward substantially improves semantic preservation
over both SFT and Style-only GRPO. However, it does not improve
detoxification over SFT. Its slightly lower observed detoxification success
is not distinguishable from the SFT baseline in the paired fixed-test-set
analysis.

---

## Statistical Analysis

Because every system is evaluated on the same 1,194 test sources,
comparisons are paired by source.

Test-set uncertainty is estimated using **10,000 paired bootstrap
resamples** with seed `42`. Percentile 95% confidence intervals are computed
for aggregate metrics and pairwise system differences.

For binary detoxification success, paired outcomes are additionally compared
using an exact McNemar test.

For SFT versus Style+Semantic GRPO, the observed detoxification-success
difference is 0.84 percentage points in favor of SFT, with a paired
bootstrap 95% confidence interval of `[-0.25, 1.93]` percentage points.
The exact McNemar test gives `p = .164`.

These statistical analyses quantify uncertainty over the fixed test set.
They should **not** be interpreted as training-run variability because each
training condition was run only once.

---

## Example Outputs

### Semantic drift under Style-only GRPO

**Source**

```text
as far as the q situation they ain't got they shit together
```

**Style-only GRPO**

```text
As far as the situation, it seems that things have improved for them.
```

**Style+Semantic GRPO**

```text
as far as the q situation they ain't got they together
```

The style-only output reverses the negative assessment in the source,
whereas the combined reward better preserves the original proposition.

Another example:

**Source**

```text
sumbody hacked my damn twitter
```

**Style-only GRPO**

```text
Someone has been monitoring my Twitter posts for me.
```

**Style+Semantic GRPO**

```text
sumbody hacked my twitter
```

These examples illustrate why optimizing toxicity alone is insufficient for
faithful text detoxification.

---

## Limitations

The experiments have several limitations.

First, each training condition is run only once because of computational
constraints. The statistical analysis therefore measures uncertainty over
the test examples rather than variation across independent training runs.

Second, embedding-based semantic similarity does not guarantee semantic
correctness. It may fail to detect factual changes, negation, or subtle
meaning shifts.

Third, toxicity is measured using an automatic classifier and therefore
does not fully represent human judgments of offensiveness or
appropriateness.

Finally, the experiments use one dataset and one relatively small language
model. Future work could evaluate multiple random seeds, larger models,
alternative reward formulations, non-zero KL penalties, and human
evaluation.

---

## Main Finding

The experiments demonstrate that reinforcement learning can strongly shape
detoxification behavior, but its effect depends on reward design.

Optimizing toxicity alone improves detoxification beyond SFT at the cost of
substantial semantic degradation. Explicitly including semantic preservation
in the reward largely prevents this failure mode and achieves higher
semantic similarity, but does not establish an improvement in
detoxification over the SFT baseline.

---

## References

- Logacheva et al. (2022). *ParaDetox: Detoxification with Parallel Data*.
  ACL 2022.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models*.
- Yang et al. (2024). *Qwen2.5 Technical Report*.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*.
- Reimers and Gurevych (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks*.

See the project report for the complete bibliography and experimental
discussion.

---