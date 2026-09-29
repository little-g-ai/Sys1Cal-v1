---
license: mit
language:
  - en
pretty_name: Sys1Cal-v1
size_categories:
  - n<1K
task_categories:
  - text-classification
  - question-answering
tags:
  - probability-calibration
  - benchmark
  - synthetic
  - calibration
  - system-one-models
  - jev
configs:
  - config_name: parallel_primitives
    data_files:
      - split: test
        path: data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl
  - config_name: binary
    data_files:
      - split: test
        path: data/v0.1.0_tiny_cleanvars_binary.jsonl
---

# Sys1Cal-v1

![image](figure_8_score_expectation_connections.png)

## PAPER: [Jev thinks "I don't know", but doesn't say it: Introducing Sys1Cal-v1 Dataset for Probability Calibration](https://arxiv.org/abs/2609.35342)

Sys1Cal-v1 is a synthetic benchmark for probability calibration in Jev-like typed decision models. Each item is a True/False question about a proposition `A` where the exact probability `P(A)` is known by construction. The benchmark evaluates whether a model returns probabilities with the right numerical meaning, not only whether it selects the right label.

## Dataset Summary

Sys1Cal-v1 procedurally constructs probability problems with exact gold distributions, renders equivalent forms of the same latent item, and queries the proposition through Jev-style primitives:

- `noul`: one scalar probability that the proposition is true.
- `choice`: a categorical distribution over `False` and `True`.
- `score`: a distribution over ordered truth levels, evaluated by expected normalized truth value.

The primary metric is total variation distance from the exact gold distribution. For binary True/False questions, this is exactly absolute probability error.

## Splits and Configs

Both configs expose a single `test` split.

| Config | File | Rows | Use |
|---|---:|---:|---|
| `parallel_primitives` | `data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl` | 365 | Main Noul/Choice/Score benchmark contract |
| `binary` | `data/v0.1.0_tiny_cleanvars_binary.jsonl` | 365 | Binary Choice-like probability recovery |

## Load

```python
from datasets import load_dataset

ds = load_dataset("RiccardoPorcedda/Sys1Cal-v1", "parallel_primitives", split="test")
print(ds[0])
```

For local use before upload:

```python
from datasets import load_dataset

ds = load_dataset(
    "json",
    data_files="data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl",
    split="train",
)
```

## Row Schema

Important fields:

- `instance_id`: unique rendered benchmark item id.
- `latent_instance_id`: shared id for equivalent representations of the same latent probability problem.
- `family`: generator family.
- `representation`: rendering form, such as `direct`, `counts`, `ratio`, `table`, `prose`, or `distractor`.
- `state`: sufficient information for the model.
- `proposition` and `prompt`: proposition being evaluated.
- `queries`: Noul/Choice/Score query objects for the parallel primitive config.
- `outcomes`: usually `["True", "False"]`.
- `gold_distribution`: exact target distribution.
- `gold_argmax`: exact argmax label.
- `probability_band`: bin for the true probability.

## Minimal Example

```json
{
  "instance_id": "explicit_probability__seed_000042__repr_direct__parallel_primitives",
  "latent_instance_id": "explicit_probability__seed_000042",
  "family": "explicit_probability",
  "representation": "direct",
  "proposition": "Event A is true.",
  "outcomes": ["True", "False"],
  "gold_distribution": {"False": 1.0, "True": 0.0},
  "queries": {
    "noul": {"proposition": "Event A is true."},
    "choice": {
      "question": "What is the truth status of the following proposition?",
      "proposition": "Event A is true.",
      "options": ["False", "True"]
    },
    "score": {
      "question": "To what degree is the following proposition true?",
      "proposition": "Event A is true."
    }
  }
}
```

## Evaluation

This repository includes a standalone evaluator:

```bash
python scripts/evaluate_sys1cal.py \
  --dataset data/v0.1.0_tiny_cleanvars_parallel_primitives.jsonl \
  --predictions examples/predictions_minimal.jsonl \
  --output evaluation_summary.json
```

Prediction rows must include `instance_id` and one of:

- `predicted_distribution`, `distribution`, `prediction`, or `probabilities`, with labels such as `True` and `False`;
- `probability_true` or `p_true`, for binary True/False submissions.

The evaluator reports accuracy, pointwise probability fidelity (`1 - TV`), total variation distance, and Brier regret overall and by model/group, family, representation, and probability band.

## How to cite

```bibtex
@misc{porcedda2026sys1cal,
  title = {Jev thinks "I don't know", but doesn't say it: Introducing Sys1Cal-v1 Dataset for Probability Calibration},
  author = {Riccardo Porcedda},
  year = {2026},
  eprint = {2609.35342},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi = {10.48550/arXiv.2609.35342},
  url = {https://arxiv.org/abs/2609.35342}
}
```

## GitHub Repo

```url
https://github.com/little-g-ai/Sys1Cal-v1
```