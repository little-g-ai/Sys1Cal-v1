# Jev Probability Benchmark — Agent Implementation Specification

## 1. Objective

Build a benchmark dataset and evaluation pipeline for testing whether decision models that return probability distributions — especially Jev — produce **statistically correct probabilities when the input state contains all information required to determine the ground-truth distribution**.

The benchmark must treat every evaluated model as a black box.

For each benchmark instance:

\[
x_i \longrightarrow \hat p_i
\]

where:

- \(x_i\) is a fully specified input state,
- \(p_i^*\) is the exact probability distribution implied by that state,
- \(\hat p_i\) is the probability distribution returned by the evaluated model.

The central evaluation question is:

\[
\hat p_i \overset{?}{\approx} p_i^*
\]

The benchmark must **not** assume anything about how a model internally computes probabilities.

Because Jev may return different probabilities for the same primitive/question in different runs, external Jev evaluations must repeat each identical question at least 10 times. Store every response separately with a repeat index and raw response metadata; do not overwrite or pre-average repeated calls during collection.

The benchmark must distinguish:

1. **Probability recovery** — does the model return the correct probability distribution for an individual state?
2. **Calibration** — across repeated stochastic outcomes, do events assigned probability \(q\) occur approximately \(q\) of the time?
3. **Representation sensitivity** — does the returned probability change when mathematically equivalent information is represented differently?
4. **Decision accuracy** — optional secondary metric when the model also returns a discrete decision.

Probability recovery is the primary goal.

---

## 1.1 Noul-Choice-Score Research Question

A central Jev-specific use case of this benchmark is to evaluate a question that is not directly covered by existing Jev/JevBench-style evaluations: whether Jev's returned **answer probability distributions** have stable probabilistic semantics across typed primitives.

The benchmark constructs simple probabilistic binary-classification problems where the true event probability is known exactly:

\[
p^*=P(A\mid x).
\]

Each single latent instance is queried through the same state and proposition using three primitives:

- **Noul**, which returns a probability that the proposition is true;
- **Choice**, with the mutually exclusive binary options `False` and `True`, which returns a normalized distribution over the supplied options;
- **Score**, with 10 ordered levels from completely false to completely true.

This lets the benchmark ask whether the primitives expose the same probabilistic object or whether their operational semantics differ.

The current working interpretation from the Jev experiments is:

1. **Noul appears comparatively well calibrated** against the exact event probability \(p^*\).
2. **The same binary problem expressed through Choice is not calibrated in the same way**: Choice tends to assign too much probability to `True`, especially away from the extremes.
3. A simple Noul-to-Choice power law often describes this distortion:

\[
P_C(T) \approx 1-(1-P_N(T))^\alpha.
\]

For \(\alpha\approx2\), this becomes:

\[
P_C(T)\approx 2P_N(T)-P_N(T)^2.
\]

4. Score explains why this can happen. If the Score distribution is treated as evidence for three latent truth masses,

\[
(T,U,F),
\]

then Choice can be modeled as the binary renormalization obtained after discarding an implied Unknown mass:

\[
P_C(T) \approx \frac{T}{T+F}=\frac{T}{1-U}.
\]

5. Score expectation recovers the Noul-like truth probability:

\[
\mu_S=\sum_{j=0}^9 q_j\frac{j}{9}\approx P_N(T).
\]

6. A Score-derived Unknown mass \(U_S\), learned from the 10-level Score distribution, substantially improves prediction of Choice:

\[
\widehat P_C(T)=\frac{T_S}{1-U_S}.
\]

Under the power-law calibration, the implied Unknown mass is:

\[
U(p)=1-\frac{p}{1-(1-p)^\alpha}.
\]

For \(\alpha=2\):

\[
U(p)=\frac{1-p}{2-p}.
\]

Thus the benchmark's emerging story is not merely that Choice is miscalibrated. It is that Choice may be behaving like a forced binary normalization of a richer truth-state distribution, where part of the Score mass corresponds to an implicit third state that is best described operationally as **Unknown**.

This remains a black-box behavioral interpretation. The benchmark must not claim that Jev literally implements an internal three-valued truth representation. It should report that the observed Noul-Choice-Score relationship is consistent with such a latent-state model.

---

# 2. Design Principles

The benchmark must satisfy the following requirements.

## 2.1 Exact ground truth

Every benchmark item must have an analytically or programmatically exact target distribution \(p^*\).

No human labels, LLM judges, or subjective grading should be required.

## 2.2 Procedural generation

Instances should be generated procedurally from random seeds.

Avoid relying mainly on famous textbook problems such as:

- standard dice puzzles,
- Monty Hall,
- classic disease-test examples,
- well-known card problems.

Use synthetic names and synthetic worlds where possible to reduce memorization effects.

## 2.3 Separation of reasoning difficulty

Create families with increasing inferential complexity:

- explicit probability,
- empirical frequency,
- compound probability,
- conditional probability,
- Bayes,
- sequential Bayes.

The benchmark should make it possible to locate where probability recovery begins to fail.

## 2.4 Mathematically equivalent representations

The same latent probability problem must be renderable through multiple representations.

This is essential for measuring representation sensitivity.

## 2.5 Black-box compatibility

The benchmark pipeline must support arbitrary models through adapters.

At minimum implement:

- Jev adapter,
- generic HTTP/API adapter interface,
- offline/mock adapter,
- optional OpenAI-compatible adapter if straightforward.

Do not hard-code Jev-specific assumptions into the core evaluator.

## 2.6 Reproducibility

All generated instances must be reproducible from:

- benchmark version,
- family,
- seed,
- generator parameters.

---

# 3. Repository Structure

Use a structure similar to:

```text
jev-prob-bench/
├── README.md
├── pyproject.toml
├── config/
│   ├── benchmark.yaml
│   └── models.example.yaml
├── src/
│   └── jev_prob_bench/
│       ├── __init__.py
│       ├── schemas.py
│       ├── generators/
│       │   ├── base.py
│       │   ├── explicit_probability.py
│       │   ├── frequency.py
│       │   ├── compound.py
│       │   ├── conditional.py
│       │   ├── bayes.py
│       │   └── sequential_bayes.py
│       ├── renderers/
│       │   ├── base.py
│       │   ├── direct.py
│       │   ├── counts.py
│       │   ├── table.py
│       │   ├── ratio.py
│       │   ├── prose.py
│       │   └── distractor.py
│       ├── adapters/
│       │   ├── base.py
│       │   ├── jev.py
│       │   ├── mock.py
│       │   └── generic_http.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── calibration.py
│       │   ├── representation.py
│       │   ├── runner.py
│       │   └── aggregation.py
│       ├── reporting/
│       │   ├── tables.py
│       │   ├── plots.py
│       │   └── report.py
│       └── cli.py
├── datasets/
│   ├── generated/
│   └── releases/
├── results/
├── tests/
└── scripts/
    ├── generate_dataset.py
    ├── run_benchmark.py
    └── build_report.py
```

---

# 4. Core Dataset Schema

Use JSONL as the canonical exchange format.

Each row should represent one **rendered benchmark item**.

Recommended schema:

```json
{
  "benchmark_version": "0.1.0",
  "instance_id": "bayes__seed_000124__repr_counts",
  "latent_instance_id": "bayes__seed_000124",
  "family": "bayes",
  "difficulty": 4,
  "seed": 124,

  "representation": "counts",

  "state": {
    "population_A": 100,
    "population_not_A": 900,
    "event_given_A": 0.8,
    "event_given_not_A": 0.2,
    "observed_event": true
  },

  "prompt": "Given the state, what is the probability that the item belongs to group A?",

  "outcomes": [
    "A",
    "not_A"
  ],

  "gold_distribution": {
    "A": 0.3076923077,
    "not_A": 0.6923076923
  },

  "gold_argmax": "not_A",

  "generator_metadata": {
    "parameters": {
      "prior_A": 0.1,
      "likelihood_event_given_A": 0.8,
      "likelihood_event_given_not_A": 0.2
    }
  }
}
```

## 4.1 Latent instance vs rendered item

A latent instance defines the mathematical problem.

A rendered item defines one particular representation of that problem.

For example:

```text
latent_instance_id = bayes__seed_000124
```

may produce:

```text
bayes__seed_000124__repr_direct
bayes__seed_000124__repr_counts
bayes__seed_000124__repr_table
bayes__seed_000124__repr_prose
bayes__seed_000124__repr_distractor
```

All of these must share exactly the same gold distribution.

This pairing is critical for representation-sensitivity analysis.

## 4.2 Parallel primitive binary schema

For experiments comparing Jev primitives, each single latent probabilistic instance must be queryable in parallel through Noul, Choice, and Score. The dataset hierarchy is:

\[
\text{latent probabilistic problem}
\rightarrow
\text{representation}
\rightarrow
\{\text{Noul},\text{Choice},\text{Score}\}.
\]

Each rendered binary row must contain one shared `state`, one shared proposition \(A\), exact gold probabilities \(P(A)\) and \(P(\neg A)\), and three primitive-specific query definitions. The same proposition and evidence must be preserved across all three primitives; only the minimum wording required by each primitive may differ.

Recommended schema:

```json
{
  "instance_id": "frequency__000124__repr_counts",
  "latent_instance_id": "frequency__000124",
  "family": "frequency",
  "representation": "counts",
  "state": {"zor": 37, "nif": 63, "sampling": "uniform"},
  "proposition": "The sampled object is a zor.",
  "gold": {
    "probability_true": 0.37,
    "probability_false": 0.63
  },
  "queries": {
    "noul": {
      "proposition": "The sampled object is a zor."
    },
    "choice": {
      "question": "What is the truth status of the following proposition?",
      "proposition": "The sampled object is a zor.",
      "options": ["False", "True"]
    },
    "score": {
      "question": "To what degree is the following proposition true?",
      "proposition": "The sampled object is a zor.",
      "levels": [
        "Completely false",
        "Very strongly false",
        "Strongly false",
        "Moderately false",
        "Slightly false",
        "Slightly true",
        "Moderately true",
        "Strongly true",
        "Very strongly true",
        "Completely true"
      ],
      "normalized_level_values": [0.0, 0.1111111111, 0.2222222222, 0.3333333333, 0.4444444444, 0.5555555556, 0.6666666667, 0.7777777778, 0.8888888889, 1.0]
    }
  }
}
```

The Score primitive always uses exactly 10 ordered levels mapped to \(z_j=j/9\), \(j=0,\ldots,9\). Level 0 means completely false, level 9 means completely true, and levels 1-8 are progressively increasing degrees of truth. Do not introduce a dedicated `unknown` category.

Do not define Score ground-truth distributions in the dataset. The only normative gold quantity required at dataset-construction time is the exact binary probability \(P(A)=p^*\), \(P(\neg A)=1-p^*\). Any later interpretation of the Score output distribution belongs in the evaluation pipeline.

Legacy evaluator compatibility fields such as `prompt`, `outcomes`, and `gold_distribution` may be duplicated in generated artifacts, but the normative parallel primitive contract is the `state`/`proposition`/`gold`/`queries` block above.

---

# 5. Experimental Families

Implement the following families.

---

## 5.1 Family A — Explicit Probability

### Purpose

Test whether a model preserves a probability that is directly supplied in the state.

This is the benchmark's simplest sanity check.

### Example latent state

```json
{
  "event_probability": 0.37
}
```

Question:

```text
Will event A occur?
```

Gold:

\[
P(A)=0.37,\qquad P(\neg A)=0.63.
\]

### Sampling

Generate values over the full interval, for example:

\[
p \in [0.01, 0.99].
\]

Use both:

- uniform random samples,
- a deterministic grid.

Recommended grid:

```text
0.01, 0.02, ..., 0.99
```

### Important analysis

Estimate the empirical probability-transfer function:

\[
f(p)=\hat p_{\text{model}}.
\]

Produce a scatter/line plot of:

\[
p^* \quad \text{vs} \quad \hat p.
\]

The ideal relationship is the identity line.

Also test complement symmetry:

\[
f(1-p) \approx 1-f(p).
\]

---

## 5.2 Family B — Frequency Recovery

### Purpose

Test whether the model can recover a probability from counts.

### Example

```json
{
  "zor": 37,
  "nif": 63,
  "sampling": "uniform"
}
```

Gold:

\[
P(\text{zor})=\frac{37}{100}=0.37.
\]

### Variants

Generate:

- two-category counts,
- multiclass counts,
- small counts,
- large counts preserving the same ratio.

Examples of mathematically equivalent states:

```text
3 / 10
30 / 100
300 / 1000
30000 / 100000
```

This family should contribute directly to representation-sensitivity experiments.

---

## 5.3 Family C — Compound Probability

### Purpose

Test elementary composition rules.

Include:

### Complement

\[
P(\neg A)=1-P(A)
\]

### Independent intersection

\[
P(A\cap B)=P(A)P(B)
\]

### Independent union

\[
P(A\cup B)=P(A)+P(B)-P(A)P(B)
\]

### Repeated Bernoulli trials

Examples:

- at least one success,
- exactly \(k\) successes,
- no successes.

### Sampling without replacement

Use hypergeometric cases.

Avoid overly large combinatorial values if numerical instability becomes unnecessary.

---

## 5.4 Family D — Conditional Probability

### Purpose

Test whether the model selects and uses the correct conditioning population.

### Example contingency table

|            | B  | not B |
|------------|---:|------:|
| A          | 30 | 10    |
| not A      | 20 | 40    |

Question:

\[
P(A\mid B)
\]

Gold:

\[
P(A\mid B)=\frac{30}{30+20}=0.6.
\]

### Required subfamilies

1. Clean contingency table.
2. Nested state dictionaries.
3. Relevant and irrelevant fields.
4. Multiple possible reference classes.
5. Temporal filters.
6. Boundary-condition filters.

This family should explicitly test the kind of reference-class mistakes that can occur even when arithmetic is simple.

---

## 5.5 Family E — Bayes

### Purpose

Test whether the model recovers exact posterior distributions.

### Binary Bayes

Generate:

\[
P(H),
\quad
P(E\mid H),
\quad
P(E\mid \neg H).
\]

Observe \(E\).

Gold:

\[
P(H\mid E)
=
\frac{P(E\mid H)P(H)}
{P(E\mid H)P(H)+P(E\mid \neg H)(1-P(H))}.
\]

### Multiclass Bayes

For hypotheses \(H_1,\dots,H_K\):

\[
P(H_i\mid E)
=
\frac{P(E\mid H_i)P(H_i)}
{\sum_j P(E\mid H_j)P(H_j)}.
\]

Recommended \(K\):

```text
3, 4, 5
```

### Parameter coverage

Sample cases spanning:

- weak evidence,
- strong evidence,
- rare priors,
- common priors,
- likelihood ratios near 1,
- likelihood ratios far from 1,
- posterior near 0.5,
- posterior near extremes.

Avoid only generating easy argmax cases.

---

## 5.6 Family F — Sequential Bayes

### Purpose

Test whether returned probabilities track mathematically correct updates as evidence accumulates.

For each latent problem define:

\[
P(H)
\]

and a sequence:

\[
E_1,E_2,\ldots,E_T.
\]

Compute:

\[
p_t^*=P(H\mid E_{1:t}).
\]

Create rendered states for each time step.

### Example sequence

Gold posterior trajectory:

```text
0.20 -> 0.43 -> 0.72 -> 0.91
```

Compare against model trajectory.

### Required analyses

Measure:

- pointwise posterior error,
- trajectory MAE,
- monotonicity when mathematically required,
- update direction correctness,
- final posterior error,
- path consistency.

When conditional independence assumptions are used, state them explicitly.

---

# 6. Representation Sensitivity Experiment

This is a required first-class experiment, not an optional appendix.

## 6.1 Goal

For the same latent probabilistic problem, determine whether the model returns materially different probabilities depending only on how the information is represented.

For latent instance \(z\), render representations:

\[
r_1,\dots,r_m.
\]

Obtain:

\[
\hat p(z,r_1),\dots,\hat p(z,r_m).
\]

All versions have the same exact gold distribution \(p^*(z)\).

---

## 6.2 Required representations

Implement at least:

### Direct probability

```json
{
  "p_A": 0.3
}
```

### Counts

```json
{
  "A": 30,
  "not_A": 70
}
```

### Scaled counts

```json
{
  "A": 30000,
  "not_A": 70000
}
```

### Ratio

```json
{
  "odds_A_to_not_A": "3:7"
}
```

### Table

Use an explicit structured table representation where relevant.

### Natural-language prose

Example:

```text
Thirty percent of objects in the container belong to class A.
```

### Distractor representation

Add irrelevant but plausible attributes while preserving the same sufficient statistics.

Example:

```json
{
  "A": 30,
  "not_A": 70,
  "container_temperature": 18.3,
  "operator_shift": "night",
  "batch_code": "ZX-14"
}
```

Distractors must not alter the true probability.

---

## 6.3 Representation-sensitivity metrics

### Pairwise total variation

For representations \(r_a,r_b\):

\[
D_{\mathrm{TV}}
\left(
\hat p(z,r_a),
\hat p(z,r_b)
\right)
=
\frac12
\sum_j
\left|
\hat p_j(z,r_a)-\hat p_j(z,r_b)
\right|.
\]

Report mean pairwise TV distance per latent instance.

### Representation variance

For binary problems:

\[
S_{\mathrm{var}}(z)
=
\operatorname{Var}_{r}
\left[
\hat p(A\mid z,r)
\right].
\]

### Max representation deviation

\[
S_{\max}(z)
=
\max_{r_a,r_b}
D_{\mathrm{TV}}
\left(
\hat p(z,r_a),
\hat p(z,r_b)
\right).
\]

### Gold-relative representation gap

For each representation:

\[
E(z,r)
=
D_{\mathrm{TV}}
\left(
p^*(z),
\hat p(z,r)
\right).
\]

Then report spread:

\[
\Delta_E(z)
=
\max_r E(z,r)-\min_r E(z,r).
\]

This shows whether certain representations systematically make the model less statistically correct.

---

## 6.4 Representation invariance score

Define an interpretable aggregate score:

\[
RIS
=
1
-
\frac{1}{N}
\sum_z
\operatorname{MeanPairwiseTV}(z).
\]

Keep the raw TV values in all reports.

Do not rely solely on the transformed score.

---

# 7. Metrics

The evaluator must compute both per-item and aggregate metrics.

---

## 7.1 Total Variation Distance

Primary distributional metric:

\[
D_{\mathrm{TV}}(p^*,\hat p)
=
\frac12
\sum_j |p_j^*-\hat p_j|.
\]

Aggregate:

```text
mean TV
median TV
std TV
p90 TV
p95 TV
max TV
```

---

## 7.2 Mean Absolute Error

For binary cases:

\[
MAE
=
\frac1N
\sum_i
|\hat p_i-p_i^*|.
\]

---

## 7.3 Root Mean Squared Error

\[
RMSE
=
\sqrt{
\frac1N
\sum_i
(\hat p_i-p_i^*)^2
}.
\]

---

## 7.4 Brier Regret

Because the true distribution is known exactly, expected Brier regret can be computed directly:

\[
R_{\text{Brier}}
=
\|\hat p-p^*\|_2^2.
\]

This does not require sampling outcomes.

---

## 7.5 Log-score Regret / KL Divergence

\[
R_{\log}
=
D_{\mathrm{KL}}
\left(
p^*\Vert\hat p
\right).
\]

Use epsilon clipping for returned zero probabilities:

```text
eps = 1e-12
```

Report the clipping policy clearly.

---

## 7.6 Argmax Accuracy

Secondary metric:

\[
\mathbb{1}
\left[
\arg\max_j \hat p_j
=
\arg\max_j p_j^*
\right].
\]

This must never be the primary headline metric.

A model can have correct argmax while returning statistically poor probabilities.

---

## 7.7 Probability-transfer diagnostics

For the explicit-probability family, estimate:

\[
f(p)=E[\hat p\mid p^*=p].
\]

Report:

- linear regression slope,
- intercept,
- \(R^2\),
- MAE from identity,
- maximum deviation from identity.

Plot:

```text
true probability vs returned probability
```

Include the identity line.

---

# 8. Frequentist Calibration Experiment

Probability recovery and calibration must be reported separately.

For each benchmark instance with known distribution:

\[
Y_i \sim p_i^*.
\]

Generate one or more stochastic realizations.

The evaluated model does **not** see the realized outcome.

Use returned probabilities to compute:

- empirical Brier score,
- empirical log loss,
- reliability curves,
- calibration bins,
- expected calibration error,
- calibration slope,
- calibration intercept.

Because \(p^*\) is already known, these empirical calibration metrics are secondary to direct probability recovery.

---

# 9. Model Adapter Interface

Define a common adapter abstraction.

Example:

```python
class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def predict_distribution(
        self,
        state: dict,
        prompt: str,
        outcomes: list[str],
    ) -> dict[str, float]:
        ...
```

The adapter must return a normalized mapping:

```python
{
    "A": 0.31,
    "not_A": 0.69
}
```

The core benchmark must not know whether the model is:

- Jev,
- an LLM,
- a classifier,
- a symbolic engine,
- a mock baseline.

---

# 10. Jev Adapter

Implement a dedicated Jev adapter.

The agent should consult the current Jev API/SDK documentation at implementation time instead of relying on hard-coded assumptions from this specification.

The adapter must:

1. construct the required typed decision/question,
2. pass the benchmark state,
3. request probabilities over the benchmark outcomes,
4. parse the returned distribution,
5. normalize labels to benchmark labels,
6. preserve raw API output,
7. record latency and errors,
8. record model/version metadata if available.

Store:

```json
{
  "raw_response": {},
  "parsed_distribution": {},
  "latency_ms": 0,
  "model_version": null,
  "api_metadata": {}
}
```

Do not confuse a model's separate confidence field with its returned outcome probability distribution.

If Jev exposes both, save both, but evaluate probability recovery using the outcome probabilities.

---

# 10.5 Open-Source Jev-Class Models

The benchmark should track open-source Jev-class decision models listed by Benchmark Heaven's JevBench page:

```text
https://benchmarkheaven.com/jev-models
```

The page reports JevBench v1.2 results for multiple Jev-class systems, including open rebuilds and open-source or open-weights alternatives. These systems should be added through normal black-box adapters rather than special-cased in the evaluator.

For v0.1, implement only:

```text
SemIf, formerly OpenJev, by TheoLeeCJ
Repository: https://github.com/TheoLeeCJ/SemIf
Default model: Qwen/Qwen3.5-4B
Default revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
Default interface: semif-score JSONL CLI
```

SemIf input rows should be constructed as:

```json
{
  "id": "benchmark item id",
  "state": {},
  "question": "benchmark prompt",
  "options": [
    {"id": "A", "description": "..."},
    {"id": "not_A", "description": "..."}
  ]
}
```

SemIf output rows are expected to contain either:

```json
{
  "option_ids": ["A", "not_A"],
  "probabilities": [0.31, 0.69]
}
```

or a direct probability mapping. The adapter must map SemIf option IDs back to benchmark outcome labels, normalize through the common evaluator path, preserve raw SemIf output, and record model/revision metadata.

SemIf is a local/open model path and may require a separate environment with GPU-capable PyTorch, Transformers, downloaded Hugging Face weights, and the `semif-score` CLI installed from the SemIf repository. The benchmark package should not install these heavy dependencies by default.

---

# 11. Baselines

Implement deterministic baselines to validate the evaluator.

## 11.1 Oracle

Returns \(p^*\).

Expected performance:

```text
TV = 0
MAE = 0
Brier regret = 0
```

Use this baseline as a pipeline unit test.

## 11.2 Uniform baseline

For \(K\) outcomes:

\[
\hat p_j = \frac1K.
\]

## 11.3 Argmax-only pseudo-probability baseline

Optional:

```text
winner = 1 - eps
others = eps / (K - 1)
```

This illustrates that high accuracy can coexist with poor probability quality.

## 11.4 Frequency heuristic

Where appropriate, implement a transparent heuristic baseline that extracts obvious normalized counts.

This can distinguish model failure from dataset-generation errors.

---

# 12. Dataset Size

For an initial research-quality release, target approximately:

```text
1,000 latent instances per family
```

For six families:

```text
6,000 latent instances
```

With multiple representations, rendered item count will be larger.

Suggested initial representation counts:

```text
Explicit Probability: 4 representations
Frequency: 6 representations
Compound: 3 representations
Conditional: 5 representations
Bayes: 5 representations
Sequential Bayes: 3 representations
```

If API cost is high, support stratified subsets:

```text
tiny
dev
standard
full
```

Suggested:

```text
tiny:      20 latent instances/family
dev:      100 latent instances/family
standard: 500 latent instances/family
full:     1000+ latent instances/family
```

---

# 13. Parameter Sampling

Avoid distributions that produce mostly trivial probabilities near 0 or 1.

Use stratified probability bands.

Suggested bins:

```text
[0.01, 0.10)
[0.10, 0.30)
[0.30, 0.45)
[0.45, 0.55]
(0.55, 0.70]
(0.70, 0.90]
(0.90, 0.99]
```

Ensure meaningful coverage of:

- near-decision-boundary cases,
- moderate probabilities,
- extreme probabilities.

For Bayes, stratify by final posterior as well as by prior.

---

# 14. Numerical Precision

The benchmark must retain high-precision gold values internally.

Use Python `decimal.Decimal` or exact rational arithmetic where convenient.

Serialized gold probabilities should retain enough precision to avoid evaluator-induced error.

If an evaluated API rounds probabilities to a fixed number of decimals, do not silently compensate.

Instead report:

1. raw error,
2. optional quantization-aware error.

For a system returning two decimal places, a quantization-aware tolerance may use approximately:

\[
\pm 0.005.
\]

Do not replace raw metrics with tolerance-based metrics.

---

# 15. Repeated Queries and Stochasticity

Some models may be stochastic.

Support repeated calls per rendered item:

```text
n_repeats = configurable
```

Recommended default:

```text
n_repeats = 1
```

Recommended stochasticity study:

```text
n_repeats = 10
```

For repeated queries compute:

- mean returned distribution,
- standard deviation,
- within-item TV variation,
- between-repeat entropy,
- representation sensitivity after averaging.

This distinguishes representation sensitivity from model sampling noise.

---

# 16. Output Schema

Each model result row should include:

```json
{
  "model": "jev",
  "model_version": "...",
  "instance_id": "...",
  "latent_instance_id": "...",
  "family": "bayes",
  "representation": "counts",

  "gold_distribution": {
    "A": 0.3076923077,
    "not_A": 0.6923076923
  },

  "predicted_distribution": {
    "A": 0.31,
    "not_A": 0.69
  },

  "metrics": {
    "tv": 0.0023076923,
    "mae_binary": 0.0023076923,
    "brier_regret": 0.0000106509,
    "kl": 0.000012
  },

  "predicted_argmax": "not_A",
  "gold_argmax": "not_A",
  "argmax_correct": true,

  "latency_ms": 0,
  "repeat_index": 0,
  "error": null,
  "raw_response": {}
}
```

Write results incrementally so long API runs can resume after failure.

---

# 17. Evaluation Pipeline

Implement the benchmark as a resumable pipeline.

Recommended flow:

```text
1. generate latent instances
2. render equivalent representations
3. validate mathematical ground truth
4. save dataset
5. load model adapter
6. query model
7. save raw responses immediately
8. parse probability distributions
9. validate normalization
10. compute per-item metrics
11. aggregate by:
    - model
    - family
    - difficulty
    - representation
    - probability band
12. compute representation-sensitivity metrics
13. compute calibration metrics
14. generate figures/tables
15. build final report
```

---

# 18. Validation Rules

Before querying any external model, validate every dataset item.

## 18.1 Distribution validation

Require:

\[
p_j \ge 0
\]

and

\[
\left|\sum_j p_j - 1\right| < 10^{-10}.
\]

## 18.2 Representation equivalence

For every latent instance, all rendered versions must have identical gold distributions within numerical tolerance.

## 18.3 Independent recomputation

Where practical, compute gold values through two independent implementations for difficult families.

Example:

- analytical Bayes formula,
- generic normalization implementation.

## 18.4 Oracle test

The Oracle adapter must achieve zero error.

If it does not, the benchmark pipeline is wrong.

---

# 19. Reporting

Produce both machine-readable and human-readable outputs.

## 19.1 Tables

Required tables:

### Overall model comparison

| Model | Mean TV | Median TV | Brier Regret | KL | Argmax Acc. |
|---|---:|---:|---:|---:|---:|

### By family

| Model | Family | Mean TV | MAE | Brier Regret | Argmax Acc. |
|---|---|---:|---:|---:|---:|

### By representation

| Model | Representation | Mean TV | Mean Gold Error |
|---|---|---:|---:|

### Representation sensitivity

| Model | Mean Pairwise TV | Mean Max TV | RIS |
|---|---:|---:|---:|

---

# 20. Required Figures

Generate at least the following.

## Figure 1 — Probability transfer curve

For explicit-probability items:

```text
x-axis: true probability
y-axis: returned probability
```

Include identity line.

## Figure 2 — Error by family

Boxplot or violin-style equivalent of TV error by family.

## Figure 3 — Representation sensitivity

For each representation type, show gold-relative error.

## Figure 4 — Equivalent-state spread

For selected latent instances, show returned probabilities across representations.

Example:

```text
direct       0.30
counts       0.31
scaled       0.42
ratio        0.28
prose        0.36
distractor   0.47
```

## Figure 5 — Sequential Bayes trajectories

Plot:

```text
gold posterior trajectory
model posterior trajectory
```

## Figure 6 — Calibration curve

Use stochastic realizations.

---

# 21. Statistical Comparisons Between Models

The comparison pipeline should support multiple models.

For paired benchmark instances, compare model errors using paired tests or bootstrap confidence intervals.

Recommended primary comparison:

\[
\Delta_i
=
D_{\mathrm{TV}}(p_i^*,\hat p_i^{(A)})
-
D_{\mathrm{TV}}(p_i^*,\hat p_i^{(B)}).
\]

Report:

- mean paired difference,
- median paired difference,
- bootstrap 95% CI,
- win/tie/loss rate.

Avoid relying only on a single aggregate score.

---

# 22. Experiment Matrix

At minimum, run the following experiments.

## Experiment E0 — Explicit Probability Recovery

Question:

> If the correct probability is directly present in the state, does the model reproduce it?

Outputs:

- transfer curve,
- MAE,
- TV,
- slope/intercept.

---

## Experiment E1 — Frequency Recovery

Question:

> Does the model recover probabilities from normalized counts?

Vary:

- numerator,
- denominator,
- scale,
- number of categories.

---

## Experiment E2 — Compound Probability

Question:

> Does the model correctly combine simple probabilities?

Subtasks:

- complement,
- independent intersection,
- union,
- repeated trials,
- sampling without replacement.

---

## Experiment E3 — Conditional Probability

Question:

> Can the model identify the correct conditioning population and return the correct conditional probability?

Include distractor reference classes.

---

## Experiment E4 — Bayesian Posterior Recovery

Question:

> Given complete prior and likelihood information, does the model return the statistically correct posterior?

Include binary and multiclass Bayes.

---

## Experiment E5 — Sequential Bayesian Updating

Question:

> As evidence accumulates, does the model's probability trajectory follow the exact Bayesian posterior trajectory?

---

## Experiment E6 — Representation Sensitivity

Question:

> Holding the latent probabilistic problem fixed, how much does the returned distribution vary across equivalent representations?

Required representations:

- direct probability,
- counts,
- scaled counts,
- ratio,
- table where applicable,
- prose,
- distractor-enriched state.

Primary outputs:

- mean pairwise TV,
- max pairwise TV,
- representation variance,
- gold-relative representation gap,
- representation invariance score.

---

## Experiment E7 — Distractor Robustness

Question:

> Does statistically irrelevant information alter the model's returned probabilities?

Create increasing distractor levels:

```text
0 distractors
2 distractors
5 distractors
10 distractors
```

Measure probability drift from the clean representation.

---

## Experiment E8 — Repeated-Query Stability

Question:

> Is the same state assigned stable probabilities across repeated identical queries?

Run repeated calls on a representative subset.

Measure:

- within-item standard deviation,
- within-item pairwise TV,
- probability range.

---

# 23. Optional Advanced Experiments

These can be implemented after the core benchmark works.

## 23.1 Label permutation invariance

Rename outcomes without changing the latent structure.

Example:

```text
A / B
```

becomes:

```text
zor / nif
```

Probabilities should permute accordingly.

## 23.2 Option-order invariance

Change only the ordering of answer options.

## 23.3 Numerical-format invariance

Equivalent forms:

```text
0.3
30%
3/10
30 out of 100
```

## 23.4 Irrelevant causal narrative

Wrap the same sufficient statistics in different domain stories.

## 23.5 Missing-information controls

Deliberately omit required information.

These should be separated from the core benchmark because the gold posterior may no longer be uniquely defined.

The model should ideally signal uncertainty or insufficiency instead of inventing precision.

---

# 24. Configuration

Support YAML configuration.

Example:

```yaml
benchmark:
  version: "0.1.0"
  seed: 42

dataset:
  preset: "standard"
  families:
    - explicit_probability
    - frequency
    - compound
    - conditional
    - bayes
    - sequential_bayes

representations:
  direct: true
  counts: true
  scaled_counts: true
  ratio: true
  table: true
  prose: true
  distractor: true

evaluation:
  repeats: 1
  kl_epsilon: 1.0e-12
  save_raw_responses: true

models:
  - name: jev
    adapter: jev
```

Secrets must come from environment variables, never committed configuration files.

---

# 25. CLI

Provide commands similar to:

```bash
jev-prob-bench generate \
  --config config/benchmark.yaml \
  --output datasets/generated/v0.1.0.jsonl
```

```bash
jev-prob-bench run \
  --dataset datasets/generated/v0.1.0.jsonl \
  --models config/models.yaml \
  --output results/run_001/
```

```bash
jev-prob-bench report \
  --results results/run_001/ \
  --output results/run_001/report/
```

Also support resume:

```bash
jev-prob-bench run \
  --resume results/run_001/
```

---

# 26. Testing Requirements

Implement unit tests for:

- every generator,
- every renderer,
- every gold-probability calculation,
- normalization,
- metric calculations,
- representation pairing,
- Oracle adapter,
- checkpoint/resume logic.

Add property-based tests where useful.

Examples:

### Binary complement property

\[
P(A)+P(\neg A)=1.
\]

### Bayes normalization

\[
\sum_i P(H_i\mid E)=1.
\]

### Representation equivalence

All renderers for the same latent instance must share exactly the same gold distribution.

---

# 27. Acceptance Criteria for v0.1

The initial implementation is complete when:

1. All six core families are implemented.
2. At least 100 latent instances per family can be generated reproducibly.
3. Equivalent representations are paired through `latent_instance_id`.
4. Representation sensitivity is fully implemented.
5. Oracle and Uniform baselines work.
6. Jev can be evaluated through an adapter.
7. The pipeline resumes safely after interruption.
8. Raw responses and parsed responses are stored.
9. TV, MAE, RMSE, Brier regret, KL, and argmax accuracy are computed.
10. Representation-sensitivity metrics are computed.
11. Frequentist calibration can be run on simulated outcomes.
12. Standard comparison tables and required figures are automatically generated.
13. Unit tests pass.
14. A README explains reproduction from dataset generation through final report.

---

# 28. Scientific Interpretation

The benchmark should avoid overclaiming.

A low probability-recovery error supports the claim:

> Given a state that fully specifies a known stochastic process, the model's returned probability distribution closely matches the mathematically correct conditional distribution.

It does **not** establish how the model internally obtained that probability.

A high probability-recovery error supports the claim:

> The model's returned probability should not be interpreted naively as the exact probability implied by the supplied state, at least for the tested problem family.

Representation sensitivity adds a separate diagnostic:

> Even when two states encode the same probabilistic information, the model may assign different probabilities depending on representation.

This distinction should appear explicitly in the final benchmark report.

---

# 29. Primary Research Questions

The final benchmark should answer:

### RQ1 — Probability recovery

When the state contains sufficient information to determine a probability exactly, how close is the model's returned distribution to the true distribution?

### RQ2 — Complexity

How does probability-recovery error change from explicit probability to frequency, conditional probability, Bayes, and sequential Bayes?

### RQ3 — Representation sensitivity

How invariant are returned probabilities to mathematically equivalent representations of the same state?

### RQ4 — Calibration

Are returned probabilities empirically calibrated when outcomes are sampled from the known generating process?

### RQ5 — Stability

Does the same model assign stable probabilities to identical repeated queries?

### RQ6 — Model comparison

Which evaluated models most closely reproduce the known probability distributions, and on which problem families do their behaviors diverge?

---

# 30. Recommended First Release

For the first meaningful release, prioritize correctness and interpretability over breadth.

Recommended order:

```text
1. Explicit Probability
2. Frequency
3. Conditional Probability
4. Bayes
5. Sequential Bayes
6. Compound Probability
7. Representation Sensitivity
8. Calibration
9. Multi-model comparison/reporting
```

A strong first paper-quality benchmark does not require thousands of task templates.

It requires:

- exact probabilistic ground truth,
- procedural variation,
- controlled equivalent representations,
- reproducible model calls,
- transparent metrics,
- paired model comparison,
- and clear separation between probability recovery and calibration.
