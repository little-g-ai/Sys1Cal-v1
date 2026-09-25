# Semantic Experiments

JevCal's core benchmark is fixed pointwise probability fidelity, representation sensitivity, and Noul/Choice/Score consistency. The power-law and unknown-mass analyses are included as research experiments for studying Jev semantics, not as leaderboard requirements.

## Power-Law Mapping

The Noul-to-Choice experiment asks whether binary Choice behaves like the same event probability as Noul. The descriptive mapping is:

```text
p_choice ~= 1 - (1 - p_noul) ** alpha
```

The report writes the fitted result to `powerlaw_calibration.csv` and the richer Noul-to-Choice mapping comparison to `noul_choice_calibration.csv`.

## Score-Derived Unknown Mass

The unknown-mass experiment treats a Score distribution as evidence for three operational masses:

```text
T + U + F = 1
p_choice ~= T / (T + F) = T / (1 - U)
```

Here `T` is true-like mass, `F` is false-like mass, and `U` is an implied middle or unknown mass. This is a black-box behavioral model. It should not be read as a claim that the evaluated model internally stores a three-valued truth representation.

The report writes the relevant Python experiment outputs to:

- `score_unknown_mass_records.csv`
- `score_unknown_mass_summary.csv`
- `score_three_state_summary.csv`
- `score_three_state_predictions.csv`
- `score_choice_tuf_summary.csv`
- `score_choice_tuf_predictions.csv`
- `choice_from_score_recovery.csv`
- `score_choice_entropy_scan.csv`

The implementation lives in [../src/jev_prob_bench/evaluation/noul_choice_score.py](../src/jev_prob_bench/evaluation/noul_choice_score.py).
