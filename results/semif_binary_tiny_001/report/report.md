# Jev Probability Benchmark Report

Log-score regret uses epsilon clipping at 1e-12. Argmax accuracy is secondary; TV distance is the primary recovery metric. For repeated Jev runs, headline tables use the per-question mean prediction over repeats unless stated otherwise.

## Repeat Coverage
| model | n_items | min_repeats | median_repeats | max_repeats | mean_repeats |
| --- | --- | --- | --- | --- | --- |
| semif_qwen35_4b | 365 | 1 | 1 | 1 | 1 |

## Repeat-Aggregated Model Comparison
| repeat_aggregation | model | n | mean_tv | median_tv | mean_brier_regret | mean_kl | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean | semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |
| median | semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |

## Overall Model Comparison
| model | n | mean_tv | median_tv | mean_brier_regret | mean_kl | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |

## By Family
| model | family | n | mean_tv | mean_mae | mean_brier_regret | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| semif_qwen35_4b | bayes | 35 | 0.297581 | 0.297581 | 0.334318 | 0.571429 |
| semif_qwen35_4b | compound | 60 | 0.51476 | 0.51476 | 0.678121 | 0.266667 |
| semif_qwen35_4b | conditional | 100 | 0.257032 | 0.257032 | 0.188647 | 0.6 |
| semif_qwen35_4b | explicit_probability | 80 | 0.42038 | 0.42038 | 0.484699 | 0.525 |
| semif_qwen35_4b | frequency | 30 | 0.314929 | 0.314929 | 0.271692 | 0.4 |
| semif_qwen35_4b | sequential_bayes | 60 | 0.420967 | 0.420967 | 0.500263 | 0.45 |

## By Representation
| model | representation | n | mean_tv | mean_brier_regret |
| --- | --- | --- | --- | --- |
| semif_qwen35_4b | counts | 12 | 0.25588 | 0.189633 |
| semif_qwen35_4b | direct | 87 | 0.393729 | 0.457837 |
| semif_qwen35_4b | distractor | 52 | 0.344391 | 0.344643 |
| semif_qwen35_4b | nested | 20 | 0.229195 | 0.158033 |
| semif_qwen35_4b | prose | 92 | 0.406028 | 0.481358 |
| semif_qwen35_4b | ratio | 25 | 0.376374 | 0.367954 |
| semif_qwen35_4b | scaled_counts | 5 | 0.274477 | 0.194024 |
| semif_qwen35_4b | table | 72 | 0.380375 | 0.424336 |

## Representation Sensitivity
| model | mean_pairwise_tv | mean_max_tv | ris | mean_gold_relative_gap |
| --- | --- | --- | --- | --- |
| semif_qwen35_4b | 0.117446 | 0.19365 | 0.882554 | 0.180125 |

## Power-Law Calibration
| model | n | alpha | identity_mae | power_mae_to_returned | inverse_calibrated_mae | identity_rmse | power_rmse_to_returned | inverse_calibrated_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semif_qwen35_4b | 365 | 21.2 | 0.370796 | 0.282099 | 0.470686 | 0.450564 | 0.404482 | 0.551058 |

## Noul-Choice-Score Evaluation
_No complete Noul/Choice/Score triples found._

## Calibration
ECE: 0.354363

Empirical Brier: 0.364694

Empirical log loss: 1.09925

## Figures
- results/semif_binary_tiny_001/report/figures/figure_1_probability_transfer.png
- results/semif_binary_tiny_001/report/figures/figure_16_powerlaw_calibration.png
- results/semif_binary_tiny_001/report/figures/figure_16b_powerlaw_calibration_explicit_probability.png
- results/semif_binary_tiny_001/report/figures/figure_2_error_by_family.png
- results/semif_binary_tiny_001/report/figures/figure_3_representation_sensitivity.png
- results/semif_binary_tiny_001/report/figures/figure_5_sequential_bayes_trajectories.png
- results/semif_binary_tiny_001/report/figures/figure_6_calibration_curve.png
- results/semif_binary_tiny_001/report/figures/figure_15_calibration_by_representation.png
