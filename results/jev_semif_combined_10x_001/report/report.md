# Jev Probability Benchmark Report

Log-score regret uses epsilon clipping at 1e-12. Argmax accuracy is secondary; TV distance is the primary recovery metric. For repeated Jev runs, headline tables use the per-question mean prediction over repeats unless stated otherwise.

## Benchmark Summary
| model | response_type | n | argmax_accuracy | pointwise_probability_fidelity | mean_tv_error | noul_choice_r2 | noul_score_r2 | score_choice_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev | noul_like | 365 | 0.893151 | 0.918318 | 0.0816815 | 0.78129 | 0.920978 | 0.786213 |
| jev | choice_like | 365 | 0.693151 | 0.763756 | 0.236244 | 0.78129 | 0.920978 | 0.786213 |
| jev | score_like | 365 | 0.893151 | 0.886134 | 0.113866 | 0.78129 | 0.920978 | 0.786213 |
| jev | overall | 1095 | 0.826484 | 0.856069 | 0.143931 | 0.78129 | 0.920978 | 0.786213 |
| semif_qwen35_4b | overall | 365 | 0.484932 | 0.629204 | 0.370796 |  |  |  |

## Repeat Coverage
| model | n_items | min_repeats | median_repeats | max_repeats | mean_repeats |
| --- | --- | --- | --- | --- | --- |
| jev_choice | 365 | 10 | 10 | 10 | 10 |
| jev_noul | 365 | 10 | 10 | 10 | 10 |
| jev_score | 365 | 10 | 10 | 10 | 10 |
| semif_qwen35_4b | 365 | 10 | 10 | 10 | 10 |

## Repeat-Aggregated Model Comparison
| repeat_aggregation | model | n | mean_tv | median_tv | mean_brier_regret | mean_kl | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean | jev_choice | 365 | 0.236244 | 0.229158 | 0.156661 | 0.24255 | 0.693151 |
| mean | jev_noul | 365 | 0.0816815 | 0.0508143 | 0.03393 | 0.0467231 | 0.893151 |
| mean | jev_score | 365 | 0.113866 | 0.0813275 | 0.0516731 | 0.0698758 | 0.893151 |
| mean | semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |
| median | jev_choice | 365 | 0.236497 | 0.233158 | 0.157001 | 0.243704 | 0.690411 |
| median | jev_noul | 365 | 0.0815982 | 0.0516161 | 0.033869 | 0.0465904 | 0.887671 |
| median | jev_score | 365 | 0.113822 | 0.0793356 | 0.0517902 | 0.0699902 | 0.893151 |
| median | semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |

## Overall Model Comparison
| model | n | mean_tv | median_tv | mean_brier_regret | mean_kl | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| jev_choice | 365 | 0.236244 | 0.229158 | 0.156661 | 0.24255 | 0.693151 |
| jev_noul | 365 | 0.0816815 | 0.0508143 | 0.03393 | 0.0467231 | 0.893151 |
| jev_score | 365 | 0.113866 | 0.0813275 | 0.0516731 | 0.0698758 | 0.893151 |
| semif_qwen35_4b | 365 | 0.370796 | 0.335335 | 0.406015 | 0.631981 | 0.484932 |

## By Family
| model | family | n | mean_tv | mean_mae | mean_brier_regret | argmax_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| jev_choice | bayes | 35 | 0.198286 | 0.198286 | 0.124442 | 0.771429 |
| jev_choice | compound | 60 | 0.322786 | 0.322786 | 0.280949 | 0.483333 |
| jev_choice | conditional | 100 | 0.239023 | 0.239023 | 0.136748 | 0.63 |
| jev_choice | explicit_probability | 80 | 0.251161 | 0.251161 | 0.177493 | 0.6875 |
| jev_choice | frequency | 30 | 0.208367 | 0.208367 | 0.107197 | 0.8 |
| jev_choice | sequential_bayes | 60 | 0.161266 | 0.161266 | 0.0813137 | 0.916667 |
| jev_noul | bayes | 35 | 0.133445 | 0.133445 | 0.0674896 | 0.885714 |
| jev_noul | compound | 60 | 0.204582 | 0.204582 | 0.130583 | 0.716667 |
| jev_noul | conditional | 100 | 0.0478846 | 0.0478846 | 0.00670741 | 0.86 |
| jev_noul | explicit_probability | 80 | 0.0361191 | 0.0361191 | 0.00389509 | 1 |
| jev_noul | frequency | 30 | 0.0299 | 0.0299 | 0.0029118 | 1 |
| jev_noul | sequential_bayes | 60 | 0.071555 | 0.071555 | 0.0186275 | 0.933333 |
| jev_score | bayes | 35 | 0.159658 | 0.159658 | 0.0798813 | 0.885714 |
| jev_score | compound | 60 | 0.248942 | 0.248942 | 0.169515 | 0.633333 |
| jev_score | conditional | 100 | 0.0896001 | 0.0896001 | 0.0305507 | 0.91 |
| jev_score | explicit_probability | 80 | 0.0532878 | 0.0532878 | 0.00877574 | 1 |
| jev_score | frequency | 30 | 0.0447903 | 0.0447903 | 0.00579665 | 1 |
| jev_score | sequential_bayes | 60 | 0.107827 | 0.107827 | 0.0327154 | 0.933333 |
| semif_qwen35_4b | bayes | 35 | 0.297581 | 0.297581 | 0.334318 | 0.571429 |
| semif_qwen35_4b | compound | 60 | 0.51476 | 0.51476 | 0.678121 | 0.266667 |
| semif_qwen35_4b | conditional | 100 | 0.257032 | 0.257032 | 0.188647 | 0.6 |
| semif_qwen35_4b | explicit_probability | 80 | 0.42038 | 0.42038 | 0.484699 | 0.525 |
| semif_qwen35_4b | frequency | 30 | 0.314929 | 0.314929 | 0.271692 | 0.4 |
| semif_qwen35_4b | sequential_bayes | 60 | 0.420967 | 0.420967 | 0.500263 | 0.45 |

## By Representation
| model | representation | n | mean_tv | mean_brier_regret |
| --- | --- | --- | --- | --- |
| jev_choice | counts | 12 | 0.181813 | 0.0840367 |
| jev_choice | direct | 87 | 0.235492 | 0.16118 |
| jev_choice | distractor | 52 | 0.262942 | 0.170949 |
| jev_choice | nested | 20 | 0.255373 | 0.158964 |
| jev_choice | prose | 92 | 0.272232 | 0.190266 |
| jev_choice | ratio | 25 | 0.143194 | 0.0696234 |
| jev_choice | scaled_counts | 5 | 0.163 | 0.062682 |
| jev_choice | table | 72 | 0.213042 | 0.146155 |
| jev_noul | counts | 12 | 0.0867843 | 0.0306905 |
| jev_noul | direct | 87 | 0.091339 | 0.0407282 |
| jev_noul | distractor | 52 | 0.0539625 | 0.0169347 |
| jev_noul | nested | 20 | 0.0503774 | 0.0066904 |
| jev_noul | prose | 92 | 0.0902 | 0.0326576 |
| jev_noul | ratio | 25 | 0.0264295 | 0.00209864 |
| jev_noul | scaled_counts | 5 | 0.0154 | 0.0005404 |
| jev_noul | table | 72 | 0.110779 | 0.0610938 |
| jev_score | counts | 12 | 0.099469 | 0.0363343 |
| jev_score | direct | 87 | 0.124917 | 0.055689 |
| jev_score | distractor | 52 | 0.0933582 | 0.0346205 |
| jev_score | nested | 20 | 0.0903445 | 0.0317692 |
| jev_score | prose | 92 | 0.1222 | 0.0580893 |
| jev_score | ratio | 25 | 0.0411671 | 0.00480142 |
| jev_score | scaled_counts | 5 | 0.0296 | 0.00282526 |
| jev_score | table | 72 | 0.1447 | 0.0786901 |
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
| jev_choice | 0.0998297 | 0.181076 | 0.90017 | 0.148703 |
| jev_noul | 0.0440591 | 0.0786739 | 0.955941 | 0.0690187 |
| jev_score | 0.0656071 | 0.115899 | 0.934393 | 0.099603 |
| semif_qwen35_4b | 0.117446 | 0.19365 | 0.882554 | 0.180125 |

## Power-Law Calibration
| model | n | alpha | identity_mae | power_mae_to_returned | inverse_calibrated_mae | identity_rmse | power_rmse_to_returned | inverse_calibrated_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev_choice | 365 | 2.57 | 0.236244 | 0.108939 | 0.111942 | 0.279876 | 0.173332 | 0.153055 |
| jev_noul | 365 | 1.07 | 0.0816815 | 0.0797883 | 0.07892 | 0.13025 | 0.129051 | 0.126072 |
| jev_score | 365 | 0.97 | 0.113866 | 0.114005 | 0.114606 | 0.160737 | 0.160617 | 0.161843 |
| semif_qwen35_4b | 365 | 21.2 | 0.370796 | 0.282099 | 0.470686 | 0.450564 | 0.404482 | 0.551058 |

## Noul-Choice-Score Evaluation

### External Fidelity and Cross-Primitive Consistency
| section | metric | n | mae | rmse | mean_tv | mean | mean_abs | median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| external_probability_fidelity | noul_mae_to_p_star | 365 | 0.0816815 | 0.13025 | 0.0816815 |  |  | 0.0508143 |
| external_probability_fidelity | choice_mae_to_p_star | 365 | 0.236244 | 0.279876 | 0.236244 |  |  | 0.229158 |
| external_probability_fidelity | score_expectation_mae_to_p_star | 365 | 0.113866 | 0.160737 | 0.113866 |  |  | 0.0813275 |
| cross_primitive_consistency | abs_noul_choice | 365 | 0.192268 | 0.221132 | 0.192268 |  |  | 0.189 |
| cross_primitive_consistency | abs_noul_score_expectation | 365 | 0.0541057 | 0.0722247 | 0.0541057 |  |  | 0.0383951 |
| cross_primitive_consistency | abs_choice_score_expectation | 365 | 0.206718 | 0.23735 | 0.206718 |  |  | 0.209667 |
| choice_extremeness | choice_minus_noul_extremeness | 365 |  | 0.182794 |  | 0.10074 | 0.156603 | 0.125 |
| noul_choice_bias | choice_minus_noul_probability | 365 |  | 0.221132 |  | 0.178959 | 0.192268 | 0.189 |

### Score Projection Models
| collapse_rule | target | n_test | mae | rmse | r2 | beta | fuzzy_beta | unknown_lambda | unknown_gamma | false_max_level | true_min_level | n_test_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expected_truth | noul | 89 | 0.0534971 | 0.0780345 | 0.861122 |  |  |  |  |  |  |  |
| hard_threshold | noul | 89 | 0.127865 | 0.161526 | 0.404958 |  |  |  |  |  |  |  |
| endpoint_conditioning | noul | 71 | 0.161176 | 0.203236 | 0.153679 |  |  |  |  |  |  | 71 |
| parametric_sharpening | noul | 89 | 0.0553559 | 0.0747957 | 0.872411 | 1.25 |  |  |  |  |  |  |
| monotonic_collapse | noul | 89 | 0.0562711 | 0.0750821 | 0.871432 |  |  |  |  |  |  |  |
| full_q_ridge | noul | 89 | 0.0577942 | 0.0765848 | 0.866234 |  |  |  |  |  |  |  |
| expected_truth | choice | 89 | 0.22125 | 0.244161 | -0.0302512 |  |  |  |  |  |  |  |
| hard_threshold | choice | 89 | 0.16313 | 0.225628 | 0.120214 |  |  |  |  |  |  |  |
| endpoint_conditioning | choice | 71 | 0.322815 | 0.373781 | -1.48372 |  |  |  |  |  |  | 71 |
| parametric_sharpening | choice | 89 | 0.197341 | 0.226215 | 0.115634 | 1.5 |  |  |  |  |  |  |
| monotonic_collapse | choice | 89 | 0.0701356 | 0.104378 | 0.811717 |  |  |  |  |  |  |  |
| full_q_ridge | choice | 89 | 0.0735989 | 0.1104 | 0.789365 |  |  |  |  |  |  |  |
| crisp_unknown_conditional | choice | 89 | 0.0769053 | 0.117506 | 0.761379 |  |  |  |  | 1 | 4 |  |
| fuzzy_unknown_conditional | choice | 89 | 0.197341 | 0.226215 | 0.115634 |  | 1.5 | 0 | 0.5 |  |  |  |

### Score-Derived Unknown Mass
| measure | n_test | mae | rmse | r2 | corr | mean_predicted_unknown | mean_implied_unknown | gamma | central_lambda | weight_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_fuzzy_mass | 89 | 0.394178 | 0.421045 | -7.21491 | 0.558824 | 0.634866 | 0.240688 |  |  |  |
| score_entropy_mass | 89 | 0.441747 | 0.465096 | -9.02377 | 0.554587 | 0.682436 | 0.240688 |  |  |  |
| score_gamma_fuzzy_mass | 89 | 0.110423 | 0.161118 | -0.20292 | 0.4579 | 0.243892 | 0.240688 | 8 |  |  |
| score_central_2_mass | 89 | 0.153166 | 0.200745 | -0.867399 | 0.31493 | 0.136896 | 0.240688 |  |  |  |
| score_central_decay_mass | 89 | 0.11346 | 0.168109 | -0.309573 | 0.464553 | 0.294598 | 0.240688 |  | 0.39 |  |
| score_middle_2_7_mass | 89 | 0.444282 | 0.480676 | -9.70659 | 0.53418 | 0.680335 | 0.240688 |  |  |  |
| score_crisp_unknown_2_3_mass | 89 | 0.142554 | 0.182257 | -0.539274 | 0.42709 | 0.167374 | 0.240688 |  |  |  |
| score_non_extreme_mass | 89 | 0.617893 | 0.637409 | -17.8271 | 0.446334 | 0.858581 | 0.240688 |  |  |  |
| learned_score_unknown_weights | 89 | 0.0984118 | 0.147968 | -0.0145661 | 0.488829 | 0.284612 | 0.240688 |  |  | 0.000,0.121,0.652,0.000,0.954,1.000,0.203,0.266,0.000,0.000 |
| learned_symmetric_monotone_weights | 89 | 0.107424 | 0.159001 | -0.17151 | 0.421884 | 0.29353 | 0.240688 |  |  | 0.000,0.161,0.236,0.236,1.000,1.000,0.236,0.236,0.161,0.000 |

### Score-Derived Unknown Mass by Family
| family | measure | n_test | mae | rmse | r2 | corr | mean_predicted_unknown | mean_implied_unknown | gamma | central_lambda | weight_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayes | score_fuzzy_mass | 15 | 0.367243 | 0.378444 | -7.90129 | 0.695757 | 0.596872 | 0.229628 |  |  |  |
| bayes | score_entropy_mass | 15 | 0.431277 | 0.441804 | -11.1313 | 0.668058 | 0.660905 | 0.229628 |  |  |  |
| bayes | score_gamma_fuzzy_mass | 15 | 0.0724931 | 0.0889968 | 0.507736 | 0.75204 | 0.235625 | 0.229628 | 5 |  |  |
| bayes | score_central_2_mass | 15 | 0.135477 | 0.155182 | -0.496682 | 0.724409 | 0.101802 | 0.229628 |  |  |  |
| bayes | score_central_decay_mass | 15 | 0.0726289 | 0.0868872 | 0.530796 | 0.744175 | 0.230988 | 0.229628 |  | 0.39 |  |
| bayes | score_middle_2_7_mass | 15 | 0.369013 | 0.384825 | -8.20398 | 0.629543 | 0.598642 | 0.229628 |  |  |  |
| bayes | score_crisp_unknown_2_3_mass | 15 | 0.14666 | 0.16402 | -0.672032 | 0.45601 | 0.166735 | 0.229628 |  |  |  |
| bayes | score_non_extreme_mass | 15 | 0.682233 | 0.694959 | -29.017 | -0.156086 | 0.911862 | 0.229628 |  |  |  |
| bayes | learned_score_unknown_weights | 15 | 0.0988118 | 0.114916 | 0.179246 | 0.55626 | 0.202941 | 0.229628 |  |  | 0.000,0.108,0.913,0.000,0.198,1.000,0.000,0.119,0.066,0.000 |
| bayes | learned_symmetric_monotone_weights | 15 | 0.0742309 | 0.0885913 | 0.512211 | 0.718528 | 0.234724 | 0.229628 |  |  | 0.000,0.164,0.164,0.164,1.000,1.000,0.164,0.164,0.164,0.000 |
| compound | score_fuzzy_mass | 9 | 0.261482 | 0.27208 | -30.8393 | 0.698313 | 0.322945 | 0.0614634 |  |  |  |
| compound | score_entropy_mass | 9 | 0.293693 | 0.305788 | -39.2171 | 0.72117 | 0.355156 | 0.0614634 |  |  |  |
| compound | score_gamma_fuzzy_mass | 9 | 0.0796048 | 0.0886678 | -2.38145 | 0.419507 | 0.137314 | 0.0614634 | 5 |  |  |
| compound | score_central_2_mass | 9 | 0.040145 | 0.0448761 | 0.133834 | 0.389793 | 0.0550175 | 0.0614634 |  |  |  |
| compound | score_central_decay_mass | 9 | 0.102787 | 0.109575 | -4.16411 | 0.650918 | 0.16425 | 0.0614634 |  | 0.49 |  |
| compound | score_middle_2_7_mass | 9 | 0.262302 | 0.281577 | -33.1009 | 0.516153 | 0.323765 | 0.0614634 |  |  |  |
| compound | score_crisp_unknown_2_3_mass | 9 | 0.0629899 | 0.0686607 | -1.02762 | -0.493546 | 0.0790172 | 0.0614634 |  |  |  |
| compound | score_non_extreme_mass | 9 | 0.419447 | 0.439652 | -82.1358 | 0.775124 | 0.48091 | 0.0614634 |  |  |  |
| compound | learned_score_unknown_weights | 9 | 0.0857739 | 0.0996311 | -3.26934 | 0.0822512 | 0.140905 | 0.0614634 |  |  | 0.000,0.616,0.453,0.646,1.000,1.000,0.356,0.000,0.000,0.000 |
| compound | learned_symmetric_monotone_weights | 9 | 0.103413 | 0.108551 | -4.06801 | 0.73308 | 0.164876 | 0.0614634 |  |  | 0.000,0.187,0.187,0.475,1.000,1.000,0.475,0.187,0.187,0.000 |
| conditional | score_fuzzy_mass | 30 | 0.401077 | 0.414934 | -12.2799 | 0.523013 | 0.669631 | 0.268554 |  |  |  |
| conditional | score_entropy_mass | 30 | 0.441072 | 0.453203 | -14.8425 | 0.506425 | 0.709626 | 0.268554 |  |  |  |
| conditional | score_gamma_fuzzy_mass | 30 | 0.0898046 | 0.112026 | 0.0319993 | 0.581497 | 0.299437 | 0.268554 | 8 |  |  |
| conditional | score_central_2_mass | 30 | 0.112772 | 0.141586 | -0.546253 | 0.529367 | 0.175263 | 0.268554 |  |  |  |
| conditional | score_central_decay_mass | 30 | 0.0895616 | 0.110866 | 0.0519506 | 0.573499 | 0.297913 | 0.268554 |  | 0.3 |  |
| conditional | score_middle_2_7_mass | 30 | 0.458301 | 0.477816 | -16.61 | 0.467955 | 0.726855 | 0.268554 |  |  |  |
| conditional | score_crisp_unknown_2_3_mass | 30 | 0.0997785 | 0.119848 | -0.107892 | 0.682714 | 0.188296 | 0.268554 |  |  |  |
| conditional | score_non_extreme_mass | 30 | 0.584123 | 0.593917 | -26.2075 | 0.337811 | 0.852677 | 0.268554 |  |  |  |
| conditional | learned_score_unknown_weights | 30 | 0.060336 | 0.0792296 | 0.515814 | 0.755771 | 0.269632 | 0.268554 |  |  | 0.000,0.441,0.944,0.149,0.114,0.480,0.338,0.204,0.119,0.000 |
| conditional | learned_symmetric_monotone_weights | 30 | 0.0699264 | 0.0956898 | 0.293734 | 0.547924 | 0.27548 | 0.268554 |  |  | 0.000,0.082,0.286,0.334,0.523,0.523,0.334,0.286,0.082,0.000 |
| explicit_probability | score_fuzzy_mass | 20 | 0.448577 | 0.486776 | -8.67528 | 0.00125458 | 0.763884 | 0.315306 |  |  |  |
| explicit_probability | score_entropy_mass | 20 | 0.49283 | 0.524885 | -10.2495 | -0.0192728 | 0.808137 | 0.315306 |  |  |  |
| explicit_probability | score_gamma_fuzzy_mass | 20 | 0.200928 | 0.27773 | -2.14957 | 0.0337858 | 0.431425 | 0.315306 | 5 |  |  |
| explicit_probability | score_central_2_mass | 20 | 0.26841 | 0.318175 | -3.1337 | -0.0639824 | 0.193275 | 0.315306 |  |  |  |
| explicit_probability | score_central_decay_mass | 20 | 0.179218 | 0.274443 | -2.07546 | -0.00717054 | 0.420037 | 0.315306 |  | 0.44 |  |
| explicit_probability | score_middle_2_7_mass | 20 | 0.549708 | 0.581149 | -12.7905 | -0.0641943 | 0.865014 | 0.315306 |  |  |  |
| explicit_probability | score_crisp_unknown_2_3_mass | 20 | 0.2112 | 0.260504 | -1.77098 | 0.377837 | 0.187321 | 0.315306 |  |  |  |
| explicit_probability | score_non_extreme_mass | 20 | 0.64473 | 0.665839 | -17.1027 | -0.382162 | 0.960036 | 0.315306 |  |  |  |
| explicit_probability | learned_score_unknown_weights | 20 | 0.173278 | 0.256464 | -1.6857 | -0.0256766 | 0.354541 | 0.315306 |  |  | 0.000,0.350,0.604,0.000,1.000,1.000,0.084,0.294,0.000,0.000 |
| explicit_probability | learned_symmetric_monotone_weights | 20 | 0.158295 | 0.257613 | -1.70983 | -0.0742023 | 0.374999 | 0.315306 |  |  | 0.000,0.237,0.237,0.237,1.000,1.000,0.237,0.237,0.237,0.000 |
| frequency | score_fuzzy_mass | 8 | 0.330985 | 0.354978 | -12.846 | 0.0236796 | 0.62343 | 0.292445 |  |  |  |
| frequency | score_entropy_mass | 8 | 0.397526 | 0.414954 | -17.92 | 0.050707 | 0.68997 | 0.292445 |  |  |  |
| frequency | score_gamma_fuzzy_mass | 8 | 0.114931 | 0.155834 | -1.66835 | -0.0269644 | 0.225734 | 0.292445 | 5 |  |  |
| frequency | score_central_2_mass | 8 | 0.229174 | 0.254882 | -6.13839 | -0.108207 | 0.0632704 | 0.292445 |  |  |  |
| frequency | score_central_decay_mass | 8 | 0.0955027 | 0.126518 | -0.758843 | -0.0244112 | 0.283737 | 0.292445 |  | 0.49 |  |
| frequency | score_middle_2_7_mass | 8 | 0.377024 | 0.425155 | -18.8617 | -0.0540987 | 0.669468 | 0.292445 |  |  |  |
| frequency | score_crisp_unknown_2_3_mass | 8 | 0.233101 | 0.276202 | -7.38254 | 0.516967 | 0.386695 | 0.292445 |  |  |  |
| frequency | score_non_extreme_mass | 8 | 0.650545 | 0.656039 | -46.2912 | 0.566717 | 0.942989 | 0.292445 |  |  |  |
| frequency | learned_score_unknown_weights | 8 | 0.109775 | 0.138967 | -1.122 | 0.0399991 | 0.293513 | 0.292445 |  |  | 0.000,0.000,0.345,0.599,1.000,0.810,0.466,0.188,0.144,0.000 |
| frequency | learned_symmetric_monotone_weights | 8 | 0.106082 | 0.138118 | -1.09614 | -0.0430989 | 0.276424 | 0.292445 |  |  | 0.000,0.032,0.272,0.501,1.000,1.000,0.501,0.272,0.032,0.000 |
| sequential_bayes | score_fuzzy_mass | 15 | 0.4144 | 0.451598 | -8.16165 | 0.389676 | 0.618461 | 0.204061 |  |  |  |
| sequential_bayes | score_entropy_mass | 15 | 0.47429 | 0.504026 | -10.4124 | 0.396396 | 0.678351 | 0.204061 |  |  |  |
| sequential_bayes | score_gamma_fuzzy_mass | 15 | 0.129655 | 0.162679 | -0.188859 | 0.313424 | 0.178598 | 0.204061 | 8 |  |  |
| sequential_bayes | score_central_2_mass | 15 | 0.165797 | 0.196829 | -0.740389 | 0.277126 | 0.0692138 | 0.204061 |  |  |  |
| sequential_bayes | score_central_decay_mass | 15 | 0.121147 | 0.151424 | -0.0300528 | 0.33459 | 0.197208 | 0.204061 |  | 0.33 |  |
| sequential_bayes | score_middle_2_7_mass | 15 | 0.460134 | 0.517979 | -11.053 | 0.398723 | 0.636691 | 0.204061 |  |  |  |
| sequential_bayes | score_crisp_unknown_2_3_mass | 15 | 0.18021 | 0.219432 | -1.16307 | 0.13576 | 0.152586 | 0.204061 |  |  |  |
| sequential_bayes | score_non_extreme_mass | 15 | 0.704375 | 0.717714 | -22.1405 | 0.422202 | 0.908436 | 0.204061 |  |  |  |
| sequential_bayes | learned_score_unknown_weights | 15 | 0.10043 | 0.142316 | 0.0901354 | 0.427763 | 0.237116 | 0.204061 |  |  | 0.000,0.021,0.672,0.000,1.000,1.000,0.172,0.290,0.000,0.000 |
| sequential_bayes | learned_symmetric_monotone_weights | 15 | 0.102482 | 0.139132 | 0.130389 | 0.386882 | 0.216473 | 0.204061 |  |  | 0.000,0.088,0.217,0.217,1.000,1.000,0.217,0.217,0.088,0.000 |

### Score Three-State Decomposition
| target | n_test | mae | rmse | r2 | corr | mean_observed | mean_predicted | train_joint_mse | u_profile | t_profile | f_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noul_from_score_T | 89 | 0.0660763 | 0.0864436 | 0.829578 | 0.92143 | 0.580517 | 0.562137 | 0.0284451 | 0.000,0.000,0.000,0.000,0.087,0.087,0.000,0.000,0.000,0.000 | 0.000,0.083,0.197,0.197,0.314,0.599,0.803,0.803,0.917,1.000 | 1.000,0.917,0.803,0.803,0.599,0.314,0.197,0.197,0.083,0.000 |
| choice_from_score_T_over_TF | 89 | 0.200899 | 0.228059 | 0.101155 | 0.87791 | 0.764618 | 0.567815 | 0.0284451 | 0.000,0.000,0.000,0.000,0.087,0.087,0.000,0.000,0.000,0.000 | 0.000,0.083,0.197,0.197,0.314,0.599,0.803,0.803,0.917,1.000 | 1.000,0.917,0.803,0.803,0.599,0.314,0.197,0.197,0.083,0.000 |
| unknown_from_score_U | 89 | 0.231185 | 0.270061 | -2.37963 | 0.31493 | 0.240688 | 0.0118511 | 0.0284451 | 0.000,0.000,0.000,0.000,0.087,0.087,0.000,0.000,0.000,0.000 | 0.000,0.083,0.197,0.197,0.314,0.599,0.803,0.803,0.917,1.000 | 1.000,0.917,0.803,0.803,0.599,0.314,0.197,0.197,0.083,0.000 |

### Score Three-State Decomposition by Family
| family | target | n_test | mae | rmse | r2 | corr | mean_observed | mean_predicted | u_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayes | noul_from_score_T | 15 | 0.0503237 | 0.0592748 | 0.930383 | 0.973193 | 0.617867 | 0.63834 | 0.000,0.000,0.000,0.000,0.122,0.122,0.000,0.000,0.000,0.000 |
| bayes | choice_from_score_T_over_TF | 15 | 0.1473 | 0.16443 | 0.41036 | 0.952659 | 0.792067 | 0.644767 | 0.000,0.000,0.000,0.000,0.122,0.122,0.000,0.000,0.000,0.000 |
| bayes | unknown_from_score_U | 15 | 0.217166 | 0.247222 | -2.79862 | 0.724409 | 0.229628 | 0.0124621 | 0.000,0.000,0.000,0.000,0.122,0.122,0.000,0.000,0.000,0.000 |
| compound | noul_from_score_T | 9 | 0.074564 | 0.0856793 | 0.671589 | 0.903294 | 0.681333 | 0.628264 | 0.000,0.002,0.051,0.051,0.537,0.537,0.051,0.051,0.002,0.000 |
| compound | choice_from_score_T_over_TF | 9 | 0.0787538 | 0.0919631 | 0.797244 | 0.953738 | 0.717556 | 0.657154 | 0.000,0.002,0.051,0.051,0.537,0.537,0.051,0.051,0.002,0.000 |
| compound | unknown_from_score_U | 9 | 0.0412737 | 0.0463106 | 0.0775752 | 0.563518 | 0.0614634 | 0.0435188 | 0.000,0.002,0.051,0.051,0.537,0.537,0.051,0.051,0.002,0.000 |
| conditional | noul_from_score_T | 30 | 0.0713227 | 0.0934228 | 0.726488 | 0.876973 | 0.602567 | 0.578426 | 0.000,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.000 |
| conditional | choice_from_score_T_over_TF | 30 | 0.209877 | 0.229581 | -1.36845 | 0.878893 | 0.810567 | 0.60069 | 0.000,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.000 |
| conditional | unknown_from_score_U | 30 | 0.231282 | 0.2575 | -4.11438 | 0.337811 | 0.268554 | 0.0372724 | 0.000,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.044,0.000 |
| explicit_probability | noul_from_score_T | 20 | 0.0278934 | 0.0325081 | 0.96629 | 0.988476 | 0.5574 | 0.540551 | 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000 |
| explicit_probability | choice_from_score_T_over_TF | 20 | 0.285849 | 0.302008 | -0.841356 | 0.814417 | 0.81355 | 0.540551 | 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000 |
| explicit_probability | unknown_from_score_U | 20 | 0.315306 | 0.352006 | -4.05949 | 0 | 0.315306 | 0 | 0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000 |
| frequency | noul_from_score_T | 8 | 0.0236038 | 0.0274583 | 0.988528 | 0.995157 | 0.433125 | 0.422253 | 0.000,0.020,0.020,0.047,0.406,0.406,0.047,0.020,0.020,0.000 |
| frequency | choice_from_score_T_over_TF | 8 | 0.143235 | 0.151881 | 0.739246 | 0.991918 | 0.584125 | 0.44089 | 0.000,0.020,0.020,0.047,0.406,0.406,0.047,0.020,0.020,0.000 |
| frequency | unknown_from_score_U | 8 | 0.244626 | 0.263978 | -6.65699 | -0.0761622 | 0.292445 | 0.0478187 | 0.000,0.020,0.020,0.047,0.406,0.406,0.047,0.020,0.020,0.000 |
| sequential_bayes | noul_from_score_T | 15 | 0.026351 | 0.0338371 | 0.983438 | 0.994219 | 0.4694 | 0.462283 | 0.000,0.000,0.000,0.124,0.153,0.153,0.124,0.000,0.000,0.000 |
| sequential_bayes | choice_from_score_T_over_TF | 15 | 0.128102 | 0.151098 | 0.821568 | 0.977535 | 0.608267 | 0.488453 | 0.000,0.000,0.000,0.124,0.153,0.153,0.124,0.000,0.000,0.000 |
| sequential_bayes | unknown_from_score_U | 15 | 0.173688 | 0.216071 | -1.09731 | 0.285163 | 0.204061 | 0.0422873 | 0.000,0.000,0.000,0.124,0.153,0.153,0.124,0.000,0.000,0.000 |

### Score to Choice via T/U/F
| truth_model | unknown_rule | n_test | mae | rmse | r2 | corr | mean_observed_choice | mean_predicted_choice | mean_predicted_unknown | central_lambda | u_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_expectation | no_unknown | 89 | 0.22125 | 0.244161 | -0.0302512 | 0.874476 | 0.764618 | 0.551807 | 0 |  |  |
| score_expectation | central_2_unknown | 89 | 0.165457 | 0.193266 | 0.354495 | 0.78023 | 0.764618 | 0.643631 | 0.136896 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| score_expectation | central_decay_unknown | 89 | 0.095516 | 0.136461 | 0.678185 | 0.844064 | 0.764618 | 0.794004 | 0.311981 | 0.42 | 0.000,0.074,0.176,0.420,1.000,1.000,0.420,0.176,0.074,0.000 |
| score_expectation | learned_symmetric_unknown | 89 | 0.0733171 | 0.101687 | 0.821303 | 0.920353 | 0.764618 | 0.801612 | 0.348592 |  | 0.000,0.275,0.429,0.429,0.484,0.484,0.429,0.429,0.275,0.000 |
| score_to_noul_monotonic | no_unknown | 89 | 0.200822 | 0.226284 | 0.115093 | 0.879738 | 0.764618 | 0.56953 | 0 |  |  |
| score_to_noul_monotonic | central_2_unknown | 89 | 0.141728 | 0.170561 | 0.497253 | 0.823554 | 0.764618 | 0.662814 | 0.136896 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| score_to_noul_monotonic | central_decay_unknown | 89 | 0.0905704 | 0.131741 | 0.70006 | 0.855763 | 0.764618 | 0.784441 | 0.278081 | 0.36 | 0.000,0.047,0.130,0.360,1.000,1.000,0.360,0.130,0.047,0.000 |
| score_to_noul_monotonic | learned_symmetric_unknown | 89 | 0.07354 | 0.106625 | 0.803524 | 0.913912 | 0.764618 | 0.803481 | 0.336042 |  | 0.000,0.348,0.348,0.348,0.619,0.619,0.348,0.348,0.348,0.000 |

### Score to Choice via T/U/F by Family
| family | truth_model | unknown_rule | n_test | mae | rmse | r2 | corr | mean_observed_choice | mean_predicted_choice | mean_predicted_unknown | central_lambda | u_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayes | score_expectation | no_unknown | 15 | 0.180015 | 0.192735 | 0.189886 | 0.948911 | 0.792067 | 0.612052 | 0 |  |  |
| bayes | score_expectation | central_2_unknown | 15 | 0.118367 | 0.128739 | 0.638557 | 0.968394 | 0.792067 | 0.678524 | 0.101802 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| bayes | score_expectation | central_decay_unknown | 15 | 0.0405432 | 0.0619716 | 0.916245 | 0.961688 | 0.792067 | 0.810529 | 0.254668 | 0.43 | 0.000,0.080,0.185,0.430,1.000,1.000,0.430,0.185,0.080,0.000 |
| bayes | score_expectation | learned_symmetric_unknown | 15 | 0.032995 | 0.0539508 | 0.936523 | 0.971935 | 0.792067 | 0.811021 | 0.249283 |  | 0.000,0.139,0.215,0.215,0.972,0.972,0.215,0.215,0.139,0.000 |
| bayes | score_to_noul_monotonic | no_unknown | 15 | 0.128956 | 0.142549 | 0.556849 | 0.96383 | 0.792067 | 0.663111 | 0 |  |  |
| bayes | score_to_noul_monotonic | central_2_unknown | 15 | 0.0725048 | 0.0793109 | 0.862821 | 0.967937 | 0.792067 | 0.735244 | 0.101802 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| bayes | score_to_noul_monotonic | central_decay_unknown | 15 | 0.0758296 | 0.0978164 | 0.791336 | 0.949081 | 0.792067 | 0.861102 | 0.287673 | 0.48 | 0.000,0.111,0.230,0.480,1.000,1.000,0.480,0.230,0.111,0.000 |
| bayes | score_to_noul_monotonic | learned_symmetric_unknown | 15 | 0.0786689 | 0.101599 | 0.774887 | 0.946917 | 0.792067 | 0.86604 | 0.329609 |  | 0.000,0.292,0.292,0.292,0.917,0.917,0.292,0.292,0.292,0.000 |
| compound | score_expectation | no_unknown | 9 | 0.117482 | 0.145728 | 0.490865 | 0.901649 | 0.717556 | 0.615215 | 0 |  |  |
| compound | score_expectation | central_2_unknown | 9 | 0.0995717 | 0.123976 | 0.631512 | 0.887733 | 0.717556 | 0.651078 | 0.0550175 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| compound | score_expectation | central_decay_unknown | 9 | 0.0651319 | 0.082249 | 0.837816 | 0.93088 | 0.717556 | 0.714583 | 0.136404 | 0.41 | 0.000,0.069,0.168,0.410,1.000,1.000,0.410,0.168,0.069,0.000 |
| compound | score_expectation | learned_symmetric_unknown | 9 | 0.0523576 | 0.0722706 | 0.874781 | 0.945593 | 0.717556 | 0.72851 | 0.150265 |  | 0.000,0.150,0.239,0.461,0.714,0.714,0.461,0.239,0.150,0.000 |
| compound | score_to_noul_monotonic | no_unknown | 9 | 0.0918575 | 0.110604 | 0.706714 | 0.954364 | 0.717556 | 0.633534 | 0 |  |  |
| compound | score_to_noul_monotonic | central_2_unknown | 9 | 0.0741298 | 0.0856943 | 0.823944 | 0.948772 | 0.717556 | 0.670222 | 0.0550175 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| compound | score_to_noul_monotonic | central_decay_unknown | 9 | 0.0443061 | 0.0650215 | 0.898642 | 0.969792 | 0.717556 | 0.759324 | 0.16425 | 0.49 | 0.000,0.118,0.240,0.490,1.000,1.000,0.490,0.240,0.118,0.000 |
| compound | score_to_noul_monotonic | learned_symmetric_unknown | 9 | 0.0443035 | 0.0656028 | 0.896821 | 0.969021 | 0.717556 | 0.759479 | 0.16415 |  | 0.000,0.165,0.184,0.506,1.000,1.000,0.506,0.184,0.165,0.000 |
| conditional | score_expectation | no_unknown | 30 | 0.268383 | 0.280766 | -2.54228 | 0.833333 | 0.810567 | 0.542184 | 0 |  |  |
| conditional | score_expectation | central_2_unknown | 30 | 0.176042 | 0.199488 | -0.788257 | 0.569785 | 0.810567 | 0.654297 | 0.175263 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| conditional | score_expectation | central_decay_unknown | 30 | 0.118748 | 0.147029 | 0.0285913 | 0.462059 | 0.810567 | 0.851273 | 0.3685 | 0.43 | 0.000,0.080,0.185,0.430,1.000,1.000,0.430,0.185,0.080,0.000 |
| conditional | score_expectation | learned_symmetric_unknown | 30 | 0.0666291 | 0.0802375 | 0.710699 | 0.864108 | 0.810567 | 0.824886 | 0.350321 |  | 0.000,0.248,0.439,0.439,0.439,0.439,0.439,0.439,0.248,0.000 |
| conditional | score_to_noul_monotonic | no_unknown | 30 | 0.221964 | 0.23316 | -1.44289 | 0.878914 | 0.810567 | 0.588602 | 0 |  |  |
| conditional | score_to_noul_monotonic | central_2_unknown | 30 | 0.132588 | 0.147774 | 0.0187237 | 0.69872 | 0.810567 | 0.70843 | 0.175263 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| conditional | score_to_noul_monotonic | central_decay_unknown | 30 | 0.103499 | 0.125651 | 0.290545 | 0.59209 | 0.810567 | 0.827189 | 0.292971 | 0.29 | 0.000,0.024,0.084,0.290,1.000,1.000,0.290,0.084,0.024,0.000 |
| conditional | score_to_noul_monotonic | learned_symmetric_unknown | 30 | 0.0601327 | 0.0748563 | 0.748203 | 0.886558 | 0.810567 | 0.814928 | 0.284433 |  | 0.000,0.188,0.336,0.336,0.431,0.431,0.336,0.336,0.188,0.000 |
| explicit_probability | score_expectation | no_unknown | 20 | 0.274807 | 0.289523 | -0.69226 | 0.819481 | 0.81355 | 0.55371 | 0 |  |  |
| explicit_probability | score_expectation | central_2_unknown | 20 | 0.240963 | 0.266558 | -0.434453 | 0.424389 | 0.81355 | 0.69956 | 0.193275 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| explicit_probability | score_expectation | central_decay_unknown | 20 | 0.132119 | 0.199128 | 0.199493 | 0.680936 | 0.81355 | 0.869018 | 0.426975 | 0.45 | 0.000,0.091,0.203,0.450,1.000,1.000,0.450,0.203,0.091,0.000 |
| explicit_probability | score_expectation | learned_symmetric_unknown | 20 | 0.119875 | 0.182038 | 0.331003 | 0.704405 | 0.81355 | 0.885276 | 0.48329 |  | 0.000,0.467,0.467,0.467,0.650,0.650,0.467,0.467,0.467,0.000 |
| explicit_probability | score_to_noul_monotonic | no_unknown | 20 | 0.280819 | 0.294945 | -0.75624 | 0.807589 | 0.81355 | 0.54943 | 0 |  |  |
| explicit_probability | score_to_noul_monotonic | central_2_unknown | 20 | 0.253036 | 0.275518 | -0.532508 | 0.377456 | 0.81355 | 0.694985 | 0.193275 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| explicit_probability | score_to_noul_monotonic | central_decay_unknown | 20 | 0.130011 | 0.198643 | 0.203385 | 0.646405 | 0.81355 | 0.845289 | 0.373884 | 0.37 | 0.000,0.051,0.137,0.370,1.000,1.000,0.370,0.137,0.051,0.000 |
| explicit_probability | score_to_noul_monotonic | learned_symmetric_unknown | 20 | 0.111756 | 0.175821 | 0.375918 | 0.715135 | 0.81355 | 0.87529 | 0.434968 |  | 0.000,0.421,0.421,0.421,0.582,0.582,0.421,0.421,0.421,0.000 |
| frequency | score_expectation | no_unknown | 8 | 0.147952 | 0.155334 | 0.727252 | 0.992258 | 0.584125 | 0.436173 | 0 |  |  |
| frequency | score_expectation | central_2_unknown | 8 | 0.124013 | 0.132125 | 0.802669 | 0.992973 | 0.584125 | 0.460112 | 0.0632704 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| frequency | score_expectation | central_decay_unknown | 8 | 0.0410394 | 0.0502577 | 0.971448 | 0.992446 | 0.584125 | 0.614171 | 0.343549 | 0.56 | 0.000,0.176,0.314,0.560,1.000,1.000,0.560,0.314,0.176,0.000 |
| frequency | score_expectation | learned_symmetric_unknown | 8 | 0.0422411 | 0.0529951 | 0.968253 | 0.993046 | 0.584125 | 0.622516 | 0.376223 |  | 0.000,0.149,0.472,0.472,0.778,0.778,0.472,0.472,0.149,0.000 |
| frequency | score_to_noul_monotonic | no_unknown | 8 | 0.152104 | 0.164171 | 0.695338 | 0.985992 | 0.584125 | 0.432021 | 0 |  |  |
| frequency | score_to_noul_monotonic | central_2_unknown | 8 | 0.12852 | 0.141067 | 0.775054 | 0.988946 | 0.584125 | 0.455605 | 0.0632704 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| frequency | score_to_noul_monotonic | central_decay_unknown | 8 | 0.0350711 | 0.0450586 | 0.97705 | 0.994727 | 0.584125 | 0.614043 | 0.334481 | 0.55 | 0.000,0.166,0.303,0.550,1.000,1.000,0.550,0.303,0.166,0.000 |
| frequency | score_to_noul_monotonic | learned_symmetric_unknown | 8 | 0.0337415 | 0.0356828 | 0.985607 | 0.996099 | 0.584125 | 0.591346 | 0.281225 |  | 0.000,0.189,0.189,0.493,1.000,1.000,0.493,0.189,0.189,0.000 |
| sequential_bayes | score_expectation | no_unknown | 15 | 0.159071 | 0.185475 | 0.73114 | 0.977553 | 0.608267 | 0.470225 | 0 |  |  |
| sequential_bayes | score_expectation | central_2_unknown | 15 | 0.13023 | 0.14661 | 0.83201 | 0.981654 | 0.608267 | 0.508365 | 0.0692138 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| sequential_bayes | score_expectation | central_decay_unknown | 15 | 0.0834297 | 0.0965674 | 0.927118 | 0.971877 | 0.608267 | 0.572126 | 0.14774 | 0.23 | 0.000,0.012,0.053,0.230,1.000,1.000,0.230,0.053,0.012,0.000 |
| sequential_bayes | score_expectation | learned_symmetric_unknown | 15 | 0.049068 | 0.0693476 | 0.962415 | 0.981831 | 0.608267 | 0.619193 | 0.214749 |  | 0.000,0.114,0.245,0.245,0.647,0.647,0.245,0.245,0.114,0.000 |
| sequential_bayes | score_to_noul_monotonic | no_unknown | 15 | 0.159453 | 0.18679 | 0.727315 | 0.976242 | 0.608267 | 0.463237 | 0 |  |  |
| sequential_bayes | score_to_noul_monotonic | central_2_unknown | 15 | 0.128151 | 0.147801 | 0.82927 | 0.979918 | 0.608267 | 0.500868 | 0.0692138 |  | 0.000,0.000,0.000,0.000,1.000,1.000,0.000,0.000,0.000,0.000 |
| sequential_bayes | score_to_noul_monotonic | central_decay_unknown | 15 | 0.0791701 | 0.0922278 | 0.933522 | 0.969256 | 0.608267 | 0.581522 | 0.171123 | 0.28 | 0.000,0.022,0.078,0.280,1.000,1.000,0.280,0.078,0.022,0.000 |
| sequential_bayes | score_to_noul_monotonic | learned_symmetric_unknown | 15 | 0.0505057 | 0.0709864 | 0.960617 | 0.980447 | 0.608267 | 0.617273 | 0.234491 |  | 0.000,0.172,0.212,0.212,0.979,0.979,0.212,0.212,0.172,0.000 |

### Noul to Choice Calibration
| mapping | n_test | mae | rmse | r2 | alpha | a | b | c | n_blocks | mean_implied_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 89 | 0.193966 | 0.219247 | 0.169272 |  |  |  |  |  | 0 |
| power | 89 | 0.0741244 | 0.10766 | 0.79969 | 2 |  |  |  |  | 0.280512 |
| logit | 89 | 0.0720753 | 0.114351 | 0.774021 |  | 1.8384 | 1.14357 |  |  | 0.240747 |
| beta | 89 | 0.0718664 | 0.115582 | 0.769129 |  | 2.09141 | -1.60638 | 1.57079 |  | 0.247694 |
| isotonic | 89 | 0.0704961 | 0.102975 | 0.816747 |  |  |  |  | 32 | 0.247395 |

### Noul to Choice Calibration by Family
| family | mapping | n_test | mae | rmse | r2 | alpha | a | b | c | n_blocks | mean_implied_unknown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayes | identity | 15 | 0.1742 | 0.198465 | 0.141007 |  |  |  |  |  | 0 |
| bayes | power | 15 | 0.062089 | 0.0757636 | 0.874817 | 1.75 |  |  |  |  | 0.224544 |
| bayes | logit | 15 | 0.0516298 | 0.0698994 | 0.893446 |  | 1.23183 | 1.074 |  |  | 0.248233 |
| bayes | beta | 15 | 0.0580156 | 0.0728999 | 0.884102 |  | 0.993321 | -1.42949 | 0.661697 |  | 0.238479 |
| bayes | isotonic | 15 | 0.0633552 | 0.07837 | 0.866056 |  |  |  |  | 9 | 0.22841 |
| compound | identity | 9 | 0.0653333 | 0.0782886 | 0.853059 |  |  |  |  |  | 0 |
| compound | power | 9 | 0.0977905 | 0.124374 | 0.629142 | 1.5 |  |  |  |  | 0.162498 |
| compound | logit | 9 | 0.117732 | 0.144219 | 0.501359 |  | 0.950565 | 0.787328 |  |  | 0.169844 |
| compound | beta | 9 | 0.0993044 | 0.128307 | 0.605319 |  | -0.0630067 | -1.85457 | -0.679994 |  | 0.167007 |
| compound | isotonic | 9 | 0.0919219 | 0.118597 | 0.662797 |  |  |  |  | 13 | 0.162166 |
| conditional | identity | 30 | 0.208 | 0.222725 | -1.22912 |  |  |  |  |  | 0 |
| conditional | power | 30 | 0.0424678 | 0.0573193 | 0.852363 | 2 |  |  |  |  | 0.272344 |
| conditional | logit | 30 | 0.044885 | 0.0622866 | 0.825665 |  | 1.54926 | 1.24784 |  |  | 0.282349 |
| conditional | beta | 30 | 0.0506939 | 0.0755108 | 0.74378 |  | 2.95867 | -0.537032 | 3.00139 |  | 0.271855 |
| conditional | isotonic | 30 | 0.0322088 | 0.0493487 | 0.890567 |  |  |  |  | 15 | 0.269178 |
| explicit_probability | identity | 20 | 0.27365 | 0.286871 | -0.661404 |  |  |  |  |  | 0 |
| explicit_probability | power | 20 | 0.070328 | 0.128954 | 0.664285 | 2.5 |  |  |  |  | 0.346916 |
| explicit_probability | logit | 20 | 0.0888946 | 0.170708 | 0.411683 |  | 2.4212 | 1.33048 |  |  | 0.245573 |
| explicit_probability | beta | 20 | 0.0911182 | 0.165381 | 0.447828 |  | 3.06767 | -1.87299 | 2.56773 |  | 0.275492 |
| explicit_probability | isotonic | 20 | 0.0733612 | 0.13617 | 0.625664 |  |  |  |  | 29 | 0.345621 |
| frequency | identity | 8 | 0.151 | 0.15993 | 0.710876 |  |  |  |  |  | 0 |
| frequency | power | 8 | 0.0445286 | 0.0613202 | 0.957496 | 2 |  |  |  |  | 0.342832 |
| frequency | logit | 8 | 0.0595659 | 0.0776398 | 0.931861 |  | 1.70128 | 1.57809 |  |  | 0.314288 |
| frequency | beta | 8 | 0.0445577 | 0.0612451 | 0.9576 |  | 1.14014 | -2.34875 | 0.577363 |  | 0.326472 |
| frequency | isotonic | 8 | 0.0433172 | 0.0575445 | 0.962569 |  |  |  |  | 8 | 0.360934 |
| sequential_bayes | identity | 15 | 0.1566 | 0.182225 | 0.740479 |  |  |  |  |  | 0 |
| sequential_bayes | power | 15 | 0.0767087 | 0.0955151 | 0.928698 | 1.75 |  |  |  |  | 0.283696 |
| sequential_bayes | logit | 15 | 0.0662937 | 0.0888297 | 0.93833 |  | 1.40276 | 1.04227 |  |  | 0.228083 |
| sequential_bayes | beta | 15 | 0.0774604 | 0.115662 | 0.895446 |  | 2.27826 | -0.605042 | 2.69539 |  | 0.244369 |
| sequential_bayes | isotonic | 15 | 0.0669399 | 0.104288 | 0.914999 |  |  |  |  | 20 | 0.233726 |

### Choice From Score Recovery
| rule | n | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| expected_truth | 89 | 0.22125 | 0.244161 | -0.0302512 |
| hard_threshold | 89 | 0.16313 | 0.225628 | 0.120214 |
| endpoint_conditioning | 71 | 0.322815 | 0.373781 | -1.48372 |
| parametric_sharpening | 89 | 0.197341 | 0.226215 | 0.115634 |
| monotonic_collapse | 89 | 0.0701356 | 0.104378 | 0.811717 |
| full_q_ridge | 89 | 0.0735989 | 0.1104 | 0.789365 |
| crisp_unknown_conditional | 89 | 0.0769053 | 0.117506 | 0.761379 |
| fuzzy_unknown_conditional | 89 | 0.197341 | 0.226215 | 0.115634 |

### Multiscale Score-Choice Entropy
| choice_bins | nonempty_bins | h_score | h_score_given_choice_bin | mutual_information | normalized_mutual_information | mean_adjacent_wasserstein |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 2 | 3.2566 | 3.05424 | 0.202364 | 0.0621398 | 0.35913 |
| 3 | 3 | 3.2566 | 2.94826 | 0.308345 | 0.094683 | 0.222678 |
| 4 | 4 | 3.2566 | 2.88131 | 0.375292 | 0.11524 | 0.174846 |
| 5 | 5 | 3.2566 | 2.82382 | 0.432783 | 0.132894 | 0.144045 |
| 8 | 8 | 3.2566 | 2.7949 | 0.461702 | 0.141774 | 0.0930422 |
| 10 | 10 | 3.2566 | 2.78909 | 0.467516 | 0.14356 | 0.0735184 |
| 15 | 15 | 3.2566 | 2.74993 | 0.506674 | 0.155584 | 0.0598861 |
| 20 | 20 | 3.2566 | 2.72988 | 0.526719 | 0.161739 | 0.0566433 |
| 25 | 24 | 3.2566 | 2.70248 | 0.554124 | 0.170154 | 0.0630684 |
| 30 | 30 | 3.2566 | 2.67494 | 0.581658 | 0.178609 | 0.0651312 |

### Score to Choice by Representation
| group | rule | n_total | n_test | mae | rmse | r2 | false_max_level | true_min_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counts | expected_truth | 12 | 3 | 0.172074 | 0.173182 | -3.30754 |  |  |
| counts | crisp_unknown_conditional | 12 | 3 | 0.0106567 | 0.0173392 | 0.95682 | 2 | 3 |
| counts | monotonic_collapse | 12 | 3 | 0.0112871 | 0.0167453 | 0.959727 |  |  |
| counts | full_q_ridge | 12 | 3 | 0.0270593 | 0.0291187 | 0.878222 |  |  |
| direct | expected_truth | 87 | 22 | 0.232696 | 0.261509 | -0.174764 |  |  |
| direct | crisp_unknown_conditional | 87 | 22 | 0.0927653 | 0.134724 | 0.688208 | 0 | 5 |
| direct | monotonic_collapse | 87 | 22 | 0.0648598 | 0.0832603 | 0.880916 |  |  |
| direct | full_q_ridge | 87 | 22 | 0.0618664 | 0.0827436 | 0.882389 |  |  |
| distractor | expected_truth | 52 | 14 | 0.292422 | 0.298029 | -4.40631 |  |  |
| distractor | crisp_unknown_conditional | 52 | 14 | 0.0537702 | 0.072592 | 0.679254 | 1 | 3 |
| distractor | monotonic_collapse | 52 | 14 | 0.0442067 | 0.0578752 | 0.796124 |  |  |
| distractor | full_q_ridge | 52 | 14 | 0.0612876 | 0.0846836 | 0.563503 |  |  |
| nested | expected_truth | 20 | 6 | 0.310865 | 0.315589 | -5.99882 |  |  |
| nested | crisp_unknown_conditional | 20 | 6 | 0.0704351 | 0.0769334 | 0.584079 | 0 | 6 |
| nested | monotonic_collapse | 20 | 6 | 0.0757315 | 0.0818543 | 0.529171 |  |  |
| nested | full_q_ridge | 20 | 6 | 0.0529613 | 0.0870358 | 0.467676 |  |  |
| prose | expected_truth | 92 | 22 | 0.203204 | 0.21995 | -0.0480311 |  |  |
| prose | crisp_unknown_conditional | 92 | 22 | 0.0638792 | 0.0872164 | 0.835213 | 2 | 3 |
| prose | monotonic_collapse | 92 | 22 | 0.037322 | 0.0544436 | 0.935787 |  |  |
| prose | full_q_ridge | 92 | 22 | 0.0376589 | 0.0530979 | 0.938923 |  |  |
| ratio | expected_truth | 25 | 5 | 0.174484 | 0.195107 | 0.701579 |  |  |
| ratio | crisp_unknown_conditional | 25 | 5 | 0.150685 | 0.238537 | 0.553942 | 3 | 4 |
| ratio | monotonic_collapse | 25 | 5 | 0.020738 | 0.0288788 | 0.993462 |  |  |
| ratio | full_q_ridge | 25 | 5 | 0.0655739 | 0.117033 | 0.892627 |  |  |
| scaled_counts | insufficient_data | 5 | 0 |  |  |  |  |  |
| table | expected_truth | 72 | 17 | 0.161983 | 0.190276 | 0.583093 |  |  |
| table | crisp_unknown_conditional | 72 | 17 | 0.0798708 | 0.102835 | 0.878226 | 2 | 3 |
| table | monotonic_collapse | 72 | 17 | 0.0660352 | 0.082396 | 0.921822 |  |  |
| table | full_q_ridge | 72 | 17 | 0.0717738 | 0.0815715 | 0.923379 |  |  |

### Score to Choice by Family
| group | rule | n_total | n_test | mae | rmse | r2 | false_max_level | true_min_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bayes | expected_truth | 35 | 15 | 0.180015 | 0.192735 | 0.189886 |  |  |
| bayes | crisp_unknown_conditional | 35 | 15 | 0.0317531 | 0.0538997 | 0.936643 | 0 | 8 |
| bayes | monotonic_collapse | 35 | 15 | 0.0315146 | 0.0448049 | 0.95622 |  |  |
| bayes | full_q_ridge | 35 | 15 | 0.0403712 | 0.056139 | 0.931269 |  |  |
| compound | expected_truth | 60 | 9 | 0.117482 | 0.145728 | 0.490865 |  |  |
| compound | crisp_unknown_conditional | 60 | 9 | 0.0669759 | 0.076999 | 0.85786 | 2 | 3 |
| compound | monotonic_collapse | 60 | 9 | 0.0664689 | 0.0784334 | 0.852515 |  |  |
| compound | full_q_ridge | 60 | 9 | 0.0436362 | 0.0546146 | 0.928491 |  |  |
| conditional | expected_truth | 100 | 30 | 0.268383 | 0.280766 | -2.54228 |  |  |
| conditional | crisp_unknown_conditional | 100 | 30 | 0.0744356 | 0.0888525 | 0.64524 | 0 | 6 |
| conditional | monotonic_collapse | 100 | 30 | 0.0777338 | 0.0916289 | 0.622724 |  |  |
| conditional | full_q_ridge | 100 | 30 | 0.0540885 | 0.0734437 | 0.757616 |  |  |
| explicit_probability | expected_truth | 80 | 20 | 0.274807 | 0.289523 | -0.69226 |  |  |
| explicit_probability | crisp_unknown_conditional | 80 | 20 | 0.0778448 | 0.156661 | 0.504525 | 0 | 5 |
| explicit_probability | monotonic_collapse | 80 | 20 | 0.0844818 | 0.158675 | 0.491702 |  |  |
| explicit_probability | full_q_ridge | 80 | 20 | 0.140115 | 0.198453 | 0.204906 |  |  |
| frequency | expected_truth | 30 | 8 | 0.147952 | 0.155334 | 0.727252 |  |  |
| frequency | crisp_unknown_conditional | 30 | 8 | 0.126415 | 0.151596 | 0.740221 | 0 | 5 |
| frequency | monotonic_collapse | 30 | 8 | 0.0359242 | 0.049608 | 0.972182 |  |  |
| frequency | full_q_ridge | 30 | 8 | 0.0389564 | 0.0543361 | 0.966626 |  |  |
| sequential_bayes | expected_truth | 60 | 15 | 0.159071 | 0.185475 | 0.73114 |  |  |
| sequential_bayes | crisp_unknown_conditional | 60 | 15 | 0.0657857 | 0.10113 | 0.920069 | 2 | 4 |
| sequential_bayes | monotonic_collapse | 60 | 15 | 0.0457181 | 0.0659667 | 0.96599 |  |  |
| sequential_bayes | full_q_ridge | 60 | 15 | 0.054002 | 0.0684511 | 0.96338 |  |  |

### Primitive Representation Sensitivity
| primitive | n_latents | mean_pairwise_distance | median_pairwise_distance | mean_max_distance | max_distance |
| --- | --- | --- | --- | --- | --- |
| noul | 92 | 0.0424581 | 0.024 | 0.0786739 | 0.645 |
| choice | 92 | 0.0955949 | 0.031 | 0.181076 | 0.645 |
| score | 92 | 0.186232 | 0.151 | 0.30235 | 0.877 |

## Calibration
ECE: 0.133143

Empirical Brier: 0.245167

Empirical log loss: 0.740822

## Figures
- results/jev_semif_combined_10x_001/report/figures/figure_1_probability_transfer.png
- results/jev_semif_combined_10x_001/report/figures/figure_16_powerlaw_calibration.png
- results/jev_semif_combined_10x_001/report/figures/figure_16b_powerlaw_calibration_explicit_probability.png
- results/jev_semif_combined_10x_001/report/figures/figure_2_error_by_family.png
- results/jev_semif_combined_10x_001/report/figures/figure_3_representation_sensitivity.png
- results/jev_semif_combined_10x_001/report/figures/figure_4_equivalent_state_spread.png
- results/jev_semif_combined_10x_001/report/figures/figure_5_sequential_bayes_trajectories.png
- results/jev_semif_combined_10x_001/report/figures/figure_6_calibration_curve.png
- results/jev_semif_combined_10x_001/report/figures/figure_15_calibration_by_representation.png
- results/jev_semif_combined_10x_001/report/figures/figure_7_noul_choice_consistency.png
- results/jev_semif_combined_10x_001/report/figures/figure_16_noul_choice_calibration.png
- results/jev_semif_combined_10x_001/report/figures/figure_8_score_expectation_connections.png
- results/jev_semif_combined_10x_001/report/figures/figure_9_choice_from_score_recovery.png
- results/jev_semif_combined_10x_001/report/figures/figure_10_score_levels_vs_choice_heatmap.png
- results/jev_semif_combined_10x_001/report/figures/figure_11_score_choice_multiscale_entropy.png
- results/jev_semif_combined_10x_001/report/figures/figure_17_score_unknown_vs_implied_unknown.png
- results/jev_semif_combined_10x_001/report/figures/figure_18_score_unknown_level_weights.png
- results/jev_semif_combined_10x_001/report/figures/figure_21_score_choice_tuf.png
- results/jev_semif_combined_10x_001/report/figures/figure_12_choice_from_score_by_representation.png
- results/jev_semif_combined_10x_001/report/figures/figure_13_choice_from_score_by_family.png
- results/jev_semif_combined_10x_001/report/figures/figure_14_crisp_unknown_thresholds_by_representation.png
