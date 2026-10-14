# XAI Report — Task 8

## Models explained
| dataset    | ML model (Task 6 best)   | DL model     |   ML test AUC |   DNN test AUC |
|:-----------|:-------------------------|:-------------|--------------:|---------------:|
| university | XGBoost                  | DNN (Task 7) |        0.8005 |         0.7534 |
| graduate   | Logistic Regression      | DNN (Task 7) |        0.9479 |         0.9504 |

## SHAP — top features (mean |SHAP|)
| dataset    | model               | feature            |   mean_abs_shap |
|:-----------|:--------------------|:-------------------|----------------:|
| graduate   | DNN                 | admit_chance_index |          0.3567 |
| graduate   | DNN                 | toefl_score        |          0.0285 |
| graduate   | DNN                 | sop                |          0.0173 |
| graduate   | Logistic Regression | admit_chance_index |          0.3378 |
| graduate   | Logistic Regression | toefl_score        |          0.0274 |
| graduate   | Logistic Regression | sop                |          0.0044 |
| university | DNN                 | univname_freq      |          0.0676 |
| university | DNN                 | cgpa_pct           |          0.0671 |
| university | DNN                 | profile_distance   |          0.0665 |
| university | DNN                 | univname_label     |          0.0551 |
| university | DNN                 | toeflscore         |          0.028  |
| university | XGBoost             | univname_freq      |          0.1257 |
| university | XGBoost             | cgpa_pct           |          0.091  |
| university | XGBoost             | univname_label     |          0.0702 |
| university | XGBoost             | profile_distance   |          0.0393 |
| university | XGBoost             | toppercgpa         |          0.0315 |

## LIME
Three applicants per dataset (confident admit, borderline, confident reject): `lime_<dataset>_<case>.html`, `lime_<dataset>.png`, `lime_report.csv`.

## Counterfactual analysis (Graduate dataset)
- Applicants predicted "not admitted": 42; a realistic change that flips the prediction was found for 40.
- Most frequently needed improvements: cgpa (34), research (32), toefl_score (3).

## Trust score
| dataset    | trust_band   |   applicants |   accuracy |   mean_trust |
|:-----------|:-------------|-------------:|-----------:|-------------:|
| graduate   | Low          |            8 |      1     |        0.432 |
| graduate   | Medium       |           32 |      0.781 |        0.624 |
| graduate   | High         |           60 |      0.95  |        0.84  |
| university | Low          |         2216 |      0.616 |        0.433 |
| university | Medium       |         5697 |      0.719 |        0.604 |
| university | High         |         2793 |      0.84  |        0.774 |

## Fairness assessment (proxy groups — no protected attributes in these datasets)
| dataset    | attribute         | model               |   base_rate_gap |   demographic_parity_diff |   equal_opportunity_diff | flag   |
|:-----------|:------------------|:--------------------|----------------:|--------------------------:|-------------------------:|:-------|
| graduate   | research          | Logistic Regression |           0.525 |                     0.691 |                    0.335 | review |
| graduate   | research          | DNN                 |           0.525 |                     0.669 |                    0.407 | review |
| graduate   | university_rating | Logistic Regression |           0.347 |                     0.422 |                    0.222 | review |
| graduate   | university_rating | DNN                 |           0.347 |                     0.406 |                    0.192 | review |
| university | program           | XGBoost             |           0.006 |                     0.132 |                    0.123 | review |
| university | program           | DNN                 |           0.006 |                     0.114 |                    0.136 | review |
| university | is_ranked         | XGBoost             |           0.102 |                     0.085 |                    0.073 | ok     |
| university | is_ranked         | DNN                 |           0.102 |                     0.047 |                    0.068 | ok     |

## Notes / limitations
- All explanations are in original feature units; scaling is applied inside the prediction functions.
- SHAP uses a model-agnostic explainer with 100 background rows on a sample of test applicants.
- Label-encoded features (e.g. `univname_label`) have no natural order, so their SHAP/LIME direction should not be read as "higher is better".
- Graduate dataset: `admit_chance_index` already contains SOP and TOEFL, so the separate `sop` / `toefl_score` features only add what the index does not capture — their individual signs can look counter-intuitive (e.g. a high SOP slightly lowering the prediction).
- Fairness here uses available applicant groups only; a demographic fairness audit is part of Task 10.
