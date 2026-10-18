# XAI Report — Task 8

## Models explained
| dataset    | ML model (Task 6 best, by validation AUC)   | DL model     |   ML test AUC |   DNN test AUC |
|:-----------|:--------------------------------------------|:-------------|--------------:|---------------:|
| university | SVM                                         | DNN (Task 7) |        0.7602 |         0.7795 |
| graduate   | Logistic Regression                         | DNN (Task 7) |        0.9558 |         0.9463 |

## SHAP — top features (mean |SHAP|)
| dataset    | model               | feature                              |   mean_abs_shap |
|:-----------|:--------------------|:-------------------------------------|----------------:|
| graduate   | DNN                 | gre_score                            |          0.1194 |
| graduate   | DNN                 | cgpa                                 |          0.0866 |
| graduate   | DNN                 | toefl_score                          |          0.0614 |
| graduate   | DNN                 | sop                                  |          0.0592 |
| graduate   | DNN                 | lor                                  |          0.0485 |
| graduate   | Logistic Regression | gre_score                            |          0.1462 |
| graduate   | Logistic Regression | cgpa                                 |          0.1403 |
| graduate   | Logistic Regression | lor                                  |          0.051  |
| graduate   | Logistic Regression | toefl_score                          |          0.0396 |
| graduate   | Logistic Regression | sop                                  |          0.0109 |
| university | DNN                 | cgpa_pct                             |          0.0814 |
| university | DNN                 | univ_University_of_Texas_Dallas      |          0.0616 |
| university | DNN                 | admit_chance_index                   |          0.0542 |
| university | DNN                 | univ_Arizona_State_University        |          0.0353 |
| university | DNN                 | univ_University_of_Wisconsin_Madison |          0.027  |
| university | SVM                 | cgpa_pct                             |          0.0472 |
| university | SVM                 | admit_chance_index                   |          0.0385 |
| university | SVM                 | univ_University_of_Texas_Dallas      |          0.0383 |
| university | SVM                 | univ_University_of_Wisconsin_Madison |          0.0294 |
| university | SVM                 | greq                                 |          0.0152 |

## LIME
Three applicants per dataset (confident admit, borderline, confident reject): `lime_<dataset>_<case>.html`, `lime_<dataset>.png`, `lime_report.csv`.

## Counterfactual analysis (Graduate dataset)
- Applicants predicted "not admitted": 41; a realistic change that flips the prediction was found for 40.
- Most frequently needed improvements: cgpa (40), gre_score (2).

## Trust score
| dataset    | trust_band   |   applicants |   accuracy |   mean_trust |
|:-----------|:-------------|-------------:|-----------:|-------------:|
| graduate   | Low          |            5 |      1     |        0.439 |
| graduate   | Medium       |           33 |      0.818 |        0.627 |
| graduate   | High         |           62 |      0.919 |        0.847 |
| university | Low          |         2330 |      0.603 |        0.421 |
| university | Medium       |         5294 |      0.682 |        0.603 |
| university | High         |         3117 |      0.785 |        0.773 |

## Fairness assessment (proxy groups — no protected attributes in these datasets)
| dataset    | attribute         | model               |   base_rate_gap |   demographic_parity_diff |   equal_opportunity_diff | flag   |
|:-----------|:------------------|:--------------------|----------------:|--------------------------:|-------------------------:|:-------|
| graduate   | research          | Logistic Regression |           0.525 |                     0.588 |                    0.242 | review |
| graduate   | research          | DNN                 |           0.525 |                     0.61  |                    0.314 | review |
| graduate   | university_rating | Logistic Regression |           0.347 |                     0.406 |                    0.222 | review |
| graduate   | university_rating | DNN                 |           0.347 |                     0.422 |                    0.259 | review |
| university | program           | SVM                 |           0.074 |                     0.097 |                    0.134 | review |
| university | program           | DNN                 |           0.074 |                     0.1   |                    0.149 | review |
| university | is_ranked         | SVM                 |           0.057 |                     0.105 |                    0.138 | review |
| university | is_ranked         | DNN                 |           0.057 |                     0.095 |                    0.143 | review |

## Notes / limitations
- All explanations are in original feature units; scaling is applied inside the prediction functions.
- SHAP uses a model-agnostic explainer on a sample of test applicants (60 university / 100 graduate, with 25 / 100 background rows); the university sample is small because the selected SVM is slow to evaluate, so its SHAP ranking is indicative.
- University, department and specialization are one-hot columns (`univ_*`, `dept_*`, `spec_*`); a column's SHAP value is the effect of applying to / coming from that particular category.
- Graduate models use the raw applicant attributes (GRE, TOEFL, CGPA, SOP, LOR, research, university rating), so each attribute's SHAP value can be read directly.
- Test rows are the Task 5 test split (by applicant for the University dataset); the explained ML model is the Task 6 model with the best validation ROC-AUC.
- Fairness here uses available applicant groups only; a demographic fairness audit is part of Task 10.
