# Model Comparison Report — Task 6

## University Master dataset (target: `admit`)
| model               |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:--------------------|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| XGBoost             |   0.728937 |    0.750795 | 0.719069 | 0.734589 |                      0.707112 |                   0.739699 |               0.723039 |  0.800454 |
| Random Forest       |   0.697553 |    0.721876 | 0.683617 | 0.702225 |                      0.673805 |                   0.712751 |               0.692731 |  0.767123 |
| Decision Tree       |   0.673828 |    0.678553 | 0.712086 | 0.694915 |                      0.668111 |                   0.632103 |               0.649609 |  0.726332 |
| SVM                 |   0.648515 |    0.674521 | 0.630439 | 0.651735 |                      0.62377  |                   0.668229 |               0.645234 |  0.708399 |
| Logistic Regression |   0.576406 |    0.600229 | 0.562936 | 0.580985 |                      0.553584 |                   0.591095 |               0.571725 |  0.608057 |

**Best model: XGBoost** (ROC-AUC = 0.8005)

## Graduate Admission dataset (target: `admit_label`, chance_of_admit >= 0.7)
| model               |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:--------------------|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| Logistic Regression |       0.9  |    0.931034 | 0.9      | 0.915254 |                      0.857143 |                      0.9   |               0.878049 |  0.947917 |
| SVM                 |       0.89 |    0.915254 | 0.9      | 0.907563 |                      0.853659 |                      0.875 |               0.864198 |  0.93375  |
| Random Forest       |       0.87 |    0.873016 | 0.916667 | 0.894309 |                      0.864865 |                      0.8   |               0.831169 |  0.921875 |
| XGBoost             |       0.83 |    0.864407 | 0.85     | 0.857143 |                      0.780488 |                      0.8   |               0.790123 |  0.907083 |
| Decision Tree       |       0.84 |    0.892857 | 0.833333 | 0.862069 |                      0.772727 |                      0.85  |               0.809524 |  0.873125 |

**Best model: Logistic Regression** (ROC-AUC = 0.9479)

## Notes
- Linear/distance-based models (Logistic Regression, SVM) were trained on StandardScaler-scaled
  features (scaler fit on the training split only, to avoid leakage). Tree-based models
  (Decision Tree, Random Forest, XGBoost) were trained on unscaled features.
- The Graduate Admission dataset's real target is a continuous admission probability
  (`chance_of_admit`); it was binarized at 0.7 to allow classification-style evaluation
  (ROC curve, confusion matrix) as called for in the Task 6 plan. If a continuous prediction is
  preferred, swap the classifiers for regressors (`RandomForestRegressor`, `XGBRegressor`, etc.)
  and evaluate with regression metrics (MAE, RMSE, R^2) instead.
- All five trained models per dataset are saved under `models/university/` and `models/graduate/`,
  not just the best one, so later tasks can compare or swap models.
- **Class imbalance handling:** every model uses `class_weight='balanced'` (XGBoost uses the
  equivalent `scale_pos_weight`, computed automatically from each dataset's own train-split class
  ratio) so the minority class isn't ignored in favor of trivially predicting the majority class.
  The evaluation tables above report both the standard class-1 metrics and explicit
  minority-class precision/recall/F1, since for the Graduate Admission dataset class 1 (admit) is
  the majority class - the minority-class columns are the more meaningful check on
  whether the model is doing real work rather than coasting on class balance.
- The University Master train/test split is the one created in Task 5 (Fit Score's admitted-cohort
  statistics use training rows only), so test-set outcomes never influence the features.
