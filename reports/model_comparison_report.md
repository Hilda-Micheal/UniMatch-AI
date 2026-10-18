# Model Comparison Report — Task 6

## University Master dataset (target: `admit`)
| model               |   val_roc_auc |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:--------------------|--------------:|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| SVM                 |      0.769183 |   0.69528  |    0.705048 | 0.695945 | 0.700467 |                      0.6853   |                   0.694582 |               0.68991  |  0.760208 |
| XGBoost             |      0.755418 |   0.677591 |    0.657244 | 0.773777 | 0.710766 |                      0.70846  |                   0.576688 |               0.635819 |  0.745652 |
| Logistic Regression |      0.75035  |   0.68299  |    0.690919 | 0.689034 | 0.689975 |                      0.674719 |                   0.67665  |               0.675683 |  0.743425 |
| Random Forest       |      0.74106  |   0.646681 |    0.635755 | 0.725586 | 0.677707 |                      0.662038 |                   0.563907 |               0.609045 |  0.725689 |
| Decision Tree       |      0.606224 |   0.565962 |    0.56082  | 0.701764 | 0.623425 |                      0.57513  |                   0.423502 |               0.487805 |  0.597056 |

**Best model: SVM** (chosen on validation ROC-AUC = 0.7692; test ROC-AUC = 0.7602)

## Graduate Admission dataset (target: `admit_label`, chance_of_admit >= 0.7)
| model               |   val_roc_auc |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:--------------------|--------------:|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| Logistic Regression |      0.928333 |       0.89 |    0.915254 | 0.9      | 0.907563 |                      0.853659 |                      0.875 |               0.864198 |  0.955833 |
| Random Forest       |      0.919167 |       0.86 |    0.883333 | 0.883333 | 0.883333 |                      0.825    |                      0.825 |               0.825    |  0.929792 |
| SVM                 |      0.915    |       0.89 |    0.915254 | 0.9      | 0.907563 |                      0.853659 |                      0.875 |               0.864198 |  0.935417 |
| XGBoost             |      0.90375  |       0.84 |    0.892857 | 0.833333 | 0.862069 |                      0.772727 |                      0.85  |               0.809524 |  0.910833 |
| Decision Tree       |      0.7825   |       0.8  |    0.833333 | 0.833333 | 0.833333 |                      0.75     |                      0.75  |               0.75     |  0.7875   |

**Best model: Logistic Regression** (chosen on validation ROC-AUC = 0.9283; test ROC-AUC = 0.9558)

## Notes
- `val_roc_auc` is measured on the validation split and is the only number used to pick the best model.
  All other columns (including `roc_auc`) are measured on the test split, which plays no part in any choice.
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
- The University Master train/val/test split is the one created in Task 5. It is made **by applicant**,
  so no applicant has rows in more than one split, and Fit Score's admitted-cohort statistics use
  training rows only, so validation/test outcomes never influence the features.
- University, department and specialization enter as one-hot columns (plus frequency encodings), not as
  arbitrary integer codes, so the linear and kernel models can use them too.
