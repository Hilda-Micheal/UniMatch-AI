# Evaluation Report — Task 7 (Deep Learning)

## Best configurations (chosen on validation AUC)
| dataset    |   config | hidden        |   dropout |     l2 |   learning_rate |   epochs_run |   best_epoch |   train_auc |   val_auc |   train_loss |   val_loss |
|:-----------|---------:|:--------------|----------:|-------:|----------------:|-------------:|-------------:|------------:|----------:|-------------:|-----------:|
| university |        2 | (128, 64, 32) |       0.3 | 0.0001 |           0.001 |          135 |          115 |    0.765836 |  0.753384 |     0.587127 |   0.598148 |
| graduate   |        2 | (128, 64, 32) |       0.3 | 0.0001 |           0.001 |           75 |           55 |    0.93112  |  0.936921 |     0.348287 |   0.317701 |

## Test-set performance
| dataset    | model   |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:-----------|:--------|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| university | DNN     |   0.682421 |    0.716123 | 0.648165 | 0.680451 |                      0.652274 |                   0.719781 |               0.684367 |  0.753358 |
| graduate   | DNN     |   0.87     |    0.898305 | 0.883333 | 0.890756 |                      0.829268 |                   0.85     |               0.839506 |  0.950417 |

## Overfitting check (ROC-AUC)
| dataset    |   train_auc |   val_auc |   test_auc |   train_test_gap |
|:-----------|------------:|----------:|-----------:|-----------------:|
| university |    0.799064 |  0.753326 |   0.753358 |       0.0457062  |
| graduate   |    0.946114 |  0.9375   |   0.950417 |      -0.00430219 |

## DNN vs best ML model (Task 6)
| dataset    | model                             |   accuracy |       f1 |   roc_auc |
|:-----------|:----------------------------------|-----------:|---------:|----------:|
| university | XGBoost (Task 6 best)             |   0.728937 | 0.734589 |  0.800454 |
| university | DNN (Task 7)                      |   0.682421 | 0.680451 |  0.753358 |
| graduate   | Logistic Regression (Task 6 best) |   0.9      | 0.915254 |  0.947917 |
| graduate   | DNN (Task 7)                      |   0.87     | 0.890756 |  0.950417 |

## Notes
- Same features, targets and test sets as Task 6; 15% of the training rows used as a validation set.
- Overfitting prevention: dropout, L2 weight decay, early stopping on validation AUC (best weights restored), learning-rate reduction.
- Scaler fit on the training rows only; saved as `dnn_scaler_<dataset>.pkl` and must be applied before prediction.
