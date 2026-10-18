# Evaluation Report — Task 7 (Deep Learning)

## Best configurations (chosen on validation AUC)
| dataset    |   config | hidden   |   dropout |    l2 |   learning_rate |   epochs_run |   best_epoch |   train_auc |   val_auc |   train_loss |   val_loss |
|:-----------|---------:|:---------|----------:|------:|----------------:|-------------:|-------------:|------------:|----------:|-------------:|-----------:|
| university |        3 | (64, 32) |       0.3 | 0.001 |          0.0003 |          109 |           89 |    0.818357 |  0.787415 |     0.536518 |   0.57073  |
| graduate   |        3 | (64, 32) |       0.3 | 0.001 |          0.0003 |           50 |           30 |    0.927037 |  0.931667 |     0.390216 |   0.418633 |

## Test-set performance
| dataset    | model   |   accuracy |   precision |   recall |       f1 |   precision_minority(class 0) |   recall_minority(class 0) |   f1_minority(class 0) |   roc_auc |
|:-----------|:--------|-----------:|------------:|---------:|---------:|------------------------------:|---------------------------:|-----------------------:|----------:|
| university | DNN     |   0.705986 |    0.714652 | 0.708674 | 0.711651 |                      0.69705  |                   0.703167 |               0.700095 |   0.7795  |
| graduate   | DNN     |   0.88     |    0.913793 | 0.883333 | 0.898305 |                      0.833333 |                   0.875    |               0.853659 |   0.94625 |

## Overfitting check (ROC-AUC)
| dataset    |   train_auc |   val_auc |   test_auc |   train_test_gap |
|:-----------|------------:|----------:|-----------:|-----------------:|
| university |    0.837749 |  0.787447 |    0.7795  |       0.0582489  |
| graduate   |    0.941343 |  0.93125  |    0.94625 |      -0.00490741 |

## DNN vs best ML model (Task 6)
| dataset    | model                             |   val_roc_auc |   accuracy |       f1 |   roc_auc |
|:-----------|:----------------------------------|--------------:|-----------:|---------:|----------:|
| university | SVM (Task 6 best)                 |      0.769183 |   0.69528  | 0.700467 |  0.760208 |
| university | DNN (Task 7)                      |      0.787447 |   0.705986 | 0.711651 |  0.7795   |
| graduate   | Logistic Regression (Task 6 best) |      0.928333 |   0.89     | 0.907563 |  0.955833 |
| graduate   | DNN (Task 7)                      |      0.93125  |   0.88     | 0.898305 |  0.94625  |

## Notes
- Same features, targets and train / validation / test rows as Task 6 (split saved by Task 5; by applicant for the University dataset). Validation rows drive early stopping and the configuration choice; test rows are used once.
- In the comparison table `val_roc_auc` is the validation score both models were selected on; the other columns are test-set numbers.
- Overfitting prevention: dropout, L2 weight decay, early stopping on validation AUC (best weights restored), learning-rate reduction.
- Scaler fit on the training rows only; saved as `dnn_scaler_<dataset>.pkl` and must be applied before prediction.
