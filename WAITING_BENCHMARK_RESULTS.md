# Synthetic waiting-time comparison results

**All results are synthetic validation, not Nairobi operational accuracy.**

Dataset: 1,351,290 query rows; 171,810 arrivals; 8,760 continuous sessions; 365 days. Train: 943,946 rows / 255 days. Validation: 203,642 / 55 days. Test: 203,702 / 55 days. Test positive prevalence: 54.63%. Positive means next arrival within five minutes; probability cutoff 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| prior_baseline | 0.5463 | 0.5463 | 1.0000 | 0.7066 | 0.5000 | 0.0000 |
| logistic_regression | 0.8121 | 0.8146 | 0.8493 | 0.8316 | 0.8806 | 0.6199 |
| hist_gradient_boosting | 0.8145 | 0.8065 | 0.8690 | 0.8365 | 0.8877 | 0.6251 |

## Confusion matrices
Rows = actual [0,1], columns = predicted [0,1]. Each matrix is [[TN,FP],[FN,TP]].
- prior_baseline: [[0, 92424], [0, 111278]]
- logistic_regression: [[70910, 21514], [16766, 94512]]
- hist_gradient_boosting: [[69218, 23206], [14582, 96696]]

## Waiting-time regression
| Model | MAE minutes | RMSE minutes |
|---|---:|---:|
| last_headway_baseline | 2.1833 | 3.0836 |
| hist_gradient_boosting_median | 1.7313 | 2.5310 |

Existing mobile QRF model was not replaced or activated. These comparison models use the same waiting-time objective, with a derived binary research question for classification metrics. Correlated minute queries do not represent independent measured passengers. Read WAITING_BENCHMARK.md for split controls and limitations.
