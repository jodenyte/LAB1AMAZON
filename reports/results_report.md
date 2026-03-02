# Amazon Reviews Sentiment Analysis — Results Report

**Dataset**: Amazon Product Reviews (10% sample)  
**Test set size**: 11,369 reviews  
**Task**: 3-class sentiment classification (negative, neutral, positive)

---

## Model Performance Summary

| Model | Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|------------|---------------|
| MLP Classifier | 0.857 | 0.672 | **0.855** |
| Logistic Regression | 0.865 | 0.621 | 0.845 |
| Random Forest | 0.835 | 0.530 | 0.792 |

---

## Classification Reports

### Logistic Regression

```
              precision    recall  f1-score   support

    negative       0.76      0.63      0.69      1617
     neutral       0.49      0.16      0.24       853
    positive       0.89      0.98      0.93      8899

    accuracy                           0.87     11369
   macro avg       0.71      0.59      0.62     11369
weighted avg       0.84      0.87      0.85     11369
```

### Random Forest

```
              precision    recall  f1-score   support

    negative       0.91      0.32      0.47      1617
     neutral       0.96      0.12      0.21       853
    positive       0.83      1.00      0.91      8899

    accuracy                           0.83     11369
   macro avg       0.90      0.48      0.53     11369
weighted avg       0.85      0.83      0.79     11369
```

### MLP Classifier

```
              precision    recall  f1-score   support

    negative       0.70      0.69      0.70      1617
     neutral       0.40      0.38      0.39       853
    positive       0.92      0.93      0.93      8899

    accuracy                           0.86     11369
   macro avg       0.68      0.67      0.67     11369
weighted avg       0.85      0.86      0.86     11369
```

---

## Confusion Matrices

Labels: **negative** | **neutral** | **positive**

### Logistic Regression

|  | Pred: negative | Pred: neutral | Pred: positive |
|--|----------------|---------------|----------------|
| **True: negative** | 1,023 | 67 | 527 |
| **True: neutral** | 177 | 137 | 539 |
| **True: positive** | 142 | 78 | 8,679 |

### Random Forest

|  | Pred: negative | Pred: neutral | Pred: positive |
|--|----------------|---------------|----------------|
| **True: negative** | 511 | 2 | 1,104 |
| **True: neutral** | 26 | 103 | 724 |
| **True: positive** | 23 | 2 | 8,874 |

### MLP Classifier

|  | Pred: negative | Pred: neutral | Pred: positive |
|--|----------------|---------------|----------------|
| **True: negative** | 1,117 | 164 | 336 |
| **True: neutral** | 190 | 321 | 342 |
| **True: positive** | 283 | 311 | 8,305 |

---

## Conclusions

- **Best F1 (weighted)**: MLP Classifier (0.855), closely followed by Logistic Regression (0.845).
- **Best accuracy**: Logistic Regression (0.865).
- **Neutral class**: Hardest to predict across all models; Random Forest shows high precision but very low recall for neutral.
- **Positive class**: Easiest to predict; all models perform well due to class imbalance.
