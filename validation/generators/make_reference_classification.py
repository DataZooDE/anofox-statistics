#!/usr/bin/env python3
"""scikit-learn reference values for test/sql/reference/classification_reference.test (#153).

Usage: uv run --no-project --with scikit-learn python3 validation/generators/make_reference_classification.py
Datasets are identical to make_reference_classification.R and the .test file.
"""
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, log_loss, precision_recall_curve, roc_auc_score,
                             roc_curve)

# Scores double as probabilities; 0.55 is tied across labels 1, 0, 1.
y = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
s = np.array([0.9, 0.1, 0.8, 0.65, 0.4, 0.3, 0.55, 0.55, 0.7, 0.2,
              0.35, 0.95, 0.6, 0.05, 0.45, 0.5, 0.25, 0.75, 0.55, 0.15])


def rows(name, *cols):
    print(f"## {name}")
    for vals in zip(*cols):
        print("\t".join(repr(float(v)) for v in vals))


fpr, tpr, thr = roc_curve(y, s)
rows("roc_curve(y, s)  threshold fpr tpr", thr, fpr, tpr)
fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
rows("roc_curve(y, s, drop_intermediate=False)  threshold fpr tpr", thr, fpr, tpr)
prec, rec, thr = precision_recall_curve(y, s)
rows("precision_recall_curve(y, s)  threshold recall precision (last threshold: none)",
     list(thr) + [np.nan], rec, prec)
frac, mean = calibration_curve(y, s, n_bins=5, strategy="uniform")
rows("calibration_curve(y, s, n_bins=5)  mean_pred frac_pos", mean, frac)
print("## scalars")
print("roc_auc_score    ", repr(roc_auc_score(y, s)))
print("brier_score_loss ", repr(brier_score_loss(y, s)))
print("log_loss         ", repr(log_loss(y, s)))
print("log_loss(0/1 probs)", repr(log_loss([1, 0, 1], [1.0, 0.0, 0.0])))

# Constant scores and an all-positive sample.
fpr, tpr, thr = roc_curve([0, 1, 0, 1], [0.5] * 4)
rows("roc_curve constant scores", thr, fpr, tpr)
print("roc_auc_score constant", repr(roc_auc_score([0, 1, 0, 1], [0.5] * 4)))
fpr, tpr, thr = roc_curve([1, 1, 1], [0.2, 0.4, 0.6])
rows("roc_curve all positive (fpr nan)", thr, fpr, tpr)
