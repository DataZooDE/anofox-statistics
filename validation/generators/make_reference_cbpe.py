#!/usr/bin/env python3
"""NannyML 0.13.1 reference values for test/sql/reference/cbpe_reference.test (#365).

Usage (NannyML needs Python <= 3.12):
    uv venv -p 3.12 /tmp/nml && VIRTUAL_ENV=/tmp/nml uv pip install nannyml==0.13.1 \
        scikit-learn==1.9.1 statsmodels
    /tmp/nml/bin/python validation/generators/make_reference_cbpe.py

The data are the deterministic range() expressions of the .test file (quadratic
congruential uniforms, integer arithmetic, so DuckDB and numpy agree exactly):

    u1 = ((i*i*7919 + i*104729 + 13) % 1000003) / 1000003
    u2 = ((i*i*6271 + i*15485863 + 500101) % 999983) / 999983
    x  = 4*u1 - 2 + shift          (shift 0.8 in analysis chunks 2 and 3)
    score  = 1 / (1 + exp(-(0.8 x + 0.3)))    (a miscalibrated model score)
    y_true = u2 < 1 / (1 + exp(-(1.5 x - 0.7)))

Reference: i in [0, 1000), analysis: i in [1000, 2000), chunks of 250 rows.
Case 'threshold': y_pred = score >= 0.5. Case 'pred045': y_pred = score >= 0.45
(predictions that do not come from the 0.5 threshold).
"""
import warnings

import numpy as np
import pandas as pd
import nannyml as nml

warnings.filterwarnings("ignore")

METRICS = ["roc_auc", "f1", "precision", "recall", "specificity", "accuracy"]
CM = ["true_positive", "false_positive", "false_negative", "true_negative"]


def data(lo, hi, cut):
    i = np.arange(lo, hi, dtype=np.int64)
    u1 = ((i * i * 7919 + i * 104729 + 13) % 1000003) / 1000003
    u2 = ((i * i * 6271 + i * 15485863 + 500101) % 999983) / 999983
    shift = np.where(i >= 1500, 0.8, 0.0)
    x = 4 * u1 - 2 + shift
    score = 1 / (1 + np.exp(-(0.8 * x + 0.3)))
    y = (u2 < 1 / (1 + np.exp(-(1.5 * x - 0.7)))).astype(int)
    return pd.DataFrame({"y_pred_proba": score, "y_pred": (score >= cut).astype(int), "y_true": y})


def case(name, cut):
    ref, ana = data(0, 1000, cut), data(1000, 2000, cut)
    est = nml.CBPE(y_pred_proba="y_pred_proba", y_pred="y_pred", y_true="y_true",
                   problem_type="classification_binary", metrics=METRICS + ["confusion_matrix"],
                   chunk_size=250)
    est.fit(ref)
    print(f"## {name}: needs_calibration = {est.needs_calibration}")
    res = est.estimate(ana).to_df()
    for period in ["reference", "analysis"]:
        p = res[res[("chunk", "period")] == period]
        for m in METRICS + CM:
            for col in ["value", "sampling_error", "lower_confidence_boundary",
                        "upper_confidence_boundary", "realized"]:
                vals = p[(m, col)].astype(float).to_numpy()
                print(f"{period}\t{m}\t{col}\t" + "\t".join(repr(float(v)) for v in vals))


case("threshold", 0.5)
case("pred045", 0.45)
