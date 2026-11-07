# Performance Estimation Without Labels (CBPE)

Confidence-based performance estimation (CBPE) estimates how well a binary classifier performs
on new data before the labels arrive (#365). It follows NannyML's
`CBPE(problem_type = "classification_binary")`.

On a labelled **reference** period, the predicted scores are calibrated against the true
labels, and the sampling-error components of each metric are stored. On a **monitored** chunk
without labels, each row with calibrated probability `p` adds `p` to the expected true
positives (when predicted positive) or false negatives (when predicted negative), and `1 - p`
to the expected false positives or true negatives. Accuracy, precision, recall, F1 and
specificity follow from this expected confusion matrix. ROC AUC comes from the calibrated
probabilities ordered by the raw scores. Each estimate has NannyML's sampling error and the band
`value ∓ 3 · sampling_error`, clipped to `[0, 1]` (counts: to `≥ 0`).

All statistics come from upstream anofox-statistics `monitoring::cbpe`, which is validated
against NannyML 0.13.1. The extension tests are in `test/sql/reference/cbpe_reference.test`
(NannyML 0.13.1, `validation/generators/make_reference_cbpe.py`).

## Functions

| Function | Type | Returns |
|----------|------|---------|
| `cbpe_fit_agg(y_true, score [, options])` | Aggregate | model `STRUCT` |
| `cbpe_fit_agg(y_true, score, prediction, options)` | Aggregate | model `STRUCT` |
| `cbpe_estimate_agg(model, score [, prediction])` | Aggregate | `STRUCT` of estimated metrics |
| `cbpe_realized_agg(model, y_true, score [, prediction])` | Aggregate | `STRUCT` of realized metrics |
| `cbpe_by(source, chunk_col, y_col, score_col, is_reference, options := NULL, prediction := NULL)` | Table macro | one row per chunk and metric |

## Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `y_true` | BOOLEAN, or numeric 0/1 | The true class; any other value is an error. Rows with a NULL `y_true` are skipped. |
| `score` | DOUBLE | The model's score or predicted probability. Rows with a NULL score are skipped. A non-finite score makes the result NULL. |
| `prediction` | BOOLEAN | Optional: the model's predicted class (NannyML's `y_pred`). A NULL prediction falls back to `score >= threshold`. Without the argument, every row uses the threshold. |
| `model` | STRUCT | The result of `cbpe_fit_agg`, e.g. joined to every row of the monitored data. It is taken from the first row of a group where it is not NULL. No model gives NULL. |

### cbpe_fit_agg options

| Key | Default | Description |
|-----|---------|-------------|
| `calibration` | `'isotonic'` | `'isotonic'` (NannyML's default), `'platt'` (Platt's sigmoid), or `'none'` (use the scores as probabilities) |
| `threshold` | `0.5` | A row is predicted positive when `score >= threshold` (finite) |

NannyML decides with a randomised cross-validation whether to calibrate at all. Here you choose:
pass `'calibration': 'none'` when the scores are already calibrated.

A reference period with only one class gives NULL.

## Results

**Model** (`cbpe_fit_agg`): `STRUCT(calibration, threshold, isotonic_x, isotonic_y, platt_a,
platt_b, sampling, n_reference, model_type)`. The isotonic map's knots are `isotonic_x` /
`isotonic_y` (NULL for the other calibrations), and the Platt parameters are `platt_a` /
`platt_b` (NULL otherwise). `sampling` holds the 15 sampling-error components:
`accuracy_std`, `precision_std`, `precision_fraction`, `recall_std`, `recall_fraction`,
`specificity_std`, `specificity_fraction`, `f1_std`, `f1_fraction`, `roc_auc_std`,
`roc_auc_fraction`, `true_positive_std`, `false_positive_std`, `false_negative_std`,
`true_negative_std`. A chunk of `n` rows has sampling error `std / sqrt(n · fraction)` (accuracy:
`std / sqrt(n)`, counts: `std · sqrt(n)`). The model can be stored in a table and is rebuilt
with upstream's `CbpeModel::from_parts`.

**Estimate** (`cbpe_estimate_agg`): `STRUCT(n, roc_auc, f1, precision, recall, specificity,
accuracy, true_positive, false_positive, false_negative, true_negative)`. Each metric is
`STRUCT(value, sampling_error, lower_confidence, upper_confidence)`. The four counts are the
expected confusion matrix. A rate whose expected denominator is zero is 0 (as NannyML). ROC AUC
is NULL when all expected labels are of one class.

**Realized** (`cbpe_realized_agg`): `STRUCT(n, roc_auc, f1, precision, recall, specificity,
accuracy, true_positive, false_positive, false_negative, true_negative)` over the labelled rows,
with NannyML's rules. ROC AUC is NULL with one label class. F1, precision and recall are NULL
when the labels or the predictions have one class. Specificity is NULL without actual
negatives.

**`cbpe_by`** fits the model on the rows where the BOOLEAN expression `is_reference` is true (e.g. `period = 'reference'`). It
estimates every chunk of `chunk_col` and returns one row per chunk and metric, reference chunks
first:
`chunk_col, is_reference, metric, n, estimate, sampling_error, lower_confidence,
upper_confidence, n_labelled, realized`. `realized` is NULL in chunks without labels.

## Example

```sql
-- A miscalibrated score; the second half of the monitored rows has drifted.
CREATE OR REPLACE TABLE scored AS
SELECT i, i // 250 AS chunk, i < 1000 AS is_reference,
       1 / (1 + exp(-(0.8 * x + 0.3))) AS score,
       CASE WHEN i < 1000 THEN u2 < 1 / (1 + exp(-(1.5 * x - 0.7))) END AS y_true
FROM (
    SELECT i, 4 * u1 - 2 + CASE WHEN i >= 1500 THEN 0.8 ELSE 0.0 END AS x, u2
    FROM (SELECT i, ((i * i * 7919 + i * 104729 + 13) % 1000003) / 1000003.0 AS u1,
                 ((i * i * 6271 + i * 15485863 + 500101) % 999983) / 999983.0 AS u2
          FROM range(2000) r(i))
);

-- Estimated ROC AUC and accuracy per chunk; realized only where labels exist.
SELECT chunk, is_reference, metric, round(estimate, 3) AS estimate,
       round(lower_confidence, 3) AS lower, round(upper_confidence, 3) AS upper,
       round(realized, 3) AS realized
FROM cbpe_by('scored', chunk, y_true, score, is_reference)
WHERE metric IN ('roc_auc', 'accuracy')
ORDER BY chunk, metric;

-- The same with the aggregates: fit once, store the model, estimate later.
CREATE OR REPLACE TABLE cbpe_model AS
SELECT cbpe_fit_agg(y_true, score, {'calibration': 'isotonic'}) AS m
FROM scored WHERE is_reference;

SELECT chunk, (e).n, round((e).accuracy.value, 3) AS accuracy,
       round((e).accuracy.sampling_error, 4) AS accuracy_se,
       round((e).true_positive.value, 1) AS expected_tp
FROM (SELECT chunk, cbpe_estimate_agg(m, score) AS e
      FROM scored CROSS JOIN cbpe_model
      WHERE NOT is_reference GROUP BY chunk)
ORDER BY chunk;
```

Multiclass CBPE is not available yet.
