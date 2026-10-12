# Classifier Evaluation

ROC, precision-recall and calibration curves, ROC AUC, Brier score and log loss
for a binary classifier (#153). Each function aggregates the true class and the
classifier's score (or predicted probability) of every row, so it works per
group with `GROUP BY` and over windows.

The curves follow scikit-learn (`roc_curve`, `precision_recall_curve`,
`calibration_curve`) and are validated against it and against R `pROC`
(`test/sql/reference/classification_reference.test`).

## Functions

| Function | Type | Returns |
|----------|------|---------|
| `roc_agg(y_true, score [, drop_intermediate])` | Aggregate | `LIST(STRUCT(threshold, fpr, tpr))` |
| `pr_agg(y_true, score)` | Aggregate | `LIST(STRUCT(threshold, recall, precision))` |
| `calibration_agg(y_true, prob, n_bins [, conf_level])` | Aggregate | `LIST(STRUCT(bin, mean_pred, frac_pos, n, conf_low, conf_high))` |
| `auc_agg(y_true, score)` | Aggregate | `DOUBLE` |
| `brier_agg(y_true, prob)` | Aggregate | `DOUBLE` |
| `log_loss_agg(y_true, prob)` | Aggregate | `DOUBLE` |
| `roc_curve_by`, `pr_curve_by`, `calibration_curve_by` | Table macro | the `curve` schema, one row per point |

## Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `y_true` | BOOLEAN, or numeric 0/1 | The true class; `true`/1 is the positive class. Any other value is an error. |
| `score` | DOUBLE | Any score where larger means "more positive"; `±inf` is an ordinary extreme score (as pROC; sklearn rejects it) |
| `prob` | DOUBLE | A predicted probability in [0, 1]; a value outside is an error |
| `drop_intermediate` | BOOLEAN constant | Drop collinear ROC points (default `true`, sklearn's default) |
| `n_bins` | BIGINT constant | Number of equal-width bins on [0, 1] (>= 1) |
| `conf_level` | DOUBLE constant | Level of the Wilson intervals (default 0.95) |

## Semantics

- **NULLs.** A row whose `y_true` or score is NULL is skipped. No rows left gives
  NULL. A NaN score also gives NULL.
- **Ties.** Tied scores form a single curve point (as sklearn and pROC).
- **ROC.** Points run in decreasing threshold order from `(fpr, tpr) = (0, 0)` at
  threshold `+inf` to `(1, 1)` at the smallest score; a point at threshold `t`
  classifies `score >= t` as positive. With `drop_intermediate = false` every
  distinct score is a point, which equals `pROC::coords(roc, "all")` in fpr/tpr
  (pROC reports midpoints between scores as thresholds).
- **PR.** Points run in increasing threshold order, followed by the point
  `(recall 0, precision 1)` whose `threshold` is NULL.
- **Calibration.** Bin 0 is `[0, 1/n_bins]`, bin `i > 0` is
  `(i/n_bins, (i+1)/n_bins]` (zero-based, as sklearn's `searchsorted`); empty
  bins are omitted. `conf_low`/`conf_high` are the Wilson score interval of
  `frac_pos`, R `prop.test(x, n, correct = FALSE)$conf.int`.
- **Constant scores** give the two ROC points `(0, 0)` and `(1, 1)` and AUC 0.5.
- **One class only.** The rate that is undefined (fpr without negatives, tpr
  without positives) is NULL where sklearn returns `nan`, and `auc_agg` is NULL
  (sklearn raises). Brier score and log loss are still defined.
- **AUC** is the trapezoidal area under the full ROC curve; ties count one half
  (sklearn `roc_auc_score`, pROC `auc`).
- **Log loss** clips probabilities to `[eps, 1 - eps]` with `eps` the double
  machine epsilon (sklearn's default), so 0 and 1 give a finite loss.

```sql
CREATE TABLE scored AS SELECT * FROM (VALUES
  ('a', true, 0.9), ('a', false, 0.1), ('a', true, 0.8), ('a', true, 0.65), ('a', false, 0.4),
  ('a', false, 0.3), ('a', true, 0.55), ('a', false, 0.55), ('a', true, 0.7), ('a', false, 0.2)
) t(model, label, score);

SELECT auc_agg(label, score) AS auc, brier_agg(label, score) AS brier FROM scored;

SELECT unnest(roc_agg(label, score), recursive := true) FROM scored;

SELECT unnest(calibration_agg(label, score, 5), recursive := true) FROM scored;
```

## Curve table macros

```text
roc_curve_by(source, group_col, y_true_col, score_col, drop_intermediate := true)
pr_curve_by(source, group_col, y_true_col, score_col)
calibration_curve_by(source, group_col, y_true_col, prob_col, n_bins := 10, conf_level := 0.95)
```

Each computes the curve per group of `source` and returns the integration
contract's `curve` schema, one row per point in curve order:

| Column | ROC | PR | Calibration |
|--------|-----|----|-------------|
| group column | as in `source` | | |
| `model_id` | the group value as VARCHAR | | |
| `series` | NULL | NULL | NULL |
| `curve_type` | `'roc'` | `'pr'` | `'calibration'` |
| `x` | false positive rate | recall | mean predicted probability |
| `y` | true positive rate | precision | fraction of positives |
| `y_low`, `y_high` | NULL | NULL | Wilson interval |
| `threshold` | threshold | threshold (NULL at the last point) | NULL |
| `weight` | NULL | NULL | bin count |
| `n_censor` | NULL | NULL | NULL |

The three macros share this shape, so curves of several models or curve types
combine with `UNION ALL`:

```sql
SELECT * FROM roc_curve_by('scored', model, label, score)
UNION ALL
SELECT * FROM calibration_curve_by('scored', model, label, score, n_bins := 5);
```
