# Window Fit-Predict Functions

Eight regressions have a window-aggregate form that fits a model over the
current window frame and returns the prediction, with a prediction interval,
for the frame's last row.

| Function | Model | Model page |
|----------|-------|------------|
| `ols_fit_predict` | Ordinary least squares | [OLS](ols.md) |
| `ridge_fit_predict` | Ridge (L2) | [Ridge](ridge.md) |
| `wls_fit_predict` | Weighted least squares | [WLS](wls.md) |
| `rls_fit_predict` | Recursive least squares | [RLS](rls.md) |
| `elasticnet_fit_predict` | Elastic Net (L1+L2) | [Elastic Net](elasticnet.md) |
| `huber_fit_predict` | Huber M-estimator | [Huber](huber.md) |
| `ransac_fit_predict` | RANSAC | [RANSAC](ransac.md) |
| `theil_sen_fit_predict` | Theil-Sen | [Theil-Sen](theil_sen.md) |

## Signatures

```text
ols_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
ridge_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
rls_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
elasticnet_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
huber_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
ransac_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
theil_sen_fit_predict(y DOUBLE, x DOUBLE[] [, options MAP]) OVER (...)
wls_fit_predict(y DOUBLE, x DOUBLE[], weight DOUBLE [, options MAP]) OVER (...)

-> STRUCT(yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE)
```

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `yhat` | `DOUBLE` | Prediction for the last row of the frame |
| `yhat_lower` | `DOUBLE` | Lower prediction-interval bound |
| `yhat_upper` | `DOUBLE` | Upper prediction-interval bound |

The result is NULL until the frame contains more training rows than
parameters (`n_features + 1` with an intercept), or when the fit fails.

## How the window frame is used

The model is fitted on the training rows of the frame (rows whose `y` is not
NULL), and the prediction is made for the **last row of the frame**. That
makes the useful pattern an ordered frame that ends at the current row:

```sql
CREATE OR REPLACE TABLE fpw_demo AS
SELECT
    i AS t,
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    i::DOUBLE AS x,
    -- the last four periods are unknown and get forecasts
    CASE WHEN i <= 26 THEN 2.0 + 1.5 * i + sin(i) END AS y
FROM range(1, 31) t(i);

-- Expanding-window in-sample fits; rows with NULL y get out-of-sample predictions
SELECT
    t, y,
    round((ols_fit_predict(y, [x]) OVER (
        ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).yhat, 3) AS yhat
FROM fpw_demo
ORDER BY t;
```

A rolling window works the same way; only the frame start changes:

```sql
-- Rolling 10-row window (frame ends at the current row)
SELECT t, round((ols_fit_predict(y, [x]) OVER (
    ORDER BY t ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
)).yhat, 3) AS yhat
FROM fpw_demo
ORDER BY t;
```

### Correct and incorrect frames

Correct:

- Frames that **end at `CURRENT ROW`** over a **unique ordering** (`ORDER BY t
  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` for expanding windows,
  `ROWS BETWEEN 9 PRECEDING AND CURRENT ROW` for rolling ones). The row's own
  `y` is part of the fit when it is known, so these are in-sample fits; rows
  with NULL `y` get genuine out-of-sample predictions.

Incorrect:

- **Frames that end before the current row** (`... AND 1 PRECEDING`): the
  prediction is made for the last row of the frame, not for the current row.
  Use the one-step-ahead idiom below instead.
- **`OVER (PARTITION BY g)` or `OVER ()` without `ORDER BY`**: every row in a
  partition shares one frame, so every row gets the **same** value (the
  prediction for the partition's last row). For one prediction per row over a
  whole group use the [fit-predict aggregates](fit_predict_agg.md) or the
  [table macros](../macros/table_macros.md).
- **`RANGE` frames, or `ORDER BY` on a column with ties**: the "last row" of a
  frame is not well defined.

### One-step-ahead predictions

To predict each row from a model fitted only on strictly earlier rows, fit
with the aggregate over a frame ending at `1 PRECEDING` and apply the
coefficients to the current row with `predict`:

<!-- TODO(lead): model-aware predict -->

```sql skip
SELECT
    t, y,
    predict([[x]],
            (ols_fit_agg(y, [x]) OVER w).coefficients,
            (ols_fit_agg(y, [x]) OVER w).intercept)[1] AS yhat_one_step
FROM fpw_demo
WINDOW w AS (ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
ORDER BY t;
```

```sql
-- Per-group expanding-window predictions
SELECT
    grp, t, y,
    round((ridge_fit_predict(y, [x], {'alpha': 0.1}) OVER (
        PARTITION BY grp ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).yhat, 3) AS yhat
FROM fpw_demo
ORDER BY grp, t;
```

```sql
-- WLS: weight is the third positional argument
SELECT
    t,
    (wls_fit_predict(y, [x], 1.0 / t) OVER (
        ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).yhat AS yhat
FROM fpw_demo
ORDER BY t
LIMIT 5;
```

## Options

Each function accepts the model options listed on its model page (for example
`alpha` for ridge, `epsilon` for Huber, `forgetting_factor` for RLS), plus:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Level of the prediction interval |
| `null_policy` | VARCHAR | `'drop'` | `'drop'`: train on rows with non-NULL `y`. `'drop_y_zero_x'`: additionally exclude rows where any feature is 0 from training |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |

`compute_inference` has no effect in the window form. Option keys the function
does not support raise an error.

## NULL handling

- Rows with NULL `y` are not used for training but still get a prediction.
- Rows whose `x` list is NULL are skipped.
- The model is refitted for every frame, so cost grows with frame size times
  row count. For large partitions prefer the fit-predict aggregates.
- `WHERE` is applied before window functions. To look at a few rows, compute
  the window in a subquery and filter outside it; filtering in the same query
  shrinks the frames.

## See Also

- [Fit-predict aggregates](fit_predict_agg.md) - One fit per group, a prediction for every row
- [Table macros](../macros/table_macros.md) - Long-format per-group predictions
