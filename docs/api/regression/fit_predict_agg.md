# Fit-Predict Aggregates

`*_fit_predict_agg` functions fit one model per group on the training rows and
return a prediction for **every** row of the group, as a list in input order.
Rows with a NULL `y` are not used for fitting but are predicted, which makes
these functions the building block for forecasting and hold-out evaluation.
The [table macros](../macros/table_macros.md) (`*_fit_predict_by`) wrap them
and return one output row per input row.

## Functions

| Function | `x` argument | Intervals | Model options |
|----------|--------------|-----------|---------------|
| `ols_fit_predict_agg` | `DOUBLE[]` | yes | [OLS](ols.md#options) |
| `ridge_fit_predict_agg` | `DOUBLE[]` | yes | [Ridge](ridge.md#options) |
| `wls_fit_predict_agg` | `DOUBLE[]`, plus `weights DOUBLE` | yes | [WLS](wls.md#options) |
| `rls_fit_predict_agg` | `DOUBLE[]` | yes | [RLS](rls.md#options) |
| `elasticnet_fit_predict_agg` | `DOUBLE[]` | yes | [Elastic Net](elasticnet.md#options) |
| `huber_fit_predict_agg` | `DOUBLE[]` | yes | [Huber](huber.md#options) |
| `ransac_fit_predict_agg` | `DOUBLE[]` | yes | [RANSAC](ransac.md#options) |
| `theil_sen_fit_predict_agg` | `DOUBLE[]` | yes | [Theil-Sen](theil_sen.md#options) |
| `bls_fit_predict_agg` | `DOUBLE[]` | yes | [BLS](bls.md#bls_fit_agg) |
| `alm_fit_predict_agg` | `DOUBLE[]` | yes | [ALM](../glm/alm.md) |
| `poisson_fit_predict_agg` | `DOUBLE[]` | yes | [Poisson](../glm/poisson.md) |
| `pls_fit_predict_agg` | `DOUBLE[]` | no | [PLS](pls.md#options) |
| `quantile_fit_predict_agg` | `DOUBLE[]` | no | [Quantile](quantile.md#options) |
| `isotonic_fit_predict_agg` | `DOUBLE` (single feature) | no | [Isotonic](isotonic.md#options) |

No deprecated aliases are registered; the names above are the only ones.

## Signatures

```text
<method>_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP])
<method>_fit_predict_agg(y DOUBLE, x DOUBLE[], split_col VARCHAR [, options MAP])

wls_fit_predict_agg(y DOUBLE, x DOUBLE[], weights DOUBLE [, split_col VARCHAR] [, options MAP])
isotonic_fit_predict_agg(y DOUBLE, x DOUBLE [, split_col VARCHAR] [, options MAP])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE` | Response; NULL marks a row to predict |
| `x` | `DOUBLE[]` | Feature values for the row |
| `split_col` | `VARCHAR` | Optional. `'train'` or `'training'` (case-insensitive) marks training rows; any other value, or NULL, marks a prediction row |
| `options` | `MAP` / `STRUCT` | Model options plus the keys below |

## Returns

A list with one element per input row (in input order; add `ORDER BY` inside
the call to fix the order):

```text
STRUCT(y DOUBLE, yhat DOUBLE, yhat_lower DOUBLE, yhat_upper DOUBLE, is_training BOOLEAN)[]
```

`pls_fit_predict_agg`, `quantile_fit_predict_agg` and
`isotonic_fit_predict_agg` return `STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]`.

| Field | Type | Description |
|-------|------|-------------|
| `y` | `DOUBLE` | Input `y` (NULL for prediction rows) |
| `yhat` | `DOUBLE` | Prediction (NULL if the row has a NULL feature or the fit failed) |
| `yhat_lower` | `DOUBLE` | Lower prediction-interval bound at `confidence_level` |
| `yhat_upper` | `DOUBLE` | Upper prediction-interval bound |
| `is_training` | `BOOLEAN` | Whether the row was used to fit the model |

The interval is leverage-aware: `yhat ± t(n − p) · s · sqrt(1 + x₀ᵀ M x₀)`
with `M = (XᵀX)⁻¹` for OLS (the same interval as R's `predict.lm`),
`(XᵀWX)⁻¹` for WLS (the new point at unit weight), the ridge sandwich
`A XᵀX A`, `A = (XᵀX + λI)⁻¹`, for Ridge, and the OLS leverage of the training
rows as an approximation for Huber, RANSAC, Theil-Sen, RLS, BLS, ALM and
Elastic Net (active columns only). Intervals are wider for rows far from the
training data. The bounds are NULL when no interval exists (zero residual
degrees of freedom, singular design) and equal to `yhat` for an exact fit. See
[Methodology](../../METHODOLOGY.md#prediction-intervals).

## Common options

Accepted by every function except PLS, quantile and isotonic, in addition to
the model options:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Prediction-interval level |
| `null_policy` | VARCHAR | `'drop'` | See [null_policy](#null_policy) |

`compute_inference` has no effect here. Option keys the function does not
support raise an error.

### null_policy

| Value | Training rows | Predicted rows |
|-------|---------------|----------------|
| `'drop'` (default) | `y` is not NULL | all rows |
| `'drop_y_zero_x'` | `y` is not NULL and no feature equals 0 | all rows |

## Examples

```sql
CREATE OR REPLACE TABLE fpa_demo AS
SELECT
    i AS t,
    CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS region,
    i::DOUBLE AS price,
    (i % 4)::DOUBLE AS promo,
    CASE WHEN i <= 32 THEN 50.0 + 1.2 * i + 3.0 * (i % 4) + sin(i) END AS sales,
    CASE WHEN i <= 24 THEN 'train' ELSE 'test' END AS split
FROM range(1, 41) t(i);

-- Forecast the rows with unknown sales, per region
SELECT region, p.yhat, p.yhat_lower, p.yhat_upper
FROM (
    SELECT region, unnest(ols_fit_predict_agg(sales, [price, promo] ORDER BY t)) AS p
    FROM fpa_demo
    GROUP BY region
)
WHERE NOT p.is_training
ORDER BY region;
```

```sql
-- Hold-out evaluation with a split column: rows 25-32 have a known y but are not trained on
SELECT
    round(avg(abs(p.y - p.yhat)), 3) AS test_mae
FROM (
    SELECT unnest(ridge_fit_predict_agg(sales, [price, promo], split, {'alpha': 0.1} ORDER BY t)) AS p
    FROM fpa_demo
)
WHERE NOT p.is_training AND p.y IS NOT NULL;
```

```sql
-- As a window over a partition: the same list is attached to every row,
-- index it with the row number to get the row's own prediction
SELECT
    t, region, sales,
    (pred[rn]).yhat AS yhat
FROM (
    SELECT *,
        row_number() OVER (PARTITION BY region ORDER BY t) AS rn,
        huber_fit_predict_agg(sales, [price, promo]) OVER (
            PARTITION BY region ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS pred
    FROM fpa_demo
)
ORDER BY region, t
LIMIT 6;
```

## NULL handling

- NULL `y`: prediction row (not trained on).
- A NULL element inside `x`: the row is kept in the output, not trained on,
  and its `yhat` is NULL.
- A NULL `x` list: the row is skipped and does **not** appear in the output
  list.
- With `split_col`, a training row whose `y` is NULL is treated as a
  prediction row.

## See Also

- [Table macros](../macros/table_macros.md) - One output row per input row
- [Window fit-predict](fit_predict_window.md) - Expanding/rolling window predictions
- [NULL semantics](../../NULL_SEMANTICS.md)
