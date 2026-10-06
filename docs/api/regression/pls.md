# PLS (Partial Least Squares)

Partial least squares regression (SIMPLS) for high-dimensional or strongly
collinear predictors. PLS extracts a small number of latent components that
maximise the covariance between the feature scores and `y`, then regresses `y`
on those components.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pls_fit_agg` | Aggregate | Fit a PLS model per group <!-- TODO(lead): verify after *_fit_agg lands --> |
| `pls_fit_predict_agg` | Aggregate | Fit and predict every row of a group |
| `pls_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#pls_fit_predict_by) |

## pls_fit_agg

<!-- TODO(lead): verify after *_fit_agg lands -->

**Signature:**

```text
pls_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

Options: `n_components`, `fit_intercept` (see [Options](#options)).

```sql skip
-- TODO(lead): un-skip once pls_fit_agg is registered
SELECT (pls_fit_agg(y, [x1, x2, x3], {'n_components': 2})).coefficients
FROM pls_demo;
```

## pls_fit_predict_agg

**Signature:**

```text
pls_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
pls_fit_predict_agg(y DOUBLE, x DOUBLE[], split_col VARCHAR [, options MAP]) -> STRUCT(y DOUBLE, yhat DOUBLE, is_training BOOLEAN)[]
```

Rows with a NULL `y` (or, with the split form, rows whose `split_col` is not
`'train'`/`'training'`) are not used for fitting but still receive a
prediction. PLS returns point predictions only; there are no `yhat_lower` /
`yhat_upper` fields. See [Fit-predict aggregates](fit_predict_agg.md) for the
general behaviour.

**Example:**

```sql
CREATE OR REPLACE TABLE pls_demo AS
SELECT
    i AS id,
    i::DOUBLE AS x1,
    i::DOUBLE * 0.98 + (i % 3) * 0.1 AS x2,    -- strongly collinear with x1
    (i % 4)::DOUBLE AS x3,
    CASE WHEN i <= 25 THEN 1.0 + 0.5 * i + 0.3 * (i % 4) END AS y   -- last 5 rows unknown
FROM range(1, 31) t(i);

SELECT p.y, round(p.yhat, 3) AS yhat, p.is_training
FROM (
    SELECT unnest(pls_fit_predict_agg(y, [x1, x2, x3], {'n_components': 2} ORDER BY id)) AS p
    FROM pls_demo
)
WHERE NOT p.is_training;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_components` (alias `components`) | INTEGER | `1` | Number of latent components to extract |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |

Option keys the function does not support raise an error.

## Choosing n_components

- **1**: a single direction of maximal covariance (default)
- **2-3**: typical for moderate-dimensional data
- At most `min(n_observations - 1, n_features)`

Use hold-out validation (the `split_col` form) to choose the number of
components.

## NULL handling

Rows where `x` is NULL are skipped. Rows with NULL `y` are prediction rows.

## Use Cases

- More features than observations
- Strongly correlated predictors
- Chemometrics and spectroscopy (NIR, Raman)
- Gene expression and other wide data

## See Also

- [Ridge](ridge.md) - L2 regularization alternative
- [Elastic Net](elasticnet.md) - L1+L2 regularization
- [Fit-predict aggregates](fit_predict_agg.md)
