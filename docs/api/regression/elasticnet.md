# Elastic Net Regression

Elastic Net regression (combined L1 and L2 penalty) fitted by coordinate
descent, with optional glmnet-compatible lambda scaling.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `elasticnet_fit` | Scalar | Fit on complete arrays in a single call |
| `elasticnet_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `elasticnet_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `elasticnet_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `elasticnet_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#elasticnet_fit_predict_by) |

## elasticnet_fit

**Signature:**

```text
elasticnet_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

`alpha` and `l1_ratio` are options, not positional arguments.

**Example:**

```sql
SELECT elasticnet_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'alpha': 0.1, 'l1_ratio': 0.5}
) AS fit;
```

## elasticnet_fit_agg

**Signature:**

```text
elasticnet_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE enet_demo AS
SELECT
    i::DOUBLE AS x1,
    (i % 5)::DOUBLE AS x2,
    ((i * 7) % 11)::DOUBLE AS noise_feature,
    2.0 + 1.5 * i + 0.8 * (i % 5) AS y
FROM range(1, 51) t(i);

-- Mostly L1: irrelevant features are pushed toward zero
SELECT (elasticnet_fit_agg(
    y, [x1, x2, noise_feature],
    {'alpha': 0.5, 'l1_ratio': 0.9}
)).coefficients AS coefficients
FROM enet_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `alpha` (alias `lambda`) | DOUBLE | `1.0` | Overall penalty strength, >= 0 |
| `l1_ratio` | DOUBLE | `0.5` | Mix between L1 and L2: 0 = ridge, 1 = lasso |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `max_iterations` (alias `max_iter`) | INTEGER | `1000` | Maximum coordinate-descent iterations |
| `tolerance` (alias `tol`) | DOUBLE | `1e-6` | Convergence tolerance |
| `lambda_scaling` | VARCHAR | `'raw'` | `'raw'` or `'glmnet'` (match R's glmnet scaling) |

Elastic Net has no inference output. Option keys the function does not support
raise an error.

```sql
SELECT (elasticnet_fit_agg(
    y, [x1, x2],
    {'alpha': 0.01, 'l1_ratio': 0.5, 'lambda_scaling': 'glmnet'}
)).coefficients AS coefficients
FROM enet_demo;
```

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | One coefficient per feature (zero for features removed by the L1 penalty) |
| `intercept` | `DOUBLE` | Intercept |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `adj_r_squared` | `DOUBLE` | Adjusted R² |
| `residual_std_error` | `DOUBLE` | Residual standard error |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

## Understanding l1_ratio

- **l1_ratio = 0**: pure ridge (L2 only)
- **l1_ratio = 0.5**: equal mix of L1 and L2
- **l1_ratio = 1**: pure lasso (L1 only)

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `elasticnet_fit` drops positions with NaN or infinite values.

## See Also

- [Ridge](ridge.md) - L2 penalty only
- [LARS](lars.md) - Least angle regression
- [Table macros](../macros/table_macros.md#elasticnet_fit_predict_by) - Per-group predictions
