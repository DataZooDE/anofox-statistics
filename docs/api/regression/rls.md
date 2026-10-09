# RLS (Recursive Least Squares)

Recursive least squares for online and adaptive regression, with exponential
forgetting of old observations.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `rls_fit` | Scalar | Fit on complete arrays in a single call |
| `rls_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `rls_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `rls_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `rls_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#rls_fit_predict_by) |

## rls_fit

**Signature:**

```text
rls_fit(y DOUBLE[], x DOUBLE[][] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values, in time order |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
SELECT rls_fit(
    [3.0, 5.0, 7.1, 8.9, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    {'forgetting_factor': 0.99}
) AS fit;
```

## rls_fit_agg

**Signature:**

```text
rls_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

RLS is order dependent. Pass an `ORDER BY` inside the aggregate call so the
observations are processed in time order:

```sql
CREATE OR REPLACE TABLE rls_demo AS
SELECT
    i AS t,
    i::DOUBLE AS x,
    CASE WHEN i <= 20 THEN 1.0 * i ELSE 3.0 * i - 40.0 END AS y   -- slope changes at t = 20
FROM range(1, 41) t(i);

SELECT
    (rls_fit_agg(y, [x] ORDER BY t)).coefficients AS no_forgetting,
    (rls_fit_agg(y, [x], {'forgetting_factor': 0.9} ORDER BY t)).coefficients AS adaptive
FROM rls_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `forgetting_factor` | DOUBLE | `1.0` | Exponential forgetting λ in (0, 1]; 1.0 means no forgetting |
| `initial_p_diagonal` (alias `p_diagonal`) | DOUBLE | `100.0` | Initial diagonal of the covariance matrix P |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |

RLS has no inference output. Option keys the function does not support raise
an error.

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Final coefficient estimates |
| `intercept` | `DOUBLE` | Intercept |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `adj_r_squared` | `DOUBLE` | Adjusted R² |
| `residual_std_error` | `DOUBLE` | Residual standard error |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

## Understanding the forgetting factor

- **λ = 1.0**: no forgetting; all observations weigh the same (converges to OLS)
- **λ = 0.99**: slight forgetting
- **λ = 0.95**: moderate forgetting, adapts to recent trends
- **λ = 0.90**: strong forgetting, rapid adaptation

The effective memory is roughly `1 / (1 - λ)` observations.

## NULL handling

The aggregate skips rows where `y` or `x` is NULL, or where `x` contains a NULL
element. `rls_fit` drops positions with NaN or infinite values.

## See Also

- [OLS](ols.md) - Static regression
- [WLS](wls.md) - Weighted regression
- [Table macros](../macros/table_macros.md#rls_fit_predict_by) - Per-group predictions
