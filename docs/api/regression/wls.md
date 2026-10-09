# WLS (Weighted Least Squares)

Weighted least squares for heteroscedastic data or observations of unequal
reliability. Supports SVD, QR or Cholesky solvers and HC standard errors.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `wls_fit` | Scalar | Fit on complete arrays in a single call |
| `wls_fit_agg` | Aggregate | Row-by-row accumulation; works with `GROUP BY` |
| `wls_fit_predict` | Window aggregate | Fit and predict per window frame, see [Window fit-predict](fit_predict_window.md) |
| `wls_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `wls_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#wls_fit_predict_by) |

WLS is the only regression whose calling convention has an extra positional
argument: the weights come after `x` and before the options.

## wls_fit

**Signature:**

```text
wls_fit(y DOUBLE[], x DOUBLE[][], weights DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Response values |
| `x` | `DOUBLE[][]` | Feature columns: each inner list is one feature |
| `weights` | `DOUBLE[]` | Observation weights, same length as `y` |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
SELECT wls_fit(
    [3.0, 5.1, 7.0, 9.2, 10.9],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    [1.0, 2.0, 3.0, 2.0, 1.0]   -- higher weight for middle observations
) AS fit;
```

## wls_fit_agg

**Signature:**

```text
wls_fit_agg(y DOUBLE, x DOUBLE[], weight DOUBLE [, options MAP]) -> STRUCT
```

**Example:**

```sql
CREATE OR REPLACE TABLE wls_demo AS
SELECT
    i::DOUBLE AS x,
    3.0 + 2.0 * i + (i % 4) * sqrt(i) * 0.3 AS y,
    1.0 / i AS weight                       -- noise grows with x
FROM range(1, 31) t(i);

SELECT (wls_fit_agg(y, [x], weight, {'compute_inference': true})).p_values AS p_values
FROM wls_demo;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `compute_inference` (alias `inference`) | BOOLEAN | `false` | Add standard errors, t-values, p-values, confidence intervals and the F-test |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | Level of the coefficient confidence intervals |
| `solver` | VARCHAR | `'svd'` | `'svd'`, `'qr'` or `'cholesky'` |
| `hc_type` | VARCHAR | `'none'` | Robust standard errors: `'none'`, `'hc0'`, `'hc1'`, `'hc2'`, `'hc3'` |

Option keys the function does not support raise an error.

```sql
SELECT (wls_fit_agg(y, [x], weight, {'solver': 'cholesky', 'hc_type': 'hc1', 'compute_inference': true})).std_errors
FROM wls_demo;
```

## Returns

Same structure as [OLS](ols.md#returns): `coefficients DOUBLE[]`,
`intercept DOUBLE`, `r_squared DOUBLE`, `adj_r_squared DOUBLE`,
`residual_std_error DOUBLE`, `n_observations BIGINT`, `n_features BIGINT`; with
`compute_inference = true` also `std_errors`, `t_values`, `p_values`,
`ci_lower`, `ci_upper` (`DOUBLE[]`), `f_statistic` and `f_pvalue` (`DOUBLE`).

`r_squared` is the weighted R².

## Choosing weights

- **Inverse variance**: `weight = 1 / variance` when the variance is known
- **Group size**: `weight = n` when each observation is a group mean
- **Reliability**: larger weights for more trustworthy observations

## NULL handling

The aggregate skips rows where `y`, `x` or `weight` is NULL, or where `x`
contains a NULL element. `wls_fit` drops positions with NaN or infinite
values.

## See Also

- [OLS](ols.md) - Equal-weighted regression
- [RLS](rls.md) - Recursive, adaptive regression
- [Table macros](../macros/table_macros.md#wls_fit_predict_by) - Per-group predictions
