# LARS (Least Angle Regression)

Least angle regression builds the model forward, one feature at a time: at
each step it moves the coefficients in the direction equiangular to the
features most correlated with the current residual. Features are standardized
internally; coefficients are returned on the original scale.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `lars_fit_agg` | Aggregate | Fit a LARS model; works with `GROUP BY` |

There is no scalar, window, fit-predict or table-macro form for LARS. To
predict, combine the returned coefficients with the `predict` scalar or with
arithmetic on the feature columns.

## lars_fit_agg

**Signature:**

```text
lars_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE` | Response value |
| `x` | `DOUBLE[]` | Feature values for the row |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
CREATE OR REPLACE TABLE lars_demo AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    i::DOUBLE AS x1,
    (i % 7)::DOUBLE AS x2,
    ((i * 3) % 11)::DOUBLE AS x3,
    4.0 + 0.8 * i - 1.2 * (i % 7) + 0.05 * sin(i) AS y
FROM range(1, 61) t(i);

SELECT
    grp,
    (lars_fit_agg(y, [x1, x2, x3])).coefficients AS coefficients,
    (lars_fit_agg(y, [x1, x2, x3])).r_squared AS r_squared
FROM lars_demo
GROUP BY grp
ORDER BY grp;
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an intercept term |
| `alpha` (alias `lambda`) | DOUBLE | `0.0` | Regularization parameter passed to the LARS solver; must be >= 0 |

Option keys the function does not support raise an error. LARS has no
inference output.

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | One coefficient per feature |
| `intercept` | `DOUBLE` | Intercept |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `adj_r_squared` | `DOUBLE` | Adjusted R²; NULL while the LARS backend does not compute it |
| `residual_std_error` | `DOUBLE` | Residual standard error; NULL while the LARS backend does not compute it |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |

## NULL handling

Rows where `y` or `x` is NULL are skipped. Rows containing NaN or infinite
values are dropped before fitting. Constant features get a NaN coefficient.

## Predicting with the fitted coefficients

```sql
WITH fit AS (
    SELECT lars_fit_agg(y, [x1, x2, x3]) AS m FROM lars_demo
)
SELECT
    x1, x2, x3, y,
    round(m.intercept + list_dot_product(m.coefficients, [x1, x2, x3]), 3) AS yhat
FROM lars_demo, fit
LIMIT 5;
```

## Use Cases

- Forward feature selection on wide data
- Inspecting which predictors enter the model first
- Data with more features than observations

## See Also

- [Elastic Net](elasticnet.md) - L1+L2 penalty
- [Ridge](ridge.md) - L2 penalty
- [OLS](ols.md) - Unregularized baseline
