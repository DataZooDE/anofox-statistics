# Diagnostic Functions

Model diagnostics: multicollinearity (VIF), information criteria (AIC/BIC),
residual normality (Jarque-Bera) and residual analysis. Most come in a scalar
form that takes arrays and an aggregate form that takes one row at a time.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `vif` | Scalar | Variance inflation factors from feature arrays |
| `vif_agg` | Aggregate | Variance inflation factors from rows |
| `aic` | Scalar | Akaike information criterion from RSS |
| `bic` | Scalar | Bayesian information criterion from RSS |
| `jarque_bera` | Scalar | Jarque-Bera normality test on an array |
| `jarque_bera_agg` | Aggregate | Jarque-Bera normality test on rows |
| `residuals_diagnostics` | Scalar | Raw, standardized and studentized residuals, leverage |
| `residuals_diagnostics_agg` | Aggregate | Same, from rows |

## Variance Inflation Factor

### vif

```text
vif(x DOUBLE[][]) -> DOUBLE[]
```

`x` holds one inner list per feature (feature-major). Returns one VIF per
feature.

```sql
SELECT vif([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [2.0, 4.1, 6.2, 7.9, 10.1],   -- almost 2 * the first feature
    [5.0, 3.0, 4.0, 1.0, 2.0]
]) AS vif_values;
```

### vif_agg

```text
vif_agg(x DOUBLE[]) -> DOUBLE[]
```

```sql
CREATE OR REPLACE TABLE diag_demo AS
SELECT
    i AS id,
    i::DOUBLE AS x1,
    2.0 * i + (i % 3) * 0.1 AS x2,
    ((i * 7) % 5)::DOUBLE AS x3,
    1.0 + 0.5 * i + ((i * 7) % 5) + 0.3 * sin(i) AS y
FROM range(1, 31) t(i);

SELECT vif_agg([x1, x2, x3]) AS vif_values FROM diag_demo;
```

| VIF | Interpretation |
|-----|----------------|
| 1 | No correlation with the other features |
| 1-5 | Moderate correlation |
| > 5 | High correlation (warning) |
| > 10 | Very high correlation (problematic) |

## Information Criteria

### aic

```text
aic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `rss` | `DOUBLE` | Residual sum of squares |
| `n` | `BIGINT` | Number of observations |
| `k` | `BIGINT` | Number of parameters, including the intercept |

AIC = n · ln(RSS / n) + 2k. Lower is better.

```sql
SELECT aic(100.0, 50, 3) AS aic_value;
```

### bic

```text
bic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

BIC = n · ln(RSS / n) + k · ln(n). Lower is better; penalizes extra parameters
more than AIC once n > 7.

```sql
SELECT bic(100.0, 50, 3) AS bic_value;
```

| Criterion | Best for |
|-----------|----------|
| AIC | Prediction, when the true model may not be among the candidates |
| BIC | Model identification; consistent as n grows |

## Normality

### jarque_bera

```text
jarque_bera(values DOUBLE[]) -> STRUCT
```

```sql
SELECT jarque_bera([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 3.0, 2.0, 8.0, 1.0]) AS jb;
```

### jarque_bera_agg

```text
jarque_bera_agg(value DOUBLE) -> STRUCT
```

```sql
-- Normality of OLS residuals
WITH fit AS (SELECT ols_fit_agg(y, [x1, x3]) AS m FROM diag_demo)
SELECT (jarque_bera_agg(y - (m.intercept + list_dot_product(m.coefficients, [x1, x3])))).p_value AS p_value
FROM diag_demo, fit;
```

**Returns** (both forms):

| Field | Type | Description |
|-------|------|-------------|
| `statistic` | `DOUBLE` | Jarque-Bera statistic |
| `p_value` | `DOUBLE` | p-value (chi-squared, 2 df) |
| `skewness` | `DOUBLE` | Sample skewness |
| `kurtosis` | `DOUBLE` | Sample excess kurtosis (0 for a normal distribution) |
| `n` | `BIGINT` | Number of values used |

NULL values are ignored by the aggregate.

## Residual Analysis

### residuals_diagnostics

```text
residuals_diagnostics(y DOUBLE[], y_hat DOUBLE[]) -> STRUCT
residuals_diagnostics(y DOUBLE[], y_hat DOUBLE[], x DOUBLE[][],
                      residual_std_error DOUBLE, include_studentized BOOLEAN) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE[]` | Observed values |
| `y_hat` | `DOUBLE[]` | Fitted values |
| `x` | `DOUBLE[][]` | Feature columns (feature-major); needed for leverage |
| `residual_std_error` | `DOUBLE` | Residual standard error of the model; needed for standardized residuals |
| `include_studentized` | `BOOLEAN` | Also compute studentized residuals |

With only `y` and `y_hat`, just the raw residuals are filled; the other fields
are NULL.

```sql
SELECT residuals_diagnostics(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.1, 1.9, 3.2, 3.8, 5.1],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.2,
    true
) AS diagnostics;
```

### residuals_diagnostics_agg

```text
residuals_diagnostics_agg(y DOUBLE, y_hat DOUBLE [, x DOUBLE[]]) -> STRUCT
```

The aggregate computes raw residuals and, when `x` is given, leverage.

```sql
SELECT residuals_diagnostics_agg(y, y_hat, [x]) AS diagnostics
FROM (VALUES (1.0, 1.1, 1.0), (2.0, 1.9, 2.0), (3.0, 3.2, 3.0),
             (4.0, 3.8, 4.0), (5.0, 5.1, 5.0)) t(y, y_hat, x);
```

**Returns** (both forms):

| Field | Type | Description |
|-------|------|-------------|
| `raw` | `DOUBLE[]` | Raw residuals `y - y_hat` |
| `standardized` | `DOUBLE[]` | `raw / residual_std_error` (NULL unless `residual_std_error` is given) |
| `studentized` | `DOUBLE[]` | Internally studentized residuals `raw / (σ · √(1 - h))` (NULL unless requested) |
| `leverage` | `DOUBLE[]` | Diagonal of the hat matrix (NULL unless `x` is given) |

## Detecting problems

- **High leverage**: leverage > 2(k + 1) / n suggests an influential point;
  check its studentized residual.
- **Outliers**: |studentized residual| > 3.
- **Non-normal residuals**: small Jarque-Bera p-value.
- **Multicollinearity**: VIF > 5 (moderate) or > 10 (severe); consider
  [Ridge](../regression/ridge.md) or dropping features.

## See Also

- [OLS](../regression/ols.md) - Standard regression
- [Ridge](../regression/ridge.md) - Regularization for multicollinearity
- [Hypothesis tests](../statistics/hypothesis.md) - Statistical tests
