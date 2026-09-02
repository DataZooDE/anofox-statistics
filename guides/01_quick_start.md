# Anofox Statistics - Quick Start Guide

This guide helps you get started with regression analysis using the Anofox Statistics DuckDB extension.

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/anofox-statistics.git
cd anofox-statistics

# Build the extension
make release
```

### Loading the Extension

```sql skip
-- Load the extension (adjust path to your local build)
LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';
```

---

## Example 1: Simple Linear Regression

Fit a basic OLS model to find the relationship between variables.

```sql
-- Create sample data: y = 2x + 1
WITH sample_data AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0] as x_values,
        [3.0, 5.0, 7.0, 9.0, 11.0] as y_values
)
SELECT
    (ols_fit(y_values, [x_values])).coefficients[1] as slope,
    (ols_fit(y_values, [x_values])).intercept as intercept,
    (ols_fit(y_values, [x_values])).r_squared as r_squared
FROM sample_data;
```

**Expected Output:**
```
┌───────┬───────────┬───────────┐
│ slope │ intercept │ r_squared │
├───────┼───────────┼───────────┤
│ 2.0   │ 1.0       │ 1.0       │
└───────┴───────────┴───────────┘
```

---

## Example 2: Statistical Inference

Get p-values and confidence intervals for hypothesis testing.

```sql
WITH sample_data AS (
    SELECT
        [3.1, 4.9, 7.2, 8.8, 11.1] as y_values,
        [1.0, 2.0, 3.0, 4.0, 5.0] as x_values
)
SELECT
    fit.coefficients[1] as slope,
    fit.p_values[1] as slope_pvalue,
    fit.ci_lower[1] as ci_lower,
    fit.ci_upper[1] as ci_upper,
    fit.f_statistic,
    fit.f_pvalue
FROM (
    SELECT ols_fit(
        y_values,
        [x_values],
        {'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) as fit
    FROM sample_data
);
```

**Interpretation:**
- If `slope_pvalue < 0.05`, the slope is statistically significant
- The true slope lies within `[ci_lower, ci_upper]` with 95% confidence
- `f_pvalue` tests if the overall model is significant

---

## Example 3: Per-Group Regression

Use aggregate functions with GROUP BY for segmented analysis.

```sql
-- Create sample data with groups
WITH sales_data AS (
    SELECT * FROM (VALUES
        ('North', 100, 10),
        ('North', 150, 15),
        ('North', 200, 20),
        ('North', 250, 25),
        ('South', 120, 10),
        ('South', 180, 15),
        ('South', 240, 20),
        ('South', 300, 25)
    ) AS t(region, sales, marketing_spend)
)
SELECT
    region,
    ROUND((ols_fit_agg(
        sales::DOUBLE,
        [marketing_spend::DOUBLE]
    )).coefficients[1], 2) as sales_per_dollar,
    ROUND((ols_fit_agg(
        sales::DOUBLE,
        [marketing_spend::DOUBLE]
    )).intercept, 2) as base_sales,
    ROUND((ols_fit_agg(
        sales::DOUBLE,
        [marketing_spend::DOUBLE]
    )).r_squared, 3) as r_squared
FROM sales_data
GROUP BY region
ORDER BY region;
```

**Expected Output:**
```
┌────────┬─────────────────┬────────────┬───────────┐
│ region │ sales_per_dollar│ base_sales │ r_squared │
├────────┼─────────────────┼────────────┼───────────┤
│ North  │ 10.0            │ 0.0        │ 1.0       │
│ South  │ 12.0            │ 0.0        │ 1.0       │
└───────┴─────────────────┴────────────┴───────────┘
```

---

## Example 4: Rolling Regression (Time Series)

Use window functions for rolling/moving regression analysis.

```sql
-- Generate time series data
WITH time_series AS (
    SELECT
        row_number() OVER () as day,
        i::DOUBLE as x,
        (2.0 * i + 1.0 + (random() - 0.5))::DOUBLE as y
    FROM generate_series(1, 30) t(i)
)
SELECT
    day,
    ROUND(y, 2) as y,
    ROUND((ols_fit_agg(y, [x]) OVER (
        ORDER BY day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )).coefficients[1], 3) as rolling_7day_slope
FROM time_series
ORDER BY day;
```

---

## Example 5: Predictions

Generate predictions from a fitted model.

```sql
-- Fit model and predict
WITH training_data AS (
    SELECT
        [3.0, 5.0, 7.0, 9.0, 11.0] as y_train,
        [[1.0, 2.0, 3.0, 4.0, 5.0]] as x_train
),
model AS (
    SELECT ols_fit(y_train, x_train) as fit
    FROM training_data
)
SELECT
    predict(
        [[6.0, 7.0, 8.0]],  -- new x values
        model.fit.coefficients,
        model.fit.intercept
    ) as predictions
FROM model;
```

**Expected Output:** `[13.0, 15.0, 17.0]`

---

## Example 6: Model Comparison with Information Criteria

Compare models using AIC and BIC.

```sql
-- Compare two models using AIC and BIC (lower = better fit for complexity)
-- AIC/BIC take residual sum of squares, n observations, k parameters
SELECT
    'Model 1 (linear, k=2)' as model,
    ROUND(aic(45.2, 50, 2), 2) as aic_value,
    ROUND(bic(45.2, 50, 2), 2) as bic_value
UNION ALL
SELECT
    'Model 2 (quadratic, k=3)' as model,
    ROUND(aic(22.1, 50, 3), 2) as aic_value,
    ROUND(bic(22.1, 50, 3), 2) as bic_value;
```

**Interpretation:** Lower AIC indicates better model fit (balancing complexity and fit).

---

## Example 7: Residual Diagnostics

Check model assumptions with residual analysis.

```sql
WITH fitted AS (
    SELECT
        [3.1, 4.9, 7.2, 8.8, 11.1] as y,
        [3.0, 5.0, 7.0, 9.0, 11.0] as y_hat
)
SELECT
    diag.raw as raw_residuals,
    (jarque_bera(diag.raw)).p_value as normality_pvalue
FROM (
    SELECT residuals_diagnostics(y, y_hat) as diag
    FROM fitted
);
```

**Interpretation:**
- If `normality_pvalue > 0.05`, residuals are approximately normal
- Raw residuals should be randomly distributed around 0

---

## Aggregate Function Types

### OLS Aggregate
Basic regression for per-group analysis.
```sql skip
SELECT ols_fit_agg(y, [x1, x2]) FROM data GROUP BY category;
```

### WLS Aggregate
Weighted regression for heteroscedastic data.
```sql skip
SELECT wls_fit_agg(y, [x], weight) FROM data GROUP BY category;
```

### Ridge Aggregate
L2-regularized regression for multicollinearity.
```sql skip
SELECT ridge_fit_agg(y, [x1, x2], {'alpha': 0.5}) FROM data GROUP BY category;
```

### RLS Aggregate
Recursive least squares for adaptive/streaming analysis.
```sql skip
SELECT rls_fit_agg(y, [x], {'forgetting_factor': 0.95}) FROM streaming_data;
```

---

## Common Patterns

### Quick Coefficient Extraction
```sql skip
SELECT (ols_fit(y_array, x_array)).coefficients[1] as beta1;
```

### Grouped Analysis
```sql skip
SELECT
    category,
    (ols_fit_agg(y, [x])).*
FROM data
GROUP BY category;
```

### Rolling Window
```sql skip
SELECT
    date,
    (ols_fit_agg(y, [x]) OVER (
        ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )).coefficients[1] as rolling_30day_beta
FROM time_series;
```

---

## Troubleshooting

### Extension Loading Issues
```sql skip
-- Check if loaded
SELECT * FROM duckdb_extensions() WHERE extension_name = 'anofox_statistics';

-- Reload if needed
LOAD 'path/to/anofox_statistics.duckdb_extension';
```

### Type Errors
Ensure all numeric values are DOUBLE:
```sql skip
-- Illustrative: integer arrays — prefer explicit DOUBLE to avoid relying on implicit casts
SELECT ols_fit([1, 2, 3], [[1, 2, 3]]);
```
```sql
-- Correct: explicit DOUBLE
SELECT ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]);
```
From a real table, cast to DOUBLE:
```sql skip
SELECT ols_fit(array_agg(y::DOUBLE), [array_agg(x::DOUBLE)]) FROM my_table;
```

### Insufficient Observations
Minimum 3 observations required for single-feature regression with intercept:
```sql skip
-- Illustrative: too few observations (n < n_features + 1)
SELECT ols_fit([1.0, 2.0], [[1.0, 2.0]]);
```
```sql
-- Need at least 3 points
SELECT ols_fit([1.0, 2.0, 3.0], [[1.0, 2.0, 3.0]]);
```

---

## Next Steps

- See [API Reference](../docs/API_REFERENCE.md) for complete function documentation
- See [Technical Guide](02_technical_guide.md) for implementation details
- See [Business Guide](03_business_guide.md) for real-world applications
- See [Advanced Use Cases](04_advanced_use_cases.md) for complex scenarios
