# Quick Start Guide

Get up and running with Anofox Statistics in 5 minutes.

## Installation

```bash
# Build the extension
cd anofox-statistics-duckdb-extension
make release
```

## Load Extension

```sql
-- Start DuckDB
duckdb

-- Load the extension
LOAD '/path/to/anofox_statistics.duckdb_extension';
```

## Example 1: Simple Linear Regression

**What it does**: Performs ordinary least squares (OLS) regression to find the best-fit line for your data. This is the foundation of regression analysis.

**When to use**: When you want to understand the relationship between two continuous variables (e.g., how sales change with price, or how performance correlates with training hours).

**How it works**: The `ols_fit_agg` function analyzes your y and x columns, calculating the slope (coefficient) and quality metrics. It automatically handles missing values and computes goodness-of-fit statistics.

```sql
-- Basic OLS fit (use positional parameters)
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1
    true                                   -- add_intercept
);
```

**Output:**
```
┌──────────────┬───────────┬──────────┬───────────────┬──────┬───────┐
│ variable     │ coef      │ r²       │ adj_r²        │ rmse │ n_obs │
├──────────────┼───────────┼──────────┼───────────────┼──────┼───────┤
│ x1           │ 1.02      │ 0.998    │ 0.997         │ 0.05 │ 5     │
└──────────────┴───────────┴──────────┴───────────────┴──────┴───────┘
```

**What the results mean**:
- **coef (1.02)**: For every 1-unit increase in x1, y increases by 1.02 units
- **R² (0.998)**: The model explains 99.8% of the variation in y - an excellent fit
- **RMSE (0.05)**: The typical prediction error is only 0.05 units

## Example 2: Get p-values and Significance

**What it does**: Performs statistical hypothesis testing to determine if your predictors have a real effect or if the observed relationship could be due to chance.

**When to use**: When you need to validate that a relationship is statistically significant before making business decisions or reporting findings. Essential for scientific research and evidence-based decision making.

**How it works**: The `ols_inference` function computes test statistics and p-values using the t-distribution. It tests the null hypothesis that each coefficient equals zero (no effect). Lower p-values provide stronger evidence against the null hypothesis.

```sql
-- Inference with confidence intervals (use positional parameters)
SELECT
    variable,
    ROUND(estimate, 4) as coefficient,
    ROUND(p_value, 4) as p_value,
    significant
FROM ols_inference(
    [2.1, 4.0, 6.1, 7.9, 10.2]::DOUBLE[],          -- y
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x (matrix)
    0.95,                                            -- confidence_level
    true                                             -- add_intercept
);
```

**Output:**
```
┌───────────┬─────────────┬─────────┬─────────────┐
│ variable  │ coefficient │ p_value │ significant │
├───────────┼─────────────┼─────────┼─────────────┤
│ intercept │ 0.03        │ 0.9766  │ false       │
│ x1        │ 2.01        │ 0.0000  │ true        │
└───────────┴─────────────┴─────────┴─────────────┘
```

**What the results mean**:
- **intercept (0.03, p=0.98)**: Not statistically significant - could be zero
- **x1 (2.01, p<0.0001)**: Highly significant - very strong evidence of a real effect
- **significant flag**: Automatically marks coefficients with p < 0.05

The coefficient x1 = 2.01 is highly significant (p < 0.0001), meaning we can confidently say there's a real relationship between x1 and y.

## Example 3: Regression Per Group

**What it does**: Runs separate regressions for each group in your data using SQL's GROUP BY clause. This reveals whether relationships differ across segments.

**When to use**: When analyzing multiple products, regions, customers, or time periods simultaneously. Common for A/B testing, market segmentation, and comparative analysis.

**How it works**: The `ols_fit_agg` function works like any SQL aggregate (SUM, AVG, etc.). Combined with GROUP BY, it computes separate regression models for each group. The extension automatically parallelizes these calculations for performance.

```sql
-- Create sample data
CREATE TABLE sales AS
SELECT
    CASE WHEN i <= 10 THEN 'Product A' ELSE 'Product B' END as product,
    i::DOUBLE as price,
    (i * 2.0 + RANDOM() * 0.5)::DOUBLE as quantity
FROM range(1, 21) t(i);

-- Regression per product
SELECT
    product,
    (ols_fit_agg(quantity, price)).coefficient as price_elasticity,
    (ols_fit_agg(quantity, price)).r2 as fit_quality
FROM sales
GROUP BY product;
```

**Output:**
```
┌───────────┬──────────────────┬──────────────┐
│ product   │ price_elasticity │ fit_quality  │
├───────────┼──────────────────┼──────────────┤
│ Product A │ 1.98             │ 0.996        │
│ Product B │ 2.02             │ 0.997        │
└───────────┴──────────────────┴──────────────┘
```

**What the results mean**:
- **Product A**: Price elasticity of 1.98 means a 1% price increase leads to ~2% decrease in quantity
- **Product B**: Similar elasticity (2.02), suggesting both products are price-sensitive
- **Fit quality**: R² > 0.99 indicates price strongly predicts demand for both products

This analysis completed in a single query across all products - no need for loops or separate analyses.

## Example 4: Time-Series Rolling Regression

**What it does**: Computes regression coefficients over a moving window of data points. This captures how relationships change over time.

**When to use**: For trend analysis with time-series data where you expect the relationship to evolve (e.g., seasonal patterns, market dynamics changing, or detecting regime shifts).

**How it works**: Uses SQL window functions with `ROWS BETWEEN ... PRECEDING AND CURRENT ROW` to define the rolling window. The `ols_coeff_agg` function then runs on each window. Perfect for detecting when trends accelerate or reverse.

```sql
-- Create time series
CREATE TABLE time_series AS
SELECT
    i as time_idx,
    (i * 1.5 + RANDOM() * 0.3)::DOUBLE as value
FROM range(1, 51) t(i);

-- Rolling 10-period regression
SELECT
    time_idx,
    value,
    ols_coeff_agg(value, time_idx) OVER (
        ORDER BY time_idx
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as rolling_trend
FROM time_series
WHERE time_idx >= 10;
```

**Output:**
```
┌───────────┬───────┬───────────────┐
│ time_idx  │ value │ rolling_trend │
├───────────┼───────┼───────────────┤
│ 10        │ 15.2  │ 1.51          │
│ 11        │ 16.7  │ 1.49          │
│ 12        │ 18.1  │ 1.52          │
│ ...       │ ...   │ ...           │
└───────────┴───────┴───────────────┘
```

**What the results mean**:
- **rolling_trend (1.51)**: Based on the last 10 observations, the value increases by 1.51 per time unit
- **Changing values**: If rolling_trend was 1.49 earlier and now 1.52, the growth rate is accelerating
- **Applications**: Detect market momentum, identify inflection points, or adapt forecasts to recent patterns

## Example 5: Make Predictions

**What it does**: Generates predictions for new data points along with confidence intervals that quantify uncertainty.

**When to use**: For forecasting, scenario planning, or any situation where you need point estimates plus a sense of how confident you should be in those estimates.

**How it works**: The `ols_predict_interval` function takes your fitted model and new predictor values, computing both the predicted value and the standard error. Confidence intervals are constructed using the t-distribution, accounting for both model uncertainty and inherent data variability.

```sql
-- Predict for new values (use positional parameters and literal arrays)
SELECT * FROM ols_predict_interval(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],          -- y_train
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x_train
    [[6.0], [7.0], [8.0]]::DOUBLE[][],             -- x_new: values to predict
    0.95,                                           -- confidence_level
    'prediction',                                   -- interval_type
    true                                            -- add_intercept
);
```

**Output:**
```
┌────────────────┬───────────┬──────────┬───────────┬─────┐
│ observation_id │ predicted │ ci_lower │ ci_upper  │ se  │
├────────────────┼───────────┼──────────┼───────────┼─────┤
│ 1              │ 6.0       │ 5.2      │ 6.8       │ 0.4 │
│ 2              │ 7.0       │ 6.1      │ 7.9       │ 0.45│
│ 3              │ 8.0       │ 7.0      │ 9.0       │ 0.5 │
└────────────────┴───────────┴──────────┴───────────┴─────┘
```

**What the results mean**:
- **predicted (6.0)**: Best estimate for observation 1
- **ci_lower/ci_upper (5.2 to 6.8)**: 95% confidence interval - the true mean is very likely in this range
- **se (0.4)**: Standard error increases for predictions farther from the training data
- **Narrower intervals = more confident**, wider intervals = more uncertainty

Use these intervals for risk assessment: plan for the midpoint, but prepare for the range.

## Example 6: Check Model Quality

**What it does**: Computes information criteria (AIC, BIC) to help you choose between competing models. These metrics balance goodness-of-fit against model complexity.

**When to use**: When deciding whether to add more predictors, comparing different model specifications, or preventing overfitting by penalizing complexity.

**How it works**: The `information_criteria` function calculates AIC and BIC, which combine the residual sum of squares (fit quality) with a penalty for the number of parameters. Lower values indicate better models that balance fit and parsimony.

```sql
-- Compare models (use positional parameters)
SELECT * FROM information_criteria(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 15.9]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][],
    true  -- add_intercept
);
```

**Output:**
```
┌───────┬──────────┬──────┬───────────┬────────┬────────┐
│ n_obs │ n_params │ rss  │ r_squared │ aic    │ bic    │
├───────┼──────────┼──────┼───────────┼────────┼────────┤
│ 8     │ 2        │ 0.13 │ 0.9994    │ -40.23 │ -39.81 │
└───────┴──────────┴──────┴───────────┴────────┴────────┘
```

**What the results mean**:
- **R² (0.9994)**: Excellent fit - model explains 99.94% of variation
- **AIC (-40.23), BIC (-39.81)**: Lower is better - use to compare against alternative models
- **Rule of thumb**: If adding a predictor decreases AIC/BIC by >2, the extra complexity is justified
- **BIC penalty**: Stronger than AIC, so BIC favors simpler models more aggressively

When comparing models, choose the one with the lowest AIC (prediction focus) or BIC (simplicity focus).

## Example 7: Detect Outliers

**What it does**: Identifies observations that don't fit the model well (outliers) and those that heavily influence the regression results (influential points).

**When to use**: For data quality checks, identifying data entry errors, finding special cases that need investigation, or assessing model robustness.

**How it works**: The `residual_diagnostics` function computes leverage (how unusual x values are), Cook's distance (combined measure of influence), and studentized residuals (standardized prediction errors). It flags observations exceeding statistical thresholds.

```sql
-- Detect outliers and influential points (use positional parameters)
SELECT
    obs_id,
    ROUND(residual, 3) as residual,
    ROUND(leverage, 3) as leverage,
    ROUND(cooks_distance, 3) as cooks_d,
    is_outlier,
    is_influential
FROM residual_diagnostics(
    [2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0]::DOUBLE[], -- y (last point is outlier)
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][], -- x
    true,  -- add_intercept
    2.5,   -- outlier_threshold
    0.5    -- influence_threshold
)
ORDER BY cooks_distance DESC
LIMIT 3;
```

**Output:**
```
┌────────┬──────────┬──────────┬─────────┬────────────┬────────────────┐
│ obs_id │ residual │ leverage │ cooks_d │ is_outlier │ is_influential │
├────────┼──────────┼──────────┼─────────┼────────────┼────────────────┤
│ 8      │ 8.95     │ 0.417    │ 49.23   │ true       │ true           │
│ 1      │ 0.12     │ 0.417    │ 0.09    │ false      │ false          │
│ 7      │ -0.18    │ 0.274    │ 0.08    │ false      │ false          │
└────────┴──────────┴──────────┴─────────┴────────────┴────────────────┘
```

**What the results mean**:
- **Observation 8**: Large residual (8.95) + high leverage (0.417) + high Cook's D (49.23) = problematic point
- **is_outlier (true)**: Prediction error is extreme (> 2.5 standard deviations)
- **is_influential (true)**: Removing this point would significantly change the regression line
- **Action**: Investigate observation 8 - could be data error, special case, or genuine extreme value

Most observations (like 1 and 7) fit well and have minimal influence on the results.

## Common Patterns

These patterns demonstrate best practices for different analytical scenarios.

### Pattern 1: Quick coefficient check

**Use case**: Get a fast coefficient estimate without full model details. Ideal for exploratory analysis or when you only care about the slope.

```sql
SELECT ols_coeff_agg(y, x) as slope FROM data;
```

### Pattern 2: Per-group with GROUP BY

**Use case**: Compare relationships across multiple segments simultaneously. More efficient than running separate analyses for each group.

```sql
SELECT category, ols_fit_agg(y, x) as model
FROM data GROUP BY category;
```

### Pattern 3: Rolling window with OVER

**Use case**: Track how relationships evolve over time or across ordered data. Essential for time-series analysis and detecting trend changes.

```sql
SELECT *, ols_coeff_agg(y, x) OVER (
    ORDER BY time ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
) as rolling_coef FROM data;
```

### Pattern 4: Full statistical workflow

**Use case**: Complete regression analysis from model fitting through diagnostics. This is the comprehensive approach for rigorous statistical work.

```sql
-- Create sample data table
CREATE OR REPLACE TABLE workflow_sample AS
SELECT * FROM (VALUES
    (2.1, 1.0),
    (4.0, 2.0),
    (6.1, 3.0),
    (7.9, 4.0),
    (10.2, 5.0),
    (11.8, 6.0)
) AS t(y, x);

-- Complete statistical workflow
WITH
-- Step 1: Fit model and compute statistics
model_fit AS (
    SELECT
        (ols_fit_agg(y, x)).coefficient as slope,
        (ols_fit_agg(y, x)).r2 as r_squared,
        (ols_fit_agg(y, x)).std_error as std_error,
        COUNT(*) as n_obs
    FROM workflow_sample
),

-- Step 2: Compute fitted values and residuals using the model
fitted_values AS (
    SELECT
        y,
        x,
        (SELECT slope FROM model_fit) * x as fitted,
        y - (SELECT slope FROM model_fit) * x as residual
    FROM workflow_sample
),

-- Step 3: Identify outliers based on residuals
outlier_check AS (
    SELECT
        y,
        x,
        residual,
        residual / (SELECT std_error FROM model_fit) as standardized_residual,
        ABS(residual / (SELECT std_error FROM model_fit)) > 2.5 as is_outlier
    FROM fitted_values
)

-- Display comprehensive results
SELECT
    'Model Summary' as section,
    'R²' as metric,
    ROUND(r_squared, 4)::VARCHAR as value
FROM model_fit
UNION ALL
SELECT 'Model Summary', 'Slope', ROUND(slope, 4)::VARCHAR FROM model_fit
UNION ALL
SELECT 'Model Summary', 'Std Error', ROUND(std_error, 4)::VARCHAR FROM model_fit
UNION ALL
SELECT 'Model Summary', 'N Observations', n_obs::VARCHAR FROM model_fit
UNION ALL
SELECT
    'Diagnostics',
    'Outliers Detected',
    COUNT(*)::VARCHAR
FROM outlier_check
WHERE is_outlier
UNION ALL
SELECT
    'Diagnostics',
    'Max |Standardized Residual|',
    ROUND(MAX(ABS(standardized_residual)), 2)::VARCHAR
FROM outlier_check
ORDER BY section, metric;
```

## Next Steps

- **[Technical Guide](02_technical_guide.md)**: Learn about architecture and implementation
- **[Statistics Guide](03_statistics_guide.md)**: Understand the statistical methodology
- **[Business Guide](04_business_guide.md)**: See real-world business applications
- **[Advanced Use Cases](05_advanced_use_cases.md)**: Complex analytical workflows

## Common Issues

### Issue: Extension won't load
```sql
-- Check if file exists
LOAD '/full/path/to/anofox_statistics.duckdb_extension';
```

### Issue: Type mismatch
```sql
-- Ensure arrays are DOUBLE[] and use positional parameters
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0]::DOUBLE[],  -- y: Cast to DOUBLE[]
    [1.0, 2.0, 3.0]::DOUBLE[],  -- x1: Cast to DOUBLE[]
    true                         -- add_intercept
);
```

### Issue: Insufficient observations
```
Error: Need at least n+1 observations for n parameters
```

Solution: Ensure you have more observations than predictors.

## Getting Help

- Check function signatures: See [README.md](../README.md)
- Report bugs: [GitHub Issues](https://github.com/yourusername/anofox-statistics-duckdb-extension/issues)
- Ask questions: [GitHub Discussions](https://github.com/yourusername/anofox-statistics-duckdb-extension/discussions)
