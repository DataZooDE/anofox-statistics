# Quick Start Guide

This guide demonstrates the core regression analysis capabilities of the Anofox Statistics extension for DuckDB.

## Table of Contents

- [Installation](#installation)
- [Load Extension](#load-extension)
- [Example 1: Simple Linear Regression](#example-1-simple-linear-regression)
- [Example 2: Get p-values and Significance](#example-2-get-p-values-and-significance)
- [Example 3: Regression Per Group](#example-3-regression-per-group)
- [Example 4: Time-Series Rolling Regression](#example-4-time-series-rolling-regression)
- [Example 5: Make Predictions](#example-5-make-predictions)
- [Example 6: Check Model Quality](#example-6-check-model-quality)
- [Example 7: Detect Outliers](#example-7-detect-outliers)
- [Aggregate Functions for GROUP BY](#aggregate-functions-for-group-by)
  - [OLS Aggregate](#ols-aggregate-basic-per-group-regression)
  - [WLS Aggregate](#wls-aggregate-weighted-analysis)
  - [Ridge Aggregate](#ridge-aggregate-handling-multicollinearity)
  - [RLS Aggregate](#rls-aggregate-adaptive-online-learning)
- [Common Patterns](#common-patterns)
  - [Pattern 1: Quick coefficient check](#pattern-1-quick-coefficient-check)
  - [Pattern 2: Per-group with GROUP BY](#pattern-2-per-group-with-group-by)
  - [Pattern 3: Rolling window with OVER](#pattern-3-rolling-window-with-over)
  - [Pattern 4: Full statistical workflow](#pattern-4-full-statistical-workflow)
- [Common Issues](#common-issues)

## Installation

```bash
# Build the extension
cd anofox-statistics-duckdb-extension
make release
```

## Load Extension


```sql
-- DISABLED: This is a documentation example, not a runnable test
-- Shows command-line usage, not SQL

SELECT 'guide01_load_extension.sql - DISABLED - documentation example only' as status;
```

## Example 1: Simple Linear Regression

**How it works**: The `anofox_statistics_ols_fit` table function fits an OLS regression model using QR decomposition from libanostat. It takes array inputs for y (response) and X (feature matrix), along with configuration options in a MAP.


```sql

-- Simple linear regression using OLS
WITH input AS (
    SELECT
        LIST_VALUE(1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0) as y,
        LIST_VALUE(LIST_VALUE(1.1::DOUBLE, 2.1, 2.9, 4.2, 4.8)) as X
)
SELECT result.* FROM input,
LATERAL anofox_statistics_ols_fit(
    input.y,
    input.X,
    {'intercept': true}
) as result;
```

**Technical interpretation**:

- **coefficients[1]**: Estimated slope parameter (β₁)
- **r_squared**: Coefficient of determination (proportion of variance explained)
- **adj_r_squared**: Adjusted R² penalizing additional predictors
- **mse**: Mean squared error (residual variance)
- **n_obs**: Sample size used in estimation

## Example 2: Get p-values and Significance

**How it works**: The aggregate fit functions compute t-statistics and p-values for hypothesis testing. Each coefficient is tested against H₀: β = 0 using the t-distribution with df_residual degrees of freedom.


```sql

-- Get statistical inference using fit with full_output
WITH model AS (
    SELECT * FROM anofox_statistics_ols_fit(
        [2.1, 4.0, 6.1, 7.9, 10.2]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95::DOUBLE}
    )
)
SELECT
    'x1' as variable,
    ROUND(coefficients[1], 4) as coefficient,
    ROUND(coefficient_p_values[1], 4) as p_value,
    coefficient_p_values[1] < 0.05 as significant
FROM model
UNION ALL
SELECT
    'intercept' as variable,
    ROUND(intercept, 4) as coefficient,
    ROUND(intercept_p_value, 4) as p_value,
    intercept_p_value < 0.05 as significant
FROM model;
```

**Technical interpretation**:

- **coefficient_p_values**: Two-tailed p-values for testing H₀: β_i = 0
- **intercept_p_value**: Two-tailed p-value for testing H₀: intercept = 0
- **Interpretation**: p < α (e.g., 0.05) provides evidence to reject the null hypothesis
- **coefficient_t_statistics**: Test statistics computed as β̂ / SE(β̂)

## Example 3: Regression Per Group

**How it works**: The `anofox_statistics_ols_fit_agg` aggregate function supports GROUP BY to fit separate models per group. Each group accumulates observations independently and computes a complete regression result struct.


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
    (anofox_statistics_ols_fit_agg(quantity, [price], {'intercept': true})).coefficients[1] as price_elasticity,
    (anofox_statistics_ols_fit_agg(quantity, [price], {'intercept': true})).r2 as fit_quality
FROM sales
GROUP BY product;
```

**Technical interpretation**:

- **GROUP BY semantics**: Each group receives a separate OLS estimation
- **coefficients[1]**: Slope estimate per group
- **r2**: Within-group coefficient of determination
- **n_obs**: Sample size per group used in estimation

## Example 4: Time-Series Rolling Regression

**How it works**: Aggregate functions support OVER clauses for window-based computation. The `ROWS BETWEEN n PRECEDING AND CURRENT ROW` frame defines a sliding window, with regression re-computed at each row.


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
    (anofox_statistics_ols_fit_agg(value, [time_idx], {'intercept': true}) OVER (
        ORDER BY time_idx
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    )).coefficients[1] as rolling_trend
FROM time_series
WHERE time_idx >= 10;
```

**Technical interpretation**:

- **OVER (ORDER BY ... ROWS BETWEEN ...)**: Window frame specification
- **coefficients[1]**: Slope estimate using observations in the current window
- **Window size**: Number of preceding rows + current row determines sample size
- **Computation**: Each row gets a new regression using only the windowed data

## Example 5: Make Predictions

**How it works**: The `anofox_statistics_predict_ols` table function fits a model on training data (y_train, x_train) and generates predictions for new observations (x_new) with confidence or prediction intervals using the t-distribution.


```sql

-- Predict for new values using the predict function
SELECT * FROM anofox_statistics_predict_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],          -- y_train
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x_train
    [[6.0], [7.0], [8.0]]::DOUBLE[][],             -- x_new
    0.95,                                           -- confidence_level
    'prediction',                                   -- interval_type
    true                                            -- add_intercept
);
```

**Technical interpretation**:

- **predicted**: Point prediction ŷ = X_new × β̂
- **ci_lower/ci_upper**: Interval bounds using t-critical values at specified confidence level
- **se**: Standard error of prediction SE(ŷ) = √(MSE × x'(X'X)⁻¹x)
- **interval_type='prediction'**: Includes residual variance; 'confidence' for mean prediction only

## Example 6: Check Model Quality

**How it works**: Aggregate fit functions return information criteria (AIC, BIC, AICc) computed from the log-likelihood and number of parameters. These metrics penalize model complexity to prevent overfitting.


```sql

-- Extract model quality metrics using aggregate fit function
WITH data AS (
    SELECT UNNEST([2.1, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 15.9]::DOUBLE[]) as y,
           UNNEST([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]::DOUBLE[][]) as x
)
SELECT
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).n_obs as n_obs,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).r2 as r2,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).adj_r2 as adj_r2,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).aic as aic,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).aicc as aicc,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).bic as bic,
    (anofox_statistics_ols_fit_agg(y, x, {'intercept': true})).log_likelihood as log_likelihood
FROM data;
```

**Technical interpretation**:

- **aic**: Akaike Information Criterion = -2×log(L) + 2k
- **bic**: Bayesian Information Criterion = -2×log(L) + k×log(n)
- **aicc**: Corrected AIC for small samples = AIC + 2k(k+1)/(n-k-1)
- **Model comparison**: Lower values indicate better trade-off between fit and complexity

## Example 7: Detect Outliers

**How it works**: The `anofox_statistics_residual_diagnostics` table function computes standardized residuals and flags outliers based on a threshold. It returns per-observation diagnostics for assessing model fit and data quality.


```sql

-- Detect outliers and influential points using residual diagnostics
WITH data AS (
    SELECT
        [2.1::DOUBLE, 4.0, 6.1, 7.9, 10.2, 11.8, 14.1, 25.0] as y_actual,  -- Last point is outlier
        [2.0::DOUBLE, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0] as y_predicted  -- Fitted values
)
SELECT
    result.obs_id,
    ROUND(result.residual, 3) as residual,
    ROUND(result.std_residual, 3) as std_residual,
    result.is_outlier
FROM data,
LATERAL anofox_statistics_residual_diagnostics(data.y_actual, data.y_predicted, 2.5) as result
ORDER BY ABS(result.std_residual) DESC
LIMIT 3;
```

**Technical interpretation**:

- **observation_id**: Row index (1-indexed)
- **residual**: Raw residual e_i = y_i - ŷ_i
- **standardized_residual**: Residual divided by standard deviation
- **is_outlier**: Boolean flag where |standardized_residual| > threshold
- **is_influential**: Boolean flag for high-influence observations

## Aggregate Functions for GROUP BY

The extension provides aggregate functions that support GROUP BY and OVER clauses for per-group and windowed regression analysis.

### OLS Aggregate: Basic Per-Group Regression

**How it works**: `anofox_statistics_ols_fit_agg` is a DuckDB aggregate that accumulates observations per group and computes OLS regression coefficients using QR decomposition. Returns a complete statistics struct per group.


```sql

-- Quick Start Example: Simple OLS Aggregate with GROUP BY
-- Demonstrates basic per-group regression analysis

-- Sample data: sales by product category
CREATE TEMP TABLE product_sales AS
SELECT
    'electronics' as category, 100 as price, 250 as units_sold
UNION ALL SELECT 'electronics', 120, 230
UNION ALL SELECT 'electronics', 140, 210
UNION ALL SELECT 'electronics', 160, 190
UNION ALL SELECT 'electronics', 180, 170
UNION ALL SELECT 'furniture', 200, 180
UNION ALL SELECT 'furniture', 250, 165
UNION ALL SELECT 'furniture', 300, 150
UNION ALL SELECT 'furniture', 350, 135
UNION ALL SELECT 'furniture', 400, 120;

-- Run OLS regression for each category
SELECT
    category,
    result.coefficients[1] as price_elasticity,
    result.intercept,
    result.r2,
    result.n_obs
FROM (
    SELECT
        category,
        anofox_statistics_ols_fit_agg(
            units_sold::DOUBLE,
            [price::DOUBLE],
            {'intercept': true}
        ) as result
    FROM product_sales
    GROUP BY category
) sub;
```

**Technical interpretation**:

- **Per-group estimation**: Each GROUP BY group gets independent coefficient estimates
- **coefficients[1]**: Slope β₁ estimated from group observations only
- **r2**: Proportion of within-group variance explained
- **n_obs**: Sample size used for group's estimation

### WLS Aggregate: Weighted Analysis

**How it works**: `anofox_statistics_wls_fit_agg` implements weighted least squares by transforming inputs with √W before applying OLS. Each observation i receives weight w_i, modifying its contribution to the objective function.


```sql

-- Quick Start Example: Weighted Least Squares Aggregate
-- Demonstrates regression with observation weights (for heteroscedasticity)

-- Sample data: customer transactions with varying reliability
CREATE TEMP TABLE customer_transactions AS
SELECT
    'premium' as segment, 1 as month, 1000.0 as spend, 500.0 as income, 1.0 as reliability_weight
UNION ALL SELECT 'premium', 2, 1100.0, 510.0, 1.0
UNION ALL SELECT 'premium', 3, 1200.0, 520.0, 1.0
UNION ALL SELECT 'premium', 4, 1300.0, 530.0, 1.0
UNION ALL SELECT 'standard', 1, 300.0, 400.0, 0.8
UNION ALL SELECT 'standard', 2, 320.0, 410.0, 0.8
UNION ALL SELECT 'standard', 3, 340.0, 420.0, 0.8
UNION ALL SELECT 'standard', 4, 360.0, 430.0, 0.8
UNION ALL SELECT 'budget', 1, 100.0, 300.0, 0.5
UNION ALL SELECT 'budget', 2, 110.0, 305.0, 0.5
UNION ALL SELECT 'budget', 3, 120.0, 310.0, 0.5
UNION ALL SELECT 'budget', 4, 130.0, 315.0, 0.5;

-- WLS regression weighted by reliability
SELECT
    segment,
    result.coefficients[1] as income_sensitivity,
    result.intercept,
    result.r2,
    result.weighted_mse
FROM (
    SELECT
        segment,
        anofox_statistics_wls_fit_agg(
            spend,
            [income],
            reliability_weight,
            {'intercept': true}
        ) as result
    FROM customer_transactions
    GROUP BY segment
) sub;
```

**Technical interpretation**:

- **Weighted estimation**: Minimizes Σ w_i(y_i - ŷ_i)²
- **coefficients**: β̂_WLS = (X'WX)⁻¹X'Wy
- **mse**: Weighted mean squared error
- **Heteroscedasticity**: Weights inversely proportional to variance (w_i = 1/σ²_i) yield efficient estimates

### Ridge Aggregate: Handling Multicollinearity

**How it works**: `anofox_statistics_ridge_fit_agg` adds L2 regularization to the OLS objective function. The penalty term λ||β||² shrinks coefficient estimates, reducing variance at the cost of bias.


```sql

-- Quick Start Example: Ridge Regression Aggregate
-- Demonstrates L2 regularization for handling multicollinearity

-- Sample data: stock returns with correlated factors
CREATE TEMP TABLE stock_returns AS
SELECT
    'tech_stock' as ticker,
    1 as period,
    0.05 as return,
    0.03 as market_return,
    0.04 as sector_return,  -- Highly correlated with market_return
    0.02 as momentum
UNION ALL SELECT 'tech_stock', 2, 0.08, 0.06, 0.07, 0.05
UNION ALL SELECT 'tech_stock', 3, -0.02, -0.01, -0.01, -0.03
UNION ALL SELECT 'tech_stock', 4, 0.12, 0.10, 0.11, 0.08
UNION ALL SELECT 'tech_stock', 5, 0.06, 0.04, 0.05, 0.03
UNION ALL SELECT 'finance_stock', 1, 0.04, 0.03, 0.02, 0.01
UNION ALL SELECT 'finance_stock', 2, 0.07, 0.06, 0.05, 0.04
UNION ALL SELECT 'finance_stock', 3, -0.01, -0.01, -0.02, -0.02
UNION ALL SELECT 'finance_stock', 4, 0.09, 0.10, 0.08, 0.06
UNION ALL SELECT 'finance_stock', 5, 0.05, 0.04, 0.03, 0.02;

-- Ridge regression with regularization parameter lambda=1.0
SELECT
    ticker,
    result.coefficients[1] as market_beta,
    result.coefficients[2] as sector_beta,
    result.coefficients[3] as momentum_factor,
    result.r2,
    result.lambda
FROM (
    SELECT
        ticker,
        anofox_statistics_ridge_fit_agg(
            return,
            [market_return, sector_return, momentum],
            {'lambda': 1.0, 'intercept': true}
        ) as result
    FROM stock_returns
    GROUP BY ticker
) sub;
```

**Technical interpretation**:

- **Regularized estimation**: Minimizes ||y - Xβ||² + λ||β||²
- **coefficients**: β̂_ridge = (X'X + λI)⁻¹X'y
- **lambda**: Regularization parameter controlling shrinkage strength
- **Bias-variance trade-off**: Larger λ increases bias but reduces variance

### RLS Aggregate: Adaptive Online Learning

**How it works**: `anofox_statistics_rls_fit_agg` implements recursive least squares with exponential forgetting. Observations are weighted by λ^(n-i) where λ is the forgetting factor and i is the observation index.


```sql

-- Quick Start Example: Recursive Least Squares Aggregate
-- Demonstrates adaptive regression for changing relationships (online learning)

-- Sample data: sensor readings with evolving calibration
CREATE TEMP TABLE sensor_readings AS
SELECT
    'sensor_a' as sensor_id,
    1 as time_index,
    100.0 as raw_reading,
    98.0 as true_value
UNION ALL SELECT 'sensor_a', 2, 105.0, 103.0
UNION ALL SELECT 'sensor_a', 3, 110.0, 107.0
UNION ALL SELECT 'sensor_a', 4, 115.0, 112.0  -- Drift starts here
UNION ALL SELECT 'sensor_a', 5, 120.0, 116.5
UNION ALL SELECT 'sensor_a', 6, 125.0, 121.0
UNION ALL SELECT 'sensor_b', 1, 200.0, 202.0
UNION ALL SELECT 'sensor_b', 2, 205.0, 207.5
UNION ALL SELECT 'sensor_b', 3, 210.0, 213.0
UNION ALL SELECT 'sensor_b', 4, 215.0, 218.5
UNION ALL SELECT 'sensor_b', 5, 220.0, 224.0
UNION ALL SELECT 'sensor_b', 6, 225.0, 229.5;

-- RLS with forgetting_factor=0.95 (emphasizes recent observations)
SELECT
    sensor_id,
    result.coefficients[1] as calibration_slope,
    result.intercept as calibration_offset,
    result.r2,
    result.forgetting_factor,
    result.n_obs
FROM (
    SELECT
        sensor_id,
        anofox_statistics_rls_fit_agg(
            true_value,
            [raw_reading],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) as result
    FROM sensor_readings
    GROUP BY sensor_id
) sub;
```

**Technical interpretation**:

- **Sequential update**: Coefficients updated recursively as new observations arrive
- **lambda** (forgetting_factor): Exponential decay parameter (0 < λ ≤ 1)
- **Effective window**: Approximately 1/(1-λ) observations contribute significantly
- **Adaptation**: Lower λ values (e.g., 0.90-0.95) track non-stationary relationships

## Common Patterns

### Pattern 1: Quick coefficient check

Extract a single coefficient from the aggregate result struct by accessing the coefficients array directly.


```sql

-- Generate sample data for pattern demonstration
CREATE TEMP TABLE data AS
SELECT
    i::DOUBLE as y,
    (i * 2.5 + 10)::DOUBLE as x
FROM generate_series(1, 20) t(i);

SELECT (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as slope FROM data;
```

### Pattern 2: Per-group with GROUP BY

Fit independent regression models for each category using GROUP BY semantics.


```sql

-- Generate sample data with multiple categories
CREATE TEMP TABLE data AS
SELECT
    CASE
        WHEN i <= 10 THEN 'A'
        WHEN i <= 20 THEN 'B'
        ELSE 'C'
    END as category,
    (i + random() * 5)::DOUBLE as y,
    (i * 1.5 + 5)::DOUBLE as x
FROM generate_series(1, 30) t(i);

SELECT category, anofox_statistics_ols_fit_agg(y, [x], {'intercept': true}) as model
FROM data GROUP BY category;
```

### Pattern 3: Rolling window with OVER

Apply regression over a sliding window using OVER clause with ROWS BETWEEN frame specification.


```sql

-- Generate time-series data
CREATE TEMP TABLE data AS
SELECT
    i as time,
    (10 + i * 0.5 + random() * 3)::DOUBLE as y,
    (5 + i * 0.3)::DOUBLE as x
FROM generate_series(1, 100) t(i);

SELECT *, (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true}) OVER (
    ORDER BY time ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
)).coefficients[1] as rolling_coef FROM data;
```

### Pattern 4: Full statistical workflow

Combine model fitting, residual computation, and diagnostic checks in a single CTE chain.


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
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as slope,
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).r2 as r2,
        (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).residual_standard_error as std_error,
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
    ROUND(r2, 4)::VARCHAR as value
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

## Common Issues

### Issue: Extension won't load

```sql
-- DISABLED: This is a troubleshooting example with placeholder path
-- Not a runnable test

SELECT 'guide01_issue_extension_wont_load.sql - DISABLED - documentation example' as status;
```

### Issue: Type mismatch

```sql

-- Ensure arrays are DOUBLE[] and use new API with 2D array + MAP
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0]::DOUBLE[],         -- y: Cast to DOUBLE[]
    [[1.0, 2.0, 3.0]]::DOUBLE[][],     -- X: 2D array (one feature)
    {'intercept': true}              -- options in MAP
);
```

### Issue: Insufficient observations

```
Error: Need at least n+1 observations for n parameters
```

Solution: Ensure you have more observations than predictors.
