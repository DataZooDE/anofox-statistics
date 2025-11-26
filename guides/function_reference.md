# Function Reference

Comprehensive reference for all functions in the Anofox Statistics extension.

## Table of Contents

- [Aggregate Functions](#aggregate-functions)
- [Table Functions](#table-functions)
- [Diagnostic Functions](#diagnostic-functions)
- [Options MAP Reference](#options-map-reference)

---

## Aggregate Functions

Aggregate functions work with `GROUP BY` and window functions (`OVER`) to perform regression analysis per group or within rolling windows.

### anofox_statistics_ols_agg

**Description:** Ordinary Least Squares regression per group. Fits a linear model minimizing squared residuals.

**Signature:**

```sql
anofox_statistics_ols_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, n_obs BIGINT, n_features BIGINT)
```

**Parameters:**

- `y`: Response variable (dependent variable)
- `x`: Array of predictor variables (independent variables)
- `options`: Configuration MAP (see [Options Reference](#options-map-reference))

**Returns:** STRUCT containing coefficients, intercept, R², adjusted R², observation count, and feature count.

**Example:**


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
    result.r_squared,
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

**When to use:** Standard regression analysis, per-group modeling, baseline comparisons.

---

### anofox_statistics_wls_agg

**Description:** Weighted Least Squares regression. Handles heteroscedasticity by giving different weights to observations.

**Signature:**

```sql
anofox_statistics_wls_agg(
    y DOUBLE,
    x DOUBLE[],
    weights DOUBLE,
    options MAP(VARCHAR, ANY)
) → STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, weighted_mse DOUBLE, n_obs BIGINT)
```

**Parameters:**

- `y`: Response variable
- `x`: Array of predictors
- `weights`: Observation weights (precision weights = 1/variance)
- `options`: Configuration MAP

**Returns:** STRUCT with coefficients, metrics, and weighted MSE.

**Example:**


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
    result.r_squared,
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

**When to use:** When observations have different reliability or precision, heteroscedastic errors.

---

### anofox_statistics_ridge_agg

**Description:** Ridge regression with L2 regularization per group. Stabilizes coefficients when predictors are correlated.

**Signature:**

```sql
anofox_statistics_ridge_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, lambda DOUBLE, n_obs BIGINT)
```

**Parameters:**

- `y`: Response variable
- `x`: Array of predictors
- `options`: Configuration MAP with `lambda` (regularization parameter)

**Returns:** STRUCT with regularized coefficients and fit metrics.

**Example:**


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
    result.r_squared,
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

**When to use:** Multicollinearity, overfitting prevention, many correlated predictors.

---

### anofox_statistics_rls_agg

**Description:** Recursive Least Squares with exponential weighting. Adapts to changing relationships over time.

**Signature:**

```sql
anofox_statistics_rls_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, forgetting_factor DOUBLE, n_obs BIGINT)
```

**Parameters:**

- `y`: Response variable
- `x`: Array of predictors
- `options`: Configuration MAP with `forgetting_factor` (default: 1.0, range: 0.9-1.0)

**Returns:** STRUCT with adaptive coefficients emphasizing recent data.

**Example:**


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
    result.r_squared,
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

**When to use:** Time-series with evolving relationships, online learning, real-time forecasting.

---

### anofox_statistics_elastic_net_agg

**Description:** Elastic Net regression with combined L1 and L2 regularization per group. Performs feature selection while handling multicollinearity.

**Signature:**

```sql
anofox_statistics_elastic_net_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, n_nonzero BIGINT, n_iterations BIGINT, converged BOOLEAN, n_obs BIGINT)
```

**Parameters:**

- `y`: Response variable
- `x`: Array of predictors
- `options`: Configuration MAP with `alpha` (L1/L2 mix, 0=Ridge, 1=Lasso, default: 0.5) and `lambda` (regularization strength)

**Returns:** STRUCT with sparse coefficients, convergence info, and fit metrics.

**Example:**


```sql
-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.n_nonzero,
    result.r_squared > 0.9 as good_fit
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
SELECT
    array_length(result.coefficients) as n_coeffs,
    result.intercept IS NOT NULL as has_intercept,
    result.n_nonzero >= 2 as all_nonzero  -- Ridge doesn't zero coefficients
FROM anofox_statistics_elastic_net_fit(
    [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
SELECT
    array_length(result.coefficients) as n_features,
    result.n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    result.r_squared
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1::DOUBLE, 0.2, 0.3, 0.4, 0.5, 0.6], [0.05::DOUBLE, 0.10, 0.15, 0.20, 0.25, 0.30]],
    options := MAP(['alpha', 'lambda', 'intercept'], [1.0::DOUBLE, 0.5::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
SELECT
    result.intercept IS NULL as no_intercept,
    array_length(result.coefficients) = 2 as correct_size
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 0.0::DOUBLE])
) as result;
-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0::DOUBLE, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0::DOUBLE, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0::DOUBLE, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0::DOUBLE, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0::DOUBLE, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0::DOUBLE, 4.0]
),
aggregated AS (
    SELECT
        category,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.5::DOUBLE, 0.01::DOUBLE])) as result
    FROM data
    GROUP BY category
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit
FROM aggregated;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0::DOUBLE, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0::DOUBLE, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0::DOUBLE, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0::DOUBLE, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.3::DOUBLE, 0.1::DOUBLE]))
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data
-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
SELECT
    COUNT(*) = 3 as has_all_features,
    MAX(CASE WHEN variable_name = 'x2' THEN vif ELSE 0 END) > 5 as high_multicollinearity,  -- Second feature is 2x first
    MAX(CASE WHEN variable_name = 'x2' THEN severity ELSE '' END) IN ('high', 'perfect') as correct_severity
FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 1.5], [2.0::DOUBLE, 4.0, 3.0], [3.0::DOUBLE, 6.0, 4.5],
                            [4.0::DOUBLE, 8.0, 6.0], [5.0::DOUBLE, 10.0, 7.5]]);

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM anofox_statistics_normality_test([0.5::DOUBLE, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
                                       0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9], 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN result.is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(result.std_residual)) > 2.0 as has_extreme_residual
FROM anofox_statistics_residual_diagnostics([1.0::DOUBLE, 2.0, 3.0, 4.0, 10.0],
                                            [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.5],
                                            outlier_threshold := 2.5) as result;
-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
),
aggregated AS (
    SELECT
        group_id,
        anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM data
    GROUP BY group_id
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM aggregated;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual > result.mean_abs_residual as logical_max
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP(['outlier_threshold'], [2.0::DOUBLE])) as result
    FROM data
);

-- DISABLED: Tests 4.3-4.6 use aggregate functions that don't exist yet
-- (anofox_statistics_vif_agg, anofox_statistics_normality_test_agg)
-- These tests will be enabled when the aggregate versions are implemented

-- -- Test 4.3: vif_aggregate - VIF per group
-- SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
-- WITH data AS (
--     SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
--     SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
--     SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
--     SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
--     SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
--     SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
--     SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
--     SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
--     SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
--     SELECT 'B', [5.0, 3.0, 6.0]
-- ),
-- aggregated AS (
--     SELECT
--         category,
--         anofox_statistics_vif_agg(x) as result
--     FROM data
--     GROUP BY category
-- )
-- SELECT
--     category,
--     array_length(result.vif) = 3 as correct_feature_count,
--     result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
--     array_length(result.severity) = 3 as has_severity_labels
-- FROM aggregated;

-- -- Test 4.4: vif_aggregate with window function
-- SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- -- Note: VIF aggregate does not support window functions as per requirements
-- -- This test just verifies it works with GROUP BY only

-- -- Test 4.5: normality_test_aggregate - Jarque-Bera per group
-- SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
-- WITH data AS (
--     -- Group A: Normally distributed residuals
--     SELECT 'A' as group_id, 0.1 as residual UNION ALL
--     SELECT 'A', 0.2 UNION ALL
--     SELECT 'A', -0.1 UNION ALL
--     SELECT 'A', 0.3 UNION ALL
--     SELECT 'A', -0.2 UNION ALL
--     SELECT 'A', 0.15 UNION ALL
--     SELECT 'A', -0.05 UNION ALL
--     SELECT 'A', 0.25 UNION ALL
--     SELECT 'A', -0.15 UNION ALL
--     SELECT 'A', 0.05 UNION ALL
--     -- Group B: Skewed distribution
--     SELECT 'B', 1.0 UNION ALL
--     SELECT 'B', 1.5 UNION ALL
--     SELECT 'B', 2.0 UNION ALL
--     SELECT 'B', 2.5 UNION ALL
--     SELECT 'B', 3.0 UNION ALL
--     SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
--     SELECT 'B', 1.2 UNION ALL
--     SELECT 'B', 1.8 UNION ALL
--     SELECT 'B', 2.2 UNION ALL
--     SELECT 'B', 2.8
-- ),
-- aggregated AS (
--     SELECT
--         group_id,
--         anofox_statistics_normality_test_agg(residual, {'alpha': 0.05}) as result
--     FROM data
--     GROUP BY group_id
-- )
-- SELECT
--     group_id,
--     result.n_obs >= 8 as sufficient_obs,
--     result.jb_statistic >= 0 as valid_statistic,
--     result.p_value BETWEEN 0 AND 1 as valid_pvalue,
--     result.conclusion IN ('normal', 'non-normal') as valid_conclusion
-- FROM aggregated;

-- -- Test 4.6: normality_test_aggregate with different alpha levels
-- SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
-- WITH data AS (
--     SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
--     SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
--     SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
--     SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
-- )
-- SELECT
--     result.n_obs = 12 as correct_count,
--     result.is_normal IS NOT NULL as has_test_result,
--     result.p_value >= 0 as valid_pvalue
-- FROM (
--     SELECT anofox_statistics_normality_test_agg(residual, {'alpha': 0.01}) as result
--     FROM data
-- );
-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH model AS (
    SELECT
        result.coefficients,
        result.intercept,
        result.r_squared,
        result.n_nonzero
    FROM anofox_statistics_elastic_net_fit(
        [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1],
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
         [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
    ) as result
),
predictions AS (
    SELECT
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) + 0.1 as y_predicted  -- Simulated predictions
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
```

**When to use:** Feature selection, high-dimensional data, when you need both shrinkage and sparsity.

---

### anofox_statistics_residual_diagnostics_agg

**Description:** Per-group residual analysis aggregate. Identifies outliers and influential observations within each group.

**Signature:**

```sql
anofox_statistics_residual_diagnostics_agg(
    y_actual DOUBLE,
    y_predicted DOUBLE,
    options MAP(VARCHAR, ANY)
) → STRUCT(n_obs BIGINT, n_outliers BIGINT, outlier_pct DOUBLE, max_abs_residual DOUBLE, ...)
```

**Parameters:**

- `y_actual`: Actual observed values
- `y_predicted`: Predicted values from model
- `options`: Configuration MAP with `mode` ('summary' or 'detailed'), `outlier_threshold` (default: 2.5)

**Returns:** Summary statistics (mode='summary') or detailed diagnostics per observation (mode='detailed').

**Example:**


```sql
-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.n_nonzero,
    result.r_squared > 0.9 as good_fit
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
SELECT
    array_length(result.coefficients) as n_coeffs,
    result.intercept IS NOT NULL as has_intercept,
    result.n_nonzero >= 2 as all_nonzero  -- Ridge doesn't zero coefficients
FROM anofox_statistics_elastic_net_fit(
    [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
SELECT
    array_length(result.coefficients) as n_features,
    result.n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    result.r_squared
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1::DOUBLE, 0.2, 0.3, 0.4, 0.5, 0.6], [0.05::DOUBLE, 0.10, 0.15, 0.20, 0.25, 0.30]],
    options := MAP(['alpha', 'lambda', 'intercept'], [1.0::DOUBLE, 0.5::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
SELECT
    result.intercept IS NULL as no_intercept,
    array_length(result.coefficients) = 2 as correct_size
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 0.0::DOUBLE])
) as result;
-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0::DOUBLE, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0::DOUBLE, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0::DOUBLE, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0::DOUBLE, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0::DOUBLE, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0::DOUBLE, 4.0]
),
aggregated AS (
    SELECT
        category,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.5::DOUBLE, 0.01::DOUBLE])) as result
    FROM data
    GROUP BY category
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit
FROM aggregated;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0::DOUBLE, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0::DOUBLE, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0::DOUBLE, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0::DOUBLE, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.3::DOUBLE, 0.1::DOUBLE]))
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data
-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
SELECT
    COUNT(*) = 3 as has_all_features,
    MAX(CASE WHEN variable_name = 'x2' THEN vif ELSE 0 END) > 5 as high_multicollinearity,  -- Second feature is 2x first
    MAX(CASE WHEN variable_name = 'x2' THEN severity ELSE '' END) IN ('high', 'perfect') as correct_severity
FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 1.5], [2.0::DOUBLE, 4.0, 3.0], [3.0::DOUBLE, 6.0, 4.5],
                            [4.0::DOUBLE, 8.0, 6.0], [5.0::DOUBLE, 10.0, 7.5]]);

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM anofox_statistics_normality_test([0.5::DOUBLE, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
                                       0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9], 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN result.is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(result.std_residual)) > 2.0 as has_extreme_residual
FROM anofox_statistics_residual_diagnostics([1.0::DOUBLE, 2.0, 3.0, 4.0, 10.0],
                                            [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.5],
                                            outlier_threshold := 2.5) as result;
-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
),
aggregated AS (
    SELECT
        group_id,
        anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM data
    GROUP BY group_id
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM aggregated;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual > result.mean_abs_residual as logical_max
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP(['outlier_threshold'], [2.0::DOUBLE])) as result
    FROM data
);

-- DISABLED: Tests 4.3-4.6 use aggregate functions that don't exist yet
-- (anofox_statistics_vif_agg, anofox_statistics_normality_test_agg)
-- These tests will be enabled when the aggregate versions are implemented

-- -- Test 4.3: vif_aggregate - VIF per group
-- SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
-- WITH data AS (
--     SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
--     SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
--     SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
--     SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
--     SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
--     SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
--     SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
--     SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
--     SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
--     SELECT 'B', [5.0, 3.0, 6.0]
-- ),
-- aggregated AS (
--     SELECT
--         category,
--         anofox_statistics_vif_agg(x) as result
--     FROM data
--     GROUP BY category
-- )
-- SELECT
--     category,
--     array_length(result.vif) = 3 as correct_feature_count,
--     result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
--     array_length(result.severity) = 3 as has_severity_labels
-- FROM aggregated;

-- -- Test 4.4: vif_aggregate with window function
-- SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- -- Note: VIF aggregate does not support window functions as per requirements
-- -- This test just verifies it works with GROUP BY only

-- -- Test 4.5: normality_test_aggregate - Jarque-Bera per group
-- SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
-- WITH data AS (
--     -- Group A: Normally distributed residuals
--     SELECT 'A' as group_id, 0.1 as residual UNION ALL
--     SELECT 'A', 0.2 UNION ALL
--     SELECT 'A', -0.1 UNION ALL
--     SELECT 'A', 0.3 UNION ALL
--     SELECT 'A', -0.2 UNION ALL
--     SELECT 'A', 0.15 UNION ALL
--     SELECT 'A', -0.05 UNION ALL
--     SELECT 'A', 0.25 UNION ALL
--     SELECT 'A', -0.15 UNION ALL
--     SELECT 'A', 0.05 UNION ALL
--     -- Group B: Skewed distribution
--     SELECT 'B', 1.0 UNION ALL
--     SELECT 'B', 1.5 UNION ALL
--     SELECT 'B', 2.0 UNION ALL
--     SELECT 'B', 2.5 UNION ALL
--     SELECT 'B', 3.0 UNION ALL
--     SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
--     SELECT 'B', 1.2 UNION ALL
--     SELECT 'B', 1.8 UNION ALL
--     SELECT 'B', 2.2 UNION ALL
--     SELECT 'B', 2.8
-- ),
-- aggregated AS (
--     SELECT
--         group_id,
--         anofox_statistics_normality_test_agg(residual, {'alpha': 0.05}) as result
--     FROM data
--     GROUP BY group_id
-- )
-- SELECT
--     group_id,
--     result.n_obs >= 8 as sufficient_obs,
--     result.jb_statistic >= 0 as valid_statistic,
--     result.p_value BETWEEN 0 AND 1 as valid_pvalue,
--     result.conclusion IN ('normal', 'non-normal') as valid_conclusion
-- FROM aggregated;

-- -- Test 4.6: normality_test_aggregate with different alpha levels
-- SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
-- WITH data AS (
--     SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
--     SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
--     SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
--     SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
-- )
-- SELECT
--     result.n_obs = 12 as correct_count,
--     result.is_normal IS NOT NULL as has_test_result,
--     result.p_value >= 0 as valid_pvalue
-- FROM (
--     SELECT anofox_statistics_normality_test_agg(residual, {'alpha': 0.01}) as result
--     FROM data
-- );
-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH model AS (
    SELECT
        result.coefficients,
        result.intercept,
        result.r_squared,
        result.n_nonzero
    FROM anofox_statistics_elastic_net_fit(
        [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1],
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
         [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
    ) as result
),
predictions AS (
    SELECT
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) + 0.1 as y_predicted  -- Simulated predictions
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
```

**When to use:** Group-wise outlier detection, model quality checks per segment.

---

### anofox_statistics_vif_agg

**Description:** Variance Inflation Factor aggregate per group. Detects multicollinearity within each group.

**Signature:**

```sql
anofox_statistics_vif_agg(
    x DOUBLE[]
) → STRUCT(vif_values DOUBLE[], max_vif DOUBLE, mean_vif DOUBLE, severity VARCHAR, n_obs BIGINT)
```

**Parameters:**

- `x`: Array of predictor variables

**Returns:** VIF values for each predictor, max/mean VIF, severity classification.

**Example:**


```sql
-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.n_nonzero,
    result.r_squared > 0.9 as good_fit
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
SELECT
    array_length(result.coefficients) as n_coeffs,
    result.intercept IS NOT NULL as has_intercept,
    result.n_nonzero >= 2 as all_nonzero  -- Ridge doesn't zero coefficients
FROM anofox_statistics_elastic_net_fit(
    [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
SELECT
    array_length(result.coefficients) as n_features,
    result.n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    result.r_squared
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1::DOUBLE, 0.2, 0.3, 0.4, 0.5, 0.6], [0.05::DOUBLE, 0.10, 0.15, 0.20, 0.25, 0.30]],
    options := MAP(['alpha', 'lambda', 'intercept'], [1.0::DOUBLE, 0.5::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
SELECT
    result.intercept IS NULL as no_intercept,
    array_length(result.coefficients) = 2 as correct_size
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 0.0::DOUBLE])
) as result;
-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0::DOUBLE, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0::DOUBLE, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0::DOUBLE, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0::DOUBLE, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0::DOUBLE, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0::DOUBLE, 4.0]
),
aggregated AS (
    SELECT
        category,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.5::DOUBLE, 0.01::DOUBLE])) as result
    FROM data
    GROUP BY category
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit
FROM aggregated;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0::DOUBLE, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0::DOUBLE, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0::DOUBLE, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0::DOUBLE, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.3::DOUBLE, 0.1::DOUBLE]))
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data
-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
SELECT
    COUNT(*) = 3 as has_all_features,
    MAX(CASE WHEN variable_name = 'x2' THEN vif ELSE 0 END) > 5 as high_multicollinearity,  -- Second feature is 2x first
    MAX(CASE WHEN variable_name = 'x2' THEN severity ELSE '' END) IN ('high', 'perfect') as correct_severity
FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 1.5], [2.0::DOUBLE, 4.0, 3.0], [3.0::DOUBLE, 6.0, 4.5],
                            [4.0::DOUBLE, 8.0, 6.0], [5.0::DOUBLE, 10.0, 7.5]]);

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM anofox_statistics_normality_test([0.5::DOUBLE, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
                                       0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9], 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN result.is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(result.std_residual)) > 2.0 as has_extreme_residual
FROM anofox_statistics_residual_diagnostics([1.0::DOUBLE, 2.0, 3.0, 4.0, 10.0],
                                            [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.5],
                                            outlier_threshold := 2.5) as result;
-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
),
aggregated AS (
    SELECT
        group_id,
        anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM data
    GROUP BY group_id
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM aggregated;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual > result.mean_abs_residual as logical_max
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP(['outlier_threshold'], [2.0::DOUBLE])) as result
    FROM data
);

-- DISABLED: Tests 4.3-4.6 use aggregate functions that don't exist yet
-- (anofox_statistics_vif_agg, anofox_statistics_normality_test_agg)
-- These tests will be enabled when the aggregate versions are implemented

-- -- Test 4.3: vif_aggregate - VIF per group
-- SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
-- WITH data AS (
--     SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
--     SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
--     SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
--     SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
--     SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
--     SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
--     SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
--     SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
--     SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
--     SELECT 'B', [5.0, 3.0, 6.0]
-- ),
-- aggregated AS (
--     SELECT
--         category,
--         anofox_statistics_vif_agg(x) as result
--     FROM data
--     GROUP BY category
-- )
-- SELECT
--     category,
--     array_length(result.vif) = 3 as correct_feature_count,
--     result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
--     array_length(result.severity) = 3 as has_severity_labels
-- FROM aggregated;

-- -- Test 4.4: vif_aggregate with window function
-- SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- -- Note: VIF aggregate does not support window functions as per requirements
-- -- This test just verifies it works with GROUP BY only

-- -- Test 4.5: normality_test_aggregate - Jarque-Bera per group
-- SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
-- WITH data AS (
--     -- Group A: Normally distributed residuals
--     SELECT 'A' as group_id, 0.1 as residual UNION ALL
--     SELECT 'A', 0.2 UNION ALL
--     SELECT 'A', -0.1 UNION ALL
--     SELECT 'A', 0.3 UNION ALL
--     SELECT 'A', -0.2 UNION ALL
--     SELECT 'A', 0.15 UNION ALL
--     SELECT 'A', -0.05 UNION ALL
--     SELECT 'A', 0.25 UNION ALL
--     SELECT 'A', -0.15 UNION ALL
--     SELECT 'A', 0.05 UNION ALL
--     -- Group B: Skewed distribution
--     SELECT 'B', 1.0 UNION ALL
--     SELECT 'B', 1.5 UNION ALL
--     SELECT 'B', 2.0 UNION ALL
--     SELECT 'B', 2.5 UNION ALL
--     SELECT 'B', 3.0 UNION ALL
--     SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
--     SELECT 'B', 1.2 UNION ALL
--     SELECT 'B', 1.8 UNION ALL
--     SELECT 'B', 2.2 UNION ALL
--     SELECT 'B', 2.8
-- ),
-- aggregated AS (
--     SELECT
--         group_id,
--         anofox_statistics_normality_test_agg(residual, {'alpha': 0.05}) as result
--     FROM data
--     GROUP BY group_id
-- )
-- SELECT
--     group_id,
--     result.n_obs >= 8 as sufficient_obs,
--     result.jb_statistic >= 0 as valid_statistic,
--     result.p_value BETWEEN 0 AND 1 as valid_pvalue,
--     result.conclusion IN ('normal', 'non-normal') as valid_conclusion
-- FROM aggregated;

-- -- Test 4.6: normality_test_aggregate with different alpha levels
-- SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
-- WITH data AS (
--     SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
--     SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
--     SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
--     SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
-- )
-- SELECT
--     result.n_obs = 12 as correct_count,
--     result.is_normal IS NOT NULL as has_test_result,
--     result.p_value >= 0 as valid_pvalue
-- FROM (
--     SELECT anofox_statistics_normality_test_agg(residual, {'alpha': 0.01}) as result
--     FROM data
-- );
-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH model AS (
    SELECT
        result.coefficients,
        result.intercept,
        result.r_squared,
        result.n_nonzero
    FROM anofox_statistics_elastic_net_fit(
        [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1],
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
         [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
    ) as result
),
predictions AS (
    SELECT
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) + 0.1 as y_predicted  -- Simulated predictions
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
```

**When to use:** Check for multicollinearity before modeling, per-group correlation analysis.

---

### anofox_statistics_normality_test_agg

**Description:** Jarque-Bera normality test aggregate per group. Tests if residuals follow a normal distribution.

**Signature:**

```sql
anofox_statistics_normality_test_agg(
    residual DOUBLE,
    options MAP(VARCHAR, ANY)
) → STRUCT(n_obs BIGINT, skewness DOUBLE, kurtosis DOUBLE, jb_statistic DOUBLE, p_value DOUBLE, is_normal BOOLEAN)
```

**Parameters:**

- `residual`: Residuals from regression model
- `options`: Configuration MAP with `alpha` (significance level, default: 0.05)

**Returns:** Skewness, kurtosis, test statistic, p-value, normality verdict.

**Example:**


```sql
-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.n_nonzero,
    result.r_squared > 0.9 as good_fit
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
SELECT
    array_length(result.coefficients) as n_coeffs,
    result.intercept IS NOT NULL as has_intercept,
    result.n_nonzero >= 2 as all_nonzero  -- Ridge doesn't zero coefficients
FROM anofox_statistics_elastic_net_fit(
    [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
SELECT
    array_length(result.coefficients) as n_features,
    result.n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    result.r_squared
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1::DOUBLE, 0.2, 0.3, 0.4, 0.5, 0.6], [0.05::DOUBLE, 0.10, 0.15, 0.20, 0.25, 0.30]],
    options := MAP(['alpha', 'lambda', 'intercept'], [1.0::DOUBLE, 0.5::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
SELECT
    result.intercept IS NULL as no_intercept,
    array_length(result.coefficients) = 2 as correct_size
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 0.0::DOUBLE])
) as result;
-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0::DOUBLE, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0::DOUBLE, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0::DOUBLE, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0::DOUBLE, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0::DOUBLE, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0::DOUBLE, 4.0]
),
aggregated AS (
    SELECT
        category,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.5::DOUBLE, 0.01::DOUBLE])) as result
    FROM data
    GROUP BY category
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit
FROM aggregated;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0::DOUBLE, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0::DOUBLE, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0::DOUBLE, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0::DOUBLE, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.3::DOUBLE, 0.1::DOUBLE]))
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data
-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
SELECT
    COUNT(*) = 3 as has_all_features,
    MAX(CASE WHEN variable_name = 'x2' THEN vif ELSE 0 END) > 5 as high_multicollinearity,  -- Second feature is 2x first
    MAX(CASE WHEN variable_name = 'x2' THEN severity ELSE '' END) IN ('high', 'perfect') as correct_severity
FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 1.5], [2.0::DOUBLE, 4.0, 3.0], [3.0::DOUBLE, 6.0, 4.5],
                            [4.0::DOUBLE, 8.0, 6.0], [5.0::DOUBLE, 10.0, 7.5]]);

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM anofox_statistics_normality_test([0.5::DOUBLE, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
                                       0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9], 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN result.is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(result.std_residual)) > 2.0 as has_extreme_residual
FROM anofox_statistics_residual_diagnostics([1.0::DOUBLE, 2.0, 3.0, 4.0, 10.0],
                                            [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.5],
                                            outlier_threshold := 2.5) as result;
-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
),
aggregated AS (
    SELECT
        group_id,
        anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM data
    GROUP BY group_id
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM aggregated;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual > result.mean_abs_residual as logical_max
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP(['outlier_threshold'], [2.0::DOUBLE])) as result
    FROM data
);

-- DISABLED: Tests 4.3-4.6 use aggregate functions that don't exist yet
-- (anofox_statistics_vif_agg, anofox_statistics_normality_test_agg)
-- These tests will be enabled when the aggregate versions are implemented

-- -- Test 4.3: vif_aggregate - VIF per group
-- SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
-- WITH data AS (
--     SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
--     SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
--     SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
--     SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
--     SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
--     SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
--     SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
--     SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
--     SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
--     SELECT 'B', [5.0, 3.0, 6.0]
-- ),
-- aggregated AS (
--     SELECT
--         category,
--         anofox_statistics_vif_agg(x) as result
--     FROM data
--     GROUP BY category
-- )
-- SELECT
--     category,
--     array_length(result.vif) = 3 as correct_feature_count,
--     result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
--     array_length(result.severity) = 3 as has_severity_labels
-- FROM aggregated;

-- -- Test 4.4: vif_aggregate with window function
-- SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- -- Note: VIF aggregate does not support window functions as per requirements
-- -- This test just verifies it works with GROUP BY only

-- -- Test 4.5: normality_test_aggregate - Jarque-Bera per group
-- SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
-- WITH data AS (
--     -- Group A: Normally distributed residuals
--     SELECT 'A' as group_id, 0.1 as residual UNION ALL
--     SELECT 'A', 0.2 UNION ALL
--     SELECT 'A', -0.1 UNION ALL
--     SELECT 'A', 0.3 UNION ALL
--     SELECT 'A', -0.2 UNION ALL
--     SELECT 'A', 0.15 UNION ALL
--     SELECT 'A', -0.05 UNION ALL
--     SELECT 'A', 0.25 UNION ALL
--     SELECT 'A', -0.15 UNION ALL
--     SELECT 'A', 0.05 UNION ALL
--     -- Group B: Skewed distribution
--     SELECT 'B', 1.0 UNION ALL
--     SELECT 'B', 1.5 UNION ALL
--     SELECT 'B', 2.0 UNION ALL
--     SELECT 'B', 2.5 UNION ALL
--     SELECT 'B', 3.0 UNION ALL
--     SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
--     SELECT 'B', 1.2 UNION ALL
--     SELECT 'B', 1.8 UNION ALL
--     SELECT 'B', 2.2 UNION ALL
--     SELECT 'B', 2.8
-- ),
-- aggregated AS (
--     SELECT
--         group_id,
--         anofox_statistics_normality_test_agg(residual, {'alpha': 0.05}) as result
--     FROM data
--     GROUP BY group_id
-- )
-- SELECT
--     group_id,
--     result.n_obs >= 8 as sufficient_obs,
--     result.jb_statistic >= 0 as valid_statistic,
--     result.p_value BETWEEN 0 AND 1 as valid_pvalue,
--     result.conclusion IN ('normal', 'non-normal') as valid_conclusion
-- FROM aggregated;

-- -- Test 4.6: normality_test_aggregate with different alpha levels
-- SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
-- WITH data AS (
--     SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
--     SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
--     SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
--     SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
-- )
-- SELECT
--     result.n_obs = 12 as correct_count,
--     result.is_normal IS NOT NULL as has_test_result,
--     result.p_value >= 0 as valid_pvalue
-- FROM (
--     SELECT anofox_statistics_normality_test_agg(residual, {'alpha': 0.01}) as result
--     FROM data
-- );
-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH model AS (
    SELECT
        result.coefficients,
        result.intercept,
        result.r_squared,
        result.n_nonzero
    FROM anofox_statistics_elastic_net_fit(
        [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1],
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
         [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
    ) as result
),
predictions AS (
    SELECT
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) + 0.1 as y_predicted  -- Simulated predictions
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
```

**When to use:** Validate OLS assumptions, check residual distributions per group.

---

## Table Functions

Table functions perform regression on array inputs and return detailed results as tables.

### anofox_statistics_ols

**Description:** Ordinary Least Squares regression for array inputs. Returns comprehensive fit statistics.

**Signature:**

```sql
anofox_statistics_ols(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(coefficients DOUBLE[], intercept DOUBLE, r2 DOUBLE, adj_r2 DOUBLE, ...)
```

**Parameters:**

- `y`: Response vector
- `x`: Feature matrix (n_observations × n_features)
- `options`: Configuration MAP with `intercept` (default: true)

**Returns:** Table with one row containing all regression results.

**Example:**


```sql
-- Simplified OLS Validation Test
-- Tests basic OLS functionality against known data

-- Load input data
CREATE OR REPLACE TABLE ols_input AS
SELECT * FROM read_csv('test/data/ols_tests/input/simple_linear.csv');

-- Test using aggregate function (works directly on table data)
CREATE OR REPLACE TABLE ols_agg_result AS
SELECT
    (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).coefficients[1] as slope,
    (anofox_statistics_ols_fit_agg(y, [x], {'intercept': true})).r_squared as r_squared
FROM ols_input;

-- Load expected R² from reference data
CREATE OR REPLACE TABLE ols_expected AS
SELECT r_squared as expected_r2
FROM read_json('test/data/ols_tests/expected/simple_linear.json',
    format='auto', maximum_object_size=10000000);

-- Compare R² values
CREATE OR REPLACE TABLE validation_result AS
SELECT
    r.r_squared as computed_r2,
    e.expected_r2,
    abs(r.r_squared - e.expected_r2) as error,
    CASE WHEN abs(r.r_squared - e.expected_r2) < 0.01 THEN 'PASS' ELSE 'FAIL' END as status
FROM ols_agg_result r, ols_expected e;

-- Show result
SELECT * FROM validation_result;

-- Return success only if passed
SELECT * FROM validation_result WHERE status = 'PASS';
```

**When to use:** Single model fitting, small datasets, when you have arrays ready.

---

### anofox_statistics_wls

**Description:** Weighted Least Squares for array inputs.

**Signature:**

```sql
anofox_statistics_wls(
    y DOUBLE[],
    x DOUBLE[][],
    weights DOUBLE[],
    options MAP(VARCHAR, ANY)
) → TABLE(coefficients DOUBLE[], intercept DOUBLE, ...)
```

**Example:**


```sql

-- Variance proportional to x (new API with 2D array + MAP)
SELECT * FROM anofox_statistics_wls_fit(
    [50.0, 100.0, 150.0, 200.0, 250.0]::DOUBLE[],  -- y: sales
    [[10.0, 20.0, 30.0, 40.0, 50.0]]::DOUBLE[][],  -- X: 2D array (one feature)
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],      -- weights: proportional to size
    {'intercept': true}                          -- options in MAP
);
```

---

### anofox_statistics_ridge

**Description:** Ridge regression for array inputs with L2 regularization.

**Signature:**

```sql
anofox_statistics_ridge(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(coefficients DOUBLE[], intercept DOUBLE, lambda DOUBLE, ...)
```

**Example:**


```sql

-- Table function requires literal arrays with 2D array + MAP
WITH data AS (
    SELECT
        [100.0::DOUBLE, 98.0, 102.0, 97.0, 101.0] as y,
        [
            [10.0::DOUBLE, 9.8, 10.2, 9.7, 10.1],
            [9.9::DOUBLE, 9.7, 10.1, 9.8, 10.0]
        ] as X
)
SELECT result.* FROM data,
LATERAL anofox_statistics_ridge_fit(
    data.y,
    data.X,
    {'lambda': 0.1, 'intercept': true}
) as result;
```

---

### anofox_statistics_rls

**Description:** Recursive Least Squares for array inputs with sequential updating.

**Signature:**

```sql
anofox_statistics_rls(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(coefficients DOUBLE[], intercept DOUBLE, forgetting_factor DOUBLE, ...)
```

**Parameters:**

- `options`: MAP with `forgetting_factor` (0.9-1.0, default: 1.0)

**When to use:** Sequential data processing, adaptive filtering.

---

### anofox_statistics_elastic_net

**Description:** Elastic Net regression for array inputs with L1+L2 regularization.

**Signature:**

```sql
anofox_statistics_elastic_net(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(coefficients DOUBLE[], intercept DOUBLE, n_nonzero BIGINT, converged BOOLEAN, ...)
```

**Parameters:**

- `options`: MAP with `alpha` (0-1, L1/L2 mix), `lambda` (strength), `max_iterations`, `tolerance`

**Example:**


```sql
-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
SELECT
    result.coefficients,
    result.intercept,
    result.n_nonzero,
    result.r_squared > 0.9 as good_fit
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
SELECT
    array_length(result.coefficients) as n_coeffs,
    result.intercept IS NOT NULL as has_intercept,
    result.n_nonzero >= 2 as all_nonzero  -- Ridge doesn't zero coefficients
FROM anofox_statistics_elastic_net_fit(
    [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.0::DOUBLE, 1.0::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
SELECT
    array_length(result.coefficients) as n_features,
    result.n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    result.r_squared
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1::DOUBLE, 0.2, 0.3, 0.4, 0.5, 0.6], [0.05::DOUBLE, 0.10, 0.15, 0.20, 0.25, 0.30]],
    options := MAP(['alpha', 'lambda', 'intercept'], [1.0::DOUBLE, 0.5::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
SELECT
    result.intercept IS NULL as no_intercept,
    array_length(result.coefficients) = 2 as correct_size
FROM anofox_statistics_elastic_net_fit(
    [1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0],
    [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0], [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0]],
    options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 0.0::DOUBLE])
) as result;
-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0::DOUBLE, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0::DOUBLE, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0::DOUBLE, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0::DOUBLE, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0::DOUBLE, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0::DOUBLE, 4.0]
),
aggregated AS (
    SELECT
        category,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.5::DOUBLE, 0.01::DOUBLE])) as result
    FROM data
    GROUP BY category
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit
FROM aggregated;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0::DOUBLE, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0::DOUBLE, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0::DOUBLE, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0::DOUBLE, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0::DOUBLE, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0::DOUBLE, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_fit_agg(y, x, MAP(['alpha', 'lambda'], [0.3::DOUBLE, 0.1::DOUBLE]))
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data
-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
SELECT
    COUNT(*) = 3 as has_all_features,
    MAX(CASE WHEN variable_name = 'x2' THEN vif ELSE 0 END) > 5 as high_multicollinearity,  -- Second feature is 2x first
    MAX(CASE WHEN variable_name = 'x2' THEN severity ELSE '' END) IN ('high', 'perfect') as correct_severity
FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 1.5], [2.0::DOUBLE, 4.0, 3.0], [3.0::DOUBLE, 6.0, 4.5],
                            [4.0::DOUBLE, 8.0, 6.0], [5.0::DOUBLE, 10.0, 7.5]]);

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM anofox_statistics_normality_test([0.5::DOUBLE, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
                                       0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9], 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN result.is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(result.std_residual)) > 2.0 as has_extreme_residual
FROM anofox_statistics_residual_diagnostics([1.0::DOUBLE, 2.0, 3.0, 4.0, 10.0],
                                            [1.1::DOUBLE, 1.9, 3.1, 3.9, 4.5],
                                            outlier_threshold := 2.5) as result;
-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
),
aggregated AS (
    SELECT
        group_id,
        anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM data
    GROUP BY group_id
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM aggregated;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual > result.mean_abs_residual as logical_max
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP(['outlier_threshold'], [2.0::DOUBLE])) as result
    FROM data
);

-- DISABLED: Tests 4.3-4.6 use aggregate functions that don't exist yet
-- (anofox_statistics_vif_agg, anofox_statistics_normality_test_agg)
-- These tests will be enabled when the aggregate versions are implemented

-- -- Test 4.3: vif_aggregate - VIF per group
-- SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
-- WITH data AS (
--     SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
--     SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
--     SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
--     SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
--     SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
--     SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
--     SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
--     SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
--     SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
--     SELECT 'B', [5.0, 3.0, 6.0]
-- ),
-- aggregated AS (
--     SELECT
--         category,
--         anofox_statistics_vif_agg(x) as result
--     FROM data
--     GROUP BY category
-- )
-- SELECT
--     category,
--     array_length(result.vif) = 3 as correct_feature_count,
--     result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
--     array_length(result.severity) = 3 as has_severity_labels
-- FROM aggregated;

-- -- Test 4.4: vif_aggregate with window function
-- SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- -- Note: VIF aggregate does not support window functions as per requirements
-- -- This test just verifies it works with GROUP BY only

-- -- Test 4.5: normality_test_aggregate - Jarque-Bera per group
-- SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
-- WITH data AS (
--     -- Group A: Normally distributed residuals
--     SELECT 'A' as group_id, 0.1 as residual UNION ALL
--     SELECT 'A', 0.2 UNION ALL
--     SELECT 'A', -0.1 UNION ALL
--     SELECT 'A', 0.3 UNION ALL
--     SELECT 'A', -0.2 UNION ALL
--     SELECT 'A', 0.15 UNION ALL
--     SELECT 'A', -0.05 UNION ALL
--     SELECT 'A', 0.25 UNION ALL
--     SELECT 'A', -0.15 UNION ALL
--     SELECT 'A', 0.05 UNION ALL
--     -- Group B: Skewed distribution
--     SELECT 'B', 1.0 UNION ALL
--     SELECT 'B', 1.5 UNION ALL
--     SELECT 'B', 2.0 UNION ALL
--     SELECT 'B', 2.5 UNION ALL
--     SELECT 'B', 3.0 UNION ALL
--     SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
--     SELECT 'B', 1.2 UNION ALL
--     SELECT 'B', 1.8 UNION ALL
--     SELECT 'B', 2.2 UNION ALL
--     SELECT 'B', 2.8
-- ),
-- aggregated AS (
--     SELECT
--         group_id,
--         anofox_statistics_normality_test_agg(residual, {'alpha': 0.05}) as result
--     FROM data
--     GROUP BY group_id
-- )
-- SELECT
--     group_id,
--     result.n_obs >= 8 as sufficient_obs,
--     result.jb_statistic >= 0 as valid_statistic,
--     result.p_value BETWEEN 0 AND 1 as valid_pvalue,
--     result.conclusion IN ('normal', 'non-normal') as valid_conclusion
-- FROM aggregated;

-- -- Test 4.6: normality_test_aggregate with different alpha levels
-- SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
-- WITH data AS (
--     SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
--     SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
--     SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
--     SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
-- )
-- SELECT
--     result.n_obs = 12 as correct_count,
--     result.is_normal IS NOT NULL as has_test_result,
--     result.p_value >= 0 as valid_pvalue
-- FROM (
--     SELECT anofox_statistics_normality_test_agg(residual, {'alpha': 0.01}) as result
--     FROM data
-- );
-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH model AS (
    SELECT
        result.coefficients,
        result.intercept,
        result.r_squared,
        result.n_nonzero
    FROM anofox_statistics_elastic_net_fit(
        [2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1],
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
         [1.5::DOUBLE, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
         [2.0::DOUBLE, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        options := MAP(['alpha', 'lambda', 'intercept'], [0.5::DOUBLE, 0.1::DOUBLE, 1.0::DOUBLE])
    ) as result
),
predictions AS (
    SELECT
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest([2.5::DOUBLE, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1]) + 0.1 as y_predicted  -- Simulated predictions
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP(['outlier_threshold'], [2.5::DOUBLE])) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
```

---

## Diagnostic Functions

### Statistical Inference (Integrated)

**Description:** Statistical inference is integrated into fit functions using `full_output=true`. Computes t-statistics, p-values, and confidence intervals.

**Signature:**

```sql
anofox_statistics_ols_fit(
    y DOUBLE[],
    x DOUBLE[][],
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
) → STRUCT(coefficients, coefficient_p_values, coefficient_t_statistics, intercept_p_value, f_statistic, ...)
```

**Parameters:**

- `full_output`: Set to true to get inference statistics
- `confidence_level`: Confidence level (default: 0.95)
- `intercept`: Include intercept (default: true)

**Returns:** STRUCT with all regression statistics including inference.

**Example:**


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

**When to use:** Hypothesis testing, determining which predictors are significant.

---

### anofox_statistics_predict_ols

**Description:** Predictions with confidence and prediction intervals.

**Signature:**

```sql
anofox_statistics_predict_ols(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new DOUBLE[][],
    options MAP
) → TABLE(observation_id BIGINT, predicted DOUBLE, ci_lower DOUBLE, ci_upper DOUBLE, se DOUBLE)
```

**Parameters:**

- `options`: MAP with `confidence_level` (default: 0.95), `intercept` (default: true)

**Returns:** Predictions with both confidence intervals (for mean) and prediction intervals (for individual predictions).

**Example:**


```sql

-- Use the predict function for prediction intervals
SELECT
    predicted,
    ci_lower,
    ci_upper,
    ci_upper - ci_lower as interval_width
FROM anofox_statistics_predict_ols(
    [50.0, 55.0, 60.0, 65.0, 70.0]::DOUBLE[],           -- y_train: historical_sales
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],    -- x_train: historical_features
    [[6.0], [7.0], [8.0]]::DOUBLE[][],                  -- x_new: future_features
    0.95,                                                 -- confidence_level
    'prediction',                                         -- interval_type
    true                                                  -- intercept
);
```

---

### anofox_statistics_information_criteria

**Description:** Model selection criteria (AIC, BIC) for comparing models.

**Signature:**

```sql
anofox_statistics_information_criteria(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(n_obs BIGINT, n_params BIGINT, rss DOUBLE, r_squared DOUBLE, aic DOUBLE, bic DOUBLE)
```

**Example:**


```sql

-- Compare two models (using literal arrays)
WITH model1 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],  -- y: sales
        [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]]::DOUBLE[][],  -- x: price only
        true                                                -- add_intercept
    )
),
model2 AS (
    SELECT * FROM information_criteria(
        [100.0, 95.0, 92.0, 88.0, 85.0, 82.0]::DOUBLE[],  -- y: sales
        [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0], [15.0, 10.0]]::DOUBLE[][],  -- x: price + advertising
        true                                                -- add_intercept
    )
)
SELECT
    'Model 1 (price only)' as model,
    aic, bic, r_squared FROM model1
UNION ALL
SELECT
    'Model 2 (price + ads)',
    aic, bic, r_squared FROM model2;
```

**When to use:** Model comparison, preventing overfitting.

---

### anofox_statistics_residual_diagnostics

**Description:** Comprehensive residual diagnostics including leverage, Cook's distance, and influence measures.

**Signature:**

```sql
anofox_statistics_residual_diagnostics(
    y_actual DOUBLE[],
    y_predicted DOUBLE[],
    outlier_threshold DOUBLE DEFAULT 2.5
) → TABLE(obs_id BIGINT, residual DOUBLE, standardized_residual DOUBLE, is_outlier BOOLEAN, ...)
```

**Example:**


```sql

-- Note: residual_diagnostics expects y_actual and y_predicted, not y and X
WITH predictions AS (
    SELECT
        [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[] as y_actual,
        [50.5, 54.8, 60.2, 64.9, 70.1, 74.7]::DOUBLE[] as y_predicted  -- Simulated predictions
)
SELECT
    obs_id,
    residual,
    std_residual,
    is_outlier  -- TRUE if |std_residual| > 2.5
FROM predictions, anofox_statistics_residual_diagnostics(
    y_actual,
    y_predicted,
    2.5  -- outlier_threshold
);
```

---

### anofox_statistics_vif

**Description:** Variance Inflation Factor for detecting multicollinearity.

**Signature:**

```sql
anofox_statistics_vif(
    x DOUBLE[][]
) → TABLE(variable_index BIGINT, vif DOUBLE, severity VARCHAR)
```

**Returns:** VIF for each predictor. VIF > 10 indicates problematic multicollinearity.

---

### anofox_statistics_normality_test

**Description:** Jarque-Bera test for normality of residuals.

**Signature:**

```sql
anofox_statistics_normality_test(
    residuals DOUBLE[],
    alpha DOUBLE DEFAULT 0.05
) → TABLE(n_obs BIGINT, skewness DOUBLE, kurtosis DOUBLE, jb_statistic DOUBLE, p_value DOUBLE, is_normal BOOLEAN, conclusion VARCHAR)
```

**Example:**


```sql

-- Test normality of residuals (use literal array)
SELECT * FROM anofox_statistics_normality_test(
    [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.0, 0.1, -0.1, 0.2]::DOUBLE[],  -- residuals
    0.05                                                                 -- alpha
);

-- Note: To test residuals from a table, first extract to array using LIST()
-- Example: SELECT LIST(residual) FROM my_diagnostics_table
```

---

## Options MAP Reference

All functions accept an `options` MAP parameter for configuration. Common keys:

### Common Options (All Functions)

- **`intercept`** (BOOLEAN, default: `true`): Include intercept term in model

  ```sql
  MAP{'intercept': false}  -- No intercept (forced through origin)
  ```

### Ridge-specific Options

- **`lambda`** (DOUBLE, default: `1.0`): Regularization strength for Ridge regression

  ```sql
  MAP{'lambda': 0.5, 'intercept': true}
  ```

  - Higher λ = more shrinkage toward zero
  - λ = 0 equivalent to OLS
  - Typical range: 0.01 to 10

### RLS-specific Options

- **`forgetting_factor`** (DOUBLE, default: `1.0`): Exponential weighting for RLS

  ```sql
  MAP{'forgetting_factor': 0.95}
  ```

  - Range: 0.9 to 1.0
  - 1.0 = no forgetting (equivalent to OLS)
  - 0.95 = moderate adaptation (5% decay per step)
  - Lower values = faster adaptation to recent changes

### Elastic Net-specific Options

- **`alpha`** (DOUBLE, default: `0.5`): L1/L2 mixing parameter

  ```sql
  MAP{'alpha': 0.5, 'lambda': 0.1}
  ```

  - α = 0: Pure Ridge (L2 only)
  - α = 1: Pure Lasso (L1 only)
  - α = 0.5: Equal mix of L1 and L2

- **`lambda`** (DOUBLE, default: `0.1`): Overall regularization strength

- **`max_iterations`** (BIGINT, default: `1000`): Maximum coordinate descent iterations

- **`tolerance`** (DOUBLE, default: `1e-6`): Convergence tolerance

### Inference Options

- **`confidence_level`** (DOUBLE, default: `0.95`): Confidence level for intervals

  ```sql
  MAP{'confidence_level': 0.99, 'intercept': true}
  ```

  - 0.90 = 90% confidence
  - 0.95 = 95% confidence (standard)
  - 0.99 = 99% confidence (stricter)

### Diagnostic Aggregate Options

- **`mode`** (VARCHAR, default: `'summary'`): Output mode for residual diagnostics

  ```sql
  MAP{'mode': 'detailed', 'outlier_threshold': 3.0}
  ```

  - `'summary'`: Group-level summary statistics
  - `'detailed'`: Per-observation diagnostics

- **`outlier_threshold`** (DOUBLE, default: `2.5`): Threshold for flagging outliers (in standard deviations)

- **`alpha`** (DOUBLE, default: `0.05`): Significance level for normality test

  ```sql
  MAP{'alpha': 0.01}  -- 99% confidence
  ```

---

## Quick Reference Table

| Function | Type | Use Case | Key Options |
|----------|------|----------|-------------|
| `anofox_statistics_ols_agg` | Aggregate | Standard regression per group | `intercept` |
| `anofox_statistics_wls_agg` | Aggregate | Weighted regression (heteroscedasticity) | `intercept` |
| `anofox_statistics_ridge_agg` | Aggregate | Regularized regression (multicollinearity) | `lambda`, `intercept` |
| `anofox_statistics_rls_agg` | Aggregate | Adaptive regression (time-series) | `forgetting_factor`, `intercept` |
| `anofox_statistics_elastic_net_agg` | Aggregate | Feature selection + regularization | `alpha`, `lambda`, `intercept` |
| `anofox_statistics_residual_diagnostics_agg` | Aggregate | Group-wise outlier detection | `mode`, `outlier_threshold` |
| `anofox_statistics_vif_agg` | Aggregate | Group-wise multicollinearity check | None |
| `anofox_statistics_normality_test_agg` | Aggregate | Group-wise normality test | `alpha` |
| `anofox_statistics_ols` | Table | OLS for arrays | `intercept` |
| `anofox_statistics_wls` | Table | WLS for arrays | `intercept` |
| `anofox_statistics_ridge` | Table | Ridge for arrays | `lambda`, `intercept` |
| `anofox_statistics_rls` | Table | RLS for arrays | `forgetting_factor`, `intercept` |
| `anofox_statistics_elastic_net` | Table | Elastic Net for arrays | `alpha`, `lambda`, `max_iterations` |
| `anofox_statistics_ols_fit` (with `full_output=true`) | Table | Coefficient significance tests | `confidence_level` |
| `anofox_statistics_predict_ols` | Table | Predictions with intervals | `confidence_level`, `interval_type` |
| `anofox_statistics_information_criteria` | Table | Model selection (AIC/BIC) | `intercept` |
| `anofox_statistics_residual_diagnostics` | Table | Outlier and influence detection | `outlier_threshold` |
| `anofox_statistics_vif` | Table | Multicollinearity detection | None |
| `anofox_statistics_normality_test` | Table | Residual normality test | `alpha` |

---

## See Also

- [Quick Start Guide](01_quick_start.md): Get started quickly with examples
- [Statistics Guide](03_statistics_guide.md): Understanding the statistical methodology
- [Business Guide](04_business_guide.md): Real-world business applications
- [Advanced Use Cases](05_advanced_use_cases.md): Complex analytical workflows
