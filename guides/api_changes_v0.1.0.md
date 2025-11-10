# API Changes Since v0.1.0

This document provides a comprehensive overview of all API changes between v0.1.0 and the current version of the Anofox Statistics extension.

## Table of Contents

- [Breaking Changes](#breaking-changes)
- [New Features](#new-features)
- [Removed Functions](#removed-functions)
- [Unchanged Features](#unchanged-features)
- [Migration Guide](#migration-guide)

## Breaking Changes

### 1. Function Naming - Removed `_fit` Suffix

All table functions had their `_fit` suffix removed for consistency and brevity.

**Old (v0.1.0):**

```sql
anofox_statistics_ols_fit(...)
anofox_statistics_ridge_fit(...)
anofox_statistics_wls_fit(...)
anofox_statistics_rls_fit(...)
```

**New (Current):**

```sql
anofox_statistics_ols(...)
anofox_statistics_ridge(...)
anofox_statistics_wls(...)
anofox_statistics_rls(...)
```

**Impact:** All existing queries must update function names by removing the `_fit` suffix.

### 2. Parameter Format - Matrix Input for Features

Changed from individual predictor arrays to a unified matrix format.

**Old (v0.1.0):**

```sql
-- Multiple individual arrays for each predictor
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [1.1, 2.1, 2.9, 4.2, 4.8]::DOUBLE[],  -- x1
    [2.0, 3.0, 4.0, 5.0, 6.0]::DOUBLE[],  -- x2
    [3.0, 4.0, 5.0, 6.0, 7.0]::DOUBLE[],  -- x3
    true                                   -- add_intercept
);
```

**New (Current):**

```sql
-- Single 2D matrix for all features
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [[1.1, 2.0, 3.0],                      -- x matrix (n_obs x n_features)
     [2.1, 3.0, 4.0],
     [2.9, 4.0, 5.0],
     [4.2, 5.0, 6.0],
     [4.8, 6.0, 7.0]]::DOUBLE[][],
    MAP{'intercept': true}                 -- options
);
```

**Impact:** All calls must restructure feature arrays into a 2D matrix format where each row represents one observation and each column represents one feature.

### 3. Options API - MAP-based Instead of Positional

Switched from positional boolean/scalar parameters to MAP-based options for all regression functions.

**Old (v0.1.0):**

```sql
-- Positional parameters
anofox_statistics_ols_fit(y, x1, x2, true)  -- add_intercept as boolean
anofox_statistics_ridge_fit(y, x1, x2, 1.0) -- lambda as double
```

**New (Current):**

```sql
-- MAP-based options
anofox_statistics_ols(y, x, MAP{'intercept': true})
anofox_statistics_ridge(y, x, MAP{'intercept': true, 'lambda': 1.0})
anofox_statistics_elastic_net(y, x, MAP{
    'intercept': true,
    'alpha': 0.5,
    'lambda': 0.1,
    'max_iterations': 1000,
    'tolerance': 1e-6
})
```

**Available Options:**

- `intercept` (BOOLEAN): Include intercept term (default: true)
- `lambda` (DOUBLE): Regularization parameter (Ridge, Elastic Net)
- `alpha` (DOUBLE): L1/L2 mix for Elastic Net (0=Ridge, 1=Lasso)
- `forgetting_factor` (DOUBLE): Exponential weighting for RLS (default: 1.0)
- `confidence_level` (DOUBLE): For inference functions (default: 0.95)
- `full_output` (BOOLEAN): Include extended metadata for model prediction
- `outlier_threshold` (DOUBLE): For residual diagnostics
- `leverage_threshold` (DOUBLE): For residual diagnostics

**Impact:** All function calls must use MAP syntax instead of positional parameters.

### 4. Aggregate Function Signature Changes

The simple scalar aggregate functions have been replaced with structured output aggregates.

**Old (v0.1.0):**

```sql
-- Single coefficient aggregate (simple output)
SELECT category, ols_coeff_agg(sales, price) as coeff
FROM data GROUP BY category;
```

**New (Current):**

```sql
-- Unified aggregate with structured output
SELECT
    category,
    anofox_statistics_ols_agg(
        sales,
        [price],
        MAP{'intercept': true}
    ) as result
FROM data GROUP BY category;

-- Access fields from result struct
SELECT
    category,
    result.coefficients[1] as price_effect,
    result.intercept,
    result.r2
FROM (
    SELECT category, anofox_statistics_ols_agg(sales, [price], MAP{'intercept': true}) as result
    FROM data GROUP BY category
) sub;
```

**Impact:** Queries must use new function names and access struct fields for results.

### 5. Structured Output Types

All aggregate functions now return rich STRUCT types instead of simple scalars.

**Old (v0.1.0):**

```sql
-- Simple coefficient output
ols_coeff_agg(y, x) → DOUBLE
```

**New (Current):**

```sql
-- Rich structured output
anofox_statistics_ols_agg(y, [x], options) → STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    mse DOUBLE,
    n_obs BIGINT,
    x_train_means DOUBLE[],
    coefficient_std_errors DOUBLE[],
    intercept_std_error DOUBLE,
    df_residual BIGINT
)
```

**Impact:** Results must be accessed using struct field notation (`.field`) or extracted in subqueries.

## New Features

### 1. Elastic Net Regression

Combined L1 + L2 regularization for feature selection and stability.

```sql
-- Table function
SELECT * FROM anofox_statistics_elastic_net(
    y_array,
    x_matrix,
    MAP{
        'alpha': 0.5,        -- Mix of L1 and L2 (0=Ridge, 1=Lasso)
        'lambda': 0.1,       -- Regularization strength
        'intercept': true,
        'max_iterations': 1000,
        'tolerance': 1e-6
    }
);

-- Aggregate function (per group or window)
SELECT category,
       anofox_statistics_elastic_net_agg(y, [x1, x2], MAP{'alpha': 0.5, 'lambda': 0.1})
FROM data GROUP BY category;
```

**Returns:** coefficients, intercept, r2, adj_r2, mse, alpha, lambda, n_nonzero (sparsity), n_iterations, converged

### 2. Model-Based Prediction

Efficient prediction on new data using pre-fitted models, with confidence/prediction intervals.

```sql
-- 1. Fit model with full_output to get all metadata
CREATE TABLE model AS
SELECT * FROM anofox_statistics_ols(
    y_array,
    x_matrix,
    MAP{'intercept': true, 'full_output': true}
);

-- 2. Make predictions with confidence intervals
SELECT p.*
FROM model m,
LATERAL anofox_statistics_model_predict(
    m.intercept,
    m.coefficients,
    m.mse,
    m.x_train_means,
    m.coefficient_std_errors,
    m.intercept_std_error,
    m.df_residual,
    [[29.99, 5000.0], [34.99, 6000.0]]::DOUBLE[][],  -- new data
    0.95,           -- confidence level
    'confidence'    -- interval type: 'confidence', 'prediction', or 'none'
) p;

-- Returns: observation_id, predicted, ci_lower, ci_upper, se

-- 3. Prediction intervals (wider than confidence intervals)
SELECT
    observation_id,
    predicted,
    ci_lower,
    ci_upper
FROM model m,
LATERAL anofox_statistics_model_predict(
    m.intercept, m.coefficients, m.mse, m.x_train_means,
    m.coefficient_std_errors, m.intercept_std_error, m.df_residual,
    [[29.99, 5000.0]]::DOUBLE[][],
    0.95,
    'prediction'  -- Prediction intervals for individual observations
) p;
```

**Benefits:**

- No refitting required
- Works with all regression types
- Confidence intervals (for mean predictions) or prediction intervals (for individual predictions)
- Fast batch scoring

### 3. Window Functions Support for All Aggregates

All five regression aggregates now support SQL window functions with `OVER` clause.

**At v0.1.0:** Had separate `rolling_ols` and `expanding_ols` table functions.

**Current:** Unified window function support through aggregates:

```sql
-- Rolling window (30-period)
SELECT
    date,
    anofox_statistics_ols_agg(value, [x1, x2], MAP{'intercept': true})
        OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as model
FROM data;

-- Expanding window (cumulative)
SELECT
    date,
    anofox_statistics_ridge_agg(value, [x1], MAP{'lambda': 1.0})
        OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as model
FROM data;

-- Partitioned rolling (per category)
SELECT
    category, date,
    anofox_statistics_wls_agg(outcome, [pred1, pred2], weight, MAP{'intercept': true})
        OVER (PARTITION BY category ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
FROM data;
```

**Supported Aggregates with Window Functions:**

- `anofox_statistics_ols_agg`
- `anofox_statistics_wls_agg`
- `anofox_statistics_ridge_agg`
- `anofox_statistics_rls_agg`
- `anofox_statistics_elastic_net_agg`

**Impact:** The old `rolling_ols` and `expanding_ols` table functions are removed. Use window functions instead.

### 4. New Diagnostic Aggregate Functions

Expanded from 1 aggregate function to 8 total aggregates.

**Regression Aggregates (support both GROUP BY and OVER):**

- `anofox_statistics_ols_agg(y, x[], options)` - OLS per group/window
- `anofox_statistics_wls_agg(y, x[], weights, options)` - WLS per group/window
- `anofox_statistics_ridge_agg(y, x[], options)` - Ridge per group/window
- `anofox_statistics_rls_agg(y, x[], options)` - RLS per group/window
- `anofox_statistics_elastic_net_agg(y, x[], options)` - Elastic Net per group/window

**Diagnostic Aggregates (GROUP BY only, no window functions):**

- `anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted, options)` - Per-group residual analysis
- `anofox_statistics_vif_agg(x[])` - Per-group VIF multicollinearity detection
- `anofox_statistics_normality_test_agg(residual, options)` - Per-group Jarque-Bera test

**Example - Diagnostic Aggregates:**

```sql
-- Per-group residual diagnostics
SELECT
    category,
    diag.n_obs,
    diag.n_outliers,
    diag.outlier_rate,
    diag.max_std_residual
FROM (
    SELECT
        category,
        anofox_statistics_residual_diagnostics_agg(
            y_actual,
            y_predicted,
            MAP{'mode': 'summary', 'threshold': 2.5}
        ) as diag
    FROM predictions
    GROUP BY category
) sub;

-- Per-group VIF detection
SELECT
    region,
    vif.max_vif,
    vif.severity
FROM (
    SELECT
        region,
        anofox_statistics_vif_agg([price, advertising, competition]) as vif
    FROM features
    GROUP BY region
) sub;
```

### 5. Lateral Join Support

All regression table functions now support lateral joins with column references.

```sql
-- v0.1.0: Required arrays constructed beforehand
CREATE TEMP TABLE temp_arrays AS
SELECT
    category,
    LIST(y) as y_arr,
    LIST(x1) as x1_arr,
    LIST(x2) as x2_arr
FROM data
GROUP BY category;

-- Current: Direct lateral join
SELECT
    d.category,
    r.*
FROM data d
LATERAL anofox_statistics_ols(
    d.y,
    [[d.x1, d.x2]],
    MAP{'intercept': true}
) r;
```

### 6. Full Model Output Option

New `full_output` option stores extended metadata for model-based prediction.

```sql
SELECT * FROM anofox_statistics_ols(
    y, x,
    MAP{'intercept': true, 'full_output': true}
);
```

**Additional fields returned:**

- `x_train_means` (DOUBLE[]): Mean of each feature (for centering)
- `coefficient_std_errors` (DOUBLE[]): Standard errors of coefficients
- `intercept_std_error` (DOUBLE): Standard error of intercept
- `df_residual` (BIGINT): Degrees of freedom for residuals

These fields enable efficient prediction with `anofox_statistics_model_predict()`.

## Removed Functions

### 1. Removed Table Functions

The following dedicated table functions were removed in favor of unified window function support:

- `anofox_statistics_rolling_ols()` → Use `anofox_statistics_ols_agg()` with `OVER` clause
- `anofox_statistics_expanding_ols()` → Use `anofox_statistics_ols_agg()` with `OVER` clause
- `anofox_statistics_rolling_ols_fit()` → Same as above
- `anofox_statistics_expanding_ols_fit()` → Same as above

**Migration:**

```sql
-- v0.1.0: Dedicated rolling function
SELECT * FROM anofox_statistics_rolling_ols(y, x, 30, true);

-- Current: Window function
SELECT
    anofox_statistics_ols_agg(y, [x], MAP{'intercept': true})
        OVER (ORDER BY row_number ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
FROM data;
```

### 2. Removed Aggregate Functions

- `ols_coeff_agg(y, x)` → Replaced by `anofox_statistics_ols_agg(y, [x], MAP{})`
- `ols_fit_agg(y, x)` → Same replacement
- `ols_fit_agg_array(y, x[])` → Same replacement

**Migration:**

```sql
-- v0.1.0: Simple coefficient aggregate
SELECT category, ols_coeff_agg(sales, price) as coeff
FROM data GROUP BY category;

-- Current: Structured output aggregate
SELECT
    category,
    result.coefficients[1] as coeff
FROM (
    SELECT
        category,
        anofox_statistics_ols_agg(sales, [price], MAP{'intercept': true}) as result
    FROM data GROUP BY category
) sub;
```

## Unchanged Features

These functions exist in both v0.1.0 and current version with no breaking changes:

### Basic Metrics (unchanged)

- `ols_r2(y, x)` - R-squared
- `ols_rmse(y, x)` - Root Mean Squared Error
- `ols_mse(y, x)` - Mean Squared Error

### Inference & Diagnostics (signature unchanged)

- `ols_inference(y, x, confidence_level, add_intercept)` - Coefficient inference
- `ols_predict_interval(...)` - Predictions with intervals
- `information_criteria(y, x, add_intercept)` - AIC, BIC
- `residual_diagnostics(y_actual, y_predicted, outlier_threshold)` - Outlier detection
- `vif(x[][])` - Variance Inflation Factor
- `normality_test(residuals[], alpha)` - Jarque-Bera test

**Note:** While these function signatures are unchanged, they work better with the new matrix format for multi-variable regression.

## Migration Guide

### Step-by-Step Migration from v0.1.0

#### 1. Rename Functions

Remove `_fit` suffix from all table function calls:

```sql
-- Old
anofox_statistics_ols_fit(...)
anofox_statistics_ridge_fit(...)

-- New
anofox_statistics_ols(...)
anofox_statistics_ridge(...)
```

#### 2. Restructure Features to Matrix Format

Convert individual arrays to 2D matrix:

```sql
-- Old: Individual arrays per predictor
SELECT * FROM anofox_statistics_ols_fit(
    [1, 2, 3]::DOUBLE[],
    [1, 2, 3]::DOUBLE[],  -- x1
    [4, 5, 6]::DOUBLE[],  -- x2
    true
);

-- New: Single matrix with all features
SELECT * FROM anofox_statistics_ols(
    [1, 2, 3]::DOUBLE[],
    [[1, 4], [2, 5], [3, 6]]::DOUBLE[][],  -- Each row is [x1, x2]
    MAP{'intercept': true}
);
```

#### 3. Use MAP Options

Replace positional parameters with MAP:

```sql
-- Old
anofox_statistics_ridge_fit(y, x1, x2, 1.0)

-- New
anofox_statistics_ridge(y, [[x1], [x2]], MAP{'intercept': true, 'lambda': 1.0})
```

#### 4. Update Aggregates

Replace old aggregate functions:

```sql
-- Old
SELECT category, ols_coeff_agg(sales, price) as coeff
FROM data GROUP BY category;

-- New
SELECT
    category,
    result.coefficients[1] as coeff,
    result.r2,
    result.intercept
FROM (
    SELECT
        category,
        anofox_statistics_ols_agg(sales, [price], MAP{'intercept': true}) as result
    FROM data GROUP BY category
) sub;
```

#### 5. Replace Rolling/Expanding Functions

Use window functions instead:

```sql
-- Old: Dedicated rolling function
SELECT * FROM anofox_statistics_rolling_ols(y, x, 30, true);

-- New: Window function
SELECT
    anofox_statistics_ols_agg(y, [x], MAP{'intercept': true})
        OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as model
FROM data;

-- Old: Expanding window
SELECT * FROM anofox_statistics_expanding_ols(y, x, true);

-- New: Window function
SELECT
    anofox_statistics_ols_agg(y, [x], MAP{'intercept': true})
        OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as model
FROM data;
```

#### 6. Access Struct Fields

Extract results from structured output:

```sql
-- Access individual fields
SELECT
    result.coefficients,
    result.intercept,
    result.r2,
    result.adj_r2,
    result.mse
FROM (SELECT anofox_statistics_ols_agg(...) as result FROM data) sub;

-- Access array elements
SELECT
    result.coefficients[1] as x1_coeff,
    result.coefficients[2] as x2_coeff
FROM (SELECT anofox_statistics_ols_agg(...) as result FROM data) sub;
```

### Complete Example Migration

**v0.1.0:**

```sql
-- Multi-variable regression with rolling window
SELECT
    date,
    ols_coeff_agg(sales, price) as price_effect
FROM (
    SELECT * FROM anofox_statistics_rolling_ols_fit(
        sales_array,
        price_array,
        ad_spend_array,
        30,
        true
    )
) sub
GROUP BY date;
```

**Current:**

```sql
-- Same analysis with new API
SELECT
    date,
    model.coefficients[1] as price_effect,
    model.coefficients[2] as ad_effect,
    model.r2
FROM (
    SELECT
        date,
        anofox_statistics_ols_agg(
            sales,
            [price, ad_spend],
            MAP{'intercept': true}
        ) OVER (
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as model
    FROM sales_data
) sub;
```

## Summary

The v0.1.0 to current migration represents a significant API evolution focused on:

- **Consistency**: Unified function naming and parameter patterns
- **Flexibility**: MAP-based options allow extensibility without breaking changes
- **SQL Idioms**: Window functions and structured types align with SQL standards
- **Usability**: Lateral joins and structured output improve query ergonomics
- **Performance**: Model-based prediction and rich aggregates reduce recomputation

While these changes require updating existing queries, they provide a more robust and maintainable API for long-term use.
