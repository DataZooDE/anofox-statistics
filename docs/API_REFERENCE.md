# Anofox Statistics Extension - API Reference

**Version:** 0.4.0
**DuckDB Version:** 1.4.3+
**Backend:** Rust (anofox-regression 0.4.0, faer)

## Overview

The Anofox Statistics Extension provides comprehensive regression analysis capabilities for DuckDB. Built with Rust for performance and reliability, it supports multiple regression methods including linear models, generalized linear models (GLM), augmented linear models (ALM), and constrained optimization (BLS/NNLS).

## Table of Contents

1. [Function Types](#function-types)
2. [OLS Functions](#ols-functions)
3. [Ridge Functions](#ridge-functions)
4. [Elastic Net Functions](#elastic-net-functions)
5. [WLS Functions](#wls-functions)
6. [RLS Functions](#rls-functions)
7. [GLM Functions](#glm-functions) *(NEW)*
8. [ALM Functions](#alm-functions) *(NEW)*
9. [BLS/NNLS Functions](#blsnnls-functions) *(NEW)*
10. [Fit-Predict Window Functions](#fit-predict-window-functions)
11. [Predict Aggregate Functions](#predict-aggregate-functions)
12. [Predict Function](#predict-function)
13. [Diagnostic Functions](#diagnostic-functions)
14. [Common Options](#common-options)
15. [Return Types](#return-types)
16. [Short Aliases](#short-aliases)

---

## Function Types

### Scalar Functions (Array-based)
Process complete arrays of data in a single call. Best for batch operations.
```sql
SELECT anofox_stats_ols_fit(y_array, x_arrays);
```

### Aggregate Functions (Streaming)
Accumulate data row-by-row. Support `GROUP BY` and window functions via `OVER`.
```sql
SELECT anofox_stats_ols_fit_agg(y, [x1, x2]) FROM table GROUP BY category;
```

---

## OLS Functions

### anofox_stats_ols_fit
Ordinary Least Squares regression using QR decomposition.

**Signature:**
```sql
anofox_stats_ols_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| y | LIST(DOUBLE) | Response variable values |
| x | LIST(LIST(DOUBLE)) | Feature arrays (each inner list is one feature) |
| fit_intercept | BOOLEAN | Include intercept term (default: true) |
| compute_inference | BOOLEAN | Compute t-tests, p-values, CIs (default: false) |
| confidence_level | DOUBLE | CI confidence level (default: 0.95) |

**Returns:** [FitResult](#fitresult-structure) STRUCT

**Example:**
```sql
-- Simple regression: y = 2x + 1
SELECT anofox_stats_ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]]
);

-- With inference
SELECT anofox_stats_ols_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    true, true, 0.95
);
```

### anofox_stats_ols_fit_agg
Streaming OLS regression aggregate function.

**Signature:**
```sql
anofox_stats_ols_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Example:**
```sql
-- Per-group regression
SELECT
    category,
    (anofox_stats_ols_fit_agg(sales, [price, ads])).r_squared
FROM data
GROUP BY category;

-- Rolling regression (window function)
SELECT
    date,
    (anofox_stats_ols_fit_agg(y, [x]) OVER (
        ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    )).coefficients[1] as rolling_beta
FROM time_series;
```

---

## Ridge Functions

### anofox_stats_ridge_fit
Ridge regression with L2 regularization.

**Signature:**
```sql
anofox_stats_ridge_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    alpha DOUBLE,
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| alpha | DOUBLE | L2 regularization strength (>= 0) |

**Example:**
```sql
SELECT anofox_stats_ridge_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1  -- alpha
);
```

### anofox_stats_ridge_fit_agg
Streaming Ridge regression aggregate function.

```sql
SELECT
    (anofox_stats_ridge_fit_agg(y, [x1, x2], 0.5)).coefficients
FROM data;
```

---

## Elastic Net Functions

### anofox_stats_elasticnet_fit
Elastic Net regression with combined L1/L2 regularization.

**Signature:**
```sql
anofox_stats_elasticnet_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    alpha DOUBLE,
    l1_ratio DOUBLE,
    [fit_intercept BOOLEAN DEFAULT true],
    [max_iterations INTEGER DEFAULT 1000],
    [tolerance DOUBLE DEFAULT 1e-6]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| alpha | DOUBLE | Regularization strength (>= 0) |
| l1_ratio | DOUBLE | L1 ratio: 0=Ridge, 1=Lasso (range: 0-1) |
| max_iterations | INTEGER | Max coordinate descent iterations |
| tolerance | DOUBLE | Convergence tolerance |

**Example:**
```sql
SELECT anofox_stats_elasticnet_fit(
    [2.1, 4.0, 5.9, 8.1, 10.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.1,  -- alpha
    0.5   -- l1_ratio (50% L1, 50% L2)
);
```

### anofox_stats_elasticnet_fit_agg
Streaming Elastic Net aggregate function.

---

## WLS Functions

### anofox_stats_wls_fit
Weighted Least Squares regression.

**Signature:**
```sql
anofox_stats_wls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    weights LIST(DOUBLE),
    [fit_intercept BOOLEAN DEFAULT true],
    [compute_inference BOOLEAN DEFAULT false],
    [confidence_level DOUBLE DEFAULT 0.95]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| weights | LIST(DOUBLE) | Observation weights (same length as y) |

**Example:**
```sql
SELECT anofox_stats_wls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    [1.0, 2.0, 3.0, 2.0, 1.0]  -- higher weight for middle observations
);
```

### anofox_stats_wls_fit_agg
Streaming WLS aggregate function.

```sql
SELECT anofox_stats_wls_fit_agg(y, [x], weight) FROM data;
```

---

## RLS Functions

### anofox_stats_rls_fit
Recursive Least Squares for online/adaptive regression.

**Signature:**
```sql
anofox_stats_rls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [forgetting_factor DOUBLE DEFAULT 1.0],
    [fit_intercept BOOLEAN DEFAULT true],
    [initial_p_diagonal DOUBLE DEFAULT 100.0]
) -> STRUCT
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| forgetting_factor | DOUBLE | Exponential forgetting (0.95-1.0 typical) |
| initial_p_diagonal | DOUBLE | Initial covariance matrix diagonal |

**Example:**
```sql
SELECT anofox_stats_rls_fit(
    [3.0, 5.0, 7.0, 9.0, 11.0],
    [[1.0, 2.0, 3.0, 4.0, 5.0]],
    0.99,  -- forgetting_factor
    true,  -- fit_intercept
    100.0  -- initial_p_diagonal
);
```

### anofox_stats_rls_fit_agg
Streaming RLS aggregate function. Ideal for adaptive/online learning.

```sql
-- Adaptive regression with exponential forgetting
SELECT anofox_stats_rls_fit_agg(y, [x], 0.95) FROM streaming_data;
```

---

## GLM Functions

Generalized Linear Models for count data and other non-normal response distributions.

### anofox_stats_poisson_fit_agg / poisson_fit_agg
Poisson regression for count data using maximum likelihood estimation.

**Signature:**
```sql
anofox_stats_poisson_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| link | VARCHAR | 'log' | Link function: 'log', 'identity', 'sqrt' |
| max_iterations | INTEGER | 100 | Maximum IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Compute z-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | CI confidence level |

**Returns:** [GlmFitResult](#glmfitresult-structure) STRUCT

**Example:**
```sql
-- Basic Poisson regression for count data
SELECT poisson_fit_agg(count, [x1, x2])
FROM event_counts;

-- With inference and custom link
SELECT poisson_fit_agg(
    accidents,
    [traffic_volume, weather_score],
    {'compute_inference': true, 'link': 'log'}
)
FROM daily_accidents;

-- Per-group Poisson regression
SELECT
    region,
    (poisson_fit_agg(sales_count, [price, ads])).coefficients
FROM sales_data
GROUP BY region;
```

**Use Cases:**
- Modeling count data (events, occurrences, frequencies)
- Rate modeling with exposure offsets
- Insurance claims, website visits, defect counts

---

## ALM Functions

Augmented Linear Models with 24 error distribution families for flexible regression.

### anofox_stats_alm_fit_agg / alm_fit_agg
Fit an Augmented Linear Model with choice of distribution and loss function.

**Signature:**
```sql
anofox_stats_alm_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| distribution | VARCHAR | 'normal' | Error distribution (see below) |
| loss | VARCHAR | 'likelihood' | Loss function: 'likelihood', 'mse', 'mae', 'ham', 'role' |
| max_iterations | INTEGER | 100 | Maximum iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| quantile | DOUBLE | 0.5 | Quantile for asymmetric_laplace |
| role_trim | DOUBLE | 0.05 | Trim parameter for ROLE loss |
| compute_inference | BOOLEAN | false | Compute t-tests, p-values, CIs |
| confidence_level | DOUBLE | 0.95 | CI confidence level |

**Supported Distributions:**
| Category | Distributions |
|----------|--------------|
| Continuous (unbounded) | `normal`, `laplace`, `student_t`, `logistic`, `asymmetric_laplace`, `generalised_normal`, `s` |
| Continuous (positive) | `log_normal`, `log_laplace`, `log_s`, `log_generalised_normal`, `gamma`, `inverse_gaussian`, `exponential` |
| Continuous (bounded) | `folded_normal`, `rectified_normal`, `box_cox_normal`, `beta`, `logit_normal` |
| Count | `poisson`, `negative_binomial`, `binomial`, `geometric` |
| Ordinal | `cumulative_logistic`, `cumulative_normal` |

**Returns:** [AlmFitResult](#almfitresult-structure) STRUCT

**Example:**
```sql
-- Robust regression with Laplace distribution (median regression)
SELECT alm_fit_agg(y, [x1, x2], {'distribution': 'laplace'})
FROM data_with_outliers;

-- Quantile regression (75th percentile)
SELECT alm_fit_agg(
    price,
    [sqft, bedrooms],
    {'distribution': 'asymmetric_laplace', 'quantile': 0.75}
)
FROM housing;

-- Gamma regression for positive data
SELECT alm_fit_agg(
    claim_amount,
    [age, risk_score],
    {'distribution': 'gamma', 'compute_inference': true}
)
FROM insurance_claims;

-- Beta regression for proportions (0-1)
SELECT alm_fit_agg(
    conversion_rate,
    [ad_spend, page_views],
    {'distribution': 'beta'}
)
FROM marketing_data;
```

**Use Cases:**
- Robust regression (Laplace, Student-t)
- Quantile regression (asymmetric_laplace)
- Positive outcomes (gamma, log_normal)
- Proportions/rates (beta, logit_normal)
- Count data alternatives (negative_binomial)

---

## BLS/NNLS Functions

Bounded Least Squares and Non-Negative Least Squares for constrained optimization.

### anofox_stats_bls_fit_agg / bls_fit_agg
Bounded Least Squares with box constraints on coefficients.

**Signature:**
```sql
anofox_stats_bls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | false | Include intercept term |
| lower_bound | DOUBLE | - | Lower bound for all coefficients |
| upper_bound | DOUBLE | - | Upper bound for all coefficients |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-10 | Convergence tolerance |

**Returns:** [BlsFitResult](#blsfitresult-structure) STRUCT

**Example:**
```sql
-- Coefficients bounded between 0 and 1
SELECT bls_fit_agg(
    y,
    [x1, x2, x3],
    {'lower_bound': 0.0, 'upper_bound': 1.0}
)
FROM portfolio_data;

-- Only lower bound (coefficients >= 0)
SELECT bls_fit_agg(
    y,
    [x1, x2],
    {'lower_bound': 0.0}
)
FROM data;
```

### anofox_stats_nnls_fit_agg / nnls_fit_agg
Non-Negative Least Squares - all coefficients constrained to be >= 0.

**Signature:**
```sql
anofox_stats_nnls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | false | Include intercept term |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-10 | Convergence tolerance |

**Returns:** [BlsFitResult](#blsfitresult-structure) STRUCT

**Example:**
```sql
-- Non-negative coefficients (e.g., mixture models)
SELECT nnls_fit_agg(spectrum, [component1, component2, component3])
FROM spectral_data;

-- Portfolio weights (no short selling)
SELECT nnls_fit_agg(returns, [stock1, stock2, stock3])
FROM portfolio_data;

-- Per-group NNLS
SELECT
    category,
    (nnls_fit_agg(y, [x1, x2])).coefficients
FROM data
GROUP BY category;
```

**Use Cases:**
- Spectral unmixing / mixture models
- Portfolio optimization without short selling
- Physical constraints (concentrations, weights must be positive)
- Image processing (non-negative matrix factorization)

---

## Fit-Predict Window Functions

Window-based aggregate functions that fit a model incrementally and predict for each row. Use with `OVER` clause for rolling/expanding window regression.

### anofox_stats_ols_fit_predict / ols_fit_predict
OLS regression with per-row predictions using window semantics.

**Signature:**
```sql
anofox_stats_ols_fit_predict(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) OVER (window_spec) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| confidence_level | DOUBLE | 0.95 | Prediction interval confidence |
| null_policy | VARCHAR | 'drop' | NULL handling: 'drop' or 'drop_y_zero_x' |

**Returns:**
```
STRUCT(
    yhat DOUBLE,        -- Predicted value
    yhat_lower DOUBLE,  -- Lower prediction interval bound
    yhat_upper DOUBLE   -- Upper prediction interval bound
)
```

**Example:**
```sql
-- Expanding window: train on all previous rows, predict current
SELECT
    date,
    y,
    pred.yhat,
    pred.yhat_lower,
    pred.yhat_upper
FROM (
    SELECT
        date, y,
        ols_fit_predict(y, [x1, x2]) OVER (
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as pred
    FROM time_series
);

-- Rolling 30-day window
SELECT
    date,
    ols_fit_predict(y, [x]) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as pred
FROM daily_data;

-- Per-group expanding regression
SELECT
    category,
    date,
    ols_fit_predict(y, [x], {'confidence_level': 0.99}) OVER (
        PARTITION BY category
        ORDER BY date
    ) as pred
FROM grouped_data;
```

### anofox_stats_ridge_fit_predict / ridge_fit_predict
Ridge regression with per-row predictions.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | L2 regularization strength |

```sql
SELECT ridge_fit_predict(y, [x], {'alpha': 0.5}) OVER (ORDER BY date) FROM data;
```

### anofox_stats_wls_fit_predict / wls_fit_predict
Weighted Least Squares with per-row predictions.

**Signature:**
```sql
wls_fit_predict(y DOUBLE, x LIST(DOUBLE), weight DOUBLE, [options MAP]) OVER (...)
```

```sql
SELECT wls_fit_predict(y, [x], weight) OVER (ORDER BY date) FROM data;
```

### anofox_stats_rls_fit_predict / rls_fit_predict
Recursive Least Squares with per-row predictions.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| forgetting_factor | DOUBLE | 1.0 | Exponential forgetting (0.95-1.0) |
| initial_p_diagonal | DOUBLE | 100.0 | Initial covariance diagonal |

```sql
SELECT rls_fit_predict(y, [x], {'forgetting_factor': 0.99}) OVER (ORDER BY date) FROM data;
```

### anofox_stats_elasticnet_fit_predict / elasticnet_fit_predict
Elastic Net with per-row predictions.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | Regularization strength |
| l1_ratio | DOUBLE | 0.5 | L1 ratio (0=Ridge, 1=Lasso) |
| max_iterations | INTEGER | 1000 | Max iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

```sql
SELECT elasticnet_fit_predict(y, [x], {'alpha': 0.1, 'l1_ratio': 0.7}) OVER (ORDER BY date) FROM data;
```

---

## Predict Aggregate Functions

Non-rolling aggregate functions that fit a model once on training data (rows where y IS NOT NULL) and return predictions for ALL rows including out-of-sample predictions.

### anofox_stats_ols_predict_agg / ols_predict_agg
Fit OLS on training rows, predict all rows.

**Signature:**
```sql
anofox_stats_ols_predict_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> LIST(STRUCT)
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include intercept term |
| confidence_level | DOUBLE | 0.95 | Prediction interval confidence |
| null_policy | VARCHAR | 'drop' | NULL handling: 'drop' or 'drop_y_zero_x' |

**Returns:**
```
LIST(STRUCT(
    y DOUBLE,           -- Original y value (NULL for out-of-sample)
    x LIST(DOUBLE),     -- Original x values
    yhat DOUBLE,        -- Predicted value
    yhat_lower DOUBLE,  -- Lower prediction interval bound
    yhat_upper DOUBLE,  -- Upper prediction interval bound
    is_training BOOLEAN -- True if row was used for training
))
```

**Example:**
```sql
-- Basic usage: fit on rows where y IS NOT NULL, predict all
CREATE TABLE data AS
SELECT
    CASE WHEN i <= 80 THEN i * 2.0 ELSE NULL END as y,
    i::DOUBLE as x,
    i as id
FROM range(1, 101) t(i);

-- Get predictions with training indicator
SELECT
    (p).y as original_y,
    (p).x as features,
    (p).yhat as predicted,
    (p).is_training
FROM (
    SELECT UNNEST(ols_predict_agg(y, [x])) AS p
    FROM data
);

-- Per-group predictions
SELECT
    segment,
    UNNEST(ols_predict_agg(y, [x1, x2], {'confidence_level': 0.99})) AS pred
FROM sales_data
GROUP BY segment;
```

### anofox_stats_ridge_predict_agg / ridge_predict_agg
Ridge regression predict aggregate.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | L2 regularization strength |

```sql
SELECT UNNEST(ridge_predict_agg(y, [x], {'alpha': 0.5})) FROM data;
```

### anofox_stats_wls_predict_agg / wls_predict_agg
Weighted Least Squares predict aggregate.

**Signature:**
```sql
wls_predict_agg(y DOUBLE, x LIST(DOUBLE), weight DOUBLE, [options MAP]) -> LIST(STRUCT)
```

```sql
SELECT UNNEST(wls_predict_agg(y, [x], weight)) FROM data;
```

### anofox_stats_rls_predict_agg / rls_predict_agg
Recursive Least Squares predict aggregate.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| forgetting_factor | DOUBLE | 1.0 | Exponential forgetting |
| initial_p_diagonal | DOUBLE | 100.0 | Initial covariance diagonal |

```sql
SELECT UNNEST(rls_predict_agg(y, [x], {'forgetting_factor': 0.99})) FROM data;
```

### anofox_stats_elasticnet_predict_agg / elasticnet_predict_agg
Elastic Net predict aggregate.

**Additional Options:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| alpha | DOUBLE | 1.0 | Regularization strength |
| l1_ratio | DOUBLE | 0.5 | L1 ratio (0=Ridge, 1=Lasso) |
| max_iterations | INTEGER | 1000 | Max iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

```sql
SELECT UNNEST(elasticnet_predict_agg(y, [x], {'alpha': 0.1, 'l1_ratio': 0.5})) FROM data;
```

---

## Predict Function

### anofox_stats_predict
Generate predictions using fitted coefficients.

**Signature:**
```sql
anofox_stats_predict(
    x LIST(LIST(DOUBLE)),
    coefficients LIST(DOUBLE),
    intercept DOUBLE
) -> LIST(DOUBLE)
```

**Example:**
```sql
-- First fit a model
WITH model AS (
    SELECT anofox_stats_ols_fit(y_values, x_values) as fit FROM training_data
)
-- Then predict
SELECT anofox_stats_predict(
    [[6.0, 7.0, 8.0]],  -- new x values
    model.fit.coefficients,
    model.fit.intercept
) as predictions
FROM model;
```

---

## Diagnostic Functions

### anofox_stats_vif / vif
Compute Variance Inflation Factor for multicollinearity detection.

**Signature:**
```sql
anofox_stats_vif(x LIST(LIST(DOUBLE))) -> LIST(DOUBLE)
```

**Interpretation:**
- VIF = 1: No correlation
- VIF > 5: Moderate correlation (warning)
- VIF > 10: High correlation (problematic)

**Example:**
```sql
SELECT vif([[x1_vals], [x2_vals], [x3_vals]]) as vif_values;
```

### anofox_stats_vif_agg / vif_agg
Streaming VIF aggregate function.

```sql
SELECT vif_agg([x1, x2, x3]) FROM data;
```

### anofox_stats_aic / aic
Compute Akaike Information Criterion.

**Signature:**
```sql
anofox_stats_aic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| rss | DOUBLE | Residual Sum of Squares |
| n | BIGINT | Number of observations |
| k | BIGINT | Number of parameters (including intercept) |

**Example:**
```sql
SELECT aic(100.0, 50, 3) as aic_value;
```

### anofox_stats_bic / bic
Compute Bayesian Information Criterion.

**Signature:**
```sql
anofox_stats_bic(rss DOUBLE, n BIGINT, k BIGINT) -> DOUBLE
```

**Example:**
```sql
SELECT bic(100.0, 50, 3) as bic_value;
```

### anofox_stats_jarque_bera / jarque_bera
Jarque-Bera test for normality of residuals.

**Signature:**
```sql
anofox_stats_jarque_bera(data LIST(DOUBLE)) -> STRUCT
```

**Returns:**
```
STRUCT(
    statistic DOUBLE,
    p_value DOUBLE,
    skewness DOUBLE,
    kurtosis DOUBLE,
    n BIGINT
)
```

**Example:**
```sql
SELECT jarque_bera(residuals).p_value as normality_pvalue;
```

### anofox_stats_jarque_bera_agg / jarque_bera_agg
Streaming Jarque-Bera aggregate function.

```sql
SELECT jarque_bera_agg(residual) FROM fitted_data;
```

### anofox_stats_residuals_diagnostics / residuals_diagnostics
Compute comprehensive residual diagnostics.

**Signature:**
```sql
anofox_stats_residuals_diagnostics(
    y LIST(DOUBLE),
    y_hat LIST(DOUBLE),
    [x LIST(LIST(DOUBLE))],
    [residual_std_error DOUBLE],
    [include_studentized BOOLEAN]
) -> STRUCT
```

**Returns:**
```
STRUCT(
    raw LIST(DOUBLE),
    standardized LIST(DOUBLE),
    studentized LIST(DOUBLE),
    leverage LIST(DOUBLE)
)
```

**Example:**
```sql
SELECT residuals_diagnostics(
    actual_values,
    predicted_values
) as diagnostics;
```

### anofox_stats_residuals_diagnostics_agg / residuals_diagnostics_agg
Streaming residuals diagnostics aggregate function.

```sql
SELECT residuals_diagnostics_agg(y, y_hat, [x]) FROM data;
```

---

## Common Options

### null_policy Parameter

The `null_policy` option controls how NULL values and zero x values are handled during model training. Available in all fit_predict window functions and predict_agg functions.

| Value | Training Set | Predictions |
|-------|--------------|-------------|
| `'drop'` (default) | Rows where y IS NOT NULL | All rows get predictions |
| `'drop_y_zero_x'` | Rows where y IS NOT NULL AND all x != 0 | All rows get predictions |

**Use Cases:**
- `'drop'`: Standard approach - use all valid observations for training
- `'drop_y_zero_x'`: Exclude zero values which may represent missing data or invalid measurements

**Example:**
```sql
-- Default: only exclude NULL y from training
SELECT ols_fit_predict(y, [x]) OVER (ORDER BY date) FROM data;

-- Exclude both NULL y and rows where any x is 0
SELECT ols_fit_predict(y, [x], {'null_policy': 'drop_y_zero_x'}) OVER (ORDER BY date) FROM data;

-- With predict_agg
SELECT UNNEST(ols_predict_agg(y, [x], {'null_policy': 'drop_y_zero_x'})) FROM data;
```

---

## Return Types

### FitResult Structure

All linear model fit functions (OLS, Ridge, Elastic Net, WLS, RLS) return a STRUCT with the following fields:

```
STRUCT(
    coefficients LIST(DOUBLE),      -- Feature coefficients
    intercept DOUBLE,               -- Intercept (NaN if fit_intercept=false)
    r_squared DOUBLE,               -- R² goodness of fit
    adj_r_squared DOUBLE,           -- Adjusted R²
    residual_std_error DOUBLE,      -- Residual standard error
    n_observations BIGINT,          -- Number of observations
    n_features BIGINT,              -- Number of features
    -- If compute_inference=true:
    std_errors LIST(DOUBLE),        -- Standard errors
    t_values LIST(DOUBLE),          -- t-statistics
    p_values LIST(DOUBLE),          -- p-values
    ci_lower LIST(DOUBLE),          -- CI lower bounds
    ci_upper LIST(DOUBLE),          -- CI upper bounds
    f_statistic DOUBLE,             -- F-statistic
    f_pvalue DOUBLE                 -- F-test p-value
)
```

### GlmFitResult Structure

GLM functions (Poisson) return a STRUCT with:

```
STRUCT(
    coefficients LIST(DOUBLE),      -- Feature coefficients
    intercept DOUBLE,               -- Intercept (NaN if fit_intercept=false)
    deviance DOUBLE,                -- Residual deviance
    null_deviance DOUBLE,           -- Null model deviance
    pseudo_r_squared DOUBLE,        -- McFadden's pseudo R²
    aic DOUBLE,                     -- Akaike Information Criterion
    dispersion DOUBLE,              -- Dispersion parameter
    n_observations BIGINT,          -- Number of observations
    n_features BIGINT,              -- Number of features
    iterations INTEGER,             -- IRLS iterations
    -- If compute_inference=true:
    std_errors LIST(DOUBLE),        -- Standard errors
    z_values LIST(DOUBLE),          -- z-statistics (Wald)
    p_values LIST(DOUBLE),          -- p-values
    ci_lower LIST(DOUBLE),          -- CI lower bounds
    ci_upper LIST(DOUBLE)           -- CI upper bounds
)
```

### AlmFitResult Structure

ALM functions return a STRUCT with:

```
STRUCT(
    coefficients LIST(DOUBLE),      -- Feature coefficients
    intercept DOUBLE,               -- Intercept (NaN if fit_intercept=false)
    log_likelihood DOUBLE,          -- Log-likelihood
    aic DOUBLE,                     -- Akaike Information Criterion
    bic DOUBLE,                     -- Bayesian Information Criterion
    scale DOUBLE,                   -- Scale parameter
    n_observations BIGINT,          -- Number of observations
    n_features BIGINT,              -- Number of features
    iterations INTEGER,             -- Optimization iterations
    -- If compute_inference=true:
    std_errors LIST(DOUBLE),        -- Standard errors
    t_values LIST(DOUBLE),          -- t-statistics
    p_values LIST(DOUBLE),          -- p-values
    ci_lower LIST(DOUBLE),          -- CI lower bounds
    ci_upper LIST(DOUBLE)           -- CI upper bounds
)
```

### BlsFitResult Structure

BLS and NNLS functions return a STRUCT with:

```
STRUCT(
    coefficients LIST(DOUBLE),      -- Feature coefficients (constrained)
    intercept DOUBLE,               -- Intercept (NaN if fit_intercept=false)
    ssr DOUBLE,                     -- Sum of squared residuals
    r_squared DOUBLE,               -- R² goodness of fit
    n_observations BIGINT,          -- Number of observations
    n_features BIGINT,              -- Number of features
    n_active_constraints BIGINT,    -- Number of active constraints
    at_lower_bound LIST(BOOLEAN),   -- Which coefficients are at lower bound
    at_upper_bound LIST(BOOLEAN)    -- Which coefficients are at upper bound
)
```

### Accessing Results

```sql
-- Extract single value
SELECT (anofox_stats_ols_fit(y, x)).r_squared;

-- Extract coefficient
SELECT (anofox_stats_ols_fit(y, x)).coefficients[1];

-- Multiple extractions
SELECT
    fit.coefficients[1] as beta1,
    fit.intercept,
    fit.r_squared
FROM (SELECT anofox_stats_ols_fit(y, x) as fit FROM data);
```

---

## Short Aliases

For convenience, the following short aliases are available:

| Full Name | Short Alias |
|-----------|-------------|
| anofox_stats_ols_fit | ols_fit |
| anofox_stats_ridge_fit | ridge_fit |
| anofox_stats_elasticnet_fit | elasticnet_fit |
| anofox_stats_wls_fit | wls_fit |
| anofox_stats_rls_fit | rls_fit |
| anofox_stats_ols_fit_predict | ols_fit_predict |
| anofox_stats_ridge_fit_predict | ridge_fit_predict |
| anofox_stats_wls_fit_predict | wls_fit_predict |
| anofox_stats_rls_fit_predict | rls_fit_predict |
| anofox_stats_elasticnet_fit_predict | elasticnet_fit_predict |
| anofox_stats_ols_predict_agg | ols_predict_agg |
| anofox_stats_ridge_predict_agg | ridge_predict_agg |
| anofox_stats_wls_predict_agg | wls_predict_agg |
| anofox_stats_rls_predict_agg | rls_predict_agg |
| anofox_stats_elasticnet_predict_agg | elasticnet_predict_agg |
| anofox_stats_poisson_fit_agg | poisson_fit_agg |
| anofox_stats_alm_fit_agg | alm_fit_agg |
| anofox_stats_bls_fit_agg | bls_fit_agg |
| anofox_stats_nnls_fit_agg | nnls_fit_agg |
| anofox_stats_vif | vif |
| anofox_stats_vif_agg | vif_agg |
| anofox_stats_aic | aic |
| anofox_stats_bic | bic |
| anofox_stats_jarque_bera | jarque_bera |
| anofox_stats_jarque_bera_agg | jarque_bera_agg |
| anofox_stats_residuals_diagnostics | residuals_diagnostics |
| anofox_stats_residuals_diagnostics_agg | residuals_diagnostics_agg |

---

## Error Handling

The extension validates inputs and returns clear error messages:

- **Insufficient data**: Minimum 3 observations required for single feature
- **Dimension mismatch**: All feature arrays must have same length as y
- **Singular matrix**: Occurs with perfectly collinear features
- **Invalid parameters**: Alpha must be >= 0, l1_ratio must be in [0, 1]

```sql
-- This will error: insufficient observations
SELECT anofox_stats_ols_fit([1.0, 2.0], [[1.0, 2.0]]);
-- Error: Insufficient data: need at least 3 observations, got 2
```

---

## Performance Notes

1. **Scalar vs Aggregate**: Use scalar functions for batch processing, aggregates for GROUP BY/window operations
2. **Inference overhead**: Setting `compute_inference=true` adds ~30% computation time
3. **Memory**: Aggregate functions accumulate state; consider partitioning large datasets
4. **RLS**: Best for streaming/online scenarios; use OLS for batch analysis

---

## Version History

- **0.4.0**: Added GLM (Poisson), ALM (24 distributions), BLS/NNLS constrained optimization
- **0.3.0**: Added fit_predict window functions, predict_agg aggregate functions, null_policy parameter
- **0.2.0**: Added RLS, Jarque-Bera, residuals diagnostics, VIF aggregate
- **0.1.0**: Initial release with OLS, Ridge, Elastic Net, WLS
