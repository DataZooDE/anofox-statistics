# Anofox Statistics Extension - API Reference

**Version:** 0.2.0
**DuckDB Version:** ≥ v1.4.1
**Statistical Engine:** libanostat

---

## Overview

The Anofox Statistics extension provides comprehensive statistical regression analysis capabilities directly within DuckDB. All statistical computations are performed by the **libanostat** library, which implements efficient linear algebra operations using Eigen3.

### Key Features

- **5 Regression Methods**: OLS, Ridge, WLS, RLS, Elastic Net
- **Flexible Function Types**: Table functions, aggregates, window functions
- **Complete Statistical Inference**: Coefficient testing, prediction intervals
- **Diagnostic Tools**: VIF, residual analysis, normality tests
- **Model Selection**: AIC, BIC, adjusted R²

### Function Naming Conventions

All functions follow the pattern: `anofox_statistics_{method}[_operation][_agg]`

- `{method}`: Regression method (ols, ridge, wls, rls, elastic_net)
- `_operation`: Optional operation (inference, predict_interval, etc.)
- `_agg`: Suffix for aggregate functions

### Parameter Conventions

**Important**: All functions use **positional parameters**, NOT named parameters (`:=` syntax).

**Common Parameter Types**:
- `y`: Response variable - `DOUBLE[]` for table functions, `DOUBLE` for aggregates
- `x`: Feature matrix - `DOUBLE[][]` for table functions, `DOUBLE[]` for aggregates
- `weights`: Observation weights - `DOUBLE[]` for table functions, `DOUBLE` for aggregates
- `options`: Configuration - `MAP` with string keys and various value types

**Standard Options MAP Keys**:
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `intercept` | BOOLEAN | true | Include intercept term |
| `confidence_level` | DOUBLE | 0.95 | Confidence level for intervals (0-1) |
| `interval_type` | VARCHAR | 'prediction' | Type: 'confidence' or 'prediction' |
| `full_output` | BOOLEAN | false | Include extended metadata for model_predict |
| `lambda` | DOUBLE | - | Ridge regularization strength |
| `alpha` | DOUBLE | - | Elastic Net L1/L2 mix (0=Ridge, 1=Lasso) |
| `forgetting_factor` | DOUBLE | 1.0 | RLS exponential weighting |
| `max_iterations` | INTEGER | 1000 | Elastic Net convergence iterations |
| `tolerance` | DOUBLE | 1e-6 | Elastic Net convergence threshold |
| `outlier_threshold` | DOUBLE | - | Residual diagnostics threshold |
| `detailed` | BOOLEAN | false | Detailed diagnostic output mode |

---

## Table of Contents

1. [Regression Table Functions](#regression-table-functions)
2. [Regression Aggregate Functions](#regression-aggregate-functions)
3. [Fit-Predict Window Aggregates](#fit-predict-window-aggregates)
4. [Inference Functions](#inference-functions)
5. [Prediction Interval Functions](#prediction-interval-functions)
6. [Diagnostic Functions](#diagnostic-functions)
7. [Model Selection Functions](#model-selection-functions)
8. [Scalar Functions](#scalar-functions)
9. [Function Coverage Matrix](#function-coverage-matrix)

---

## Regression Table Functions

These functions fit regression models on array inputs and return comprehensive statistics.

### Common Signature Pattern

```sql
anofox_statistics_{method}(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

### Common Return Structure

```sql
STRUCT(
    coefficients        DOUBLE[],     -- Feature coefficients (β)
    intercept           DOUBLE,        -- Model intercept
    r_squared           DOUBLE,        -- R² statistic
    adj_r_squared       DOUBLE,        -- Adjusted R²
    mse                 DOUBLE,        -- Mean squared error
    rmse                DOUBLE,        -- Root mean squared error
    n_obs               BIGINT,        -- Number of observations
    n_features          BIGINT,        -- Number of features

    -- With full_output=true:
    x_train_means             DOUBLE[],
    coefficient_std_errors    DOUBLE[],
    intercept_std_error       DOUBLE,
    df_residual               BIGINT,
    rank                      BIGINT
)
```

---

### anofox_statistics_ols

**Ordinary Least Squares Regression**

Fits a linear model by minimizing the sum of squared residuals.

**Signature:**
```sql
anofox_statistics_ols(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

**Parameters:**
- `y`: Response variable array (n observations)
- `x`: Feature matrix (n rows × p columns)
- `options`: Configuration MAP
  - `intercept` (BOOLEAN, default=true): Include intercept term
  - `full_output` (BOOLEAN, default=false): Include extended metadata

**Returns:** STRUCT with regression statistics (see Common Return Structure)

**Example:**
```sql
SELECT * FROM anofox_statistics_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    MAP{'intercept': true}
);
```

**Statistical Implementation:** libanostat OLSSolver with QR decomposition

---

### anofox_statistics_ridge

**Ridge Regression (L2 Regularization)**

Fits a linear model with L2 penalty to reduce multicollinearity effects.

**Signature:**
```sql
anofox_statistics_ridge(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

**Parameters:**
- `y`: Response variable array
- `x`: Feature matrix
- `options`: Configuration MAP
  - `intercept` (BOOLEAN, default=true)
  - `lambda` (**required** DOUBLE): Regularization strength (≥ 0)
  - `full_output` (BOOLEAN, default=false)

**Returns:** STRUCT with regression statistics + `lambda` field

**Example:**
```sql
SELECT * FROM anofox_statistics_ridge(
    y_array,
    x_matrix,
    MAP{'intercept': true, 'lambda': 1.0}
);
```

**Statistical Implementation:** libanostat Ridge solver with (X'X + λI)⁻¹X'y

---

### anofox_statistics_wls

**Weighted Least Squares**

Fits a linear model with observation-specific weights for heteroscedasticity.

**Signature:**
```sql
anofox_statistics_wls(
    y       DOUBLE[],
    x       DOUBLE[][],
    weights DOUBLE[],
    options MAP
) → STRUCT
```

**Parameters:**
- `y`: Response variable array
- `x`: Feature matrix
- `weights`: Observation weights (must be > 0)
- `options`: Configuration MAP
  - `intercept` (BOOLEAN, default=true)
  - `full_output` (BOOLEAN, default=false)

**Returns:** STRUCT with regression statistics + `sum_weights` field

**Example:**
```sql
SELECT * FROM anofox_statistics_wls(
    y_array,
    x_matrix,
    [1.0, 2.0, 1.5, 3.0, 2.0]::DOUBLE[],
    MAP{'intercept': true}
);
```

**Statistical Implementation:** libanostat WLS solver with W½X and W½y

---

### anofox_statistics_elastic_net

**Elastic Net (L1 + L2 Regularization)**

Fits a linear model with combined L1 and L2 penalties for feature selection and stability.

**Signature:**
```sql
anofox_statistics_elastic_net(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

**Parameters:**
- `y`: Response variable array
- `x`: Feature matrix
- `options`: Configuration MAP
  - `intercept` (BOOLEAN, default=true)
  - `alpha` (**required** DOUBLE): L1/L2 mix ratio (0=Ridge, 1=Lasso, 0-1 range)
  - `lambda` (**required** DOUBLE): Overall regularization strength
  - `max_iterations` (INTEGER, default=1000)
  - `tolerance` (DOUBLE, default=1e-6)
  - `full_output` (BOOLEAN, default=false)

**Returns:** STRUCT with regression statistics + additional fields:
- `alpha`: L1/L2 mix ratio used
- `lambda`: Regularization strength used
- `n_nonzero`: Number of non-zero coefficients
- `n_iterations`: Iterations to convergence
- `converged`: Whether algorithm converged (BOOLEAN)

**Example:**
```sql
SELECT * FROM anofox_statistics_elastic_net(
    y_array,
    x_matrix,
    MAP{
        'alpha': 0.5,
        'lambda': 0.1,
        'intercept': true
    }
);
```

**Statistical Implementation:** Coordinate descent algorithm with soft-thresholding

---

### anofox_statistics_rls

**Recursive Least Squares (Online Learning)**

Fits a linear model sequentially with optional exponential forgetting.

**Signature:**
```sql
anofox_statistics_rls(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

**Parameters:**
- `y`: Response variable array
- `x`: Feature matrix
- `options`: Configuration MAP
  - `intercept` (BOOLEAN, default=true)
  - `forgetting_factor` (DOUBLE, default=1.0): Exponential weighting (0 < λ ≤ 1)
    - 1.0 = equal weights (standard RLS)
    - < 1.0 = more weight to recent observations
  - `full_output` (BOOLEAN, default=false)

**Returns:** STRUCT with regression statistics + `lambda` (forgetting_factor) field

**Example:**
```sql
SELECT * FROM anofox_statistics_rls(
    y_array,
    x_matrix,
    MAP{'intercept': true, 'forgetting_factor': 0.99}
);
```

**Statistical Implementation:** Sequential update with P = (X'X)⁻¹ recursion

---

## Regression Aggregate Functions

These functions compute regression per group (GROUP BY) or window (OVER clause).

### Common Signature Pattern

```sql
anofox_statistics_{method}_agg(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) [GROUP BY ...] [OVER (...)] → STRUCT
```

**Key Differences from Table Functions**:
- `y` is scalar (per row), not array
- `x` is `DOUBLE[]` (feature vector per row), not `DOUBLE[][]` (matrix)
- Supports both `GROUP BY` and `OVER` (window functions)

### Common Return Structure

Same as table functions, but computed per group/window.

---

### anofox_statistics_ols_agg

**OLS Aggregate Function**

**Signature:**
```sql
anofox_statistics_ols_agg(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) → STRUCT
```

**Supports:**
- ✅ GROUP BY
- ✅ OVER (window functions)

**Example (GROUP BY):**
```sql
SELECT
    category,
    anofox_statistics_ols_agg(sales, [price, advertising], MAP{'intercept': true}) as model
FROM sales_data
GROUP BY category;
```

**Example (Window - Rolling Regression):**
```sql
SELECT
    date,
    anofox_statistics_ols_agg(value, [time_index], MAP{'intercept': true})
        OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as model
FROM time_series;
```

---

### anofox_statistics_wls_agg

**WLS Aggregate Function**

**Signature:**
```sql
anofox_statistics_wls_agg(
    y       DOUBLE,
    x       DOUBLE[],
    weights DOUBLE,
    options MAP
) → STRUCT
```

**Supports:**
- ✅ GROUP BY
- ✅ OVER (window functions)

**Example:**
```sql
SELECT
    region,
    anofox_statistics_wls_agg(
        outcome,
        [predictor1, predictor2],
        weight_column,
        MAP{'intercept': true}
    ) as model
FROM panel_data
GROUP BY region;
```

---

### anofox_statistics_ridge_agg

**Ridge Aggregate Function**

**Signature:**
```sql
anofox_statistics_ridge_agg(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) → STRUCT
```

**Supports:**
- ✅ GROUP BY
- ✅ OVER (window functions)

**Required Option:** `lambda` (DOUBLE)

**Example:**
```sql
SELECT
    date,
    anofox_statistics_ridge_agg(
        returns,
        [factor1, factor2, factor3],
        MAP{'intercept': true, 'lambda': 1.0}
    ) OVER (ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as model
FROM daily_returns;
```

---

### anofox_statistics_rls_agg

**RLS Aggregate Function**

**Signature:**
```sql
anofox_statistics_rls_agg(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) → STRUCT
```

**Supports:**
- ✅ GROUP BY
- ✅ OVER (window functions)

**Example:**
```sql
SELECT
    timestamp,
    anofox_statistics_rls_agg(
        sensor_reading,
        [temperature, humidity],
        MAP{'forgetting_factor': 0.98}
    ) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as model
FROM sensor_data;
```

---

### anofox_statistics_elastic_net_agg

**Elastic Net Aggregate Function**

**Signature:**
```sql
anofox_statistics_elastic_net_agg(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) → STRUCT
```

**Supports:**
- ✅ GROUP BY
- ✅ OVER (window functions)

**Required Options:** `alpha`, `lambda`

**Example:**
```sql
SELECT
    category,
    anofox_statistics_elastic_net_agg(
        y_value,
        [x1, x2, x3, x4, x5],
        MAP{'alpha': 0.7, 'lambda': 0.1, 'intercept': true}
    ) as sparse_model
FROM training_data
GROUP BY category;
```

---

## Fit-Predict Window Aggregates

These functions fit a model on training rows (WHERE y IS NOT NULL) and predict for ALL rows in the window.

### Common Signature Pattern

```sql
anofox_statistics_fit_predict_{method}(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Important**: These are **window-only** functions (require OVER clause).

### Common Return Structure

```sql
STRUCT(
    yhat        DOUBLE,     -- Point prediction
    yhat_lower  DOUBLE,     -- Lower interval bound
    yhat_upper  DOUBLE,     -- Upper interval bound
    std_error   DOUBLE      -- Standard error of prediction
)
```

---

### anofox_statistics_fit_predict_ols

**OLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_statistics_fit_predict_ols(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Options:**
- `intercept` (BOOLEAN, default=true)
- `confidence_level` (DOUBLE, default=0.95)
- `interval_type` (VARCHAR, default='prediction'): 'confidence' or 'prediction'

**Example:**
```sql
SELECT
    date,
    actual_value,
    pred.yhat,
    pred.yhat_lower,
    pred.yhat_upper
FROM (
    SELECT
        date,
        actual_value,
        anofox_statistics_fit_predict_ols(
            actual_value,
            [feature1, feature2],
            MAP{'intercept': true, 'interval_type': 'prediction'}
        ) OVER (
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as pred
    FROM time_series
) sub;
```

---

### anofox_statistics_fit_predict_ridge

**Ridge Fit-Predict Window Function**

**Signature:**
```sql
anofox_statistics_fit_predict_ridge(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Required Option:** `lambda`

---

### anofox_statistics_fit_predict_wls

**WLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_statistics_fit_predict_wls(
    y       DOUBLE,
    x       DOUBLE[],
    weights DOUBLE,
    options MAP
) OVER (...) → STRUCT
```

---

### anofox_statistics_fit_predict_elastic_net

**Elastic Net Fit-Predict Window Function**

**Signature:**
```sql
anofox_statistics_fit_predict_elastic_net(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Required Options:** `alpha`, `lambda`

---

### anofox_statistics_fit_predict_rls

**RLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_statistics_fit_predict_rls(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

---

## Inference Functions

Test statistical significance of model coefficients. Returns one row per parameter.

### Common Return Structure

```sql
TABLE(
    variable      VARCHAR,    -- Parameter name ("intercept", "x1", "x2", ...)
    estimate      DOUBLE,     -- Coefficient value (β)
    std_error     DOUBLE,     -- Standard error of coefficient
    t_statistic   DOUBLE,     -- t = estimate / std_error
    p_value       DOUBLE,     -- Two-tailed p-value
    ci_lower      DOUBLE,     -- Lower confidence bound
    ci_upper      DOUBLE,     -- Upper confidence bound
    significant   BOOLEAN     -- Is p_value < (1 - confidence_level)?
)
```

---

### anofox_statistics_ols_inference

**OLS Coefficient Inference**

**Signature:**
```sql
anofox_statistics_ols_inference(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → TABLE
```

**Options:**
- `confidence_level` (DOUBLE, default=0.95)
- `intercept` (BOOLEAN, default=true)

**Example:**
```sql
SELECT * FROM anofox_statistics_ols_inference(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    MAP{'confidence_level': 0.95, 'intercept': true}
);

-- Output:
-- variable | estimate | std_error | t_statistic | p_value | ci_lower | ci_upper | significant
-- intercept|   0.00   |   0.15    |    0.00     |  1.000  |  -0.47   |   0.47   | false
-- x1       |   1.00   |   0.05    |   20.00     |  0.000  |   0.88   |   1.12   | true
```

**Statistical Implementation:**
- Standard errors from (X'X)⁻¹σ²
- t-statistics with n - p - 1 degrees of freedom
- Two-tailed p-values from Student's t-distribution

---

### anofox_statistics_ridge_inference

**Ridge Coefficient Inference**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_ridge_inference(
    y       DOUBLE[],
    x       DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Option:** `lambda`

**Example:**
```sql
SELECT * FROM anofox_statistics_ridge_inference(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    MAP(['lambda'], [0.1])
);
```

**Note:** Ridge coefficients are biased by design. Standard errors account for regularization.

---

### anofox_statistics_wls_inference

**WLS Coefficient Inference**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_wls_inference(
    y       DOUBLE[],
    x       DOUBLE[][],
    weights DOUBLE[],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_statistics_wls_inference(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    MAP(['intercept'], [true])
);
```

**Note:** Uses heteroscedasticity-consistent standard errors.

---

### anofox_statistics_rls_inference

**RLS Coefficient Inference**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_rls_inference(
    y       DOUBLE[],
    x       DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_statistics_rls_inference(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    MAP(['forgetting_factor'], [0.99])
);
```

**Note:** Tests coefficients from final iteration of recursive least squares.

---

### anofox_statistics_elastic_net_inference

**Elastic Net Coefficient Inference**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_elastic_net_inference(
    y       DOUBLE[],
    x       DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Options:** `alpha`, `lambda`

**Example:**
```sql
SELECT * FROM anofox_statistics_elastic_net_inference(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    MAP(['lambda', 'alpha'], [0.1, 0.5])
);
```

**Notes:**
- Handles sparse coefficients (coefficients exactly zero appear as NULL)
- Intercept is automatically computed when `intercept=true`
- Standard errors for coefficients are NaN (require bootstrap), but intercept SE is estimated
- Coefficients may be zeroed by L1 penalty (sparse solutions)

---

## Prediction Interval Functions

Make predictions on new observations with confidence/prediction intervals. Returns one row per new observation.

### Common Return Structure

```sql
TABLE(
    observation_id  BIGINT,     -- Row number (1-indexed)
    predicted       DOUBLE,     -- Point prediction (ŷ)
    ci_lower        DOUBLE,     -- Lower interval bound
    ci_upper        DOUBLE,     -- Upper interval bound
    se              DOUBLE      -- Standard error of prediction
)
```

---

### anofox_statistics_ols_predict_interval

**OLS Prediction with Intervals**

**Signature:**
```sql
anofox_statistics_ols_predict_interval(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    options MAP
) → TABLE
```

**Parameters:**
- `y_train`: Training response
- `x_train`: Training features
- `x_new`: New observations to predict
- `options`:
  - `confidence_level` (DOUBLE, default=0.95)
  - `interval_type` (VARCHAR, default='prediction'): 'confidence' or 'prediction'
  - `intercept` (BOOLEAN, default=true)

**Interval Types:**
- **Confidence interval**: Uncertainty about **mean** response E[Y|X=x]
  - Formula: ŷ ± t* × SE(ŷ), where SE(ŷ) = √(MSE × x'(X'X)⁻¹x)
- **Prediction interval**: Uncertainty about **individual** observation
  - Formula: ŷ ± t* × √(MSE × (1 + x'(X'X)⁻¹x))
  - Wider (includes residual variance)

**Example:**
```sql
SELECT * FROM anofox_statistics_ols_predict_interval(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    [[6.0], [7.0]]::DOUBLE[][],
    MAP{'interval_type': 'prediction', 'confidence_level': 0.95}
);

-- Output:
-- observation_id | predicted | ci_lower | ci_upper | se
--       1        |    6.0    |   5.1    |   6.9    | 0.45
--       2        |    7.0    |   6.1    |   7.9    | 0.46
```

---

### anofox_statistics_ridge_predict_interval

**Ridge Prediction with Intervals**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_ridge_predict_interval(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Option:** `lambda`

**Example:**
```sql
SELECT * FROM anofox_statistics_ridge_predict_interval(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['lambda'], [0.1])
);
```

---

### anofox_statistics_wls_predict_interval

**WLS Prediction with Intervals**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_wls_predict_interval(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    weights DOUBLE[],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_statistics_wls_predict_interval(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [[6.0], [7.0]],
    MAP(['intercept'], [true])
);
```

---

### anofox_statistics_rls_predict_interval

**RLS Prediction with Intervals**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_rls_predict_interval(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_statistics_rls_predict_interval(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['forgetting_factor'], [0.99])
);
```

---

### anofox_statistics_elastic_net_predict_interval

**Elastic Net Prediction with Intervals**

**Status:** ✅ **IMPLEMENTED**

**Signature:**
```sql
anofox_statistics_elastic_net_predict_interval(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Options:** `alpha`, `lambda`

**Example:**
```sql
SELECT * FROM anofox_statistics_elastic_net_predict_interval(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['lambda', 'alpha'], [0.1, 0.5])
);
```

---

## Diagnostic Functions

### Residual Diagnostics

#### anofox_statistics_residual_diagnostics

**Table Function - Observation-Level Diagnostics**

**Signature:**
```sql
anofox_statistics_residual_diagnostics(
    y_actual        DOUBLE[],
    y_predicted     DOUBLE[],
    outlier_threshold DOUBLE
) → TABLE
```

**Returns:**
```sql
TABLE(
    observation_id       BIGINT,
    residual             DOUBLE,
    standardized_residual DOUBLE,
    is_outlier           BOOLEAN,
    is_influential       BOOLEAN
)
```

**Example:**
```sql
SELECT * FROM anofox_statistics_residual_diagnostics(
    [1.0, 2.1, 2.9, 4.2, 10.0]::DOUBLE[],
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    2.5  -- outlier_threshold
);
```

---

#### anofox_statistics_residual_diagnostics_agg

**Aggregate Function - Group-Level Diagnostics**

**Signature:**
```sql
anofox_statistics_residual_diagnostics_agg(
    y_actual    DOUBLE,
    y_predicted DOUBLE,
    options     MAP
) GROUP BY ... → STRUCT
```

**Options:**
- `outlier_threshold` (DOUBLE, default=2.5)
- `detailed` (BOOLEAN, default=false)

**Returns (Summary Mode - detailed=false):**
```sql
STRUCT(
    n_obs           BIGINT,
    mean_residual   DOUBLE,
    std_residual    DOUBLE,
    outlier_count   BIGINT,
    influential_count BIGINT
)
```

**Returns (Detailed Mode - detailed=true):**
```sql
STRUCT(
    n_obs    BIGINT,
    summary  STRUCT(mean_residual, std_residual, outlier_count, influential_count),
    details  LIST<STRUCT(observation_id, residual, standardized_residual, is_outlier, is_influential)>
)
```

**Example:**
```sql
SELECT
    category,
    anofox_statistics_residual_diagnostics_agg(
        actual,
        predicted,
        MAP{'outlier_threshold': 3.0, 'detailed': false}
    ) as diagnostics
FROM predictions
GROUP BY category;
```

**Note:** Does NOT support OVER (window functions).

---

### Multicollinearity Detection (VIF)

#### anofox_statistics_vif

**Table Function - Feature-Level VIF**

**Signature:**
```sql
anofox_statistics_vif(
    x  DOUBLE[][]
) → TABLE
```

**Returns:**
```sql
TABLE(
    variable_index  BIGINT,
    variable_name   VARCHAR,
    vif             DOUBLE,
    severity        VARCHAR  -- 'none', 'moderate', 'high', 'severe'
)
```

**VIF Interpretation:**
- 1-5: Low multicollinearity
- 5-10: Moderate multicollinearity
- 10+: High multicollinearity
- 100+: Severe multicollinearity

**Example:**
```sql
SELECT * FROM anofox_statistics_vif(
    [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]::DOUBLE[][]
);
```

---

#### anofox_statistics_vif_agg

**Aggregate Function - Group-Level VIF**

**Signature:**
```sql
anofox_statistics_vif_agg(
    x  DOUBLE[]
) GROUP BY ... → STRUCT
```

**Returns:**
```sql
STRUCT(
    variable        VARCHAR,
    vif             DOUBLE,
    severity        VARCHAR
)
```

**Example:**
```sql
SELECT
    dataset_id,
    anofox_statistics_vif_agg([x1, x2, x3]) as vif_results
FROM panel_data
GROUP BY dataset_id;
```

**Note:** Does NOT support OVER (window functions).

---

### Normality Tests

#### anofox_statistics_normality_test

**Table Function - Jarque-Bera Test**

**Signature:**
```sql
anofox_statistics_normality_test(
    residuals  DOUBLE[],
    alpha      DOUBLE
) → STRUCT
```

**Returns:**
```sql
STRUCT(
    n_obs        BIGINT,
    mean         DOUBLE,
    std          DOUBLE,
    skewness     DOUBLE,
    kurtosis     DOUBLE,
    jarque_bera  DOUBLE,    -- JB test statistic
    p_value      DOUBLE,
    is_normal    BOOLEAN    -- True if p_value > alpha
)
```

**Example:**
```sql
SELECT * FROM anofox_statistics_normality_test(
    [0.1, -0.2, 0.15, -0.1, 0.05]::DOUBLE[],
    0.05  -- significance level
);
```

---

#### anofox_statistics_normality_test_agg

**Aggregate Function - Group-Level Normality Test**

**Signature:**
```sql
anofox_statistics_normality_test_agg(
    residual  DOUBLE,
    options   MAP
) GROUP BY ... → STRUCT
```

**Options:**
- `alpha` (DOUBLE, default=0.05): Significance level

**Returns:** Same STRUCT as table function

**Example:**
```sql
SELECT
    model_id,
    anofox_statistics_normality_test_agg(
        residual,
        MAP{'alpha': 0.05}
    ) as normality_test
FROM model_residuals
GROUP BY model_id;
```

**Note:** Does NOT support OVER (window functions).

---

## Model Selection Functions

### anofox_statistics_information_criteria

**AIC, BIC, and Model Selection Metrics**

**Signature:**
```sql
anofox_statistics_information_criteria(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

**Options:**
- `intercept` (BOOLEAN, default=true)

**Returns:**
```sql
STRUCT(
    n_obs          BIGINT,
    n_params       BIGINT,
    rss            DOUBLE,     -- Residual sum of squares
    r_squared      DOUBLE,
    adj_r_squared  DOUBLE,
    aic            DOUBLE,     -- Akaike Information Criterion
    bic            DOUBLE,     -- Bayesian Information Criterion
    aicc           DOUBLE,     -- Corrected AIC (small sample)
    log_likelihood DOUBLE
)
```

**Formulas:**
- AIC = 2k - 2ln(L) = n×ln(RSS/n) + 2k
- BIC = k×ln(n) - 2ln(L) = n×ln(RSS/n) + k×ln(n)
- AICc = AIC + 2k(k+1)/(n-k-1)

Where k = number of parameters, n = number of observations, L = likelihood

**Model Selection Rule:** Lower is better (prefer model with minimum AIC/BIC)

**Example:**
```sql
-- Compare two models
WITH model1 AS (
    SELECT * FROM anofox_statistics_information_criteria(
        y_array, x_matrix_2_features, MAP{'intercept': true}
    )
),
model2 AS (
    SELECT * FROM anofox_statistics_information_criteria(
        y_array, x_matrix_5_features, MAP{'intercept': true}
    )
)
SELECT
    'Model 1' as model, model1.aic, model1.bic FROM model1
UNION ALL
SELECT
    'Model 2' as model, model2.aic, model2.bic FROM model2
ORDER BY aic;
```

---

## Model-Based Prediction

### anofox_statistics_model_predict

**Efficient Prediction from Pre-Fitted Models**

This is the **model-agnostic** prediction function that works with models fitted using `full_output=true`.

**Signature:**
```sql
anofox_statistics_model_predict(
    intercept                DOUBLE,
    coefficients             DOUBLE[],
    mse                      DOUBLE,
    x_train_means            DOUBLE[],
    coefficient_std_errors   DOUBLE[],
    intercept_std_error      DOUBLE,
    df_residual              BIGINT,
    x_new                    DOUBLE[][],
    confidence_level         DOUBLE,
    interval_type            VARCHAR
) → TABLE
```

**Parameters:**
- First 7 parameters: Model metadata (from fit function with `full_output=true`)
- `x_new`: New observations to predict
- `confidence_level`: Confidence level (e.g., 0.95)
- `interval_type`: 'confidence', 'prediction', or 'none'

**Returns:**
```sql
TABLE(
    observation_id  BIGINT,
    predicted       DOUBLE,
    ci_lower        DOUBLE,
    ci_upper        DOUBLE,
    se              DOUBLE
)
```

**Use Case:** Efficient batch prediction without refitting

**Example:**
```sql
-- Step 1: Fit model with full_output
CREATE TABLE model AS
SELECT * FROM anofox_statistics_ols(
    y_array,
    x_matrix,
    MAP{'intercept': true, 'full_output': true}
);

-- Step 2: Predict on new data (no refitting!)
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
    [[6.0], [7.0], [8.0]]::DOUBLE[][],
    0.95,
    'prediction'
) p;
```

**Works with:** All regression methods (OLS, Ridge, WLS, RLS, Elastic Net) when fitted with `full_output=true`.

---

## Scalar Functions

### Metrics Functions

#### anofox_statistics_ols_r_squared

**R² (Coefficient of Determination)**

**Signature:**
```sql
anofox_statistics_ols_r_squared(
    actual     DOUBLE,
    predicted  DOUBLE
) → DOUBLE
```

**Formula:** R² = 1 - (SS_res / SS_tot)

**Example:**
```sql
SELECT anofox_statistics_ols_r_squared(actual, predicted) as r2
FROM predictions;
```

---

#### anofox_statistics_ols_mse

**Mean Squared Error**

**Signature:**
```sql
anofox_statistics_ols_mse(
    actual     DOUBLE,
    predicted  DOUBLE
) → DOUBLE
```

**Formula:** MSE = Σ(y - ŷ)² / n

---

#### anofox_statistics_ols_rmse

**Root Mean Squared Error**

**Signature:**
```sql
anofox_statistics_ols_rmse(
    actual     DOUBLE,
    predicted  DOUBLE
) → DOUBLE
```

**Formula:** RMSE = √(MSE)

---

#### anofox_statistics_ols_mae

**Mean Absolute Error**

**Signature:**
```sql
anofox_statistics_ols_mae(
    actual     DOUBLE,
    predicted  DOUBLE
) → DOUBLE
```

**Formula:** MAE = Σ|y - ŷ| / n

---

### Prediction Functions

#### anofox_statistics_predict

**Full Prediction with Intervals (Scalar)**

**Signature:**
```sql
anofox_statistics_predict(
    intercept                DOUBLE,
    coefficients             DOUBLE[],
    x_new                    DOUBLE[],
    x_train_means            DOUBLE[],
    coefficient_std_errors   DOUBLE[],
    intercept_std_error      DOUBLE,
    mse                      DOUBLE,
    df_residual              BIGINT,
    confidence_level         DOUBLE,
    interval_type            VARCHAR
) → STRUCT
```

**Returns:**
```sql
STRUCT(
    predicted   DOUBLE,
    ci_lower    DOUBLE,
    ci_upper    DOUBLE,
    std_error   DOUBLE
)
```

**Example:**
```sql
SELECT
    observation_id,
    anofox_statistics_predict(
        m.intercept,
        m.coefficients,
        [t.x1, t.x2],
        m.x_train_means,
        m.coefficient_std_errors,
        m.intercept_std_error,
        m.mse,
        m.df_residual,
        0.95,
        'prediction'
    ) as pred
FROM test_data t, model m;
```

---

## Function Coverage Matrix

This matrix shows which functions are currently **implemented** (✅).

| Function Type | OLS | Ridge | WLS | RLS | Elastic Net |
|---------------|-----|-------|-----|-----|-------------|
| **Fit Table** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fit Aggregate** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fit-Predict Aggregate** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Inference Table** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Predict Interval Table** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Summary Statistics

- **Total Functions:** 38 (fully implemented)
- **Removed (v0.2.0):** 3 legacy functions

### Function Categories

| Category | Count |
|----------|-------|
| Regression Table Functions | 5 |
| Regression Aggregates | 5 |
| Fit-Predict Window Aggregates | 5 |
| Inference Functions | 5 (OLS, Ridge, WLS, RLS, Elastic Net) |
| Predict Interval Functions | 5 (OLS, Ridge, WLS, RLS, Elastic Net) |
| Diagnostic Functions | 6 (3 table + 3 aggregate) |
| Model Selection | 1 |
| Model-Based Prediction | 1 |
| Scalar Metrics | 4 |
| Scalar Prediction | 1 |
| **Total** | **38** |

---

## Version History

### v0.2.0 (Current)

**Breaking Changes:**
- Removed `anofox_statistics_ols_predict` (variadic)
- Removed `anofox_statistics_ols_predict_array`
- Removed `anofox_statistics_predict_simple`

**Migration:**
- Use `anofox_statistics_predict` (scalar) for simple predictions
- Use `anofox_statistics_model_predict` (table) for batch predictions

**Planned Additions:**
- Inference functions for Ridge, WLS, RLS, Elastic Net
- Predict interval functions for Ridge, WLS, RLS, Elastic Net
- Predict aggregate functions for all methods

### v0.1.0

- Initial release with 5 regression methods
- Aggregate and window function support
- OLS inference and prediction intervals
- Diagnostic tools (VIF, residuals, normality)
- Model selection (AIC, BIC)

---

## Notes

1. **All statistical calculations** are performed by the **libanostat** library using Eigen3 for linear algebra.

2. **Positional parameters only**: Functions do NOT support named parameters (`:=` syntax). Parameters must be provided in the order specified.

3. **Array indexing**: Feature arrays are 1-indexed in output (x1, x2, ...), 0-indexed internally.

4. **NULL handling**:
   - Missing values in input arrays will cause errors
   - Aliased coefficients (due to multicollinearity) are returned as NULL
   - Window functions handle NULL y values specially (fit-predict)

5. **Performance**:
   - Table functions: O(n) for small p, O(p³) for matrix inversion
   - Aggregates: Optimized for GROUP BY parallelism
   - Window functions: Cached computation when frame doesn't change

6. **Minimum sample sizes**:
   - General rule: n ≥ p + 1 (intercept=true) or n ≥ p (intercept=false)
   - For inference: n > p + 1 to have df_residual ≥ 1

---

## Support

- **Documentation**: [guides/](../guides/)
- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-statistics/issues)
- **Email**: contact@datazoo.de

---

**Last Updated:** 2025-01-25
**API Version:** 0.2.0 (pre-release)
