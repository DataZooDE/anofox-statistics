# Anofox Statistics Extension - API Reference

**Version:** 0.2.0
**DuckDB Version:** ≥ v1.4.2
**Statistical Engine:** libanostat

---

## Overview

The Anofox Statistics extension provides comprehensive statistical regression analysis capabilities directly within DuckDB. All statistical computations are performed by the **libanostat** library, which implements efficient linear algebra operations using Eigen3.

### Key Features

- **5 Regression Methods**: OLS, Ridge, WLS, RLS, Elastic Net
- **Flexible Function Types**: Table functions, aggregates, window functions
- **Integrated Statistical Inference**: F-statistic, coefficient/intercept testing, AIC/BIC (with `full_output=true`)
- **Prediction Intervals**: Confidence and prediction intervals for all methods
- **Diagnostic Tools**: VIF, residual analysis, normality tests
- **Efficient API**: Single-call fit + inference workflow

### Function Naming Conventions

All functions follow the pattern: `anofox_stats_{method}[_operation][_agg]`

- `{method}`: Regression method (ols, ridge, wls, rls, elastic_net)
- `_operation`: Optional operation (fit, fit_predict, predict, etc.)
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

## Coverage Matrix

This table shows which function types are available for each regression method:

| Method | Table Fit | Aggregate Fit | Fit-Predict Window | Predict | Aggregate Predict | Notes |
|--------|-----------|---------------|-------------------|---------|-------------------|-------|
| **OLS** | ✅ `anofox_stats_ols_fit` | ✅ `anofox_stats_ols_fit_agg` | ✅ `anofox_stats_ols_fit_predict` | ✅ `anofox_stats_predict_ols` | ❌ Not implemented | Full support |
| **Ridge** | ✅ `anofox_stats_ridge_fit` | ✅ `anofox_stats_ridge_fit_agg` | ✅ `anofox_stats_ridge_fit_predict` | ✅ `anofox_stats_predict_ridge` | ❌ Not implemented | Requires `lambda` |
| **WLS** | ✅ `anofox_stats_wls_fit` | ✅ `anofox_stats_wls_fit_agg` | ✅ `anofox_stats_wls_fit_predict` | ✅ `anofox_stats_predict_wls` | ❌ Not implemented | Requires `weights` |
| **RLS** | ✅ `anofox_stats_rls_fit` | ✅ `anofox_stats_rls_fit_agg` | ✅ `anofox_stats_rls_fit_predict` | ✅ `anofox_stats_predict_rls` | ❌ Not implemented | Optional `forgetting_factor` |
| **Elastic Net** | ✅ `anofox_stats_elastic_net_fit` | ✅ `anofox_stats_elastic_net_fit_agg` | ✅ `anofox_stats_elastic_net_fit_predict` | ✅ `anofox_stats_predict_elastic_net` | ❌ Not implemented | Requires `alpha`, `lambda` |

**Function Types:**
- **Table Fit**: Fit on array inputs `y[]`, `x[][]` → STRUCT with model statistics
- **Aggregate Fit**: Fit per group with `GROUP BY` or window with `OVER` → STRUCT with model statistics
- **Fit-Predict Window**: Window function that fits and predicts → STRUCT with `(yhat, yhat_lower, yhat_upper, std_error)`
- **Predict**: Make predictions on new data with intervals → TABLE with predictions per observation
- **Aggregate Predict**: Predict per group with `GROUP BY` using pre-fitted model → STRUCT (not yet implemented)

**Additional Functions:**
- **Model-Based Prediction**: `anofox_stats_model_predict` - Predict from pre-fitted models (any method with `full_output=true`)
- **Diagnostics**: VIF (table + aggregate), Residual diagnostics (table + aggregate), Normality tests (table + aggregate)

---

## Table of Contents

1. [Fit Functions](#fit-functions)
2. [Aggregate Fit Functions](#aggregate-fit-functions)
3. [Window Aggregate Fit-Predict Functions](#window-aggregate-fit-predict-functions)
4. [Predict Functions](#predict-functions)
5. [Diagnostic Functions](#diagnostic-functions)
6. [Model-Based Prediction](#model-based-prediction)

---

## Fit Functions

These functions fit regression models on array inputs and return comprehensive statistics.

### Common Signature Pattern

```sql
anofox_stats_{method}(
    y       DOUBLE[],
    x       DOUBLE[][],
    options MAP
) → STRUCT
```

### Common Return Structure

```sql
STRUCT(
    -- Basic model fit (always included)
    coefficients        DOUBLE[],     -- Feature coefficients (β)
    intercept           DOUBLE,        -- Model intercept
    r2                  DOUBLE,        -- R² statistic
    adj_r2              DOUBLE,        -- Adjusted R²
    n_obs               BIGINT,        -- Number of observations

    -- Extended metadata (always included)
    mse                 DOUBLE,        -- Mean squared error
    x_train_means       DOUBLE[],      -- Mean of training features
    coefficient_std_errors    DOUBLE[], -- Standard errors of coefficients
    intercept_std_error       DOUBLE,   -- Standard error of intercept
    df_residual               BIGINT,   -- Degrees of freedom (residual)

    -- Model-level inference (always included)
    residual_standard_error   DOUBLE,   -- √(MSE), matches R lm()
    f_statistic               DOUBLE,   -- Overall model F-statistic
    f_statistic_pvalue        DOUBLE,   -- p-value for F-statistic
    aic                       DOUBLE,   -- Akaike Information Criterion
    aicc                      DOUBLE,   -- Corrected AIC (small samples)
    bic                       DOUBLE,   -- Bayesian Information Criterion
    log_likelihood            DOUBLE,   -- Log-likelihood

    -- Coefficient-level inference (always included)
    coefficient_t_statistics  DOUBLE[], -- t-statistic for each coefficient
    coefficient_p_values      DOUBLE[], -- p-value for each coefficient
    coefficient_ci_lower      DOUBLE[], -- Lower confidence bound (per coef)
    coefficient_ci_upper      DOUBLE[], -- Upper confidence bound (per coef)

    -- Intercept-level inference (always included)
    intercept_t_statistic     DOUBLE,   -- t-statistic for intercept
    intercept_p_value         DOUBLE,   -- p-value for intercept
    intercept_ci_lower        DOUBLE,   -- Lower CI bound for intercept
    intercept_ci_upper        DOUBLE    -- Upper CI bound for intercept
)
```

**Note:** All regression aggregate functions return this complete structure with all statistical inference fields included.

---

### anofox_stats_ols

**Ordinary Least Squares Regression**

Fits a linear model by minimizing the sum of squared residuals.

**Signature:**
```sql
anofox_stats_ols_fit(
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

**Example (Basic):**
```sql
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    {'intercept': true}
);
```

**Example (With Statistical Inference):**
```sql
SELECT
    coefficients,
    intercept,
    r2,
    f_statistic,
    f_statistic_pvalue,
    aic,
    bic,
    coefficient_p_values,
    intercept_p_value
FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
);
```

**Statistical Implementation:** libanostat OLSSolver with QR decomposition

---

### anofox_stats_ridge

**Ridge Regression (L2 Regularization)**

Fits a linear model with L2 penalty to reduce multicollinearity effects.

**Signature:**
```sql
anofox_stats_ridge_fit(
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
SELECT * FROM anofox_stats_ridge_fit(
    y_array,
    x_matrix,
    {'intercept': true, 'lambda': 1.0}
);
```

**Statistical Implementation:** libanostat Ridge solver with (X'X + λI)⁻¹X'y

---

### anofox_stats_wls

**Weighted Least Squares**

Fits a linear model with observation-specific weights for heteroscedasticity.

**Signature:**
```sql
anofox_stats_wls_fit(
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
SELECT * FROM anofox_stats_wls_fit(
    y_array,
    x_matrix,
    [1.0, 2.0, 1.5, 3.0, 2.0]::DOUBLE[],
    {'intercept': true}
);
```

**Statistical Implementation:** libanostat WLS solver with W½X and W½y

---

### anofox_stats_elastic_net

**Elastic Net (L1 + L2 Regularization)**

Fits a linear model with combined L1 and L2 penalties for feature selection and stability.

**Signature:**
```sql
anofox_stats_elastic_net_fit(
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
SELECT * FROM anofox_stats_elastic_net_fit(
    y_array,
    x_matrix,
    {
        'alpha': 0.5,
        'lambda': 0.1,
        'intercept': true
    }
);
```

**Statistical Implementation:** Coordinate descent algorithm with soft-thresholding

---

### anofox_stats_rls

**Recursive Least Squares (Online Learning)**

Fits a linear model sequentially with optional exponential forgetting.

**Signature:**
```sql
anofox_stats_rls_fit(
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
SELECT * FROM anofox_stats_rls_fit(
    y_array,
    x_matrix,
    {'intercept': true, 'forgetting_factor': 0.99}
);
```

**Statistical Implementation:** Sequential update with P = (X'X)⁻¹ recursion

---

## Aggregate Fit Functions

These functions compute regression per group (GROUP BY) or window (OVER clause).

### Common Signature Pattern

```sql
anofox_stats_{method}_agg(
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

### anofox_stats_ols_fit_agg

**OLS Aggregate Function**

**Signature:**
```sql
anofox_stats_ols_fit_agg(
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
    anofox_stats_ols_fit_agg(sales, [price, advertising], {'intercept': true}) as model
FROM sales_data
GROUP BY category;
```

**Example (Window - Rolling Regression):**
```sql
SELECT
    date,
    anofox_stats_ols_fit_agg(value, [time_index], {'intercept': true})
        OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as model
FROM time_series;
```

---

### anofox_stats_wls_fit_agg

**WLS Aggregate Function**

**Signature:**
```sql
anofox_stats_wls_fit_agg(
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
    anofox_stats_wls_fit_agg(
        outcome,
        [predictor1, predictor2],
        weight_column,
        {'intercept': true}
    ) as model
FROM panel_data
GROUP BY region;
```

---

### anofox_stats_ridge_fit_agg

**Ridge Aggregate Function**

**Signature:**
```sql
anofox_stats_ridge_fit_agg(
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
    anofox_stats_ridge_fit_agg(
        returns,
        [factor1, factor2, factor3],
        {'intercept': true, 'lambda': 1.0}
    ) OVER (ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) as model
FROM daily_returns;
```

---

### anofox_stats_rls_fit_agg

**RLS Aggregate Function**

**Signature:**
```sql
anofox_stats_rls_fit_agg(
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
    anofox_stats_rls_fit_agg(
        sensor_reading,
        [temperature, humidity],
        {'forgetting_factor': 0.98}
    ) OVER (ORDER BY timestamp ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as model
FROM sensor_data;
```

---

### anofox_stats_elastic_net_fit_agg

**Elastic Net Aggregate Function**

**Signature:**
```sql
anofox_stats_elastic_net_fit_agg(
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
    anofox_stats_elastic_net_fit_agg(
        y_value,
        [x1, x2, x3, x4, x5],
        {'alpha': 0.7, 'lambda': 0.1, 'intercept': true}
    ) as sparse_model
FROM training_data
GROUP BY category;
```

---

## Window Aggregate Fit-Predict Functions

These functions fit a model on training rows (WHERE y IS NOT NULL) and predict for ALL rows in the window.

### Common Signature Pattern

```sql
anofox_stats_{method}_fit_predict(
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

### anofox_stats_ols_fit_predict

**OLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_stats_ols_fit_predict(
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
        anofox_stats_ols_fit_predict(
            actual_value,
            [feature1, feature2],
            {'intercept': true, 'interval_type': 'prediction'}
        ) OVER (
            ORDER BY date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as pred
    FROM time_series
) sub;
```

---

### anofox_stats_ridge_fit_predict

**Ridge Fit-Predict Window Function**

**Signature:**
```sql
anofox_stats_ridge_fit_predict(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Required Option:** `lambda`

---

### anofox_stats_wls_fit_predict

**WLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_stats_wls_fit_predict(
    y       DOUBLE,
    x       DOUBLE[],
    weights DOUBLE,
    options MAP
) OVER (...) → STRUCT
```

---

### anofox_stats_elastic_net_fit_predict

**Elastic Net Fit-Predict Window Function**

**Signature:**
```sql
anofox_stats_elastic_net_fit_predict(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

**Required Options:** `alpha`, `lambda`

---

### anofox_stats_rls_fit_predict

**RLS Fit-Predict Window Function**

**Signature:**
```sql
anofox_stats_rls_fit_predict(
    y       DOUBLE,
    x       DOUBLE[],
    options MAP
) OVER (...) → STRUCT
```

---

## Predict Functions

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

### anofox_stats_predict_ols

**OLS Prediction with Intervals**

**Signature:**
```sql
anofox_stats_predict_ols(
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
SELECT * FROM anofox_stats_predict_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],
    [[6.0], [7.0]]::DOUBLE[][],
    {'interval_type': 'prediction', 'confidence_level': 0.95}
);

-- Output:
-- observation_id | predicted | ci_lower | ci_upper | se
--       1        |    6.0    |   5.1    |   6.9    | 0.45
--       2        |    7.0    |   6.1    |   7.9    | 0.46
```

---

### anofox_stats_predict_ridge

**Ridge Prediction with Intervals**

**Signature:**
```sql
anofox_stats_predict_ridge(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Option:** `lambda`

**Example:**
```sql
SELECT * FROM anofox_stats_predict_ridge(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['lambda'], [0.1])
);
```

---

### anofox_stats_predict_wls

**WLS Prediction with Intervals**

**Signature:**
```sql
anofox_stats_predict_wls(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    weights DOUBLE[],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_stats_predict_wls(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [[6.0], [7.0]],
    MAP(['intercept'], [true])
);
```

---

### anofox_stats_predict_rls

**RLS Prediction with Intervals**

**Signature:**
```sql
anofox_stats_predict_rls(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Example:**
```sql
SELECT * FROM anofox_stats_predict_rls(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['forgetting_factor'], [0.99])
);
```

---

### anofox_stats_predict_elastic_net

**Elastic Net Prediction with Intervals**

**Signature:**
```sql
anofox_stats_predict_elastic_net(
    y_train DOUBLE[],
    x_train DOUBLE[][],
    x_new   DOUBLE[][],
    [options MAP]  -- Optional via varargs
) → TABLE
```

**Required Options:** `alpha`, `lambda`

**Example:**
```sql
SELECT * FROM anofox_stats_predict_elastic_net(
    [2.0, 5.0, 8.0, 11.0, 14.0],
    [[1.0], [2.0], [3.0], [4.0], [5.0]],
    [[6.0], [7.0]],
    MAP(['lambda', 'alpha'], [0.1, 0.5])
);
```

---

## Diagnostic Functions

### Residual Diagnostics

#### anofox_stats_residual_diagnostics

**Table Function - Observation-Level Diagnostics**

**Signature:**
```sql
anofox_stats_residual_diagnostics(
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
SELECT * FROM anofox_stats_residual_diagnostics(
    [1.0, 2.1, 2.9, 4.2, 10.0]::DOUBLE[],
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    2.5  -- outlier_threshold
);
```

---

#### anofox_stats_residual_diagnostics_agg

**Aggregate Function - Group-Level Diagnostics**

**Signature:**
```sql
anofox_stats_residual_diagnostics_agg(
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
    anofox_stats_residual_diagnostics_agg(
        actual,
        predicted,
        {'outlier_threshold': 3.0, 'detailed': false}
    ) as diagnostics
FROM predictions
GROUP BY category;
```

**Note:** Does NOT support OVER (window functions).

---

### Multicollinearity Detection (VIF)

#### anofox_stats_vif

**Table Function - Feature-Level VIF**

**Signature:**
```sql
anofox_stats_vif(
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
SELECT * FROM anofox_stats_vif(
    [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]::DOUBLE[][]
);
```

---

#### anofox_stats_vif_agg

**Aggregate Function - Group-Level VIF**

**Signature:**
```sql
anofox_stats_vif_agg(
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
    anofox_stats_vif_agg([x1, x2, x3]) as vif_results
FROM panel_data
GROUP BY dataset_id;
```

**Note:** Does NOT support OVER (window functions).

---

### Normality Tests

#### anofox_stats_normality_test

**Table Function - Jarque-Bera Test**

**Signature:**
```sql
anofox_stats_normality_test(
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
SELECT * FROM anofox_stats_normality_test(
    [0.1, -0.2, 0.15, -0.1, 0.05]::DOUBLE[],
    0.05  -- significance level
);
```

---

#### anofox_stats_normality_test_agg

**Aggregate Function - Group-Level Normality Test**

**Signature:**
```sql
anofox_stats_normality_test_agg(
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
    anofox_stats_normality_test_agg(
        residual,
        {'alpha': 0.05}
    ) as normality_test
FROM model_residuals
GROUP BY model_id;
```

**Note:** Does NOT support OVER (window functions).

---


## Model-Based Prediction

### anofox_stats_model_predict

**Efficient Prediction from Pre-Fitted Models**

This is the **model-agnostic** prediction function that works with models fitted using `full_output=true`.

**Signature:**
```sql
anofox_stats_model_predict(
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
SELECT * FROM anofox_stats_ols_fit(
    y_array,
    x_matrix,
    {'intercept': true, 'full_output': true}
);

-- Step 2: Predict on new data (no refitting!)
SELECT p.*
FROM model m,
LATERAL anofox_stats_model_predict(
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
