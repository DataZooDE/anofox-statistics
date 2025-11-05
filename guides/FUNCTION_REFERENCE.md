# Function Reference

Comprehensive reference for all functions in the Anofox Statistics extension.

## Table of Contents

- [Aggregate Functions](#aggregate-functions)
- [Table Functions](#table-functions)
- [Scalar Functions](#scalar-functions)
- [Options MAP Reference](#options-map-reference)

---

## Aggregate Functions

Aggregate functions work with `GROUP BY` and window functions (`OVER`) to perform regression analysis per group or within rolling windows.

### anofox_statistics_ols_agg

**Signature:**
```sql
anofox_statistics_ols_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    n_obs BIGINT,
    n_features BIGINT
)
```

**Description:** Ordinary Least Squares regression per group. Fits a linear model minimizing squared residuals.

**Parameters:**
- `y`: Response variable (dependent variable)
- `x`: Array of predictor variables (independent variables)
- `options`: Configuration MAP (see [Options Reference](#options-map-reference))

**Returns:** STRUCT containing:
- `coefficients`: Estimated coefficients for each predictor
- `intercept`: Intercept term (0.0 if `intercept: false`)
- `r2`: R-squared (coefficient of determination)
- `adj_r2`: Adjusted R-squared (penalizes for number of predictors)
- `n_obs`: Number of observations used
- `n_features`: Number of predictors

**Example:**
```sql
SELECT
    product_category,
    result.coefficients[1] as price_effect,
    result.r2
FROM (
    SELECT
        product_category,
        anofox_statistics_ols_agg(
            sales,
            [price, marketing_spend],
            MAP{'intercept': true}
        ) as result
    FROM sales_data
    GROUP BY product_category
) sub;
```

**When to use:** Standard regression analysis, per-group modeling, baseline comparisons.

---

### anofox_statistics_wls_agg

**Signature:**
```sql
anofox_statistics_wls_agg(
    y DOUBLE,
    x DOUBLE[],
    weights DOUBLE,
    options MAP(VARCHAR, ANY)
) → STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    weighted_mse DOUBLE,
    n_obs BIGINT
)
```

**Description:** Weighted Least Squares regression. Handles heteroscedasticity by giving different weights to observations.

**Parameters:**
- `y`: Response variable
- `x`: Array of predictors
- `weights`: Observation weights (precision weights = 1/variance)
- `options`: Configuration MAP

**Returns:** STRUCT containing standard regression outputs plus:
- `weighted_mse`: Mean squared error weighted by observation weights

**Example:**
```sql
SELECT
    customer_segment,
    result.coefficients[1] as income_sensitivity,
    result.weighted_mse
FROM (
    SELECT
        customer_segment,
        anofox_statistics_wls_agg(
            spending,
            [income],
            reliability_weight,
            MAP{'intercept': true}
        ) as result
    FROM customer_data
    GROUP BY customer_segment
) sub;
```

**When to use:**
- Heteroscedastic errors (variance changes with predictors)
- Observations have different reliability
- Combining data from multiple sources with different precision

---

### anofox_statistics_ridge_agg

**Signature:**
```sql
anofox_statistics_ridge_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    lambda DOUBLE,
    n_obs BIGINT
)
```

**Description:** Ridge regression with L2 regularization. Adds penalty term to prevent overfitting and handle multicollinearity.

**Parameters:**
- `y`: Response variable
- `x`: Array of predictors (can be highly correlated)
- `options`: Must include `lambda` parameter

**Returns:** STRUCT containing standard outputs plus:
- `lambda`: Regularization parameter used

**Example:**
```sql
SELECT
    ticker,
    result.coefficients[1] as market_beta,
    result.coefficients[2] as sector_beta,
    result.lambda
FROM (
    SELECT
        ticker,
        anofox_statistics_ridge_agg(
            stock_return,
            [market_return, sector_return, value_factor],
            MAP{'lambda': 1.0, 'intercept': true}
        ) as result
    FROM stock_returns
    GROUP BY ticker
) sub;
```

**When to use:**
- Multicollinearity (highly correlated predictors)
- Many predictors relative to observations
- Preventing overfitting in high-dimensional data

**Choosing lambda:**
- `lambda = 0`: Equivalent to OLS
- `lambda = 0.1 - 1.0`: Light regularization
- `lambda = 1.0 - 10.0`: Moderate regularization
- `lambda > 10.0`: Heavy regularization (high bias, low variance)

---

### anofox_statistics_rls_agg

**Signature:**
```sql
anofox_statistics_rls_agg(
    y DOUBLE,
    x DOUBLE[],
    options MAP(VARCHAR, ANY)
) → STRUCT(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    forgetting_factor DOUBLE,
    n_obs BIGINT
)
```

**Description:** Recursive Least Squares (online learning). Sequentially updates coefficients as new data arrives, with optional exponential weighting of past observations.

**Parameters:**
- `y`: Response variable
- `x`: Array of predictors
- `options`: Must include `forgetting_factor` parameter

**Returns:** STRUCT containing standard outputs plus:
- `forgetting_factor`: Exponential weighting factor used

**Example:**
```sql
SELECT
    sensor_id,
    result.coefficients[1] as calibration_slope,
    result.forgetting_factor
FROM (
    SELECT
        sensor_id,
        anofox_statistics_rls_agg(
            true_value,
            [raw_reading],
            MAP{'forgetting_factor': 0.95, 'intercept': true}
        ) as result
    FROM sensor_calibration
    GROUP BY sensor_id
) sub;
```

**When to use:**
- Streaming/online data processing
- Time-varying relationships
- Adaptive forecasting
- Real-time model updates

**Choosing forgetting_factor:**
- `λ = 1.0`: Equal weighting (equivalent to OLS)
- `λ = 0.98 - 0.99`: Slow adaptation, emphasizes recent data slightly
- `λ = 0.95 - 0.97`: Moderate adaptation
- `λ = 0.90 - 0.94`: Fast adaptation, strongly emphasizes recent data
- `λ < 0.90`: Very fast adaptation, may be volatile

---

## Table Functions

Table functions return table results and work with array inputs (not column references).

### anofox_statistics_ols_fit

**Signature:**
```sql
anofox_statistics_ols_fit(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(
    coefficients DOUBLE[],
    intercept DOUBLE,
    r2 DOUBLE,
    adj_r2 DOUBLE,
    mse DOUBLE,
    n_obs BIGINT,
    n_features BIGINT
)
```

**Description:** Fits OLS regression on array inputs. Returns a single-row table.

**Example:**
```sql
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.1, 2.0], [2.1, 3.0], [2.9, 4.0], [4.2, 5.0], [4.8, 6.0]]::DOUBLE[][],
    MAP{'intercept': true}
);
```

---

### anofox_statistics_ridge_fit

**Signature:**
```sql
anofox_statistics_ridge_fit(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(...)
```

**Description:** Ridge regression on array inputs with L2 regularization.

---

### anofox_statistics_wls_fit

**Signature:**
```sql
anofox_statistics_wls_fit(
    y DOUBLE[],
    x DOUBLE[][],
    weights DOUBLE[],
    options MAP(VARCHAR, ANY)
) → TABLE(...)
```

**Description:** Weighted least squares on array inputs.

---

### anofox_statistics_rls_fit

**Signature:**
```sql
anofox_statistics_rls_fit(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(...)
```

**Description:** Recursive least squares on array inputs.

---

### anofox_statistics_rolling_ols

**Signature:**
```sql
anofox_statistics_rolling_ols(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(
    window_start BIGINT,
    window_end BIGINT,
    coefficients DOUBLE[],
    intercept DOUBLE,
    r_squared DOUBLE,
    mse DOUBLE,
    n_obs BIGINT,
    n_features BIGINT
)
```

**Description:** Rolling window OLS regression. Computes regression for each sliding window.

**Parameters:**
- `y`: Response variable array
- `x`: Predictor matrix (2D array)
- `options`: Must include `window_size` (BIGINT)

**Example:**
```sql
SELECT
    window_start,
    window_end,
    coefficients[1] as slope,
    r_squared
FROM anofox_statistics_rolling_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]::DOUBLE[],
    [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]::DOUBLE[][],
    MAP{'window_size': 5, 'intercept': true}
);
```

**When to use:** Time-series analysis, adaptive trend detection, changing relationships.

---

### anofox_statistics_expanding_ols

**Signature:**
```sql
anofox_statistics_expanding_ols(
    y DOUBLE[],
    x DOUBLE[][],
    options MAP(VARCHAR, ANY)
) → TABLE(
    window_start BIGINT,
    window_end BIGINT,
    coefficients DOUBLE[],
    intercept DOUBLE,
    r_squared DOUBLE,
    mse DOUBLE,
    n_obs BIGINT,
    n_features BIGINT
)
```

**Description:** Expanding window OLS. Each window starts at index 0 and grows to include more observations.

**Parameters:**
- `options`: Must include `min_periods` (BIGINT)

**When to use:** Cumulative analysis, tracking model stability as sample size grows.

---

## Scalar Functions

### ols_r2

**Signature:**
```sql
ols_r2(y DOUBLE[], x DOUBLE[]) → DOUBLE
```

**Description:** Calculates R-squared for simple linear regression (one predictor).

---

### ols_mse

**Signature:**
```sql
ols_mse(y DOUBLE[], x DOUBLE[]) → DOUBLE
```

**Description:** Calculates mean squared error for simple linear regression.

---

### ols_rmse

**Signature:**
```sql
ols_rmse(y DOUBLE[], x DOUBLE[]) → DOUBLE
```

**Description:** Calculates root mean squared error for simple linear regression.

---

## Options MAP Reference

All regression functions accept an `options` MAP parameter for configuration.

### Common Options

**intercept** (BOOLEAN, default: `true`)
- `true`: Include intercept term (recommended for most cases)
- `false`: Force regression through origin (no intercept)

**Example:**
```sql
MAP{'intercept': true}
MAP{'intercept': false}
```

**When to use `intercept: false`:**
- Physical laws or relationships that must pass through the origin
- Theory requires zero intercept
- You've already centered your data

---

### Ridge-Specific Options

**lambda** (DOUBLE, required for Ridge)
- Regularization penalty parameter
- Higher values = more regularization = more coefficient shrinkage
- Range: 0.0 (no regularization) to 100+ (heavy regularization)

**Example:**
```sql
MAP{'lambda': 1.0, 'intercept': true}
MAP{'lambda': 10.0, 'intercept': false}
```

---

### RLS-Specific Options

**forgetting_factor** (DOUBLE, default: `1.0`)
- Exponential weighting of past observations
- Range: 0.0 to 1.0
- 1.0 = no forgetting (equivalent to OLS)
- < 1.0 = emphasize recent data

**Example:**
```sql
MAP{'forgetting_factor': 0.95, 'intercept': true}
MAP{'forgetting_factor': 1.0, 'intercept': true}  -- Same as OLS
```

---

### Window Function Options

**window_size** (BIGINT, required for rolling_ols)
- Number of observations per window
- Must be ≥ (n_features + 1)

**min_periods** (BIGINT, required for expanding_ols)
- Minimum observations before starting
- Must be ≥ (n_features + 1)

**Example:**
```sql
MAP{'window_size': 30, 'intercept': true}
MAP{'min_periods': 10, 'intercept': true}
```

---

## Complete Example: All Methods

```sql
WITH test_data AS (
    SELECT
        'product_a' as product,
        price,
        marketing,
        weight,
        sales
    FROM product_sales
)
SELECT
    product,
    'OLS' as method,
    ols.coefficients,
    ols.r2
FROM (
    SELECT product, anofox_statistics_ols_agg(sales, [price, marketing], MAP{'intercept': true}) as ols
    FROM test_data GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'WLS' as method,
    wls.coefficients,
    wls.r2
FROM (
    SELECT product, anofox_statistics_wls_agg(sales, [price, marketing], weight, MAP{'intercept': true}) as wls
    FROM test_data GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'Ridge' as method,
    ridge.coefficients,
    ridge.r2
FROM (
    SELECT product, anofox_statistics_ridge_agg(sales, [price, marketing], MAP{'lambda': 1.0, 'intercept': true}) as ridge
    FROM test_data GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'RLS' as method,
    rls.coefficients,
    rls.r2
FROM (
    SELECT product, anofox_statistics_rls_agg(sales, [price, marketing], MAP{'forgetting_factor': 0.95, 'intercept': true}) as rls
    FROM test_data GROUP BY product
) sub;
```

---

## See Also

- [Quick Start Guide](01_quick_start.md)
- [Statistics Guide](03_statistics_guide.md) - Statistical methodology and interpretation
- [Business Guide](04_business_guide.md) - Real-world use cases
- [Advanced Use Cases](05_advanced_use_cases.md) - Complex workflows
