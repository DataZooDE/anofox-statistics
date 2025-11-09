# Statistics Guide

A comprehensive guide to the statistical methodology and interpretation for statisticians, data scientists, and researchers.

## Table of Contents

1. [Regression Theory](#regression-theory)
2. [Statistical Inference](#statistical-inference)
3. [Model Diagnostics](#model-diagnostics)
4. [Model Selection](#model-selection)
5. [Assumptions and Violations](#assumptions-and-violations)
6. [Advanced Topics](#advanced-topics)

## Regression Theory

### Ordinary Least Squares (OLS)

**Model**: y = Xβ + ε

**Assumptions**:

1. **Linearity**: E[ε|X] = 0
2. **Homoscedasticity**: Var(ε|X) = σ²
3. **Independence**: Cov(εᵢ, εⱼ) = 0 for i ≠ j
4. **Normality**: ε ~ N(0, σ²) (for inference)
5. **No perfect multicollinearity**: rank(X) = p

**Estimation**:

```
β̂ = (X'X)⁻¹X'y
```

**Properties** (under assumptions):

- **BLUE**: Best Linear Unbiased Estimator (Gauss-Markov)
- **Consistency**: β̂ →ᵖ β as n → ∞
- **Asymptotic Normality**: √n(β̂ - β) →ᵈ N(0, σ²(X'X)⁻¹)

**Example**:

This example demonstrates a basic OLS fit using the extension. The function computes all standard OLS statistics including coefficients, R², and residuals.


```sql

-- Create sample products data
CREATE TEMP TABLE products AS
SELECT
    CASE
        WHEN i <= 15 THEN 'electronics'
        WHEN i <= 30 THEN 'clothing'
        ELSE 'furniture'
    END as category,
    (100 + i * 5 + random() * 20)::DOUBLE as sales,
    (50 + i * 2 + random() * 10)::DOUBLE as price
FROM generate_series(1, 45) t(i);

-- Simple OLS with aggregate function (works directly with table data)
SELECT
    category,
    (ols_fit_agg(sales, price)).coefficient as price_effect,
    (ols_fit_agg(sales, price)).r2 as r_squared
FROM products
GROUP BY category;

-- Note: ols_fit_agg works directly with table columns.
-- For table functions with multiple predictors, use literal arrays (see Quick Start Guide).
```

**Interpretation**:

- **Coefficients (β̂ⱼ)**: Marginal effect of each predictor - the expected change in y for a one-unit increase in xⱼ, holding other predictors constant
- **R² (Coefficient of Determination)**: Proportion of variance in y explained by the model. Range [0,1], where 1 = perfect fit
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals - typical prediction error in y units. Lower is better
- **Adjusted R²**: R² penalized for number of predictors - use when comparing models with different numbers of variables

#### OLS Aggregate for GROUP BY Analysis

For per-group regression analysis, use the `anofox_statistics_ols_agg` aggregate function. This computes separate OLS regressions for each group efficiently in a single query.


```sql

-- Statistics Guide: Comprehensive OLS Aggregate Example
-- Demonstrates full statistical output and interpretation

-- Create sample data: advertising effectiveness study
CREATE TEMP TABLE advertising_data AS
SELECT
    CASE WHEN i <= 20 THEN 'campaign_a' ELSE 'campaign_b' END as campaign,
    i as week,
    (1000 + i * 50 + random() * 100)::DOUBLE as tv_spend,
    (500 + i * 25 + random() * 50)::DOUBLE as digital_spend,
    (10000 + i * 200 + 0.8 * (1000 + i * 50) + 1.2 * (500 + i * 25) + random() * 500)::DOUBLE as sales
FROM generate_series(1, 40) as t(i);

-- Run comprehensive OLS analysis per campaign
SELECT
    campaign,
    -- Coefficients (interpretation: change in sales per dollar spent)
    result.coefficients[1] as tv_roi,
    result.coefficients[2] as digital_roi,
    result.intercept as baseline_sales,
    -- Model fit
    result.r2 as r_squared,
    result.adj_r2 as adjusted_r_squared,
    result.n_obs as sample_size,
    array_length(result.coefficients) as num_predictors,
    -- Interpretation flags
    CASE
        WHEN result.r2 > 0.8 THEN 'Excellent fit'
        WHEN result.r2 > 0.6 THEN 'Good fit'
        WHEN result.r2 > 0.4 THEN 'Moderate fit'
        ELSE 'Poor fit'
    END as model_quality,
    CASE
        WHEN result.coefficients[1] > result.coefficients[2] THEN 'TV more effective'
        ELSE 'Digital more effective'
    END as channel_comparison
FROM (
    SELECT
        campaign,
        anofox_statistics_ols_agg(
            sales,
            [tv_spend, digital_spend],
            {'intercept': true}
        ) as result
    FROM advertising_data
    GROUP BY campaign
) sub;
```

**Aggregate-Specific Notes**:

- Works with `GROUP BY` for per-group models
- Can be used with window functions (`OVER`) for rolling analysis
- Automatically parallelizes across groups
- Returns STRUCT with all regression statistics
- Access fields with `.` notation: `result.coefficients[1]`, `result.r2`

#### Understanding the Intercept Parameter

The intercept parameter controls whether the regression line must pass through the origin (intercept=false) or can have any y-intercept (intercept=true).


```sql

-- Statistics Guide: Understanding the Intercept Parameter
-- Demonstrates when to use intercept=true vs intercept=false

-- Example 1: Physical law (proportional relationship, no intercept)
CREATE TEMP TABLE physics_data AS
SELECT
    'force_mass_relationship' as experiment,
    i::DOUBLE as mass_kg,
    (i * 9.81)::DOUBLE as force_newtons  -- F = m * g (passes through origin)
FROM generate_series(1, 20) as t(i);

SELECT
    'Physics: With intercept' as model_type,
    result.intercept,
    result.coefficients[1] as acceleration_estimate,
    result.r2
FROM (
    SELECT anofox_statistics_ols_agg(force_newtons, [mass_kg], {'intercept': true}) as result
    FROM physics_data
) sub
UNION ALL
SELECT
    'Physics: Without intercept (correct)' as model_type,
    result.intercept,
    result.coefficients[1] as acceleration_estimate,
    result.r2
FROM (
    SELECT anofox_statistics_ols_agg(force_newtons, [mass_kg], {'intercept': false}) as result
    FROM physics_data
) sub;

-- Example 2: Business scenario (with natural intercept)
CREATE TEMP TABLE business_data AS
SELECT
    'sales_model' as model,
    i::DOUBLE as employees,
    (50000 + i * 75000)::DOUBLE as revenue  -- Base revenue + per-employee contribution
FROM generate_series(1, 15) as t(i);

SELECT
    'Business: With intercept (correct)' as model_type,
    result.intercept as fixed_costs,
    result.coefficients[1] as revenue_per_employee,
    result.r2
FROM (
    SELECT anofox_statistics_ols_agg(revenue, [employees], {'intercept': true}) as result
    FROM business_data
) sub
UNION ALL
SELECT
    'Business: Without intercept (wrong)' as model_type,
    result.intercept,
    result.coefficients[1] as biased_estimate,
    result.r2
FROM (
    SELECT anofox_statistics_ols_agg(revenue, [employees], {'intercept': false}) as result
    FROM business_data
) sub;

-- Key insight: R² comparison
SELECT
    'R² comparison' as note,
    'intercept=true uses SS from mean, intercept=false uses SS from zero' as explanation;
```

**Choosing Intercept Setting**:

- **intercept=true (default)**: Use for most business/social science applications where a natural baseline exists
- **intercept=false**: Use when theory requires zero intercept (physical laws, rates, or when data is already centered)
- **R² difference**: With intercept uses SS from mean; without intercept uses SS from zero (not directly comparable)
- **Degrees of freedom**: With intercept adds 1 to model parameters for adjusted R² calculation

### Ridge Regression

**Model**: Minimize ||y - Xβ||² + λ||β||²

**Purpose**: Handle multicollinearity by shrinking coefficients

**Solution**:

```
β̂ᵣᵢᵈᵍₑ = (X'X + λI)⁻¹X'y
```

**Properties**:

- **Biased** but lower variance than OLS
- **Shrinks** coefficients toward zero
- **Stabilizes** estimation when X'X is near-singular

**Choosing λ**:

- Cross-validation
- Generalized Cross-Validation (GCV)
- L-curve method

**Example**:

This example shows ridge regression with a regularization parameter λ=0.1. The regularization shrinks coefficients, trading some bias for reduced variance and improved prediction stability.


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
LATERAL anofox_statistics_ridge(
    data.y,
    data.X,
    MAP(['lambda', 'intercept'], [0.1::DOUBLE, 1.0::DOUBLE])
) as result;
```

**When to Use**:

- **VIF > 10**: High multicollinearity between predictors causes unstable OLS estimates
- **n < p**: More predictors than observations (OLS is undefined, ridge still works)
- **Prediction focus**: When you care more about accurate predictions than interpreting individual coefficients
- **Overfitting**: When OLS coefficients are implausibly large due to overfitting

**Practical tip**: Try λ values on a log scale (0.001, 0.01, 0.1, 1, 10) and use cross-validation to choose the best one.

#### Ridge Aggregate for GROUP BY Analysis

For per-group ridge regression, use `anofox_statistics_ridge_agg` with lambda parameter in the options MAP.


```sql

-- Statistics Guide: Ridge Regression - Handling Multicollinearity
-- Demonstrates lambda tuning and coefficient shrinkage

-- Create data with severe multicollinearity
CREATE TEMP TABLE collinear_data AS
SELECT
    'product_line_a' as product,
    i::DOUBLE as advertising,
    (i + random() * 0.5)::DOUBLE as social_media,  -- Nearly identical to advertising
    (i + random() * 0.5)::DOUBLE as influencer,     -- Nearly identical to advertising
    (1000 + 50 * i + random() * 100)::DOUBLE as sales
FROM generate_series(1, 25) as t(i);

-- Compare different lambda values
SELECT
    product,
    'OLS (lambda=0)' as method,
    result.lambda,
    result.coefficients[1] as adv_coef,
    result.coefficients[2] as social_coef,
    result.coefficients[3] as influencer_coef,
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_statistics_ridge_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 0.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'Ridge (lambda=1)' as method,
    result.lambda,
    result.coefficients[1],
    result.coefficients[2],
    result.coefficients[3],
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_statistics_ridge_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 1.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'Ridge (lambda=10)' as method,
    result.lambda,
    result.coefficients[1],
    result.coefficients[2],
    result.coefficients[3],
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_statistics_ridge_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 10.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub;

-- Interpretation note
SELECT
    'Key Insight' as note,
    'Ridge shrinks coefficients towards zero, reducing variance at the cost of small bias. Higher lambda = more shrinkage.' as interpretation;
```

**Lambda Selection Guide**:

- **λ = 0**: Equivalent to OLS (no regularization)
- **λ = 0.01-0.1**: Light regularization (slight coefficient shrinkage)
- **λ = 1-10**: Moderate regularization (recommended starting point)
- **λ > 10**: Heavy regularization (strong shrinkage, high bias)
- **Cross-validation**: Optimal λ minimizes prediction error on held-out data
- **Compare coefficients**: Ridge coefficients should be smaller but more stable than OLS

### Weighted Least Squares (WLS)

**Model**: y = Xβ + ε, where Var(εᵢ) = σ²/wᵢ

**Estimation**:

```
β̂ᵂᴸˢ = (X'WX)⁻¹X'Wy
where W = diag(w₁, ..., wₙ)
```

**Use Cases**:

1. **Heteroscedasticity**: Variance increases with x
2. **Weighted observations**: Different precision/reliability
3. **Grouped data**: Group sizes vary

**Example**:

This example demonstrates WLS when observations have different levels of precision. Observations with higher weights (more reliable) have greater influence on the fitted model.


```sql

-- Variance proportional to x (new API with 2D array + MAP)
SELECT * FROM anofox_statistics_wls(
    [50.0, 100.0, 150.0, 200.0, 250.0]::DOUBLE[],  -- y: sales
    [[10.0, 20.0, 30.0, 40.0, 50.0]]::DOUBLE[][],  -- X: 2D array (one feature)
    [10.0, 20.0, 30.0, 40.0, 50.0]::DOUBLE[],      -- weights: proportional to size
    MAP{'intercept': true}                          -- options in MAP
);
```

**Weight Selection Guidelines**:

- **Heteroscedasticity**: If Var(εᵢ) = σ²xᵢ, use weights wᵢ = 1/xᵢ
- **Measurement error**: If observations have known standard errors sᵢ, use wᵢ = 1/sᵢ²
- **Grouped data**: If observation i represents nᵢ replicates, use wᵢ = nᵢ

**Diagnostic**: Use Breusch-Pagan test or plot residuals vs. fitted values to detect heteroscedasticity and validate weight specification.

#### WLS Aggregate for GROUP BY Analysis

**What it does**: Computes Weighted Least Squares regression per group using SQL's GROUP BY clause. Each group gets its own model with observation-specific weights.

**When to use**: When analyzing multiple segments with heteroscedastic errors, combining data sources with different reliability, or when observations within groups have different precision levels.

**How it works**: The `anofox_statistics_wls_agg` function accumulates weighted data per group, applying observation weights to account for varying variance or reliability.


```sql

-- Statistics Guide: Weighted Least Squares - Handling Heteroscedasticity
-- Demonstrates when and how to use weights for non-constant variance

-- Scenario: Customer spending analysis where variance increases with income
-- High-income customers have more variable spending patterns
CREATE TEMP TABLE customer_spending AS
SELECT
    CASE
        WHEN i <= 15 THEN 'low_income'
        WHEN i <= 30 THEN 'medium_income'
        ELSE 'high_income'
    END as segment,
    i as customer_id,
    (20000 + i * 1000)::DOUBLE as annual_income,
    -- Spending with heteroscedastic errors (variance increases with income)
    (5000 + 0.3 * (20000 + i * 1000) + random() * (100 + i * 20))::DOUBLE as annual_spending,
    -- Weight by inverse variance (precision weighting)
    (1.0 / (1.0 + i * 0.1))::DOUBLE as precision_weight
FROM generate_series(1, 45) as t(i);

-- Compare OLS (ignores heteroscedasticity) vs WLS (accounts for it)
SELECT
    segment,
    'OLS (unweighted)' as method,
    result.coefficients[1] as income_propensity,
    result.intercept as base_spending,
    result.r2,
    NULL as weighted_mse
FROM (
    SELECT
        segment,
        anofox_statistics_ols_agg(
            annual_spending,
            [annual_income],
            {'intercept': true}
        ) as result
    FROM customer_spending
    GROUP BY segment
) sub
UNION ALL
SELECT
    segment,
    'WLS (precision weighted)' as method,
    result.coefficients[1] as income_propensity,
    result.intercept as base_spending,
    result.r2,
    result.weighted_mse
FROM (
    SELECT
        segment,
        anofox_statistics_wls_agg(
            annual_spending,
            [annual_income],
            precision_weight,
            {'intercept': true}
        ) as result
    FROM customer_spending
    GROUP BY segment
) sub
ORDER BY segment, method;

-- Interpretation note
SELECT
    'Statistical Note' as category,
    'WLS gives more weight to observations with lower variance (more reliable data points). This produces more efficient estimates when heteroscedasticity is present.' as explanation
UNION ALL
SELECT
    'When to use WLS',
    'Use WLS when: (1) variance changes systematically with predictors, (2) you have reliability measures for observations, or (3) you are combining data from sources with different precision.' as guidance;
```

**What the results mean**:

- **Coefficients**: Estimated while giving more influence to high-weight (reliable) observations
- **weighted_mse**: Error metric that accounts for observation weights
- **Comparison**: WLS typically produces more efficient estimates than OLS when heteroscedasticity is present

Use WLS aggregates when observations within each group have known reliability differences or non-constant variance.

## Statistical Inference

### Coefficient Tests

**Hypothesis**: H₀: βⱼ = 0 vs H₁: βⱼ ≠ 0

**Test Statistic**:

```
t = β̂ⱼ / SE(β̂ⱼ)
where SE(β̂ⱼ) = √(MSE · (X'X)⁻¹ⱼⱼ)
```

**Distribution**: t ~ t(n-p) under H₀

**Decision Rule**:

- Reject H₀ if |t| > t_{α/2,n-p}
- Or equivalently, if p-value < α

**Example**:

This query performs hypothesis tests for each coefficient, testing whether each predictor has a statistically significant relationship with the response variable.


```sql

SELECT
    variable,
    estimate,
    std_error,
    t_statistic,
    p_value,
    significant  -- TRUE if p < 0.05
FROM ols_inference(
    [65.0, 72.0, 78.0, 85.0, 92.0, 88.0]::DOUBLE[],                          -- y: exam_score
    [[3.0, 7.0], [4.0, 8.0], [5.0, 7.0], [6.0, 8.0], [7.0, 9.0], [6.5, 7.5]]::DOUBLE[][], -- x: study_hours, sleep_hours
    0.95,                                                                      -- confidence_level
    true                                                                       -- add_intercept
);
```

**Interpretation**:

- **t-statistic**: Measures how many standard errors the coefficient is from zero. |t| > 2 typically indicates significance
- **p-value**: Probability of observing this effect (or stronger) if the true coefficient is zero. p < 0.05 is commonly used as the significance threshold
- **Significant flag**: Automatically identifies predictors with p < 0.05
- **Confidence intervals**: Provide a range of plausible values for the true coefficient

**Common interpretations**:

- **p < 0.001**: Highly significant - very strong evidence of an effect
- **p < 0.05**: Significant - conventional threshold for statistical significance
- **p > 0.10**: Not significant - insufficient evidence of an effect

### Confidence Intervals

**Coefficient CI**:

```
CI₁₋α(βⱼ) = β̂ⱼ ± t_{α/2,n-p} · SE(β̂ⱼ)
```

**Interpretation**: With 95% confidence, true βⱼ lies in interval

**Example Result**:

```
variable: study_hours
estimate: 5.2
ci_lower: 4.1
ci_upper: 6.3
```

**Interpretation**: Each additional study hour increases expected score by 5.2 points (95% CI: 4.1 to 6.3).

### Prediction Intervals

Two types of intervals:

**1. Confidence Interval** (for mean prediction):

```
CI(E[y|x₀]) = x₀'β̂ ± t_{α/2,n-p} · √(MSE · x₀'(X'X)⁻¹x₀)
```

Interpretation: Uncertainty in **average** y for given x₀

**2. Prediction Interval** (for single prediction):

```
PI(y|x₀) = x₀'β̂ ± t_{α/2,n-p} · √(MSE · (1 + x₀'(X'X)⁻¹x₀))
```

Interpretation: Uncertainty in **individual** y for given x₀

**Key Difference**:

- PI is wider than CI (includes σ² term)
- CI → 0 as n → ∞, but PI → σ

**Example**:

This example demonstrates both confidence intervals (for the mean) and prediction intervals (for individual observations). Notice how prediction intervals are wider because they account for both model uncertainty and random error.


```sql

SELECT
    predicted,
    ci_lower,
    ci_upper,
    ci_upper - ci_lower as interval_width
FROM ols_predict_interval(
    [50.0, 55.0, 60.0, 65.0, 70.0]::DOUBLE[],           -- y_train: historical_sales
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][],    -- x_train: historical_features
    [[6.0], [7.0], [8.0]]::DOUBLE[][],                  -- x_new: future_features
    0.95,                                                 -- confidence_level
    'prediction',                                         -- interval_type (or 'confidence')
    true                                                  -- add_intercept
);
```

**When to use which interval**:

- **Confidence Interval**: "What's the average outcome for this input?" - Use for understanding population means
- **Prediction Interval**: "What outcome should I expect for this specific case?" - Use for forecasting individual values

**Example scenario**: Predicting house prices

- CI: "The average price for 3-bedroom houses in this neighborhood is $400K ± $20K"
- PI: "This specific 3-bedroom house will sell for $400K ± $80K"

The prediction interval is wider because individual houses vary around the average.

## Model Diagnostics

### Residual Analysis

**Residual**: eᵢ = yᵢ - ŷᵢ

**Standardized Residual**:

```
rᵢ = eᵢ / √MSE
```

**Studentized Residual**:

```
tᵢ = eᵢ / √(MSE · (1 - hᵢᵢ))
where hᵢᵢ = leverage of observation i
```

**Properties**:

- Mean ≈ 0
- Variance ≈ 1 (if homoscedastic)
- ~95% should be in [-2, 2]

**Example**:

Residual diagnostics help you assess whether the regression assumptions hold and identify problematic observations.


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

**What to look for**:

- **Pattern in residuals**: Should be randomly scattered around zero. Patterns indicate model misspecification
- **Outliers**: Standardized residuals > |2.5| are unusual and merit investigation
- **Heteroscedasticity**: If residual variance increases with fitted values, consider WLS or transformations
- **Normality**: For valid inference, residuals should be approximately normal (check with histogram or Q-Q plot)

### Leverage and Influence

**Leverage** (Hat Values):

```
hᵢᵢ = xᵢ'(X'X)⁻¹xᵢ
```

**Properties**:

- Range: 0 to 1
- Average: p/n
- Threshold: 2p/n or 3p/n

**Interpretation**: Potential to influence fitted values

**Cook's Distance**:

```
Dᵢ = (rᵢ²/p) · (hᵢᵢ/(1-hᵢᵢ))
```

**Interpretation**: Overall influence on regression coefficients

- Dᵢ > 1: Highly influential
- Dᵢ > 4/n: Potentially influential
- Dᵢ > 0.5: Use caution

**DFFITS**:

```
DFFITSᵢ = tᵢ · √(hᵢᵢ/(1-hᵢᵢ))
```

**Interpretation**: Change in fitted value when removing observation i

- |DFFITS| > 2√(p/n): Potentially influential

**Example**:

```sql

-- Find most influential observations (using literal arrays)
-- Note: residual_diagnostics expects y_actual and y_predicted, not y and X
-- Generate simple predicted values for demonstration
WITH predictions AS (
    SELECT
        [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]::DOUBLE[] as y_actual,
        [50.5, 54.8, 60.2, 64.9, 70.1, 74.7]::DOUBLE[] as y_predicted  -- Simulated predictions
)
SELECT
    obs_id,
    residual,
    std_residual,
    is_outlier
FROM predictions, anofox_statistics_residual_diagnostics(
    y_actual,
    y_predicted,
    2.5  -- outlier_threshold
)
ORDER BY ABS(std_residual) DESC;
```

### Multicollinearity

**Variance Inflation Factor (VIF)**:

```
VIFⱼ = 1 / (1 - Rⱼ²)
where Rⱼ² = R² from regressing xⱼ on other x's
```

**Interpretation**:

- VIF = 1: No correlation
- VIF < 5: Low multicollinearity ✓
- VIF = 5-10: Moderate multicollinearity ⚠️
- VIF > 10: High multicollinearity ✗

**Effects of Multicollinearity**:

- Large standard errors
- Unstable coefficients
- Insignificant t-tests despite high R²

**Example**:

```sql

SELECT
    variable_name,
    vif,
    severity
FROM anofox_statistics_vif(
    [[10.0, 9.9, 10.1], [20.0, 19.8, 20.2], [30.0, 29.9, 30.1], [40.0, 39.7, 40.3]]::DOUBLE[][]
    -- x matrix: price, competitors_price, industry_avg_price (highly correlated columns)
);
```

**Solutions**:

1. Remove correlated variables
2. Combine into single variable (PCA)
3. Use ridge regression
4. Collect more data

### Normality Tests

**Jarque-Bera Test**:

```
JB = (n/6) · (S² + (K-3)²/4)
where S = skewness, K = kurtosis
```

**Distribution**: JB ~ χ²(2) under H₀: normality

**Interpretation**:

- Skewness = 0: Symmetric
- Kurtosis = 3: Normal tails
- p > 0.05: Cannot reject normality

**Example**:

```sql

-- Test normality of residuals (use literal array)
SELECT * FROM anofox_statistics_normality_test(
    [0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.0, 0.1, -0.1, 0.2]::DOUBLE[],  -- residuals
    0.05                                                                 -- alpha
);

-- Note: To test residuals from a table, first extract to array using LIST()
-- Example: SELECT LIST(residual) FROM my_diagnostics_table
```

**Result Interpretation**:

```
skewness: 0.12 (slightly right-skewed)
kurtosis: 3.2 (slightly heavy-tailed)
jb_statistic: 0.89
p_value: 0.64
conclusion: normal ✓
```

## Model Selection

### Information Criteria

**Akaike Information Criterion (AIC)**:

```
AIC = n·ln(RSS/n) + 2k
where k = number of parameters
```

**Bayesian Information Criterion (BIC)**:

```
BIC = n·ln(RSS/n) + k·ln(n)
```

**Corrected AIC (AICc)** - for small samples:

```
AICc = AIC + 2k(k+1)/(n-k-1)
```

**Properties**:

- Lower is better
- BIC penalizes complexity more than AIC
- Use AICc when n/k < 40

**Example**:

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

**Decision**:

- ΔAIC > 10: Strong evidence for better model
- ΔAIC = 4-7: Considerable evidence
- ΔAIC < 2: Weak evidence

### Adjusted R²

**Formula**:

```
R̄² = 1 - (1-R²)·(n-1)/(n-p-1)
```

**Properties**:

- Penalizes additional predictors
- Can decrease when adding variables
- Use for model comparison

**Interpretation**:

```
R² = 0.85: Explains 85% of variance
R̄² = 0.82: Explains 82% adjusting for complexity
```

## Assumptions and Violations

### Checking Assumptions

| Assumption | Test | Remedy |
|------------|------|--------|
| Linearity | Residual plots | Add polynomials/interactions |
| Homoscedasticity | Breusch-Pagan | WLS, robust SE |
| Independence | Durbin-Watson | Time series methods |
| Normality | Jarque-Bera | Bootstrap, robust methods |
| No multicollinearity | VIF | Remove variables, ridge |

### Diagnostic Plots (Manual)


```sql

-- Create sample diagnostics data
CREATE TEMP TABLE diagnostics AS
SELECT
    i as obs_id,
    (10 + i * 0.5)::DOUBLE as predicted,
    (random() * 2 - 1)::DOUBLE as residual,
    (random() * 2 - 1)::DOUBLE as studentized_residual,
    ABS(random() * 2 - 1)::DOUBLE as sqrt_abs_residual,
    (random() * 0.5)::DOUBLE as leverage,
    (random() * 0.3)::DOUBLE as cooks_distance
FROM generate_series(1, 50) t(i);

-- 1. Residuals vs Fitted
SELECT
    predicted,
    residual
FROM diagnostics;
-- Look for: Random scatter (good), patterns (bad)

-- 2. Q-Q Plot (normal quantiles)
SELECT
    obs_id,
    studentized_residual,
    PERCENT_RANK() OVER (ORDER BY studentized_residual) as percentile
FROM diagnostics;
-- Look for: Points on diagonal line

-- 3. Scale-Location (homoscedasticity)
SELECT
    predicted,
    SQRT(ABS(studentized_residual)) as sqrt_abs_std_resid
FROM diagnostics;
-- Look for: Horizontal band (good), funnel (heteroscedastic)

-- 4. Leverage vs Residuals
SELECT
    leverage,
    studentized_residual,
    cooks_distance
FROM diagnostics;
-- Look for: High leverage + high residual = influential
```

## Advanced Topics

### Sequential/Online Estimation

**Recursive Least Squares (RLS)**:

Updates coefficients as new data arrives:

```
β̂ₜ = β̂ₜ₋₁ + Kₜ(yₜ - xₜ'β̂ₜ₋₁)
where Kₜ = Pₜ₋₁xₜ / (1 + xₜ'Pₜ₋₁xₜ)
```

**Use Cases**:

- Streaming data
- Adaptive estimation
- Real-time predictions

**Example**:

```sql

-- Recursive Least Squares (new API with 2D array + MAP)
WITH data AS (
    SELECT
        [10.0::DOUBLE, 11.0, 12.0, 13.0, 14.0, 15.0] as y,
        [[1.0::DOUBLE, 2.0, 3.0, 4.0, 5.0, 6.0]] as X
)
SELECT result.* FROM data,
LATERAL anofox_statistics_rls(
    data.y,
    data.X,
    MAP(['lambda', 'intercept'], [0.99::DOUBLE, 1.0::DOUBLE])
) as result;
```

#### RLS Aggregate for GROUP BY Analysis

**What it does**: Computes Recursive Least Squares regression per group with exponential weighting of past observations. Adapts to changing relationships over time within each group.

**When to use**: For streaming data analysis per group, when relationships evolve differently across segments, or when recent patterns are more relevant than historical data for each category.

**How it works**: The `anofox_statistics_rls_agg` function sequentially updates coefficients as it processes rows within each group. The `forgetting_factor` parameter controls adaptation speed - values below 1.0 emphasize recent observations.


```sql

-- Statistics Guide: Recursive Least Squares - Adaptive Online Learning
-- Demonstrates forgetting factor tuning for time-varying relationships

-- Scenario: Market beta estimation with regime changes
-- Stock's relationship to market changes over time
CREATE TEMP TABLE stock_market_data AS
SELECT
    CASE
        WHEN i <= 20 THEN 'regime_bull'
        WHEN i <= 40 THEN 'regime_bear'
        ELSE 'regime_recovery'
    END as market_regime,
    i as time_period,
    -- Market return (independent variable)
    (0.01 + random() * 0.02 - 0.01)::DOUBLE as market_return,
    -- Stock return (dependent variable) with changing beta
    CASE
        WHEN i <= 20 THEN (0.005 + 1.2 * (0.01 + random() * 0.02 - 0.01) + random() * 0.01)  -- Bull: high beta (1.2)
        WHEN i <= 40 THEN (0.001 + 0.8 * (0.01 + random() * 0.02 - 0.01) + random() * 0.015) -- Bear: low beta (0.8)
        ELSE (0.003 + 1.5 * (0.01 + random() * 0.02 - 0.01) + random() * 0.012)              -- Recovery: very high beta (1.5)
    END::DOUBLE as stock_return
FROM generate_series(1, 60) as t(i);

-- Compare different forgetting factors
SELECT
    'Forgetting Factor: 1.0 (OLS equivalent)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Averages all data equally - slow to adapt' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 1.0, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.98 (slow adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Gradual weight decay - moderate adaptation' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.98, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.95 (moderate adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Balanced - good for detecting regime changes' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.95, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub
UNION ALL
SELECT
    'Forgetting Factor: 0.90 (fast adaptation)' as model,
    result.forgetting_factor,
    result.coefficients[1] as estimated_beta,
    result.r2,
    'Heavy decay - very responsive to recent changes' as interpretation
FROM (
    SELECT anofox_statistics_rls_agg(
        stock_return,
        [market_return],
        {'forgetting_factor': 0.90, 'intercept': true}
    ) as result
    FROM stock_market_data
) sub;

-- Per-regime analysis
SELECT
    market_regime,
    result.coefficients[1] as regime_beta,
    result.forgetting_factor,
    result.r2,
    result.n_obs
FROM (
    SELECT
        market_regime,
        anofox_statistics_rls_agg(
            stock_return,
            [market_return],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) as result
    FROM stock_market_data
    GROUP BY market_regime
) sub
ORDER BY market_regime;

-- Guidance
SELECT
    'Choosing Forgetting Factor' as topic,
    'λ close to 1.0: More stable, slower adaptation. λ < 0.95: More responsive, tracks changes quickly. Choose based on how fast you expect relationships to change.' as guidance;
```

**What the results mean**:

- **Coefficients**: Final adaptive estimates emphasizing recent patterns per group
- **forgetting_factor**: Controls memory - 1.0 = no forgetting (OLS), 0.90-0.95 = moderate adaptation
- **Applications**: Sensor calibration drift, adaptive forecasting, regime change detection

**Choosing forgetting_factor**:

- **λ = 1.0**: Equal weighting (equivalent to OLS) - use for stable relationships
- **λ = 0.95-0.97**: Slow adaptation - relationships change gradually
- **λ = 0.90-0.94**: Fast adaptation - relationships change frequently
- **λ < 0.90**: Very fast adaptation - may be volatile, use for rapidly changing patterns

Use RLS aggregates when per-group relationships are non-stationary and recent data is more predictive.

### Rolling/Expanding Windows

**Rolling Window**: Fixed-size window moves through time

```sql

-- Generate time-series data
CREATE TEMP TABLE time_series AS
SELECT
    DATE '2023-01-01' + INTERVAL (i) DAY as date,
    (100 + i * 2 + random() * 10)::DOUBLE as sales,
    (50 + i * 0.5 + random() * 5)::DOUBLE as price
FROM generate_series(1, 100) t(i);

SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as rolling_elasticity
FROM time_series;
```

**Expanding Window**: Window starts small, grows over time

```sql

-- Generate time-series data
CREATE TEMP TABLE time_series AS
SELECT
    DATE '2023-01-01' + INTERVAL (i) DAY as date,
    (100 + i * 2 + random() * 10)::DOUBLE as sales,
    (50 + i * 0.5 + random() * 5)::DOUBLE as price
FROM generate_series(1, 100) t(i);

SELECT
    date,
    ols_coeff_agg(sales, price) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_elasticity
FROM time_series;
```

**Applications**:

- Time-varying relationships
- Structural breaks
- Forecasting

### Hypothesis Testing Framework

**General Framework**:

1. State hypotheses (H₀, H₁)
2. Choose test statistic
3. Determine distribution under H₀
4. Compute p-value
5. Make decision (reject/fail to reject)

**Type I Error**: Reject H₀ when true (α = significance level)
**Type II Error**: Fail to reject H₀ when false (β)
**Power**: 1 - β (probability of detecting true effect)

**Example Workflow**:

```sql

-- Test: Does advertising affect sales?
-- H₀: β_advertising = 0
-- H₁: β_advertising ≠ 0

SELECT
    variable,
    estimate as effect,
    p_value,
    CASE
        WHEN p_value < 0.001 THEN 'Highly significant ***'
        WHEN p_value < 0.01 THEN 'Very significant **'
        WHEN p_value < 0.05 THEN 'Significant *'
        ELSE 'Not significant'
    END as significance_level
FROM ols_inference(
    [100.0, 95.0, 92.0, 88.0, 85.0]::DOUBLE[],           -- y: sales
    [[10.0, 5.0], [11.0, 6.0], [12.0, 7.0], [13.0, 8.0], [14.0, 9.0]]::DOUBLE[][],  -- x: price, advertising
    0.95,                                                  -- confidence_level
    true                                                   -- add_intercept
)
WHERE variable = 'advertising';
```

## Best Practices

### Statistical Workflow

1. **Exploratory Analysis**
   - Visualize relationships
   - Check distributions
   - Identify outliers

2. **Model Fitting**
   - Start simple (fewer predictors)
   - Add complexity as needed
   - Consider theory/domain knowledge

3. **Diagnostics**
   - Check assumptions
   - Identify influential points
   - Test alternative specifications

4. **Inference**
   - Interpret coefficients
   - Report uncertainty (CI, p-values)
   - Check practical significance

5. **Validation**
   - Out-of-sample testing
   - Cross-validation
   - Sensitivity analysis

### Reporting Standards

**Minimum to Report**:

- Sample size (n)
- Model specification
- Coefficients with SE or CI
- R² or adjusted R²
- Diagnostic checks passed/failed

**Example Report**:

```
Linear regression of sales on price and advertising
(n = 150 stores)

Results:
  Intercept: 45.2 (SE = 2.1), p < 0.001
  Price:     -2.3 (95% CI: -2.8 to -1.8), p < 0.001
  Advertising: 5.7 (95% CI: 4.2 to 7.2), p < 0.001

Model fit: R² = 0.76, Adj R² = 0.75
Diagnostics: No violations detected
```

## References

### Textbooks

- Wooldridge (2020): *Introductory Econometrics*
- Greene (2018): *Econometric Analysis*
- Hastie et al. (2009): *Elements of Statistical Learning*

### Papers

- Gauss-Markov Theorem: Aitken (1935)
- Ridge Regression: Hoerl & Kennard (1970)
- Cook's Distance: Cook (1977)
- Information Criteria: Akaike (1974), Schwarz (1978)
