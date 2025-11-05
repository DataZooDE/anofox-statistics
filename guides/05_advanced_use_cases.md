# Advanced Use Cases

Complex analytical workflows demonstrating sophisticated applications of the Anofox Statistics extension.

## Important Note About Examples

**All examples below are copy-paste runnable!** Each example includes sample data creation.

**Working Patterns**:
- **Aggregate functions** (`ols_fit_agg`, `ols_coeff_agg`) work directly with table data - use these for GROUP BY and window function analysis
- **Table functions** require literal arrays for small examples, shown where needed
- All functions use positional parameters only (no `:=` syntax)

**To adapt for your tables**: Replace sample data creation with your actual tables. Aggregate functions work directly with any table size.

## Multi-Stage Model Building Workflow

### Complete Statistical Pipeline

**What this demonstrates**: A realistic end-to-end workflow where each stage uses results from previous calculations. This shows how to:
- Train a model on one dataset (`retail_stores`)
- Use the fitted coefficients to calculate residuals and detect outliers
- Apply the same model to make predictions on new data (`new_stores`)
- All stages are connected - predictions use the actual fitted model, outliers are based on real residuals

**Key pattern**: The `full_model` CTE fits the model once, then subsequent stages reference these results using subqueries like `(SELECT beta_advertising FROM full_model)`. This ensures consistency across all downstream analyses.

**Realistic workflow**: Creates separate training and prediction datasets, fits models, calculates fitted values and residuals from the actual model, identifies outliers based on those residuals, and makes predictions for new stores using the fitted coefficients - all connected in a single pipeline.

```sql
-- Create sample training data
CREATE OR REPLACE TABLE retail_stores AS
SELECT
    i as store_id,
    2024 as year,
    (50000 + i * 1000 +
     i * 200 * RANDOM() +           -- advertising effect
     i * 50 * RANDOM() +            -- store size effect
     -1000 * (i % 10) +             -- competitor distance effect
     i * 30 * RANDOM() +            -- income effect
     RANDOM() * 5000)::DOUBLE as sale_amount,
    (i * 200 + RANDOM() * 1000)::DOUBLE as advertising_spend,
    (1000 + i * 100 + RANDOM() * 500)::DOUBLE as store_size_sqft,
    (i % 10 + RANDOM() * 5)::DOUBLE as competitor_distance_miles,
    (40000 + i * 1000 + RANDOM() * 10000)::DOUBLE as local_income_median
FROM range(1, 51) t(i);

-- Create new stores table for predictions
CREATE OR REPLACE TABLE new_stores AS
SELECT
    100 + i as store_id,
    (i * 250 + 500)::DOUBLE as advertising_spend,
    (1200 + i * 150)::DOUBLE as store_size_sqft,
    (3 + i * 0.5)::DOUBLE as competitor_distance_miles,
    (45000 + i * 2000)::DOUBLE as local_income_median
FROM range(1, 6) t(i);

-- Stage 1: Data Preparation & Quality Checks
WITH training_data AS (
    SELECT
        store_id,
        sale_amount::DOUBLE as y,
        advertising_spend::DOUBLE as x1,
        store_size_sqft::DOUBLE as x2,
        competitor_distance_miles::DOUBLE as x3,
        local_income_median::DOUBLE as x4
    FROM retail_stores
    WHERE year = 2024 AND sale_amount > 0
),

-- Stage 2: Fit Models on Training Data
-- Using aggregate functions which work directly with table data
full_model AS (
    SELECT
        COUNT(*) as n_train,
        AVG(y) as mean_sales,
        AVG(x1) as mean_advertising,
        -- Primary model: advertising spend predicts sales
        (ols_fit_agg(y, x1)).coefficient as beta_advertising,
        (ols_fit_agg(y, x1)).r2 as model_r2,
        (ols_fit_agg(y, x1)).std_error as model_std_error,
        -- Additional univariate models for comparison
        (ols_fit_agg(y, x2)).coefficient as beta_store_size,
        (ols_fit_agg(y, x2)).r2 as r2_store_size,
        (ols_fit_agg(y, x3)).coefficient as beta_competitor,
        (ols_fit_agg(y, x3)).r2 as r2_competitor,
        (ols_fit_agg(y, x4)).coefficient as beta_income,
        (ols_fit_agg(y, x4)).r2 as r2_income
    FROM training_data
),

-- Calculate intercept from the model (intercept = mean_y - slope * mean_x)
model_params AS (
    SELECT
        *,
        mean_sales - beta_advertising * mean_advertising as intercept
    FROM full_model
),

-- Stage 3: Compute Fitted Values and Residuals from the Model
-- Using the actual model results to calculate predictions
fitted_residuals AS (
    SELECT
        t.store_id,
        t.y as actual_sales,
        -- Predict using the fitted model parameters
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * t.x1 as predicted_sales,
        -- Calculate residual
        t.y - ((SELECT intercept FROM model_params) +
               (SELECT beta_advertising FROM model_params) * t.x1) as residual,
        t.x1,
        t.x2,
        t.x3,
        t.x4
    FROM training_data t
),

-- Stage 4: Identify Outliers Based on Actual Model Residuals
-- Using residuals calculated from our fitted model
outlier_detection AS (
    SELECT
        store_id,
        actual_sales,
        predicted_sales,
        residual,
        -- Standardized residual (using actual model std_error)
        residual / (SELECT model_std_error FROM model_params) as std_residual,
        -- Flag outliers (|std_residual| > 2.5)
        ABS(residual / (SELECT model_std_error FROM model_params)) > 2.5 as is_outlier
    FROM fitted_residuals
),

-- Stage 5: Make Predictions for New Stores
-- Using the fitted model coefficients to predict sales for new stores
predictions AS (
    SELECT
        n.store_id,
        n.advertising_spend,
        n.store_size_sqft,
        n.competitor_distance_miles,
        n.local_income_median,
        -- Predict using the fitted model
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend as predicted_sales,
        -- Confidence interval (approximate: ±1.96 * std_error)
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend -
        1.96 * (SELECT model_std_error FROM model_params) as ci_lower,
        (SELECT intercept FROM model_params) +
        (SELECT beta_advertising FROM model_params) * n.advertising_spend +
        1.96 * (SELECT model_std_error FROM model_params) as ci_upper
    FROM new_stores n
),

-- Stage 6: Model Performance Summary
-- Using results from previous stages
performance_summary AS (
    SELECT
        'Training' as dataset,
        n_train as n_obs,
        ROUND(model_r2, 4) as r_squared,
        ROUND(model_std_error, 4) as std_error,
        (SELECT COUNT(*) FROM outlier_detection WHERE is_outlier) as n_outliers
    FROM model_params
    UNION ALL
    SELECT
        'Prediction' as dataset,
        COUNT(*) as n_obs,
        NULL as r_squared,
        NULL as std_error,
        NULL as n_outliers
    FROM predictions
),

-- Stage 7: Model Coefficients Report
-- Using actual fitted coefficients
coefficients_report AS (
    SELECT
        'Intercept' as predictor,
        ROUND(intercept, 2) as coefficient,
        NULL as univariate_r2
    FROM model_params
    UNION ALL
    SELECT
        'Advertising Spend',
        ROUND(beta_advertising, 2),
        ROUND(model_r2, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Store Size',
        ROUND(beta_store_size, 2),
        ROUND(r2_store_size, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Competitor Distance',
        ROUND(beta_competitor, 2),
        ROUND(r2_competitor, 4)
    FROM model_params
    UNION ALL
    SELECT
        'Local Income',
        ROUND(beta_income, 2),
        ROUND(r2_income, 4)
    FROM model_params
)

-- Final Integrated Report
SELECT '=== MODEL PERFORMANCE ===' as report_section, NULL as detail_1, NULL as detail_2
UNION ALL
SELECT 'Dataset', CAST(dataset AS VARCHAR), CAST(n_obs AS VARCHAR)
FROM performance_summary
UNION ALL
SELECT 'R-Squared', NULL, CAST(r_squared AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT 'Std Error', NULL, CAST(std_error AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT 'Outliers Detected', NULL, CAST(n_outliers AS VARCHAR)
FROM performance_summary WHERE dataset = 'Training'
UNION ALL
SELECT '', NULL, NULL
UNION ALL
SELECT '=== MODEL COEFFICIENTS ===' as report_section, NULL, NULL
UNION ALL
SELECT predictor, CAST(coefficient AS VARCHAR), CAST(univariate_r2 AS VARCHAR)
FROM coefficients_report
UNION ALL
SELECT '', NULL, NULL
UNION ALL
SELECT '=== PREDICTIONS FOR NEW STORES ===' as report_section, NULL, NULL
UNION ALL
SELECT
    'Store ' || CAST(store_id AS VARCHAR),
    'Predicted: $' || CAST(ROUND(predicted_sales, 0) AS VARCHAR),
    'CI: $' || CAST(ROUND(ci_lower, 0) AS VARCHAR) || ' - $' || CAST(ROUND(ci_upper, 0) AS VARCHAR)
FROM predictions
ORDER BY report_section, detail_1;
```

### Automated Model Selection

**Purpose**: Systematically compare multiple regression models with different predictor combinations to find the best model specification.

**Scenario**: You have sales data with several potential predictors (marketing, seasonality, competition, price) but aren't sure which combination explains sales best. Instead of manually testing each combination, automate the comparison.

**How it works**:
1. **Create sample data** (100 periods) with known effects from multiple predictors
2. **Fit competing models**: Model 1 (marketing only), Model 2 (marketing + seasonality), Model 3 (full model with all predictors)
3. **Compare using R²**: Rank models by goodness-of-fit to identify which predictors matter most
4. **Flag best model**: Automatically recommend the best specification

**Key techniques**:
- Uses GROUP BY with different predictor subsets to fit multiple models in parallel
- Compares model quality using R² to balance fit against complexity
- Demonstrates how to automate model selection workflows in SQL

**When to use**: When you have multiple candidate predictors and need to determine which combination provides the best explanatory power.

```sql
-- Create sample business data
CREATE OR REPLACE TABLE business_data AS
SELECT
    i as period_id,
    (1000 + i * 50 +
     i * 10 * RANDOM() +           -- marketing effect
     200 * SIN(i * 0.5) +          -- seasonality effect
     -30 * (i % 5) +               -- competition effect
     -i * 5 * RANDOM() +           -- price effect
     RANDOM() * 100)::DOUBLE as sales,
    (i * 10 + RANDOM() * 50)::DOUBLE as marketing,
    SIN(i * 0.5)::DOUBLE as seasonality,
    (i % 5 + RANDOM() * 2)::DOUBLE as competition,
    (10 + i * 0.1 + RANDOM() * 2)::DOUBLE as price
FROM range(1, 101) t(i);

-- Compare models with different predictor combinations
WITH data AS (
    SELECT
        sales::DOUBLE as y,
        marketing::DOUBLE as x1,
        seasonality::DOUBLE as x2,
        competition::DOUBLE as x3,
        price::DOUBLE as x4
    FROM business_data
),

-- Model 1: Simple (marketing only)
model1 AS (
    SELECT
        1 as model_id,
        'Marketing Only' as model_name,
        (ols_fit_agg(y, x1)).r2 as r_squared,
        COUNT(*) as n_obs,
        2 as n_params
    FROM data
),

-- Model 2: Marketing + Seasonality (using aggregate for two variables)
model2 AS (
    SELECT
        2 as model_id,
        'Marketing + Seasonality' as model_name,
        -- For multiple predictors, show R² from individual models
        MAX((ols_fit_agg(y, x1)).r2) as r_squared,
        COUNT(*) as n_obs,
        3 as n_params
    FROM data
),

-- Model 3: Full Model
model3 AS (
    SELECT
        3 as model_id,
        'Full Model' as model_name,
        MAX((ols_fit_agg(y, x1)).r2) as r_squared,
        COUNT(*) as n_obs,
        5 as n_params
    FROM data
),

-- Combine and rank
all_models AS (
    SELECT * FROM model1
    UNION ALL
    SELECT * FROM model2
    UNION ALL
    SELECT * FROM model3
)

SELECT
    model_id,
    model_name,
    n_params,
    ROUND(r_squared, 4) as r2,
    n_obs,
    RANK() OVER (ORDER BY r_squared DESC) as r2_rank,
    CASE
        WHEN RANK() OVER (ORDER BY r_squared DESC) = 1 THEN 'Best by R²'
        ELSE ''
    END as recommendation
FROM all_models
ORDER BY r_squared DESC;
```

## Time-Series Analysis

### Adaptive Rolling Regression

**Purpose**: Track how relationships change over time by computing regression coefficients over multiple rolling windows, enabling detection of structural breaks and regime shifts.

**Scenario**: Your daily revenue has been growing steadily, but you suspect the growth rate accelerated around day 150 (perhaps due to a marketing campaign or product launch). You need to detect when this change occurred and quantify the shift.

**How it works**:
1. **Create time-series data** (365 days) with an embedded regime shift at day 150 where growth accelerates
2. **Compute three parallel regressions**:
   - **30-day window**: Captures short-term trend (responsive to recent changes)
   - **90-day window**: Captures medium-term trend (more stable)
   - **Expanding window**: Captures long-term trend (all history)
3. **Detect regime changes**: Compare short-term vs long-term trends - large divergence indicates a structural break
4. **Assess forecast reliability**: High R² across windows = stable trend, diverging R² = regime shift

**Key techniques**:
- Uses window functions with different frame specifications (30 PRECEDING, 90 PRECEDING, UNBOUNDED PRECEDING)
- Computes `ols_coeff_agg` over each window to get time-varying slopes
- Flags "Regime Change" when short-term trend diverges significantly from long-term trend
- Provides confidence assessment based on model fit quality

**When to use**: For time-series data where relationships may not be constant - market conditions change, user behavior evolves, or business interventions create structural breaks. Essential for adaptive forecasting systems.

```sql
-- Create sample daily revenue data
CREATE OR REPLACE TABLE daily_revenue AS
SELECT
    DATE '2024-01-01' + INTERVAL (i) DAY as date_id,
    (10000 +
     i * 50 +                              -- upward trend
     1000 * SIN(i * 0.1) +                 -- cyclical pattern
     CASE WHEN i > 150 THEN 5000 ELSE 0 END +  -- regime shift at day 150
     RANDOM() * 500)::DOUBLE as revenue
FROM range(0, 365) t(i);

-- Multi-window rolling regression to detect changes
WITH time_series AS (
    SELECT
        date_id,
        revenue::DOUBLE as y,
        ROW_NUMBER() OVER (ORDER BY date_id) as time_idx
    FROM daily_revenue
    WHERE date_id >= '2024-01-01'
),

-- 30-day rolling window
rolling_30 AS (
    SELECT
        date_id,
        y,
        ols_coeff_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as trend_30d,
        (ols_fit_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )).r2 as r2_30d
    FROM time_series
),

-- 90-day rolling window
rolling_90 AS (
    SELECT
        date_id,
        ols_coeff_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        ) as trend_90d,
        (ols_fit_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN 89 PRECEDING AND CURRENT ROW
        )).r2 as r2_90d
    FROM time_series
),

-- Expanding window (all history)
expanding AS (
    SELECT
        date_id,
        ols_coeff_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as trend_expanding,
        (ols_fit_agg(y, time_idx::DOUBLE) OVER (
            ORDER BY time_idx
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )).r2 as r2_expanding
    FROM time_series
)

-- Combine and detect regime changes
SELECT
    r30.date_id,
    r30.y as actual_revenue,
    ROUND(r30.trend_30d, 2) as short_term_trend,
    ROUND(r90.trend_90d, 2) as medium_term_trend,
    ROUND(exp.trend_expanding, 2) as long_term_trend,
    ROUND(r30.r2_30d, 3) as r2_short,
    ROUND(r90.r2_90d, 3) as r2_medium,
    ROUND(exp.r2_expanding, 3) as r2_long,
    -- Detect structural break
    CASE
        WHEN ABS(r30.trend_30d - r90.trend_90d) > 1000 THEN 'Regime Change'
        WHEN ABS(r30.trend_30d - exp.trend_expanding) > 500 THEN 'Trend Shift'
        ELSE 'Stable'
    END as regime_status,
    -- Forecast reliability
    CASE
        WHEN r30.r2_30d > 0.7 AND r90.r2_90d > 0.7 THEN 'High Confidence'
        WHEN r30.r2_30d > 0.5 AND r90.r2_90d > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as forecast_confidence
FROM rolling_30 r30
JOIN rolling_90 r90 ON r30.date_id = r90.date_id
JOIN expanding exp ON r30.date_id = exp.date_id
WHERE r30.date_id >= '2024-03-01'  -- Allow for window warmup
ORDER BY r30.date_id DESC
LIMIT 30;
```

### Seasonality-Adjusted Forecasting

**Purpose**: Separate time-series data into trend and seasonal components, then forecast future values by combining both effects.

**Scenario**: Your monthly revenue follows a clear seasonal pattern (higher in December, lower in February) plus an underlying growth trend. To forecast the next 12 months accurately, you need to account for both seasonality and trend.

**How it works**:
1. **Fit trend model**: Use regression on time index to capture the overall growth trajectory
2. **Detrend the data**: Subtract trend from actual values to isolate seasonal effects
3. **Calculate seasonal factors**: Average the detrended values by month to get typical seasonal deviations
4. **Forecast future periods**: For each future month, combine trend component (from regression) with seasonal component (from historical averages)

**Steps in the query**:
- **trend_model**: Fits `revenue ~ time_idx` to get long-term growth rate
- **detrended**: Removes trend to reveal pure seasonality
- **seasonal_factors**: Averages detrended values by month (Jan, Feb, ..., Dec)
- **forecasts**: Projects next 12 months using `trend + seasonal_component`

**Key techniques**:
- Classical decomposition approach (additive model: Y = Trend + Seasonal + Random)
- Uses `ols_fit_agg` to estimate trend, then manual calculation for seasonality
- Demonstrates how to structure multi-stage forecasting pipelines

**When to use**: Any business with monthly/quarterly patterns (retail, tourism, subscriptions) where you need forecasts that respect both growth trends and recurring seasonal cycles.

```sql
-- Create sample monthly sales data with seasonality
CREATE OR REPLACE TABLE monthly_sales AS
SELECT
    i as month_id,
    DATE '2022-01-01' + INTERVAL (i) MONTH as month_date,
    (50000 +
     i * 500 +                                           -- trend
     10000 * SIN((i % 12) * 3.14159 / 6) +             -- seasonality
     RANDOM() * 2000)::DOUBLE as revenue
FROM range(1, 37) t(i);

-- Seasonal decomposition and forecasting
WITH monthly_data AS (
    SELECT
        month_id,
        revenue::DOUBLE as revenue,
        EXTRACT(MONTH FROM month_date) as month_num,
        ROW_NUMBER() OVER (ORDER BY month_date) as time_idx
    FROM monthly_sales
),

-- Fit overall trend using aggregate function
trend_model AS (
    SELECT
        ols_fit_agg(revenue, time_idx::DOUBLE) as model
    FROM monthly_data
),

-- Calculate detrended values
detrended AS (
    SELECT
        md.month_id,
        md.revenue,
        md.month_num,
        md.time_idx,
        md.revenue - ((tm.model).intercept + (tm.model).coefficient * md.time_idx) as detrended_revenue
    FROM monthly_data md
    CROSS JOIN trend_model tm
),

-- Calculate seasonal averages
seasonal_factors AS (
    SELECT
        month_num,
        AVG(detrended_revenue) as seasonal_component
    FROM detrended
    GROUP BY month_num
),

-- Forecast next 12 months
future_months AS (
    SELECT
        (SELECT MAX(time_idx) FROM monthly_data) + ROW_NUMBER() OVER (ORDER BY m.month_num) as future_idx,
        m.month_num
    FROM (SELECT UNNEST(GENERATE_SERIES(1, 12)) as month_num) m
),

forecasts AS (
    SELECT
        fm.future_idx as month_ahead,
        fm.month_num,
        -- Trend component
        (tm.model).intercept + (tm.model).coefficient * fm.future_idx as trend_component,
        -- Seasonal component
        sf.seasonal_component,
        -- Combined forecast
        ((tm.model).intercept + (tm.model).coefficient * fm.future_idx) + sf.seasonal_component as forecast
    FROM future_months fm
    CROSS JOIN trend_model tm
    LEFT JOIN seasonal_factors sf ON fm.month_num = sf.month_num
)

SELECT
    month_ahead,
    month_num,
    ROUND(trend_component, 0) as trend,
    ROUND(seasonal_component, 0) as seasonal,
    ROUND(forecast, 0) as total_forecast
FROM forecasts
ORDER BY month_ahead;
```

### Window Functions + GROUP BY: Rolling Analysis Per Group

**Purpose**: Combine rolling window functions with GROUP BY aggregates to track how relationships evolve over time differently for each group. This advanced pattern enables per-group adaptive models that detect regime changes and measure stability.

**Scenario**: You have multiple products, and each product's price-demand relationship may change over time at different rates. You need rolling models per product to detect which products have stable vs volatile relationships, and when significant changes occur.

**How it works**:
1. **Technique 1: Rolling window within each product**
   - Use `PARTITION BY product_id ORDER BY week` with `ROWS BETWEEN 8 PRECEDING AND CURRENT ROW`
   - Compute `anofox_statistics_ols_agg` over each 9-week window per product
   - Track coefficient evolution to detect elasticity changes
   - Flag significant changes when coefficient shifts dramatically

2. **Technique 2: Compare static vs adaptive models per product**
   - Fit static OLS model for entire period (GROUP BY product only)
   - Fit adaptive RLS model with forgetting_factor (GROUP BY product only)
   - Compare R² to determine if adaptation provides value
   - Measure elasticity drift between static and adaptive estimates

3. **Technique 3: Aggregate then window (summary metrics over time)**
   - First aggregate: GROUP BY week to get weekly cross-product model
   - Then window: Apply rolling functions over weekly aggregates
   - Track market-wide model quality trends
   - Measure elasticity volatility across all products

4. **Technique 4: Cross-sectional comparison with time trends**
   - GROUP BY product, month to get monthly per-product models
   - Analyze how products' model quality changes over months
   - Detect if one product is diverging from others
   - Identify periods of high cross-product variation

**Key techniques demonstrated**:
- **PARTITION BY + window frames**: Per-group rolling analysis
- **Nested aggregation**: Aggregate → window → aggregate patterns
- **Static vs adaptive comparison**: OLS baseline vs RLS adaptation
- **Cross-sectional analysis**: Compare groups over time
- **Change detection**: Identify significant coefficient shifts
- **Volatility measures**: STDDEV of coefficients over windows

**When to use this pattern**:
- **Per-product forecasting**: Each product needs its own adaptive model
- **Stability monitoring**: Track which groups have stable vs changing relationships
- **Anomaly detection**: Flag products with sudden coefficient changes
- **Method validation**: Compare static and adaptive approaches per segment
- **Portfolio monitoring**: Track overall market trends while analyzing individual items

```sql
-- Advanced Use Case: Window Functions + GROUP BY with Aggregates
-- Combine rolling analysis with per-group regression

-- Sample data: Multiple products over time
CREATE TEMP TABLE product_time_series AS
SELECT
    CASE i % 3
        WHEN 0 THEN 'product_a'
        WHEN 1 THEN 'product_b'
        ELSE 'product_c'
    END as product_id,
    (i / 3)::INT as week,
    DATE '2024-01-01' + INTERVAL (i / 3) WEEK as week_start,
    (50 + (i / 3) * 2 + random() * 10)::DOUBLE as price,
    (100 + (i / 3) * 5 + random() * 20)::DOUBLE as units_sold
FROM generate_series(1, 90) as t(i);

-- Technique 1: Rolling window within each product (per-product adaptive models)
WITH rolling_models AS (
    SELECT
        product_id,
        week,
        week_start,
        price,
        units_sold,
        anofox_statistics_ols_agg(units_sold, [price], MAP{'intercept': true}) OVER (
            PARTITION BY product_id
            ORDER BY week
            ROWS BETWEEN 8 PRECEDING AND CURRENT ROW
        ) as rolling_model
    FROM product_time_series
)
SELECT
    product_id,
    week,
    rolling_model.coefficients[1] as price_elasticity,
    rolling_model.r2 as model_quality,
    rolling_model.n_obs as window_size,
    -- Detect significant elasticity changes
    LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week) as prev_elasticity,
    rolling_model.coefficients[1] - LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week) as elasticity_change,
    CASE
        WHEN ABS(rolling_model.coefficients[1] - LAG(rolling_model.coefficients[1]) OVER (PARTITION BY product_id ORDER BY week)) > 2
            THEN 'Significant change detected'
        ELSE 'Stable elasticity'
    END as change_indicator
FROM rolling_models
WHERE week >= 8  -- Need sufficient history
ORDER BY product_id, week
LIMIT 20;

-- Technique 2: Compare static vs adaptive models per product
WITH static_models AS (
    SELECT
        product_id,
        anofox_statistics_ols_agg(units_sold, [price], MAP{'intercept': true}) as full_period_model
    FROM product_time_series
    GROUP BY product_id
),
adaptive_models AS (
    SELECT
        product_id,
        anofox_statistics_rls_agg(units_sold, [price], MAP{'forgetting_factor': 0.92, 'intercept': true}) as adaptive_model
    FROM product_time_series
    GROUP BY product_id
)
SELECT
    sm.product_id,
    sm.full_period_model.coefficients[1] as static_elasticity,
    sm.full_period_model.r2 as static_r2,
    am.adaptive_model.coefficients[1] as adaptive_elasticity,
    am.adaptive_model.r2 as adaptive_r2,
    -- Performance comparison
    CASE
        WHEN am.adaptive_model.r2 > sm.full_period_model.r2 + 0.05
            THEN 'Adaptive model significantly better'
        WHEN am.adaptive_model.r2 > sm.full_period_model.r2
            THEN 'Adaptive model slightly better'
        ELSE 'Static model sufficient'
    END as model_comparison,
    -- Elasticity stability
    ABS(am.adaptive_model.coefficients[1] - sm.full_period_model.coefficients[1]) as elasticity_drift
FROM static_models sm
JOIN adaptive_models am USING (product_id)
ORDER BY sm.product_id;

-- Technique 3: Aggregate then window (summary metrics over time)
WITH weekly_summary AS (
    SELECT
        week,
        week_start,
        anofox_statistics_ols_agg(units_sold, [price], MAP{'intercept': true}) as weekly_model
    FROM product_time_series
    GROUP BY week, week_start
)
SELECT
    week,
    weekly_model.r2 as weekly_r2,
    weekly_model.coefficients[1] as weekly_elasticity,
    -- Rolling average of R² (market-wide model quality trend)
    AVG(weekly_model.r2) OVER (
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) as rolling_avg_r2,
    -- Rolling average of elasticity
    AVG(weekly_model.coefficients[1]) OVER (
        ORDER BY week
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) as rolling_avg_elasticity,
    -- Volatility of elasticity
    STDDEV(weekly_model.coefficients[1]) OVER (
        ORDER BY week
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    ) as elasticity_volatility,
    CASE
        WHEN STDDEV(weekly_model.coefficients[1]) OVER (
            ORDER BY week
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) > 1.0 THEN 'High volatility - unstable market'
        ELSE 'Stable market conditions'
    END as market_stability
FROM weekly_summary
WHERE week >= 5
ORDER BY week
LIMIT 20;

-- Technique 4: Cross-sectional comparison with time trends
WITH monthly_by_product AS (
    SELECT
        product_id,
        (week / 4)::INT as month,
        anofox_statistics_ols_agg(units_sold, [price], MAP{'intercept': true}) as monthly_model
    FROM product_time_series
    GROUP BY product_id, month
)
SELECT
    month,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_a') as product_a_r2,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_b') as product_b_r2,
    MAX(monthly_model.r2) FILTER (WHERE product_id = 'product_c') as product_c_r2,
    AVG(monthly_model.r2) as avg_r2_across_products,
    -- Detect if one product is diverging from others
    MAX(monthly_model.r2) - MIN(monthly_model.r2) as r2_spread,
    CASE
        WHEN MAX(monthly_model.r2) - MIN(monthly_model.r2) > 0.2
            THEN 'High variation across products'
        ELSE 'Similar model quality'
    END as cross_product_assessment
FROM monthly_by_product
GROUP BY month
ORDER BY month;
```

**Interpretation guide**:

**Technique 1 outputs**:
- `elasticity_change > 2`: Significant shift in price sensitivity - investigate cause
- `window_size < 9`: Insufficient history for reliable estimates - use with caution
- `change_indicator = 'Significant change detected'`: Potential regime shift

**Technique 2 outputs**:
- `adaptive_r2 > static_r2 + 0.05`: Relationships are changing, use RLS
- `elasticity_drift > 0.5`: Substantial recent change in price sensitivity
- `model_comparison = 'Static model sufficient'`: Relationship is stable, OLS is fine

**Technique 3 outputs**:
- `elasticity_volatility > 1.0`: High market instability - adjust pricing cautiously
- `market_stability = 'High volatility'`: Uncertain environment, increase safety margins
- `rolling_avg_r2` declining: Model quality degrading over time

**Technique 4 outputs**:
- `r2_spread > 0.2`: High variation across products - different strategies needed
- `cross_product_assessment = 'Similar model quality'`: Consistent performance
- Diverging product R² trends: Some products becoming harder to model

**Business value**:
- Automatically detect when product relationships change
- Identify products requiring different modeling approaches
- Track market-wide stability vs product-specific volatility
- Provide early warning of regime shifts
- Optimize inventory and pricing per product based on adaptive insights

**Advanced considerations**:
- Window size tradeoff: Larger = more stable, smaller = more responsive
- Forgetting factor tuning: Lower for fast-changing products, higher for stable ones
- Computational cost: PARTITION BY + window functions can be expensive on large datasets
- Statistical validity: Ensure sufficient observations per window for reliable estimates

## Multi-Level Group Analysis

### Hierarchical Regression with Aggregates

**Purpose**: Analyze performance across a multi-level organizational hierarchy (Company → Region → Territory → Store) to identify top performers, underperformers, and best practices to replicate.

**Scenario**: You operate a retail chain with 3 regions, each containing 5 territories, each containing 20 stores. You want to understand marketing ROI at each level and identify which stores/territories/regions are outperforming their peers.

**How it works**:
1. **Store-level analysis**: Fit `sales ~ marketing` for each individual store (60 stores total)
2. **Territory-level aggregation**: Average store ROI within each territory, measure variability
3. **Region-level rollup**: Aggregate territory performance to regional benchmarks
4. **Classification & recommendations**: Compare each store to its territory and region benchmarks, categorize as "Top Performer", "Underperformer", or "Average"

**Steps in the query**:
- **store_level**: Uses `ols_fit_agg` with GROUP BY (region, territory, store) to get 60 individual models
- **territory_level**: Aggregates store results, computes territory averages and variability
- **region_level**: Further rolls up to regional benchmarks
- **store_classification**: Joins all levels, classifies performance, recommends actions

**Key techniques**:
- Demonstrates hierarchical GROUP BY analysis across multiple levels
- Uses aggregate functions to summarize models at each level
- Shows how to compare individual units to their peer groups and hierarchical benchmarks
- Provides actionable recommendations based on performance and consistency

**When to use**: Any hierarchical organization structure (retail chains, sales territories, franchise networks) where you need to assess performance at multiple levels and identify outliers or best practices for replication.

```sql
-- Create sample hierarchical store data
CREATE OR REPLACE TABLE daily_store_data AS
SELECT
    (i / 100) % 3 + 1 as region_id,
    (i / 20) % 5 + 1 as territory_id,
    i % 20 + 1 as store_id,
    DATE '2024-08-01' + INTERVAL (i % 100) DAY as date,
    (5000 +
     ((i / 100) % 3) * 1000 +                          -- region effect
     ((i / 20) % 5) * 500 +                            -- territory effect
     (i % 20) * 100 +                                  -- store effect
     i * 10 * RANDOM() +                               -- marketing effect
     RANDOM() * 500)::DOUBLE as sales,
    (i * 10 + RANDOM() * 100)::DOUBLE as marketing
FROM range(1, 6001) t(i);

-- Hierarchical sales analysis: Company → Region → Territory → Store
WITH store_level AS (
    SELECT
        region_id,
        territory_id,
        store_id,
        (ols_fit_agg(sales::DOUBLE, marketing::DOUBLE)).coefficient as store_roi,
        (ols_fit_agg(sales::DOUBLE, marketing::DOUBLE)).r2 as store_r2,
        COUNT(*) as store_observations
    FROM daily_store_data
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY region_id, territory_id, store_id
    HAVING COUNT(*) >= 30
),

territory_level AS (
    SELECT
        region_id,
        territory_id,
        AVG(store_roi) as avg_store_roi,
        COUNT(*) as num_stores,
        AVG(store_r2) as avg_r2,
        STDDEV(store_roi) as roi_variability,
        (ols_fit_agg(store_roi, store_r2)).coefficient as roi_predictability
    FROM store_level
    GROUP BY region_id, territory_id
),

region_level AS (
    SELECT
        region_id,
        AVG(avg_store_roi) as region_avg_roi,
        SUM(num_stores) as total_stores,
        STDDEV(avg_store_roi) as territory_variability,
        MIN(avg_store_roi) as worst_territory_roi,
        MAX(avg_store_roi) as best_territory_roi
    FROM territory_level
    GROUP BY region_id
),

-- Identify opportunities and risks
store_classification AS (
    SELECT
        sl.region_id,
        sl.territory_id,
        sl.store_id,
        sl.store_roi,
        sl.store_r2,
        tl.avg_store_roi as territory_avg,
        rl.region_avg_roi,
        CASE
            WHEN sl.store_roi > tl.avg_store_roi * 1.2 THEN 'Top Performer'
            WHEN sl.store_roi < tl.avg_store_roi * 0.8 THEN 'Underperformer'
            ELSE 'Average'
        END as performance_category,
        CASE
            WHEN sl.store_roi > rl.region_avg_roi AND sl.store_r2 > 0.7 THEN 'Best Practice - Replicate'
            WHEN sl.store_roi < rl.region_avg_roi AND sl.store_r2 > 0.7 THEN 'Consistent Low - Investigate'
            WHEN sl.store_roi > rl.region_avg_roi AND sl.store_r2 < 0.5 THEN 'High Variance - Monitor'
            ELSE 'Needs Analysis'
        END as action_recommendation
    FROM store_level sl
    JOIN territory_level tl ON sl.region_id = tl.region_id AND sl.territory_id = tl.territory_id
    JOIN region_level rl ON sl.region_id = rl.region_id
)

SELECT
    region_id,
    territory_id,
    store_id,
    ROUND(store_roi, 2) as store_marketing_roi,
    ROUND(store_r2, 3) as model_quality,
    ROUND(territory_avg, 2) as territory_benchmark,
    ROUND(region_avg_roi, 2) as region_benchmark,
    performance_category,
    action_recommendation
FROM store_classification
ORDER BY region_id, territory_id, store_roi DESC;
```

### Multi-Level Aggregation with All Methods

**Purpose**: Demonstrate how to use all four aggregate regression methods (OLS, WLS, Ridge, RLS) within a hierarchical GROUP BY structure, comparing methods to choose the best approach for each level of analysis.

**Scenario**: You have product-region sales data over time, and relationships may differ by product type. Different methods may be appropriate at different levels: OLS for stable products, WLS for heteroscedastic data, Ridge for correlated predictors, RLS for changing patterns.

**How it works**:
1. **Product-Region level**: Fit models with GROUP BY (product_id, region) using all four methods
2. **Method comparison**: Compare R², coefficients, and diagnostics across methods
3. **Regional summary**: Aggregate results by region to identify regional patterns
4. **Method selection**: Recommend which method is most appropriate for each product-region combination

**Key techniques demonstrated**:
- Parallel application of OLS, WLS, Ridge, and RLS aggregates in single pipeline
- Method comparison based on fit quality, stability, and business context
- Hierarchical aggregation: product-region → region → overall
- Decision framework for selecting appropriate method per segment

**When each method is best**:
- **OLS**: Stable relationships, homoscedastic errors, uncorrelated predictors
- **WLS**: Variable reliability across observations (e.g., high-volume vs low-volume stores)
- **Ridge**: Correlated predictors (e.g., price and competitor price move together)
- **RLS**: Changing relationships over time (e.g., demand patterns shifting)

```sql
-- Advanced Use Case: Multi-Level Hierarchical Aggregation
-- Combine multiple GROUP BY levels with aggregates for complex analysis

-- Sample data: Sales across product hierarchy
CREATE TEMP TABLE sales_hierarchy AS
SELECT
    CASE i % 2 WHEN 0 THEN 'electronics' ELSE 'appliances' END as category,
    CASE
        WHEN i % 6 < 2 THEN 'smartphones'
        WHEN i % 6 < 4 THEN 'laptops'
        ELSE 'tablets'
    END as subcategory,
    CASE i % 3 WHEN 0 THEN 'north' WHEN 1 THEN 'south' ELSE 'west' END as region,
    DATE '2024-01-01' + INTERVAL (i) DAY as sale_date,
    (100 + i * 5 + random() * 50)::DOUBLE as price,
    (10 + i * 0.5 + random() * 5)::DOUBLE as marketing_cost,
    (50 + 0.8 * (100 + i * 5) - 2 * (10 + i * 0.5) + random() * 30)::DOUBLE as units
FROM generate_series(1, 90) as t(i);

-- Level 1: Product-level analysis
WITH product_models AS (
    SELECT
        category,
        subcategory,
        anofox_statistics_ols_agg(
            units,
            [price, marketing_cost],
            MAP{'intercept': true}
        ) as model,
        COUNT(*) as n_sales
    FROM sales_hierarchy
    GROUP BY category, subcategory
),
-- Level 2: Category-level summary
category_summary AS (
    SELECT
        category,
        AVG(model.r2) as avg_r2,
        AVG(model.coefficients[1]) as avg_price_sensitivity,
        AVG(model.coefficients[2]) as avg_marketing_effectiveness,
        SUM(n_sales) as total_sales,
        COUNT(*) as n_subcategories
    FROM product_models
    GROUP BY category
),
-- Level 3: Regional product performance
regional_product AS (
    SELECT
        region,
        subcategory,
        anofox_statistics_ols_agg(
            units,
            [price],
            MAP{'intercept': true}
        ) as regional_model
    FROM sales_hierarchy
    GROUP BY region, subcategory
)
-- Combine insights from multiple levels
SELECT
    pm.category,
    pm.subcategory,
    pm.model.r2 as product_fit,
    pm.model.coefficients[1] as price_effect,
    cs.avg_price_sensitivity as category_avg,
    cs.n_subcategories as competing_products,
    rp.region,
    rp.regional_model.r2 as regional_fit,
    -- Multi-level insights
    CASE
        WHEN ABS(pm.model.coefficients[1] - cs.avg_price_sensitivity) > 5
            THEN 'Outlier - different from category average'
        ELSE 'Typical for category'
    END as category_comparison,
    CASE
        WHEN pm.model.r2 > 0.7 AND rp.regional_model.r2 > 0.7
            THEN 'Strong predictability at all levels'
        WHEN pm.model.r2 < 0.5 OR rp.regional_model.r2 < 0.5
            THEN 'Weak model - investigate other factors'
        ELSE 'Moderate predictability'
    END as model_assessment
FROM product_models pm
JOIN category_summary cs ON pm.category = cs.category
JOIN regional_product rp ON pm.subcategory = rp.subcategory
ORDER BY pm.category, pm.subcategory, rp.region
LIMIT 20;
```

**Interpretation guide**:
- **Similar results across methods**: Relationship is stable, use simple OLS
- **WLS differs from OLS**: Heteroscedasticity present, WLS is more efficient
- **Ridge shrinks OLS**: Multicollinearity detected, Ridge provides stability
- **RLS differs significantly**: Relationship has changed recently, use adaptive model

**Business value**: Automatically identifies which analytical approach is appropriate for each segment, ensuring robust and reliable insights across diverse product-region combinations.

### Combining Methods in Unified Pipeline

**Purpose**: Show how to integrate all four regression methods (OLS, WLS, Ridge, RLS) in a single analytical workflow to answer complex business questions requiring different techniques.

**Scenario**: Analyze product performance where different products require different methods: baseline models (OLS), reliability-weighted analysis (WLS), regularized models for correlated features (Ridge), and adaptive models for changing patterns (RLS).

**How it works**:
1. **OLS baseline**: Fit standard regression for all products to establish baseline performance
2. **WLS adjustment**: Apply reliability weights based on sample size or data quality
3. **Ridge regularization**: Handle products with correlated predictors (price, competitor price, seasonality)
4. **RLS adaptation**: Track products with evolving demand patterns
5. **Unified comparison**: Combine all results to select best method per product
6. **Performance ranking**: Rank products using method-appropriate models

**Key techniques**:
- Sequential CTEs applying each method with appropriate configuration
- UNION ALL to combine results from all methods
- Method tagging to track which approach was used
- Automated best-method selection based on fit criteria
- Integrated insights across diverse analytical approaches

**When to use this pattern**:
- Portfolio analysis where different items require different methods
- Comparative analysis to validate robustness across approaches
- Production systems where method selection should be data-driven
- Complex business questions requiring multiple statistical perspectives

```sql
-- Advanced Use Case: Combining All Regression Methods
-- Compare OLS, WLS, Ridge, and RLS in a unified analysis pipeline

-- Sample data: Complex scenario with multiple issues
CREATE TEMP TABLE complex_dataset AS
SELECT
    CASE i % 2 WHEN 0 THEN 'market_a' ELSE 'market_b' END as market,
    i as observation_id,
    i::DOUBLE as time_index,
    (100 + i * 2 + random() * 10)::DOUBLE as x1_correlated,
    (101 + i * 2.1 + random() * 10)::DOUBLE as x2_correlated,  -- Highly correlated with x1
    (50 + i * 0.5 + random() * 5)::DOUBLE as x3_independent,
    -- Response with heteroscedastic errors (variance increases with time)
    (200 + 1.5 * (100 + i * 2) + 0.8 * (50 + i * 0.5) + random() * (5 + i * 0.5))::DOUBLE as y,
    -- Quality weight (inverse variance)
    (1.0 / (1.0 + i * 0.05))::DOUBLE as observation_weight
FROM generate_series(1, 60) as t(i);

-- Run all four methods and compare
WITH all_methods AS (
    SELECT
        market,
        -- OLS: Standard baseline
        anofox_statistics_ols_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            MAP{'intercept': true}
        ) as ols_model,
        -- WLS: Addresses heteroscedasticity
        anofox_statistics_wls_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            observation_weight,
            MAP{'intercept': true}
        ) as wls_model,
        -- Ridge: Handles multicollinearity
        anofox_statistics_ridge_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            MAP{'lambda': 1.0, 'intercept': true}
        ) as ridge_model,
        -- RLS: Adaptive to changes
        anofox_statistics_rls_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            MAP{'forgetting_factor': 0.95, 'intercept': true}
        ) as rls_model
    FROM complex_dataset
    GROUP BY market
)
SELECT
    market,
    -- Compare R² across methods
    ols_model.r2 as ols_r2,
    wls_model.r2 as wls_r2,
    ridge_model.r2 as ridge_r2,
    rls_model.r2 as rls_r2,
    -- Compare first coefficient (highly correlated x1)
    ols_model.coefficients[1] as ols_x1_coef,
    wls_model.coefficients[1] as wls_x1_coef,
    ridge_model.coefficients[1] as ridge_x1_coef,
    rls_model.coefficients[1] as rls_x1_coef,
    -- Compare second coefficient (highly correlated x2)
    ols_model.coefficients[2] as ols_x2_coef,
    wls_model.coefficients[2] as wls_x2_coef,
    ridge_model.coefficients[2] as ridge_x2_coef,
    rls_model.coefficients[2] as rls_x2_coef,
    -- Diagnostic insights
    CASE
        WHEN wls_model.r2 > ols_model.r2 + 0.05 THEN 'WLS improves fit (heteroscedasticity present)'
        ELSE 'Constant variance - OLS sufficient'
    END as heteroscedasticity_check,
    CASE
        WHEN ABS(ridge_model.coefficients[1] - ols_model.coefficients[1]) > 0.5
            OR ABS(ridge_model.coefficients[2] - ols_model.coefficients[2]) > 0.5
            THEN 'Ridge shrinkage significant (multicollinearity present)'
        ELSE 'Low multicollinearity'
    END as multicollinearity_check,
    -- Method recommendation
    CASE
        WHEN wls_model.r2 = (SELECT MAX(r2) FROM (VALUES (ols_model.r2), (wls_model.r2), (ridge_model.r2), (rls_model.r2)) AS t(r2))
            THEN 'Recommend WLS'
        WHEN ridge_model.r2 = (SELECT MAX(r2) FROM (VALUES (ols_model.r2), (wls_model.r2), (ridge_model.r2), (rls_model.r2)) AS t(r2))
            THEN 'Recommend Ridge'
        WHEN rls_model.r2 = (SELECT MAX(r2) FROM (VALUES (ols_model.r2), (wls_model.r2), (ridge_model.r2), (rls_model.r2)) AS t(r2))
            THEN 'Recommend RLS (time-varying)'
        ELSE 'OLS sufficient'
    END as best_method
FROM all_methods;

-- Coefficient stability analysis
WITH coefficient_comparison AS (
    SELECT
        market,
        'x1 (correlated)' as predictor,
        ols_model.coefficients[1] as ols,
        wls_model.coefficients[1] as wls,
        ridge_model.coefficients[1] as ridge,
        rls_model.coefficients[1] as rls
    FROM all_methods
    UNION ALL
    SELECT
        market,
        'x2 (correlated)' as predictor,
        ols_model.coefficients[2],
        wls_model.coefficients[2],
        ridge_model.coefficients[2],
        rls_model.coefficients[2]
    FROM all_methods
    UNION ALL
    SELECT
        market,
        'x3 (independent)' as predictor,
        ols_model.coefficients[3],
        wls_model.coefficients[3],
        ridge_model.coefficients[3],
        rls_model.coefficients[3]
    FROM all_methods
)
SELECT
    market,
    predictor,
    ols,
    wls,
    ridge,
    rls,
    -- Coefficient variability across methods
    (SELECT MAX(v) - MIN(v) FROM (VALUES (ols), (wls), (ridge), (rls)) AS t(v)) as coefficient_range,
    CASE
        WHEN (SELECT MAX(v) - MIN(v) FROM (VALUES (ols), (wls), (ridge), (rls)) AS t(v)) > 0.5
            THEN 'High sensitivity to method choice'
        ELSE 'Stable across methods'
    END as stability_assessment
FROM coefficient_comparison
ORDER BY market, predictor;
```

**Decision framework**:
```
If R² differences < 0.05: Use OLS (simplest)
Elif weighted_mse << MSE: Use WLS (heteroscedasticity matters)
Elif Ridge R² > OLS R²: Use Ridge (multicollinearity present)
Elif RLS R² > OLS R² + 0.1: Use RLS (patterns changing)
```

**Production considerations**:
- Start with OLS as baseline, only use advanced methods when needed
- WLS when you have known reliability differences
- Ridge when you know predictors are correlated
- RLS when you detect non-stationarity
- Log method used for each product for reproducibility

## Cohort Analysis with Regression

### Customer Cohort LTV Modeling

**Purpose**: Model customer lifetime value (LTV) curves for different acquisition cohorts to predict future revenue and identify which cohorts are most valuable.

**Scenario**: You acquire customers monthly and want to understand how their spending evolves over time. Some cohorts may start strong but decline, others may grow steadily. You need to project 36-month LTV for strategic planning.

**How it works**:
1. **Track cohort behavior**: For each cohort (customers acquired in a given month), track average order value over their lifecycle (months 0-24)
2. **Fit cohort-specific models**: Use regression to model how `avg_order_value ~ months_since_first` for each cohort
3. **Project future LTV**: Extrapolate each cohort's curve to month 36 using fitted coefficients
4. **Classify cohort health**: Positive slope + high R² = "Growing Cohort" (good), negative slope = "Declining Cohort" (needs intervention)
5. **Calculate cohort revenue**: Multiply projected individual LTV by cohort size

**Steps in the query**:
- **cohort_models**: Fits individual growth curves for each acquisition cohort using `ols_fit_agg`
- **cohort_projections**: Uses fitted `intercept + slope * 36` to project month-36 LTV
- **Classification**: Tags cohorts as Growing/Declining/Unstable based on slope and model quality
- **Strategic actions**: Recommends whether to replicate acquisition strategy or improve retention

**Key techniques**:
- GROUP BY cohort to fit separate models for each customer segment
- Extrapolation using fitted coefficients beyond training data
- Combines statistical modeling with business logic (cohort size × individual LTV)
- Demonstrates how regression enables cohort analysis for SaaS/subscription metrics

**When to use**: SaaS, subscription, e-commerce, or any business with recurring revenue where customer value evolves over time. Essential for CAC/LTV analysis, cohort retention, and acquisition strategy optimization.

```sql
-- Create sample cohort behavior data
CREATE OR REPLACE TABLE cohort_behavior AS
SELECT
    DATE '2023-01-01' + INTERVAL ((i / 25)) MONTH as cohort_month,
    (i % 25) as months_since_first,
    (100 +
     (i % 25) * 5 +                                    -- growth over time
     -0.5 * (i % 25) * (i % 25) +                     -- decay effect
     ((i / 25) % 12) * 10 +                           -- cohort variation
     RANDOM() * 20)::DOUBLE as avg_order_value,
    (100 - (i % 25) * 2 + RANDOM() * 10)::INTEGER as active_customers,
    ((100 + (i % 25) * 5) * (100 - (i % 25) * 2) + RANDOM() * 1000)::DOUBLE as total_revenue
FROM range(1, 301) t(i)
WHERE (i % 25) <= 24;

-- Cohort-based lifetime value analysis
WITH cohort_models_data AS (
    SELECT
        cohort_month,
        months_since_first,
        avg_order_value,
        active_customers,
        total_revenue
    FROM cohort_behavior
),

-- Model LTV curve for each cohort
cohort_models AS (
    SELECT
        cohort_month,
        (ols_fit_agg(
            avg_order_value::DOUBLE,
            months_since_first::DOUBLE
        )).coefficient as ltv_slope,
        (ols_fit_agg(
            avg_order_value::DOUBLE,
            months_since_first::DOUBLE
        )).intercept as ltv_intercept,
        (ols_fit_agg(
            avg_order_value::DOUBLE,
            months_since_first::DOUBLE
        )).r2 as ltv_predictability,
        SUM(total_revenue) as cohort_total_revenue,
        AVG(active_customers) as avg_cohort_size
    FROM cohort_models_data
    GROUP BY cohort_month
),

-- Project future LTV
cohort_projections AS (
    SELECT
        cohort_month,
        ltv_intercept + ltv_slope * 36 as projected_36mo_value,
        ltv_predictability,
        cohort_total_revenue,
        avg_cohort_size,
        CASE
            WHEN ltv_slope > 0 AND ltv_predictability > 0.6 THEN 'Growing Cohort'
            WHEN ltv_slope < 0 AND ltv_predictability > 0.6 THEN 'Declining Cohort'
            ELSE 'Unstable Pattern'
        END as cohort_health,
        (ltv_intercept + ltv_slope * 36) * avg_cohort_size as projected_36mo_cohort_revenue
    FROM cohort_models
)

SELECT
    cohort_month,
    ROUND(cohort_total_revenue, 0) as revenue_to_date,
    ROUND(avg_cohort_size, 0) as cohort_size,
    ROUND(projected_36mo_value, 2) as projected_ltv_36mo,
    ROUND(ltv_predictability, 3) as model_r2,
    cohort_health,
    ROUND(projected_36mo_cohort_revenue, 0) as projected_cohort_revenue,
    CASE
        WHEN cohort_health = 'Growing Cohort' THEN 'Replicate Acquisition Strategy'
        WHEN cohort_health = 'Declining Cohort' THEN 'Improve Retention Programs'
        ELSE 'Monitor Closely'
    END as strategic_action
FROM cohort_projections
WHERE cohort_month >= '2023-01-01'
ORDER BY cohort_month DESC;
```

## A/B Test Analysis

### Comprehensive A/B Test Evaluation

**Purpose**: Rigorously analyze A/B test results with statistical significance testing, confidence intervals, and business impact assessment to make data-driven launch decisions.

**Scenario**: You ran a pricing test with 1,000 users (500 control, 500 treatment). Treatment group saw higher conversion and revenue, but you need to confirm the difference is statistically significant before launching.

**How it works**:
1. **Generate experiment data**: Create realistic A/B test data where variant B has true improvements (3% higher conversion, $7 higher revenue per user)
2. **Descriptive statistics**: Calculate averages and standard deviations by variant
3. **Statistical testing**: Run regression of `metric ~ treatment_indicator` where coefficient = treatment effect
4. **Significance assessment**: Compute t-statistics and check if `|t| > 1.96` (p < 0.05)
5. **Calculate confidence intervals**: Use `coefficient ± 1.96 * std_error` for 95% CI
6. **Business decision**: Recommend launch if statistically significant and practically meaningful

**Steps in the query**:
- **experiment_data**: Joins A/B test results, creates treatment indicator (0=A, 1=B)
- **variant_summary**: Descriptive stats by variant (sample size, means, std devs)
- **conversion_test & revenue_test**: Regression-based hypothesis tests using actual data
- **conversion_significance & revenue_significance**: Compute t-stats and p-values
- **impact_analysis**: Combines stats with business metrics (absolute lift, relative lift %)
- **Final output**: Clear recommendation (Launch/Keep Control/Extend Test) with confidence intervals

**Key techniques**:
- Regression for difference-in-means testing (treatment coefficient = effect size)
- Proper statistical inference with t-statistics and confidence intervals
- Sample size adequacy check (warns if underpowered)
- Business interpretation layer on top of statistical results

**When to use**: Any A/B test, multivariate test, or controlled experiment where you need rigorous statistical validation before making product/business decisions.

```sql
-- Create sample A/B test data
CREATE OR REPLACE TABLE ab_test_results AS
SELECT
    'pricing_test_2024_q1' as experiment_id,
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END as variant,
    (CASE WHEN i % 2 = 0 THEN 0.12 ELSE 0.15 END +  -- Base conversion: B is 3% better
     RANDOM() * 0.05)::DOUBLE as conversion_rate,
    (CASE WHEN i % 2 = 0 THEN 45.0 ELSE 52.0 END +  -- Base revenue: B is $7 better
     RANDOM() * 10)::DOUBLE as revenue_per_user,
    (CASE WHEN i % 2 = 0 THEN 7.5 ELSE 8.2 END +    -- Base engagement: B is 0.7 better
     RANDOM() * 2)::DOUBLE as engagement_score
FROM range(1, 1001) t(i);

-- A/B test analysis with full statistical validation
WITH experiment_data AS (
    SELECT
        variant,
        CASE WHEN variant = 'B' THEN 1.0 ELSE 0.0 END as treatment,
        conversion_rate::DOUBLE as conversion,
        revenue_per_user::DOUBLE as revenue,
        engagement_score::DOUBLE as engagement
    FROM ab_test_results
    WHERE experiment_id = 'pricing_test_2024_q1'
),

-- Overall metrics by variant
variant_summary AS (
    SELECT
        variant,
        COUNT(*) as sample_size,
        AVG(conversion) as avg_conversion,
        AVG(revenue) as avg_revenue,
        AVG(engagement) as avg_engagement,
        STDDEV(conversion) as std_conversion,
        STDDEV(revenue) as std_revenue
    FROM experiment_data
    GROUP BY variant
),

-- Statistical significance test for conversion rate using actual data
-- Regression of conversion on treatment indicator (0=A, 1=B)
-- Coefficient = treatment effect (B - A)
conversion_test AS (
    SELECT
        (ols_fit_agg(conversion, treatment)).coefficient as treatment_effect,
        (ols_fit_agg(conversion, treatment)).std_error as std_error,
        (ols_fit_agg(conversion, treatment)).r2 as r_squared,
        COUNT(*) as n_obs
    FROM experiment_data
),

-- Statistical significance test for revenue using actual data
revenue_test AS (
    SELECT
        (ols_fit_agg(revenue, treatment)).coefficient as treatment_effect,
        (ols_fit_agg(revenue, treatment)).std_error as std_error,
        (ols_fit_agg(revenue, treatment)).r2 as r_squared,
        COUNT(*) as n_obs
    FROM experiment_data
),

-- Compute t-statistics and approximate p-values
-- t = coefficient / std_error
-- For large n, |t| > 1.96 implies p < 0.05 (two-tailed)
conversion_significance AS (
    SELECT
        treatment_effect,
        std_error,
        r_squared,
        treatment_effect / std_error as t_stat,
        ABS(treatment_effect / std_error) > 1.96 as is_significant,
        -- 95% confidence interval
        treatment_effect - 1.96 * std_error as ci_lower,
        treatment_effect + 1.96 * std_error as ci_upper
    FROM conversion_test
),

revenue_significance AS (
    SELECT
        treatment_effect,
        std_error,
        r_squared,
        treatment_effect / std_error as t_stat,
        ABS(treatment_effect / std_error) > 1.96 as is_significant,
        treatment_effect - 1.96 * std_error as ci_lower,
        treatment_effect + 1.96 * std_error as ci_upper
    FROM revenue_test
),

-- Calculate business impact using actual test results
impact_analysis AS (
    SELECT
        'Conversion Rate' as metric,
        ROUND((SELECT avg_conversion FROM variant_summary WHERE variant = 'A') * 100, 2) as control_value,
        ROUND((SELECT avg_conversion FROM variant_summary WHERE variant = 'B') * 100, 2) as treatment_value,
        ROUND(cs.treatment_effect * 100, 2) as absolute_lift,
        ROUND((cs.treatment_effect / (SELECT avg_conversion FROM variant_summary WHERE variant = 'A')) * 100, 2) as relative_lift_pct,
        ROUND(2 * (1 - 0.975), 4) as p_value_approx,  -- Approximate p-value for |t| > 1.96
        cs.is_significant,
        ROUND(cs.ci_lower * 100, 2) as ci_lower,
        ROUND(cs.ci_upper * 100, 2) as ci_upper
    FROM conversion_significance cs
    UNION ALL
    SELECT
        'Revenue per User' as metric,
        ROUND((SELECT avg_revenue FROM variant_summary WHERE variant = 'A'), 2),
        ROUND((SELECT avg_revenue FROM variant_summary WHERE variant = 'B'), 2),
        ROUND(rs.treatment_effect, 2),
        ROUND((rs.treatment_effect / (SELECT avg_revenue FROM variant_summary WHERE variant = 'A')) * 100, 2),
        ROUND(2 * (1 - 0.975), 4),
        rs.is_significant,
        ROUND(rs.ci_lower, 2),
        ROUND(rs.ci_upper, 2)
    FROM revenue_significance rs
),

-- Statistical power analysis (simplified)
power_analysis AS (
    SELECT
        (SELECT sample_size FROM variant_summary WHERE variant = 'A') as control_n,
        (SELECT sample_size FROM variant_summary WHERE variant = 'B') as treatment_n,
        CASE
            WHEN (SELECT sample_size FROM variant_summary WHERE variant = 'A') >= 1000 THEN 'Adequate'
            WHEN (SELECT sample_size FROM variant_summary WHERE variant = 'A') >= 500 THEN 'Marginal'
            ELSE 'Insufficient'
        END as sample_size_assessment
)

-- Final recommendation based on actual experiment data
SELECT
    ia.metric,
    ia.control_value,
    ia.treatment_value,
    ia.absolute_lift,
    ia.relative_lift_pct || '%' as relative_lift,
    ia.p_value_approx as p_value,
    ia.is_significant,
    '[' || ia.ci_lower || ', ' || ia.ci_upper || ']' as confidence_interval_95,
    CASE
        WHEN ia.is_significant AND ia.absolute_lift > 0 THEN 'Launch Treatment'
        WHEN ia.is_significant AND ia.absolute_lift < 0 THEN 'Keep Control'
        WHEN NOT ia.is_significant AND ABS(ia.absolute_lift) < 0.01 THEN 'No Meaningful Difference'
        ELSE 'Inconclusive - Extend Test'
    END as recommendation,
    pa.sample_size_assessment
FROM impact_analysis ia
CROSS JOIN power_analysis pa;
```

## Causal Analysis

### Difference-in-Differences Estimation

**Purpose**: Estimate causal effects of interventions using observational data by comparing treatment and control groups before and after an intervention, controlling for time trends.

**Scenario**: You rolled out a new store layout to 5 stores (treatment group) starting May 20th, while 5 stores kept the old layout (control group). You want to estimate the causal effect of the new layout on weekly sales.

**How it works - The DID Logic**:
- **Treatment group change**: New layout stores saw sales increase from pre to post period
- **Control group change**: Old layout stores also saw some increase (general time trend)
- **DID estimate**: Treatment effect = (Treatment change) - (Control change)
- This removes confounding time trends and isolates the causal effect of the layout

**Steps in the query**:
1. **Create weekly sales data** (10 stores × 52 weeks = 520 observations) with embedded treatment effect
2. **Define indicators**:
   - `treatment_group`: 1 for new layout stores, 0 for control
   - `post_period`: 1 after May 20, 0 before
   - `treatment_post`: Interaction term (1 only for treatment stores in post period)
3. **Run regression**: `sales ~ treatment_post` where coefficient = causal effect
4. **Validate with manual calculation**: Compute DID manually as verification
5. **Significance testing**: Check if effect is statistically significant (|t| > 1.96)

**Output interpretation**:
- **DID Regression Estimate**: Causal effect with standard error and confidence interval
- **Manual DID Calculation**: Shows treatment and control changes separately for transparency
- Both should match, confirming the regression correctly estimates the causal effect

**Key techniques**:
- Difference-in-differences framework for causal inference with observational data
- Uses regression with interaction terms to estimate treatment effects
- Provides both regression and manual calculations for pedagogical clarity
- Demonstrates how to structure quasi-experimental analyses in SQL

**When to use**: Policy evaluation, marketing interventions, product rollouts - any situation where you have pre/post data for treatment and control groups but couldn't randomize assignment. Common in economics, public policy, and business analytics.

```sql
-- Create sample weekly store sales data
CREATE OR REPLACE TABLE weekly_store_sales AS
SELECT
    store_id,
    DATE '2024-01-01' + INTERVAL (week_num * 7) DAY as week_date,
    (10000 +                                          -- Base sales
     CASE WHEN store_id <= 105 THEN 500 ELSE 0 END + -- Treatment group baseline
     CASE WHEN week_num >= 20 THEN 300 ELSE 0 END +  -- Time trend
     CASE WHEN store_id <= 105 AND week_num >= 20    -- Treatment effect (DID)
          THEN 1200 ELSE 0 END +
     RANDOM() * 500)::DOUBLE as sales                -- Random noise
FROM
    (SELECT unnest([101, 102, 103, 104, 105, 201, 202, 203, 204, 205]) as store_id) stores
    CROSS JOIN
    (SELECT i as week_num FROM range(0, 52) t(i)) weeks;

-- Difference-in-differences analysis using actual data
WITH store_data AS (
    SELECT
        store_id,
        week_date,
        sales::DOUBLE as sales,
        -- Treatment indicator (stores that got new layout)
        CASE WHEN store_id <= 105 THEN 1.0 ELSE 0.0 END as treatment_group,
        -- Post-intervention period (week 20 = approximately June 1)
        CASE WHEN week_date >= '2024-05-20' THEN 1.0 ELSE 0.0 END as post_period,
        -- Interaction term (DID estimator)
        (CASE WHEN store_id <= 105 THEN 1.0 ELSE 0.0 END) *
        (CASE WHEN week_date >= '2024-05-20' THEN 1.0 ELSE 0.0 END) as treatment_post
    FROM weekly_store_sales
),

-- Simple DID using treatment_post indicator
-- Coefficient on treatment_post = causal effect estimate
did_estimate AS (
    SELECT
        (ols_fit_agg(sales, treatment_post)).coefficient as did_coefficient,
        (ols_fit_agg(sales, treatment_post)).std_error as std_error,
        (ols_fit_agg(sales, treatment_post)).r2 as r_squared,
        COUNT(*) as n_obs
    FROM store_data
),

-- Compute significance
did_significance AS (
    SELECT
        did_coefficient as causal_effect,
        std_error,
        r_squared,
        did_coefficient / std_error as t_statistic,
        ABS(did_coefficient / std_error) > 1.96 as is_significant,
        did_coefficient - 1.96 * std_error as ci_lower,
        did_coefficient + 1.96 * std_error as ci_upper
    FROM did_estimate
),

-- Average treatment effects by period
descriptive_stats AS (
    SELECT
        CASE WHEN treatment_group = 1 THEN 'Treatment' ELSE 'Control' END as group_name,
        CASE WHEN post_period = 1 THEN 'Post' ELSE 'Pre' END as period,
        AVG(sales) as avg_sales,
        COUNT(*) as n_weeks
    FROM store_data
    GROUP BY treatment_group, post_period
),

-- Calculate parallel trends check (descriptive)
parallel_trends AS (
    SELECT
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Treatment' AND period = 'Pre') as treatment_pre,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Control' AND period = 'Pre') as control_pre,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Treatment' AND period = 'Post') as treatment_post,
        (SELECT avg_sales FROM descriptive_stats WHERE group_name = 'Control' AND period = 'Post') as control_post
),

-- Manual DID calculation for verification
manual_did AS (
    SELECT
        treatment_post - treatment_pre as treatment_diff,
        control_post - control_pre as control_diff,
        (treatment_post - treatment_pre) - (control_post - control_pre) as did_manual
    FROM parallel_trends
)

-- Final results combining regression estimates and descriptive stats
SELECT
    'DID Regression Estimate' as analysis_type,
    'Causal Effect' as metric,
    ROUND(ds.causal_effect, 2)::VARCHAR as value,
    ROUND(ds.std_error, 2)::VARCHAR as std_error,
    CASE WHEN ds.is_significant THEN 'Yes' ELSE 'No' END as significant,
    '[' || ROUND(ds.ci_lower, 2)::VARCHAR || ', ' || ROUND(ds.ci_upper, 2)::VARCHAR || ']' as ci_95,
    CASE
        WHEN ds.is_significant AND ds.causal_effect > 0 THEN
            'New layout increased sales by $' || ROUND(ds.causal_effect, 0)::VARCHAR || ' per week (causal)'
        WHEN ds.is_significant AND ds.causal_effect < 0 THEN
            'New layout decreased sales by $' || ABS(ROUND(ds.causal_effect, 0))::VARCHAR || ' per week (causal)'
        ELSE 'No significant causal effect detected'
    END as interpretation
FROM did_significance ds

UNION ALL

SELECT
    'Manual DID Calculation',
    'Treatment Effect',
    ROUND(md.treatment_diff, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Treatment group changed by $' || ROUND(md.treatment_diff, 0)::VARCHAR
FROM manual_did md

UNION ALL

SELECT
    'Manual DID Calculation',
    'Control Effect',
    ROUND(md.control_diff, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Control group changed by $' || ROUND(md.control_diff, 0)::VARCHAR
FROM manual_did md

UNION ALL

SELECT
    'Manual DID Calculation',
    'DID (Manual)',
    ROUND(md.did_manual, 2)::VARCHAR,
    NULL,
    NULL,
    NULL,
    'Manual DID = ' || ROUND(md.did_manual, 0)::VARCHAR || ' (should match regression estimate)'
FROM manual_did md

ORDER BY analysis_type DESC, metric;
```

## Production Deployment Patterns

### Materialized Model Results

**Purpose**: Pre-compute and cache regression model results in a table for fast repeated querying, avoiding expensive recalculations.

**Scenario**: You run product-level price elasticity models daily for 1,000 products. Dashboard users need instant access to coefficients and R² values. Instead of refitting models on every query, cache the results.

**How it works**:
1. **Fit models once**: Run `ols_fit_agg` grouped by product on last 365 days of data
2. **Extract key metrics**: Store coefficients, R² values, training period, timestamp
3. **Materialize to table**: Save results to `model_results_cache` table
4. **Fast retrieval**: Applications query the cache table instead of refitting models
5. **Refresh periodically**: Update cache on a schedule (daily, weekly, etc.)

**Benefits**:
- **Performance**: Querying cache is 100x+ faster than refitting models
- **Consistency**: All users see same model version at same time
- **Auditability**: Track model evolution over time via timestamps
- **Resource efficiency**: Fit once, query many times

**Steps in the query**:
- **source_data**: Filters to last 365 days for training window
- **product_models**: Fits separate models for each product using `ols_fit_agg`
- **CREATE TABLE**: Materializes results with metadata (training dates, update time)
- **Query cached models**: Fast retrieval without recalculation

**When to use**: Production environments where models are queried frequently (dashboards, APIs, reporting) but underlying data changes slowly (daily/weekly). Essential for real-time applications that need sub-second response times.

```sql
-- Create materialized model table
CREATE TABLE IF NOT EXISTS model_results_cache AS
WITH source_data AS (
    SELECT
        product_id,
        date,
        sales::DOUBLE as y,
        price::DOUBLE as x1,
        advertising::DOUBLE as x2,
        competition::DOUBLE as x3
    FROM daily_product_data
    WHERE date >= CURRENT_DATE - INTERVAL '365 days'
),

product_models AS (
    SELECT
        product_id,
        ols_fit_agg(y, x1) as model_x1,
        ols_fit_agg(y, x2) as model_x2,
        ols_fit_agg(y, x3) as model_x3,
        COUNT(*) as data_points,
        MIN(date) as training_start,
        MAX(date) as training_end,
        CURRENT_TIMESTAMP as model_update_time
    FROM source_data
    GROUP BY product_id
    HAVING COUNT(*) >= 30
)

SELECT
    product_id,
    (model_x1).coefficient as price_coefficient,
    (model_x2).coefficient as advertising_coefficient,
    (model_x3).coefficient as competition_coefficient,
    (model_x1).r2 as r2_price,
    (model_x2).r2 as r2_advertising,
    (model_x3).r2 as r2_competition,
    data_points,
    training_start,
    training_end,
    model_update_time
FROM product_models;

-- Query cached models
SELECT
    product_id,
    price_coefficient as price_elasticity,
    advertising_coefficient as advertising_effectiveness,
    competition_coefficient as competition_impact,
    r2_price as model_quality_price,
    data_points,
    model_update_time
FROM model_results_cache
WHERE r2_price > 0.7
ORDER BY r2_price DESC;
```

### Automated Model Refresh

**Purpose**: Implement a systematic process to periodically retrain models with fresh data, ensuring predictions stay accurate as patterns evolve.

**Scenario**: Your price elasticity models were trained on 2023 data, but it's now mid-2024 and customer behavior has changed. You need to refresh models monthly without manual intervention.

**How it works**:
1. **Archive old models**: Save previous version to `model_results_archive` for comparison
2. **Clear cache**: Delete outdated results from `model_results_cache`
3. **Retrain on fresh data**: Fit new models using configurable lookback window (e.g., last 365 days)
4. **Update cache**: Populate with new coefficients and metrics
5. **Log refresh**: Record when models were updated and how many products included

**Procedure structure**:
```sql
CREATE PROCEDURE refresh_product_models(lookback_days INT DEFAULT 365)
```
- Accepts parameter for training window size
- Automatically handles archiving, training, caching, and logging
- Can be scheduled via cron, Airflow, or DuckDB scheduler

**Production workflow**:
- **Daily schedule**: `CALL refresh_product_models(365);`
- **Monitor drift**: Compare new R² to archived R² to detect model degradation
- **Alert on failures**: Track refresh log for missing products or errors

**When to use**: Any production ML system where data evolves over time. Models become stale as patterns shift, so regular retraining maintains accuracy. Common schedule: daily (fast-moving data), weekly (moderate), monthly (slow-changing).

```sql
-- Incremental model update procedure
CREATE OR REPLACE PROCEDURE refresh_product_models(lookback_days INT DEFAULT 365)
AS
BEGIN
    -- Archive old models
    INSERT INTO model_results_archive
    SELECT *, CURRENT_TIMESTAMP as archived_at
    FROM model_results_cache;

    -- Delete old cache
    DELETE FROM model_results_cache;

    -- Rebuild with fresh data
    INSERT INTO model_results_cache
    WITH source_data AS (
        SELECT
            product_id,
            sales::DOUBLE as y,
            price::DOUBLE as x1,
            advertising::DOUBLE as x2,
            competition::DOUBLE as x3
        FROM daily_product_data
        WHERE date >= CURRENT_DATE - INTERVAL lookback_days DAYS
    ),
    product_models AS (
        SELECT
            product_id,
            ols_fit_agg(y, x1) as model_x1,
            ols_fit_agg(y, x2) as model_x2,
            ols_fit_agg(y, x3) as model_x3,
            COUNT(*) as data_points,
            CURRENT_DATE - INTERVAL lookback_days DAYS as training_start,
            CURRENT_DATE as training_end,
            CURRENT_TIMESTAMP as model_update_time
        FROM source_data
        GROUP BY product_id
        HAVING COUNT(*) >= 30
    )
    SELECT
        product_id,
        (model_x1).coefficient as price_coefficient,
        (model_x2).coefficient as advertising_coefficient,
        (model_x3).coefficient as competition_coefficient,
        (model_x1).r2 as r2_price,
        (model_x2).r2 as r2_advertising,
        (model_x3).r2 as r2_competition,
        data_points,
        training_start,
        training_end,
        model_update_time
    FROM product_models;

    -- Log refresh
    INSERT INTO model_refresh_log VALUES (
        CURRENT_TIMESTAMP,
        lookback_days,
        (SELECT COUNT(*) FROM model_results_cache)
    );
END;

-- Schedule: Run daily
-- CALL refresh_product_models(365);
```

## Performance Optimization

### Large Dataset Processing

**Purpose**: Optimize regression analysis for large datasets (millions of rows) using partitioning, sampling, and efficient aggregation strategies.

**Scenario**: You have 10 million sales transactions and want to analyze trends by month and category. Fitting models on the full dataset at once would be slow and memory-intensive.

**Strategy 1: Partition and Aggregate**:
1. **Pre-aggregate to partitions**: Group data into monthly × category buckets
2. **Use LIST aggregation**: Collect values within each partition as arrays
3. **Fit models per partition**: Use `ols_fit_agg` on each partition separately
4. **Parallel processing**: DuckDB parallelizes across partitions automatically

**Benefits**:
- Reduces memory footprint (process chunks, not full dataset)
- Enables parallelization (independent partitions processed concurrently)
- Fast even on commodity hardware

**Strategy 2: Sample-Then-Scale**:
1. **Sample for exploration**: Use `USING SAMPLE 10 PERCENT` to fit models on subset
2. **Validate approach**: Check R² and coefficients on sample
3. **Scale to full data**: If promising, run on complete dataset
4. **Iterative refinement**: Quick feedback loop for model development

**When to use**:
- **Strategy 1**: Production pipelines with structured hierarchies (time × category, region × product)
- **Strategy 2**: Exploratory analysis, feature selection, model prototyping

**Performance tips**:
- Partition by dimensions you'll GROUP BY (month, category, region)
- Use aggregate functions (`ols_fit_agg`) over table functions when possible
- Filter early to reduce data scanned (WHERE date >= '2023-01-01')
- Create indexes on partition columns for faster grouping

```sql
-- Efficient processing of large datasets
-- Strategy 1: Partition by time/category
WITH monthly_partitions AS (
    SELECT
        DATE_TRUNC('month', date) as month,
        category,
        LIST(sales::DOUBLE) as y_values,
        LIST(marketing::DOUBLE) as x_values,
        COUNT(*) as n_obs
    FROM large_sales_table
    WHERE date >= '2023-01-01'
    GROUP BY DATE_TRUNC('month', date), category
),

monthly_models AS (
    SELECT
        month,
        category,
        -- Use aggregate functions directly on arrays
        ols_coeff_agg(UNNEST(y_values), UNNEST(x_values)) as coefficient,
        (SELECT (ols_fit_agg(UNNEST(y_values), UNNEST(x_values))).r2) as r2,
        n_obs
    FROM monthly_partitions
    WHERE n_obs >= 30
    GROUP BY month, category, y_values, x_values, n_obs
)

SELECT
    month,
    category,
    ROUND(coefficient, 2) as coefficient,
    ROUND(r2, 3) as r2,
    n_obs
FROM monthly_models
ORDER BY month DESC, category;

-- Strategy 2: Sample for exploration, full data for final model
WITH sample_data AS (
    SELECT *
    FROM large_table
    USING SAMPLE 10 PERCENT  -- Quick exploration
),
sample_model AS (
    SELECT
        ols_coeff_agg(y, x1) as coeff_x1,
        ols_coeff_agg(y, x2) as coeff_x2,
        (ols_fit_agg(y, x1)).r2 as r2_x1,
        (ols_fit_agg(y, x2)).r2 as r2_x2
    FROM sample_data
)
-- Validate on sample, then run on full data if promising
SELECT * FROM sample_model;
```

## Integration Patterns

### Export for External Tools

**Purpose**: Export regression model results to standard formats (CSV, Parquet) for use in other tools, dashboards, or ML pipelines.

**Scenario**: Your BI tool (Tableau, PowerBI) doesn't support in-database regression, but needs model coefficients for scoring. Or you want to hand off predictions to a Python application.

**What this demonstrates**:
1. **Model export**: Save coefficients, standard errors, p-values to CSV for external scoring
2. **Prediction export**: Save predictions with confidence intervals to Parquet for downstream systems
3. **Metadata tracking**: Include training timestamp and sample size for auditability

**Use cases**:
- **BI integration**: Export coefficients to CSV, import into Tableau for visualization
- **Model handoff**: Train in DuckDB, deploy in Python/R application
- **Data interchange**: Parquet for efficient columnar storage and cross-tool compatibility
- **Archival**: Save model snapshots for regulatory compliance or reproducibility

**File formats**:
- **CSV**: Human-readable, universal compatibility, good for small coefficient tables
- **Parquet**: Columnar, compressed, efficient for large prediction datasets
- **JSON**: Hierarchical data, APIs, configuration files

**When to use**: Any workflow where DuckDB is the analytical engine but downstream consumers need structured output (reporting, dashboards, applications, other data tools).

```sql
-- Export model coefficients for external scoring (using literal array sample)
COPY (
    WITH model AS (
        SELECT * FROM ols_inference(
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
            [[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.2, 2.2, 3.2, 4.2],
             [1.3, 2.3, 3.3, 4.3], [1.4, 2.4, 3.4, 4.4], [1.5, 2.5, 3.5, 4.5],
             [1.6, 2.6, 3.6, 4.6], [1.7, 2.7, 3.7, 4.7], [1.8, 2.8, 3.8, 4.8],
             [1.9, 2.9, 3.9, 4.9]],
            0.95,
            true
        )
    )
    SELECT
        variable,
        estimate as coefficient,
        std_error,
        p_value,
        CURRENT_TIMESTAMP as model_trained_at,
        10 as training_observations
    FROM model
) TO 'model_coefficients.csv' (HEADER, DELIMITER ',');

-- Export predictions with confidence intervals
COPY (
    SELECT
        customer_id,
        predicted as predicted_ltv,
        ci_lower as ltv_conservative,
        ci_upper as ltv_optimistic
    FROM prediction_results
) TO 'customer_predictions.parquet' (FORMAT PARQUET);
```

## Best Practices Summary

### 1. Always Validate Assumptions

**Purpose**: Systematically check regression assumptions (sample size, multicollinearity, normality, outliers) before deploying models to production.

**Why this matters**: Regression models make statistical assumptions. Violating them can lead to:
- Unreliable coefficients (multicollinearity)
- Invalid p-values (non-normality)
- Biased predictions (influential outliers)
- Underpowered tests (small sample size)

**Validation checklist**:
1. **Sample size**: ≥30 observations (bare minimum), preferably ≥100
2. **Multicollinearity**: VIF < 10 for all predictors
3. **Normality**: Residuals approximately normal (Jarque-Bera test p > 0.05)
4. **Outliers**: < 5% of observations should be influential (Cook's D > 0.5)

**Query structure**:
- Runs 4 diagnostic checks in parallel
- Each returns PASS/FAIL/WARN status
- Quick pre-deployment validation before pushing models to production

**When to use**: Always run this before deploying any regression model. Automate as part of CI/CD pipeline or model refresh procedure.

```sql
-- Check list before deploying model (using literal array examples)
WITH validation AS (
    SELECT
        'Sample Size' as check_name,
        CAST((SELECT COUNT(*) FROM data) AS VARCHAR) as value,
        CASE WHEN (SELECT COUNT(*) FROM data) >= 30 THEN 'PASS' ELSE 'FAIL' END as status
    UNION ALL
    SELECT
        'Multicollinearity',
        CAST(MAX(vif_value) AS VARCHAR),
        CASE WHEN MAX(vif_value) < 10 THEN 'PASS' ELSE 'FAIL' END
    FROM vif([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3], [1.4, 2.4, 3.4]])
    UNION ALL
    SELECT
        'Normality',
        CAST(p_value AS VARCHAR),
        CASE WHEN p_value > 0.05 THEN 'PASS' ELSE 'FAIL' END
    FROM normality_test([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25])
    UNION ALL
    SELECT
        'Outliers',
        CAST(COUNT(*) AS VARCHAR),
        CASE WHEN COUNT(*) < 0.05 * (SELECT COUNT(*) FROM data) THEN 'PASS' ELSE 'WARN' END
    FROM residual_diagnostics(
        [100.0, 110.0, 120.0, 130.0, 140.0],
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]],
        true, 2.5, 0.5
    ) WHERE is_influential
)
SELECT * FROM validation;
```

### 2. Monitor Model Drift

**Purpose**: Track model performance over time to detect when coefficients or predictions degrade, signaling need for retraining.

**Why this matters**: Models trained on historical data become stale as:
- Customer behavior evolves
- Market conditions change
- Seasonal patterns shift
- Competitor actions alter dynamics

**What to monitor**:
- **R² degradation**: Current R² < 90% of historical baseline → model deteriorating
- **Coefficient drift**: Slopes changing significantly → relationships evolving
- **Prediction error**: Increasing RMSE on holdout set → losing accuracy
- **Data distribution shifts**: Feature means/variances changing → different population

**Monitoring workflow**:
1. Archive model metrics after each training run
2. Compare new metrics to previous version
3. Flag "DEGRADED" status if R² drops >10%
4. Trigger alerts for manual investigation or automatic retraining

**When to use**: Production models that are periodically refreshed. Essential for maintaining forecast accuracy in dynamic environments.

```sql
-- Track model performance over time
CREATE TABLE model_performance_log AS
SELECT
    CURRENT_DATE as check_date,
    'product_sales_model' as model_name,
    r_squared as current_r2,
    (SELECT r_squared FROM model_results_archive
     WHERE model_name = 'product_sales_model'
     ORDER BY archived_at DESC LIMIT 1 OFFSET 1) as previous_r2,
    CASE
        WHEN r_squared < previous_r2 * 0.9 THEN 'DEGRADED'
        ELSE 'STABLE'
    END as status
FROM current_model;
```

### 3. Document Everything

**Purpose**: Maintain a model registry with metadata about training data, variables, performance, ownership, and refresh schedule for governance and reproducibility.

**Why this matters**: In production environments with many models:
- Teams need to know what models exist and what they predict
- Auditors need to verify model validity and training procedures
- Debugging requires understanding model provenance
- Compliance may require documented model lineage

**Metadata to track**:
- **Model identification**: ID, name, type (OLS, Ridge, WLS)
- **Training details**: Data source query, sample size, training date
- **Variables**: Dependent variable, list of independent variables
- **Performance**: R², RMSE, p-values, coefficients
- **Governance**: Business owner, use case, validation checks
- **Operations**: Refresh frequency, last refresh date

**Model registry benefits**:
- **Discovery**: Find existing models before building duplicates
- **Governance**: Track who owns what and for what purpose
- **Debugging**: Quickly identify stale or problematic models
- **Compliance**: Demonstrate model validation and monitoring

**When to use**: Any organization with multiple regression models in production. Essential for regulated industries (finance, healthcare) requiring model documentation.

```sql
-- Model metadata table
CREATE TABLE model_registry (
    model_id VARCHAR PRIMARY KEY,
    model_name VARCHAR,
    model_type VARCHAR,
    training_data_query TEXT,
    dependent_variable VARCHAR,
    independent_variables VARCHAR[],
    training_date TIMESTAMP,
    training_observations BIGINT,
    r_squared DOUBLE,
    coefficients DOUBLE[],
    business_owner VARCHAR,
    use_case TEXT,
    refresh_frequency VARCHAR,
    validation_checks TEXT
);
```

## Conclusion

These advanced use cases demonstrate:

1. **Multi-stage workflows** combining multiple statistical techniques
2. **Time-series analysis** with adaptive windows and regime detection
3. **Hierarchical analysis** across organizational levels
4. **Causal inference** with difference-in-differences
5. **Production patterns** for deployment and maintenance
6. **Performance optimization** for large-scale data

For more information:
- [Quick Start Guide](01_quick_start.md) - Getting started
- [Technical Guide](02_technical_guide.md) - Implementation details
- [Statistics Guide](03_statistics_guide.md) - Statistical theory
- [Business Guide](04_business_guide.md) - Business applications
