# Anofox Statistics - Business Guide

This guide provides practical business applications of regression analysis using the Anofox Statistics extension for DuckDB. All examples are illustrative and should be validated before making business decisions.

## Table of Contents

1. [Marketing Analytics](#marketing-analytics)
2. [Financial Analytics](#financial-analytics)
3. [Operational Analytics](#operational-analytics)
4. [Sales Analytics](#sales-analytics)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)

---

## Marketing Analytics

### Use Case 1: Marketing Mix Modeling

Determine the effectiveness of different marketing channels.

```sql
-- Sample marketing data
CREATE OR REPLACE TABLE marketing_data AS
SELECT * FROM (VALUES
    ('2024-01', 50000, 10000, 5000, 2000, 150000),
    ('2024-02', 55000, 12000, 6000, 2500, 165000),
    ('2024-03', 48000, 11000, 5500, 2200, 145000),
    ('2024-04', 60000, 15000, 7000, 3000, 180000),
    ('2024-05', 52000, 13000, 6500, 2800, 160000),
    ('2024-06', 58000, 14000, 7500, 3200, 175000),
    ('2024-07', 62000, 16000, 8000, 3500, 190000),
    ('2024-08', 55000, 14500, 7200, 3100, 170000),
    ('2024-09', 65000, 17000, 8500, 3800, 200000),
    ('2024-10', 68000, 18000, 9000, 4000, 210000),
    ('2024-11', 70000, 19000, 9500, 4200, 220000),
    ('2024-12', 75000, 20000, 10000, 4500, 235000)
) AS t(month, tv_spend, digital_spend, print_spend, radio_spend, revenue);

-- Analyze channel effectiveness
SELECT
    fit.coefficients[1] as tv_roi,
    fit.coefficients[2] as digital_roi,
    fit.coefficients[3] as print_roi,
    fit.coefficients[4] as radio_roi,
    fit.intercept as base_revenue,
    ROUND(fit.r_squared, 3) as r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(revenue::DOUBLE ORDER BY month),
        [
            array_agg(tv_spend::DOUBLE ORDER BY month),
            array_agg(digital_spend::DOUBLE ORDER BY month),
            array_agg(print_spend::DOUBLE ORDER BY month),
            array_agg(radio_spend::DOUBLE ORDER BY month)
        ],
        true, true, 0.95
    ) as fit
    FROM marketing_data
);
```

**Interpretation:**
- Each coefficient represents incremental revenue per dollar spent
- Higher coefficient = more effective channel
- Use for budget allocation optimization

### Use Case 2: Price Elasticity Analysis

Understand how price changes affect demand.

```sql
-- Sample pricing data
CREATE OR REPLACE TABLE pricing_data AS
SELECT
    (90 + i * 2)::DOUBLE as price,
    (1000 - i * 15 + (random() - 0.5) * 50)::DOUBLE as units_sold
FROM generate_series(1, 20) t(i);

-- Calculate price elasticity
SELECT
    fit.coefficients[1] as price_coefficient,
    fit.intercept as base_demand,
    fit.r_squared as model_fit,
    -- Elasticity at mean price
    fit.coefficients[1] * AVG(price) / AVG(units_sold) as price_elasticity
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(units_sold),
        [array_agg(price)]
    ) as fit
    FROM pricing_data
), pricing_data
GROUP BY fit.coefficients, fit.intercept, fit.r_squared;
```

**Interpretation:**
- Elasticity < -1: Elastic demand (price increase loses revenue)
- Elasticity > -1: Inelastic demand (price increase gains revenue)

### Use Case 3: Customer Lifetime Value Prediction

Predict customer value based on early behavior.

```sql
-- Sample customer data
CREATE OR REPLACE TABLE customer_data AS
SELECT
    'C' || LPAD(i::VARCHAR, 4, '0') as customer_id,
    (random() * 5 + 1)::DOUBLE as first_purchase_amount,
    (random() * 10 + 1)::DOUBLE as days_to_second_purchase,
    (random() * 5)::DOUBLE as support_tickets,
    (random() * 50 + first_purchase_amount * 8)::DOUBLE as ltv_12_months
FROM generate_series(1, 100) t(i);

-- Build LTV prediction model
SELECT
    fit.coefficients[1] as first_purchase_impact,
    fit.coefficients[2] as engagement_impact,
    fit.coefficients[3] as support_impact,
    fit.intercept as base_ltv,
    fit.r_squared as predictive_power
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(ltv_12_months),
        [
            array_agg(first_purchase_amount),
            array_agg(1.0 / days_to_second_purchase),  -- Engagement proxy
            array_agg(support_tickets)
        ],
        true, true, 0.95
    ) as fit
    FROM customer_data
);
```

---

## Financial Analytics

### Use Case 4: Portfolio Beta Calculation

Calculate stock beta relative to market.

```sql
-- Sample returns data
CREATE OR REPLACE TABLE returns_data AS
SELECT
    '2024-' || LPAD(i::VARCHAR, 2, '0') as month,
    ((random() - 0.5) * 10)::DOUBLE as market_return,
    ((random() - 0.5) * 15)::DOUBLE as stock_return
FROM generate_series(1, 36) t(i);

-- Calculate beta
SELECT
    fit.coefficients[1] as beta,
    fit.intercept as alpha,
    fit.r_squared as r_squared,
    fit.p_values[1] as beta_pvalue
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(stock_return ORDER BY month),
        [array_agg(market_return ORDER BY month)],
        true, true, 0.95
    ) as fit
    FROM returns_data
);
```

**Interpretation:**
- Beta > 1: Stock is more volatile than market
- Beta < 1: Stock is less volatile than market
- Alpha: Excess return above market expectation

### Use Case 5: Multi-Factor Model with Ridge Regression

Use Ridge regression to handle correlated factors.

```sql
-- Sample multi-factor data
CREATE OR REPLACE TABLE factor_data AS
SELECT
    i as period,
    ((random() - 0.5) * 10)::DOUBLE as market_factor,
    ((random() - 0.5) * 5)::DOUBLE as size_factor,
    ((random() - 0.5) * 5)::DOUBLE as value_factor,
    ((random() - 0.5) * 3)::DOUBLE as momentum_factor,
    ((random() - 0.5) * 12)::DOUBLE as portfolio_return
FROM generate_series(1, 60) t(i);

-- Ridge regression for factor exposures
SELECT
    fit.coefficients[1] as market_exposure,
    fit.coefficients[2] as size_exposure,
    fit.coefficients[3] as value_exposure,
    fit.coefficients[4] as momentum_exposure,
    fit.r_squared
FROM (
    SELECT anofox_stats_ridge_fit(
        array_agg(portfolio_return ORDER BY period),
        [
            array_agg(market_factor ORDER BY period),
            array_agg(size_factor ORDER BY period),
            array_agg(value_factor ORDER BY period),
            array_agg(momentum_factor ORDER BY period)
        ],
        0.5  -- Regularization parameter
    ) as fit
    FROM factor_data
);
```

### Use Case 6: Credit Risk Modeling

Predict default probability based on financial metrics.

```sql
-- Sample loan data
CREATE OR REPLACE TABLE loan_data AS
SELECT
    (random() * 50000 + 10000)::DOUBLE as loan_amount,
    (random() * 0.5 + 0.2)::DOUBLE as debt_to_income,
    (random() * 300 + 500)::DOUBLE as credit_score,
    (random() * 20 + 1)::DOUBLE as employment_years,
    CASE WHEN random() < 0.15 THEN 1.0 ELSE 0.0 END as defaulted
FROM generate_series(1, 500) t(i);

-- Logistic-like linear probability model
SELECT
    fit.coefficients[1] as loan_amount_effect,
    fit.coefficients[2] as dti_effect,
    fit.coefficients[3] as credit_score_effect,
    fit.coefficients[4] as employment_effect,
    fit.intercept as base_probability,
    fit.r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(defaulted),
        [
            array_agg(loan_amount / 10000),  -- Normalize
            array_agg(debt_to_income),
            array_agg(credit_score / 100),   -- Normalize
            array_agg(employment_years)
        ]
    ) as fit
    FROM loan_data
);
```

---

## Operational Analytics

### Use Case 7: Demand Forecasting

Predict inventory needs based on historical patterns.

```sql
-- Sample demand data
CREATE OR REPLACE TABLE demand_data AS
SELECT
    i as week,
    (i % 4 + 1)::DOUBLE as week_of_month,
    CASE WHEN i % 7 IN (6, 7) THEN 1.0 ELSE 0.0 END as is_weekend,
    (random() * 20)::DOUBLE as temperature,
    (100 + i * 2 + (random() - 0.5) * 30)::DOUBLE as units_sold
FROM generate_series(1, 52) t(i);

-- Build demand model
SELECT
    fit.coefficients[1] as week_of_month_effect,
    fit.coefficients[2] as weekend_effect,
    fit.coefficients[3] as temperature_effect,
    fit.intercept as base_demand,
    fit.r_squared
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(units_sold ORDER BY week),
        [
            array_agg(week_of_month ORDER BY week),
            array_agg(is_weekend ORDER BY week),
            array_agg(temperature ORDER BY week)
        ]
    ) as fit
    FROM demand_data
);
```

### Use Case 8: Adaptive Forecasting with RLS

Use RLS for real-time adaptive forecasting.

```sql
-- Streaming sensor data with regime change
CREATE OR REPLACE TABLE sensor_data AS
SELECT
    i as timestamp,
    i::DOUBLE as time_index,
    CASE
        WHEN i <= 50 THEN (2.0 * i + 10 + (random() - 0.5) * 5)::DOUBLE
        ELSE (3.0 * i - 40 + (random() - 0.5) * 5)::DOUBLE  -- Regime change at t=50
    END as measurement
FROM generate_series(1, 100) t(i);

-- RLS with forgetting factor to adapt to changes
SELECT
    timestamp,
    measurement,
    (anofox_stats_rls_fit_agg(measurement, [time_index::DOUBLE], 0.95) OVER (
        ORDER BY timestamp
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).coefficients[1] as adaptive_slope
FROM sensor_data
ORDER BY timestamp;
```

### Use Case 9: Quality Control Optimization

Identify factors affecting product quality.

```sql
-- Sample manufacturing data
CREATE OR REPLACE TABLE quality_data AS
SELECT
    (150 + random() * 20)::DOUBLE as temperature,
    (50 + random() * 10)::DOUBLE as pressure,
    (5 + random() * 2)::DOUBLE as cycle_time,
    (85 + (150 - temperature) * 0.5 + (55 - pressure) * 0.3 + (random() - 0.5) * 5)::DOUBLE as quality_score
FROM generate_series(1, 100) t(i);

-- Identify quality drivers
SELECT
    fit.coefficients[1] as temperature_impact,
    fit.coefficients[2] as pressure_impact,
    fit.coefficients[3] as cycle_time_impact,
    fit.r_squared,
    vif([[temperature], [pressure], [cycle_time]]) as multicollinearity_check
FROM (
    SELECT anofox_stats_ols_fit(
        array_agg(quality_score),
        [
            array_agg(temperature),
            array_agg(pressure),
            array_agg(cycle_time)
        ],
        true, true, 0.95
    ) as fit
    FROM quality_data
), quality_data
GROUP BY fit.coefficients, fit.intercept, fit.r_squared;
```

---

## Sales Analytics

### Use Case 10: Territory Performance Analysis

Compare sales performance across territories using grouped regression.

```sql
-- Sample territory data
CREATE OR REPLACE TABLE territory_data AS
SELECT
    CASE (i % 4)
        WHEN 0 THEN 'East'
        WHEN 1 THEN 'West'
        WHEN 2 THEN 'North'
        ELSE 'South'
    END as territory,
    (random() * 100 + 50)::DOUBLE as accounts,
    (random() * 50 + 20)::DOUBLE as calls_per_account,
    (random() * 10000 + accounts * 50 + calls_per_account * 100 + (random() - 0.5) * 2000)::DOUBLE as revenue
FROM generate_series(1, 200) t(i);

-- Per-territory analysis
SELECT
    territory,
    ROUND((anofox_stats_ols_fit_agg(revenue, [accounts, calls_per_account])).coefficients[1], 2) as revenue_per_account,
    ROUND((anofox_stats_ols_fit_agg(revenue, [accounts, calls_per_account])).coefficients[2], 2) as revenue_per_call,
    ROUND((anofox_stats_ols_fit_agg(revenue, [accounts, calls_per_account])).r_squared, 3) as r_squared,
    COUNT(*) as sample_size
FROM territory_data
GROUP BY territory
ORDER BY territory;
```

### Use Case 11: Sales Rep Effectiveness with WLS

Weight recent performance more heavily.

```sql
-- Sample sales rep data with recency
CREATE OR REPLACE TABLE rep_performance AS
SELECT
    'Rep' || (i % 10 + 1) as rep_id,
    (100 - i * 0.5)::DOUBLE as months_ago,
    (random() * 50 + 30)::DOUBLE as activities,
    (random() * 20000 + activities * 300)::DOUBLE as revenue
FROM generate_series(1, 100) t(i);

-- Weight recent data more heavily
SELECT
    rep_id,
    ROUND((anofox_stats_wls_fit_agg(
        revenue,
        [activities],
        1.0 / (months_ago + 1)  -- Higher weight for recent data
    )).coefficients[1], 2) as revenue_per_activity,
    ROUND((anofox_stats_wls_fit_agg(
        revenue,
        [activities],
        1.0 / (months_ago + 1)
    )).r_squared, 3) as r_squared
FROM rep_performance
GROUP BY rep_id
ORDER BY revenue_per_activity DESC;
```

---

## Interpreting Results

### Understanding R-squared

| R² Value | Interpretation |
|----------|----------------|
| 0.0 - 0.3 | Weak explanatory power |
| 0.3 - 0.6 | Moderate explanatory power |
| 0.6 - 0.8 | Good explanatory power |
| 0.8 - 1.0 | Strong explanatory power |

**Caution:** High R² doesn't imply causation!

### Understanding P-values

| P-value | Interpretation |
|---------|----------------|
| < 0.01 | Very strong evidence against null |
| 0.01 - 0.05 | Strong evidence against null |
| 0.05 - 0.10 | Weak evidence against null |
| > 0.10 | Insufficient evidence |

### Understanding Coefficients

```
Revenue = Base + β₁×Price + β₂×Advertising

β₁ = -50: Each $1 price increase reduces revenue by $50
β₂ = 2.5: Each $1 advertising spend increases revenue by $2.50
```

### Confidence Intervals

```sql
-- Get 95% confidence interval
SELECT
    fit.coefficients[1] as point_estimate,
    fit.ci_lower[1] as lower_bound,
    fit.ci_upper[1] as upper_bound
FROM (SELECT anofox_stats_ols_fit(y, x, true, true, 0.95) as fit FROM data);
```

**Interpretation:**
- Interval excludes 0: Effect is statistically significant
- Narrow interval: Precise estimate
- Wide interval: Uncertain estimate

---

## Best Practices

### 1. Check Statistical Significance

```sql
-- Only trust significant coefficients
SELECT
    CASE WHEN fit.p_values[1] < 0.05 THEN 'Significant' ELSE 'Not Significant' END as price_significance,
    fit.coefficients[1] as price_effect
FROM (SELECT anofox_stats_ols_fit(y, x, true, true, 0.95) as fit FROM data);
```

### 2. Validate Model Quality

```sql
-- Check R² and residual diagnostics
WITH model AS (
    SELECT anofox_stats_ols_fit(y_array, x_array) as fit FROM data
),
predictions AS (
    SELECT anofox_stats_predict(x_array, model.fit.coefficients, model.fit.intercept) as y_hat
    FROM data, model
)
SELECT
    model.fit.r_squared,
    (jarque_bera((residuals_diagnostics(y_array, y_hat)).raw)).p_value as normality_test
FROM model, predictions, data;
```

### 3. Use Confidence Intervals for Decisions

```sql
-- Conservative revenue projection
SELECT
    fit.coefficients[1] * projected_spend as expected_return,
    fit.ci_lower[1] * projected_spend as pessimistic_return,
    fit.ci_upper[1] * projected_spend as optimistic_return
FROM model_fit, (SELECT 10000 as projected_spend);
```

### 4. Consider Business Context

- Statistical significance ≠ Business significance
- A coefficient of 0.001 may be significant but meaningless
- Consider practical effect sizes

### 5. Monitor Model Performance

```sql
-- Track model accuracy over time
SELECT
    date_trunc('month', date) as month,
    SQRT(AVG((actual - predicted) * (actual - predicted))) as rmse,
    AVG(ABS(actual - predicted) / actual) * 100 as mape
FROM predictions
GROUP BY 1
ORDER BY 1;
```

### 6. Combine Multiple Models

```sql
-- Ensemble approach
SELECT
    (ols_prediction + ridge_prediction + wls_prediction) / 3 as ensemble_prediction
FROM (
    SELECT
        anofox_stats_predict(x, ols_coef, ols_int)[1] as ols_prediction,
        anofox_stats_predict(x, ridge_coef, ridge_int)[1] as ridge_prediction,
        anofox_stats_predict(x, wls_coef, wls_int)[1] as wls_prediction
    FROM models, new_data
);
```

---

## Summary

This guide covered practical applications of regression analysis for:
- **Marketing**: Channel effectiveness, price elasticity, CLV
- **Finance**: Beta calculation, factor models, credit risk
- **Operations**: Demand forecasting, quality control
- **Sales**: Territory analysis, rep effectiveness

Always validate results with domain expertise before making business decisions.
