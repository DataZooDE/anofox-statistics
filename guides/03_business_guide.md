# Business Guide

Practical business applications of regression analysis using the Anofox Statistics extension with SQL examples for common use cases.

## Introduction

This guide demonstrates regression analysis techniques for business use cases using the Anofox Statistics extension for DuckDB. Each section presents SQL implementations of statistical methods applied to specific business domains: marketing, finance, operations, and sales.

The guide covers five regression methods available in the extension:

- **OLS (Ordinary Least Squares)**: Standard linear regression for estimating relationships between variables
- **WLS (Weighted Least Squares)**: Accounts for heteroscedasticity by weighting observations differently
- **Ridge Regression**: Applies L2 regularization to handle multicollinearity in correlated predictors
- **RLS (Recursive Least Squares)**: Adapts coefficients sequentially for non-stationary time series data
- **Elastic Net**: Combines L1 and L2 penalties for variable selection and regularization

Each use case includes executable SQL code with sample data generation, statistical interpretation of results, and decision frameworks based on regression outputs. The examples use aggregate functions (`_fit_agg`) for GROUP BY and window operations, and table functions (`_fit`) for batch processing with array inputs.

All code examples execute directly in DuckDB after loading the extension. The SQL patterns demonstrate integration of regression analysis into data pipelines without requiring external statistical software.

**⚠️ Important Notice**: All examples in this guide are illustrative and for educational purposes. Results and recommendations must be validated and tested case-by-case for your specific business context before making decisions.

## Table of Contents

- [Important Note About Examples](#important-note-about-examples)
- [Marketing Analytics](#marketing-analytics)
  - [Use Case 1: Marketing Mix Modeling](#use-case-1-marketing-mix-modeling)
  - [Use Case 2: Price Elasticity Analysis](#use-case-2-price-elasticity-analysis)
  - [Use Case 3: Customer Lifetime Value Prediction](#use-case-3-customer-lifetime-value-prediction)
  - [Use Case 3b: Customer Segment Analysis with WLS Aggregate](#use-case-3b-customer-segment-analysis-with-wls-aggregate)
- [Financial Analytics](#financial-analytics)
  - [Use Case 4: Portfolio Beta Calculation](#use-case-4-portfolio-beta-calculation)
  - [Use Case 4b: Multi-Factor Portfolio Analysis with Ridge Aggregate](#use-case-4b-multi-factor-portfolio-analysis-with-ridge-aggregate)
  - [Use Case 5: Credit Risk Modeling](#use-case-5-credit-risk-modeling)
  - [Use Case 6: Revenue Forecasting with Trend Analysis](#use-case-6-revenue-forecasting-with-trend-analysis)
- [Operational Analytics](#operational-analytics)
  - [Use Case 7: Demand Forecasting for Inventory](#use-case-7-demand-forecasting-for-inventory)
  - [Use Case 7b: Adaptive Demand Forecasting with RLS Aggregate](#use-case-7b-adaptive-demand-forecasting-with-rls-aggregate)
  - [Use Case 8: Quality Control Process Optimization](#use-case-8-quality-control-process-optimization)
  - [Use Case 9: Employee Productivity Analysis](#use-case-9-employee-productivity-analysis)
- [Sales Analytics](#sales-analytics)
  - [Use Case 10: Territory Performance Analysis](#use-case-10-territory-performance-analysis)
  - [Use Case 11: Regional Sales Analysis with OLS Aggregate](#use-case-11-regional-sales-analysis-with-ols-aggregate)
- [Key Business Metrics & Interpretation](#key-business-metrics--interpretation)
  - [Understanding R² (R-Squared)](#understanding-r²-r-squared)
  - [Understanding P-Values](#understanding-p-values)
  - [Understanding Coefficients](#understanding-coefficients)
  - [Understanding Confidence Intervals](#understanding-confidence-intervals)
  - [Understanding Outliers & Influence](#understanding-outliers--influence)
- [ROI Analysis Framework](#roi-analysis-framework)
  - [Marketing Campaign ROI](#marketing-campaign-roi)
  - [Process Improvement ROI](#process-improvement-roi)
- [Best Practices for Business Users](#best-practices-for-business-users)
  - [1. Always Check Statistical Significance](#1-always-check-statistical-significance)
  - [2. Validate Model Quality](#2-validate-model-quality)
  - [3. Use Confidence Intervals for Risk Management](#3-use-confidence-intervals-for-risk-management)
  - [4. Consider Business Context](#4-consider-business-context)
  - [5. Monitor Model Performance Over Time](#5-monitor-model-performance-over-time)
  - [6. Combine Multiple Models](#6-combine-multiple-models)

## Important Note About Examples

**All examples below are copy-paste runnable!** Each example includes sample data creation.

**Working Patterns**:

- **Aggregate functions** (`anofox_statistics_ols_fit_agg`, etc.) work directly with table data - use these for GROUP BY analysis
- **Table functions** (`anofox_statistics_ols_fit`, `anofox_statistics_predict_ols`) require literal arrays - examples show small datasets with explicit arrays
- **Inference** is integrated into fit functions using `full_output=true` option
- All functions use positional parameters with MAP options (no `:=` syntax)

**To adapt for your tables**: Replace sample data creation with your actual tables. For table functions with large datasets, use the two-step approach: run `SELECT LIST(column) FROM table`, copy result, paste as literal array.

[↑ Go to Top](#business-guide)

## Marketing Analytics

### Use Case 1: Marketing Mix Modeling

**Business Question**: Which marketing channels drive the most sales?


```sql

-- Create sample marketing campaign data
CREATE OR REPLACE TABLE weekly_campaigns AS
SELECT
    i as week_id,
    2024 as year,
    (50000 + i * 2.8 * tv + i * 4.2 * digital + i * 0.3 * print + i * 1.5 * radio + RANDOM() * 5000)::DOUBLE as revenue,
    tv::DOUBLE as tv_spend,
    digital::DOUBLE as digital_spend,
    print::DOUBLE as print_spend,
    radio::DOUBLE as radio_spend
FROM (
    SELECT
        i,
        (10 + RANDOM() * 5)::DOUBLE as tv,
        (8 + RANDOM() * 4)::DOUBLE as digital,
        (5 + RANDOM() * 3)::DOUBLE as print,
        (6 + RANDOM() * 3)::DOUBLE as radio
    FROM range(1, 21) t(i)
);

-- Analyze relationship between marketing spend and revenue using aggregate function
SELECT
    'tv' as channel,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [tv_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [tv_spend], {'intercept': true})).r2, 3) as r_squared,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'digital' as channel,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [digital_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [digital_spend], {'intercept': true})).r2, 3) as r_squared,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'print' as channel,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [print_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [print_spend], {'intercept': true})).r2, 3) as r_squared,
    'Low Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'radio' as channel,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [radio_spend], {'intercept': true})).coefficients[1], 2) as roi,
    ROUND((anofox_statistics_ols_fit_agg(revenue, [radio_spend], {'intercept': true})).r2, 3) as r_squared,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
ORDER BY roi DESC;
```

**Business Interpretation**:

- **ROI (coefficient)**: For every $1 increase in channel spend, revenue increases by $X
- **R²**: How well the channel spend predicts revenue (higher is better)
- Univariate analysis shows individual channel impact

**Business Decision**: Prioritize channels with highest ROI. Digital and TV show strongest returns.

### Use Case 2: Price Elasticity Analysis

**Business Question**: How sensitive are customers to price changes?


```sql

-- Create sample sales transaction data
CREATE OR REPLACE TABLE sales_transactions AS
SELECT
    i as transaction_id,
    CURRENT_DATE - (RANDOM() * 180)::INT as transaction_date,
    category,
    price,
    quantity
FROM (
    SELECT i,
           CASE (i % 4)
               WHEN 0 THEN 'Electronics'
               WHEN 1 THEN 'Clothing'
               WHEN 2 THEN 'Groceries'
               ELSE 'Luxury Goods'
           END as category,
           CASE (i % 4)
               WHEN 0 THEN (100 + RANDOM() * 50)::DOUBLE  -- Electronics
               WHEN 1 THEN (30 + RANDOM() * 20)::DOUBLE   -- Clothing
               WHEN 2 THEN (5 + RANDOM() * 3)::DOUBLE     -- Groceries
               ELSE (500 + RANDOM() * 200)::DOUBLE        -- Luxury
           END as price,
           CASE (i % 4)
               WHEN 0 THEN (50 - (100 + RANDOM() * 50) * 0.25 + RANDOM() * 10)::DOUBLE  -- Electronics: elastic
               WHEN 1 THEN (80 - (30 + RANDOM() * 20) * 0.8 + RANDOM() * 15)::DOUBLE    -- Clothing: elastic
               WHEN 2 THEN (100 - (5 + RANDOM() * 3) * 0.5 + RANDOM() * 20)::DOUBLE     -- Groceries: inelastic
               ELSE (20 - (500 + RANDOM() * 200) * 0.01 + RANDOM() * 5)::DOUBLE         -- Luxury: inelastic
           END as quantity
    FROM range(1, 401) t(i)
);

-- Calculate price elasticity by product category
SELECT
    category,
    ROUND((anofox_statistics_ols_fit_agg(quantity, price)).coefficients[1], 3) as elasticity,
    ROUND((anofox_statistics_ols_fit_agg(quantity, price)).r2, 3) as predictability,
    CASE
        WHEN ABS((anofox_statistics_ols_fit_agg(quantity, price)).coefficients[1]) > 0.5 THEN 'Elastic'
        ELSE 'Inelastic'
    END as elasticity_type,
    CASE
        WHEN ABS((anofox_statistics_ols_fit_agg(quantity, price)).coefficients[1]) > 0.5 THEN 'Discount Strategy'
        ELSE 'Premium Pricing'
    END as pricing_recommendation
FROM sales_transactions
WHERE transaction_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY category
ORDER BY ABS(elasticity) DESC;
```

**Business Interpretation**:

- **Negative elasticity**: Lower prices → higher quantity (normal demand curve)
- **High |elasticity|**: Elastic demand - customers are price-sensitive
- **R²**: How well price predicts quantity

**Business Decision**: Use competitive pricing for elastic categories, premium pricing for inelastic categories.

### Use Case 3: Customer Lifetime Value Prediction

**Business Question**: Predict future customer value to guide acquisition spending.


```sql

-- Create sample customer data
CREATE OR REPLACE TABLE customer_summary AS
SELECT
    i as customer_id,
    (tenure * 45 + aov * 3 + freq * 80 + engagement * 500 + RANDOM() * 200)::DOUBLE as total_purchases,
    tenure,
    aov,
    freq,
    engagement,
    CURRENT_DATE - (12 + (RANDOM() * 24)::INT) * INTERVAL '1 month' as cohort_month
FROM (
    SELECT
        i,
        (6 + RANDOM() * 18)::DOUBLE as tenure,         -- months active: 6-24
        (30 + RANDOM() * 70)::DOUBLE as aov,           -- avg order value: $30-100
        (1 + RANDOM() * 5)::DOUBLE as freq,            -- purchases per month: 1-6
        (RANDOM() * 0.8)::DOUBLE as engagement         -- email engagement: 0-80%
    FROM range(1, 101) t(i)
);

-- Build CLV model using aggregate functions (works directly with table data)
SELECT
    'Model Coefficient: Tenure' as metric,
    ROUND((anofox_statistics_ols_fit_agg(total_purchases, tenure)).coefficients[1], 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: AOV' as metric,
    ROUND((anofox_statistics_ols_fit_agg(total_purchases, aov)).coefficients[1], 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: Frequency' as metric,
    ROUND((anofox_statistics_ols_fit_agg(total_purchases, freq)).coefficients[1], 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Quality (R²)' as metric,
    ROUND((anofox_statistics_ols_fit_agg(total_purchases, tenure)).r2, 3) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months';
```

**Business Interpretation**:

- **Coefficient**: Impact of each factor on customer lifetime value
- **R²**: How well features predict CLV
- **Tenure** and **frequency** typically show strongest CLV correlation

**Business Decision**: Focus acquisition on customers likely to have high tenure and purchase frequency.

### Use Case 3b: Customer Segment Analysis with WLS Aggregate

**Business Question**: How do customer segments differ in their spending patterns? Are some segments more reliable to analyze than others?

**Method**: Use `anofox_statistics_wls_agg` with GROUP BY to fit separate models per segment, weighting by customer value or data reliability.


```sql

-- Business Guide: Customer Lifetime Value by Segment (Weighted Analysis)
-- Weight analysis by customer value to focus on high-value relationships

-- Sample data: Customer cohorts with varying reliability and value
CREATE TEMP TABLE customer_cohorts AS
SELECT
    CASE
        WHEN i <= 30 THEN 'enterprise'
        WHEN i <= 60 THEN 'smb'
        ELSE 'startup'
    END as segment,
    i as customer_id,
    (i * 100)::DOUBLE as acquisition_cost,
    (1 + i / 20.0)::DOUBLE as tenure_months,
    CASE
        WHEN i <= 30 THEN (10000 + acquisition_cost * 0.8 + tenure_months * 1200 + random() * 1000)
        WHEN i <= 60 THEN (3000 + acquisition_cost * 0.6 + tenure_months * 500 + random() * 800)
        ELSE (800 + acquisition_cost * 0.4 + tenure_months * 200 + random() * 400)
    END::DOUBLE as lifetime_revenue,
    -- Weight by contract size (larger customers = more reliable data)
    CASE
        WHEN i <= 30 THEN 5.0  -- Enterprise: high weight
        WHEN i <= 60 THEN 2.0  -- SMB: medium weight
        ELSE 1.0               -- Startup: standard weight
    END as customer_value_weight
FROM generate_series(1, 90) as t(i);

-- Weighted analysis: Focus on high-value customer patterns
SELECT
    segment,
    -- Standard OLS (treats all customers equally)
    ols.coefficients[1] as ols_acq_cost_roi,
    ols.coefficients[2] as ols_tenure_value,
    ols.r2 as ols_r2,
    -- Weighted LS (emphasizes high-value customers)
    wls.coefficients[1] as wls_acq_cost_roi,
    wls.coefficients[2] as wls_tenure_value,
    wls.r2 as wls_r2,
    -- Business insights
    CASE
        WHEN wls.coefficients[1] > 1.0 THEN 'Positive ROI on acquisition'
        WHEN wls.coefficients[1] > 0.5 THEN 'Break-even on acquisition'
        ELSE 'Review acquisition strategy'
    END as acquisition_assessment,
    wls.coefficients[2] * 12 as annual_value_per_customer,
    wls.n_obs as customers_analyzed
FROM (
    SELECT
        segment,
        anofox_statistics_ols_fit_agg(
            lifetime_revenue,
            [acquisition_cost, tenure_months],
            {'intercept': true}
        ) as ols,
        anofox_statistics_wls_fit_agg(
            lifetime_revenue,
            [acquisition_cost, tenure_months],
            customer_value_weight,
            {'intercept': true}
        ) as wls
    FROM customer_cohorts
    GROUP BY segment
) sub
ORDER BY segment;

-- Calculate LTV:CAC ratio by segment
WITH ltv_analysis AS (
    SELECT
        segment,
        result.coefficients[1] as roi_per_dollar,
        result.coefficients[2] as monthly_value,
        result.intercept as base_value,
        cac.avg_cac
    FROM (
        SELECT
            segment,
            anofox_statistics_wls_fit_agg(
                lifetime_revenue,
                [acquisition_cost, tenure_months],
                customer_value_weight,
                {'intercept': true}
            ) as result
        FROM customer_cohorts
        GROUP BY segment
    ) sub
    JOIN (
        SELECT segment, AVG(acquisition_cost) as avg_cac
        FROM customer_cohorts
        GROUP BY segment
    ) cac USING (segment)
)
SELECT
    segment,
    avg_cac,
    monthly_value * 24 as estimated_24mo_ltv,
    (monthly_value * 24) / NULLIF(avg_cac, 0) as ltv_cac_ratio,
    CASE
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 3.0 THEN 'Excellent (LTV > 3x CAC)'
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 2.0 THEN 'Good (LTV > 2x CAC)'
        WHEN (monthly_value * 24) / NULLIF(avg_cac, 0) > 1.0 THEN 'Acceptable'
        ELSE 'Concerning - Review unit economics'
    END as segment_health
FROM ltv_analysis
ORDER BY ltv_cac_ratio DESC;
```

**Business Interpretation**:

- **Weighted coefficients**: Estimated while giving more influence to high-value or reliable customers
- **weighted_mse**: Model error accounting for customer weights
- **Segment differences**: Premium customers may show different behavior than budget customers

**Why use WLS here**:

- **High-value customers**: Weight by LTV - premium customers get more influence in the model
- **Data reliability**: Weight by transaction count - customers with more data points get higher weight
- **Heteroscedasticity**: Variance may differ across customer types (premium vs budget)

**Business Decision**:

- **Premium segment**: If income sensitivity is high, target promotions at income milestones
- **Budget segment**: If income sensitivity is low, focus on value messaging rather than income targeting
- **Resource allocation**: Prioritize strategies for segments with strong, reliable patterns (high R², low weighted_mse)

**Example Insights**:

```
Premium segment: income_coef = 0.15, weighted_mse = 250 → Strong income effect, reliable estimate
Budget segment: income_coef = 0.08, weighted_mse = 180 → Weaker income effect, tighter variance
```

[↑ Go to Top](#business-guide)

## Financial Analytics

### Use Case 4: Portfolio Beta Calculation

**Business Question**: Measure market sensitivity of each stock for risk management.


```sql

-- Create sample stock returns data
CREATE OR REPLACE TABLE daily_stock_returns AS
SELECT
    trade_date,
    stock_ticker,
    stock_return,
    market_return
FROM (
    SELECT
        CURRENT_DATE - (i % 504)::INT as trade_date,  -- 2 years of trading days
        ticker as stock_ticker,
        CASE
            WHEN ticker = 'TECH' THEN market_ret * 1.45 + (RANDOM() * 0.04 - 0.02)  -- High beta
            WHEN ticker = 'GROWTH' THEN market_ret * 1.18 + (RANDOM() * 0.03 - 0.015)  -- Above market
            WHEN ticker = 'INDEX' THEN market_ret * 0.98 + (RANDOM() * 0.02 - 0.01)  -- Market
            ELSE market_ret * 0.42 + (RANDOM() * 0.02 - 0.01)  -- Low beta (UTILITY)
        END::DOUBLE as stock_return,
        market_ret::DOUBLE as market_return
    FROM
        (SELECT unnest(['TECH', 'GROWTH', 'INDEX', 'UTILITY']) as ticker) tickers
        CROSS JOIN (
            SELECT
                i,
                (RANDOM() * 0.04 - 0.02)::DOUBLE as market_ret  -- Market returns: -2% to +2% daily
            FROM range(1, 505) t(i)
        ) dates
);

-- Calculate beta (market sensitivity) for each stock
SELECT
    stock_ticker,
    ROUND((anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1], 3) as beta,
    ROUND((anofox_statistics_ols_fit_agg(stock_return, market_return)).r2, 3) as r_squared,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 1.2 THEN 'High Risk'
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 0.8 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_category,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 1.0 THEN 'Aggressive'
        WHEN (anofox_statistics_ols_fit_agg(stock_return, market_return)).coefficients[1] > 0.5 THEN 'Moderate'
        ELSE 'Defensive'
    END as investor_suitability
FROM daily_stock_returns
WHERE trade_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY stock_ticker
ORDER BY beta DESC;
```

**Business Interpretation**:

- **Beta = 1.0**: Moves with market (average risk)
- **Beta > 1.0**: More volatile than market (higher risk, higher potential return)
- **Beta < 1.0**: Less volatile than market (lower risk, lower potential return)
- **R² > 0.7**: Strong correlation with market

**Example Output**:

```
┌──────────────┬──────┬───────────┬───────────────┬───────────────────────┐
│ stock_ticker │ beta │ r_squared │ risk_category │ investor_suitability  │
├──────────────┼──────┼───────────┼───────────────┼───────────────────────┤
│ TECH         │ 1.45 │ 0.823     │ High Risk     │ Aggressive            │
│ GROWTH       │ 1.18 │ 0.756     │ High Risk     │ Aggressive            │
│ INDEX        │ 0.98 │ 0.912     │ Medium Risk   │ Moderate              │
│ UTILITY      │ 0.42 │ 0.645     │ Low Risk      │ Defensive             │
└──────────────┴──────┴───────────┴───────────────┴───────────────────────┘
```

**Business Decision**: Balance portfolio with mix of high/low beta stocks based on risk tolerance.

### Use Case 4b: Multi-Factor Portfolio Analysis with Ridge Aggregate

**Business Question**: How do multiple risk factors (market, sector, value) affect each stock? Handle correlated factors without multicollinearity issues.

**Method**: Use `anofox_statistics_ridge_agg` with GROUP BY to fit factor models per stock with L2 regularization to handle correlated risk factors.


```sql

-- Business Guide: Portfolio Risk Management with Ridge Regression
-- Handle correlated securities in portfolio beta estimation

-- Sample data: Portfolio holdings with correlated market factors
CREATE TEMP TABLE portfolio_holdings AS
SELECT
    CASE
        WHEN i <= 30 THEN 'stock_tech_a'
        WHEN i <= 60 THEN 'stock_tech_b'
        ELSE 'stock_finance'
    END as ticker,
    i as trading_day,
    (random() * 0.04 - 0.02)::DOUBLE as return,
    (random() * 0.03 - 0.015)::DOUBLE as market_return,
    (random() * 0.035 - 0.0175)::DOUBLE as tech_sector_return,  -- Correlated with market
    (random() * 0.025 - 0.0125)::DOUBLE as value_factor,
    (random() * 0.03 - 0.015)::DOUBLE as momentum_factor
FROM generate_series(1, 90) as t(i);

-- Compare OLS vs Ridge for beta estimation with correlated factors
SELECT
    ticker,
    -- OLS (may have unstable coefficients due to multicollinearity)
    ols.coefficients[1] as ols_market_beta,
    ols.coefficients[2] as ols_sector_beta,
    ols.coefficients[3] as ols_value_beta,
    ols.coefficients[4] as ols_momentum_beta,
    ols.r2 as ols_r2,
    -- Ridge (stabilized coefficients)
    ridge.coefficients[1] as ridge_market_beta,
    ridge.coefficients[2] as ridge_sector_beta,
    ridge.coefficients[3] as ridge_value_beta,
    ridge.coefficients[4] as ridge_momentum_beta,
    ridge.r2 as ridge_r2,
    1.0 as lambda,
    -- Risk assessment
    CASE
        WHEN ridge.coefficients[1] > 1.2 THEN 'High systematic risk'
        WHEN ridge.coefficients[1] > 0.8 THEN 'Market-level risk'
        ELSE 'Defensive positioning'
    END as risk_profile
FROM (
    SELECT
        ticker,
        anofox_statistics_ols_fit_agg(
            return,
            [market_return, tech_sector_return, value_factor, momentum_factor],
            {'intercept': true}
        ) as ols,
        anofox_statistics_ridge_fit_agg(
            return,
            [market_return, tech_sector_return, value_factor, momentum_factor],
            {'lambda': 1.0, 'intercept': true}
        ) as ridge
    FROM portfolio_holdings
    GROUP BY ticker
) sub
ORDER BY ticker;

-- Calculate portfolio-level risk metrics
WITH stock_betas AS (
    SELECT
        ticker,
        result.coefficients[1] as market_beta,
        result.coefficients[2] as sector_beta,
        result.r2
    FROM (
        SELECT
            ticker,
            anofox_statistics_ridge_fit_agg(
                return,
                [market_return, tech_sector_return],
                {'lambda': 1.0, 'intercept': true}
            ) as result
        FROM portfolio_holdings
        GROUP BY ticker
    ) sub
)
SELECT
    ticker,
    market_beta,
    sector_beta,
    r2 as model_fit,
    -- Risk interpretation for portfolio management
    CASE
        WHEN market_beta > 1.5 THEN 'Aggressive - Consider hedging'
        WHEN market_beta > 1.0 THEN 'Growth-oriented'
        WHEN market_beta > 0.7 THEN 'Balanced'
        ELSE 'Defensive/Low volatility'
    END as investment_style,
    -- Expected volatility relative to market
    ROUND(market_beta * 100, 1) || '% of market volatility' as volatility_profile
FROM stock_betas
ORDER BY market_beta DESC;
```

**Business Interpretation**:

- **Market beta**: Stock's sensitivity to overall market movements
- **Sector beta**: Stock's sensitivity to sector-specific movements
- **Value factor**: Stock's tilt toward value vs growth stocks
- **Lambda (regularization)**: Controls shrinkage - stabilizes estimates when factors are correlated

**Why use Ridge here**:

- **Multicollinearity**: Market and sector returns are often correlated
- **Stability**: Ridge produces more stable factor exposures than OLS
- **Better out-of-sample**: Regularization prevents overfitting to historical patterns
- **Risk management**: More reliable factor exposures for hedging strategies

**Business Decision**:

- **Factor exposure**: Understand true drivers of each stock's returns
- **Portfolio construction**: Build factor-neutral or factor-tilted portfolios
- **Hedging**: Use stable factor estimates for risk management
- **Comparison**: Stocks with similar factor profiles can be treated as substitutes

**Choosing lambda**:

- **lambda = 0.1**: Light regularization (factors mostly uncorrelated)
- **lambda = 1.0**: Moderate regularization (standard choice)
- **lambda = 10.0**: Heavy regularization (factors highly correlated)

**Example Insights**:

```
TECH: market_beta = 1.2, sector_beta = 0.8, value = -0.3 → Growth stock, sector-heavy
UTIL: market_beta = 0.5, sector_beta = 0.3, value = 0.2 → Defensive stock, value tilt
```

### Use Case 5: Credit Risk Modeling

**Business Question**: Predict loan default risk to inform lending decisions.


```sql

-- Create sample loan data
CREATE OR REPLACE TABLE loans AS
SELECT
    i as loan_id,
    CURRENT_DATE - (RANDOM() * 365 * 3)::INT as origination_date,
    CASE WHEN RANDOM() < 0.15 THEN 1 ELSE 0 END as default_flag,  -- 15% default rate
    (650 + RANDOM() * 150)::DOUBLE as credit_score,  -- 650-800
    (0.15 + RANDOM() * 0.35)::DOUBLE as debt_to_income,  -- 15%-50%
    (0.60 + RANDOM() * 0.35)::DOUBLE as loan_to_value,  -- 60%-95%
    (1 + RANDOM() * 19)::DOUBLE as employment_years  -- 1-20 years
FROM range(1, 101) t(i);

-- Build default prediction model using aggregate functions
WITH risk_factors AS (
    SELECT
        'Credit Score' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, credit_score)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Debt-to-Income' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, debt_to_income)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Loan-to-Value' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, loan_to_value)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
    UNION ALL
    SELECT
        'Employment Years' as variable,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1], 5) as coefficient,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).residual_standard_error, 4) as std_error,
        CASE
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1] > 0 THEN 'Increases Risk'
            WHEN (anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficients[1] < 0 THEN 'Decreases Risk'
            ELSE 'No Effect'
        END as risk_impact,
        ROUND((anofox_statistics_ols_fit_agg(default_flag::DOUBLE, employment_years)).r2, 3) as model_quality
    FROM loans
    WHERE origination_date >= '2022-01-01'
)
SELECT *
FROM risk_factors
ORDER BY ABS(coefficient) DESC;
```

**Business Interpretation**:

- **Positive coefficient**: Factor increases default probability
- **Negative coefficient**: Factor decreases default probability
- **P-value < 0.05**: Factor is statistically significant predictor

**Business Decision**: Adjust lending criteria based on most significant risk factors.

### Use Case 6: Revenue Forecasting with Trend Analysis

**Business Question**: Forecast next quarter's revenue with confidence intervals.


```sql

-- Create sample quarterly revenue data
CREATE OR REPLACE TABLE quarterly_financials AS
SELECT
    i as quarter_id,
    (40000000 + i * 500000 + RANDOM() * 2000000)::DOUBLE as revenue  -- Growing from $40M with trend
FROM range(1, 21) t(i);

-- Time-series revenue forecasting using table function with literal arrays
-- For real tables: Run `SELECT LIST(revenue) FROM quarterly_financials` first,
-- then paste the result as a literal array below
WITH forecast AS (
    SELECT * FROM anofox_statistics_predict_ols(
        [40572849.0, 41233491.0, 42455123.0, 42789456.0, 43891234.0, 43234567.0, 44567890.0, 44123456.0, 45678901.0, 45234567.0, 46789012.0, 46345678.0, 47890123.0, 47456789.0, 48901234.0, 48567890.0, 49912345.0, 49678901.0, 50923456.0, 50789012.0]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0]]::DOUBLE[][],
        [[21.0], [22.0], [23.0], [24.0]]::DOUBLE[][],  -- Next 4 quarters
        0.90,                                           -- confidence_level
        'prediction',                                   -- interval_type
        true                                            -- intercept
    )
)
SELECT
    observation_id as quarter_offset,
    20 + observation_id as quarter_id,
    ROUND(predicted / 1000000, 2) as forecast_revenue_millions,
    ROUND(ci_lower / 1000000, 2) as conservative_estimate_millions,
    ROUND(ci_upper / 1000000, 2) as optimistic_estimate_millions,
    ROUND((ci_upper - ci_lower) / predicted * 100, 1) as uncertainty_pct
FROM forecast;
```

**Example Output**:

```
┌────────────────┬────────────┬──────────────────────────┬──────────────────────────────┬─────────────────────────────┬─────────────────┐
│ quarter_offset │ quarter_id │ forecast_revenue_millions│ conservative_estimate_millions│ optimistic_estimate_millions│ uncertainty_pct │
├────────────────┼────────────┼──────────────────────────┼──────────────────────────────┼─────────────────────────────┼─────────────────┤
│ 1              │ 21         │ 52.3                     │ 48.1                         │ 56.5                        │ 16.1            │
│ 2              │ 22         │ 54.8                     │ 49.8                         │ 59.8                        │ 18.2            │
│ 3              │ 23         │ 57.3                     │ 51.2                         │ 63.4                        │ 21.3            │
│ 4              │ 24         │ 59.8                     │ 52.4                         │ 67.2                        │ 24.7            │
└────────────────┴────────────┴──────────────────────────┴──────────────────────────────┴─────────────────────────────┴─────────────────┘
```

**Business Decision**: Plan for $52.3M revenue in Q21, with contingency plans for $48-56M range.

[↑ Go to Top](#business-guide)

## Operational Analytics

### Use Case 7: Demand Forecasting for Inventory

**Business Question**: Set inventory levels based on product demand predictions.


```sql

-- Create sample daily sales data
CREATE OR REPLACE TABLE daily_sales AS
SELECT
    i as sale_id,
    product_id,
    season,
    price,
    promotion_flag,
    competitor_price,
    units_sold
FROM (
    SELECT
        i,
        'PROD-' || ((i % 3) + 1) as product_id,
        CASE (i % 4)
            WHEN 0 THEN 'Winter'
            WHEN 1 THEN 'Spring'
            WHEN 2 THEN 'Summer'
            ELSE 'Fall'
        END as season,
        (50 + RANDOM() * 50)::DOUBLE as price,
        (RANDOM() < 0.3)::INT::DOUBLE as promotion_flag,  -- 30% promotions
        (45 + RANDOM() * 55)::DOUBLE as competitor_price,
        units_sold
    FROM (
        SELECT
            i,
            (100 - base_price * 0.8 + promotion * 15 + (competitor - 50) * 0.5 + RANDOM() * 20)::DOUBLE as units_sold,
            base_price,
            promotion,
            competitor
        FROM (
            SELECT
                i,
                (50 + RANDOM() * 50) as base_price,
                (RANDOM() < 0.3)::INT * 1.0 as promotion,
                (45 + RANDOM() * 55) as competitor
            FROM range(1, 301) t(i)
        )
    )
);

-- Analyze price sensitivity for each product/season combination
SELECT
    product_id,
    season,
    ROUND((anofox_statistics_ols_fit_agg(units_sold, price)).coefficients[1], 2) as price_sensitivity,
    ROUND((anofox_statistics_ols_fit_agg(units_sold, price)).r2, 3) as forecast_accuracy,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.8 THEN 'High Confidence'
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as forecast_reliability,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(units_sold, price)).r2 > 0.7 THEN 'Auto-Replenish'
        ELSE 'Manual Review'
    END as inventory_strategy,
    COUNT(*) as sample_size
FROM daily_sales
GROUP BY product_id, season
HAVING COUNT(*) >= 30  -- Minimum data for reliable estimates
ORDER BY forecast_accuracy DESC;
```

**Business Decision**: Automate replenishment for high-confidence products, manual review for others.

### Use Case 7b: Adaptive Demand Forecasting with RLS Aggregate

**Business Question**: How can we create forecasts that automatically adapt to changing demand patterns per product? Which products have the most volatile demand?

**Method**: Use `anofox_statistics_rls_agg` with GROUP BY to fit adaptive models per product that emphasize recent patterns over historical data.


```sql

-- Business Guide: Adaptive Demand Forecasting with RLS
-- Real-time demand prediction that adapts to market changes

-- Sample data: Product demand with evolving seasonality and trends
CREATE TEMP TABLE demand_history AS
SELECT
    CASE
        WHEN i <= 40 THEN 'product_seasonal'
        ELSE 'product_trending'
    END as product_id,
    i as week,
    -- Cyclical pattern changes over time
    CASE
        WHEN i <= 20 THEN (1000 + 200 * SIN(i * 0.5) + i * 10 + random() * 50)
        WHEN i <= 40 THEN (1200 + 300 * SIN(i * 0.5) + i * 15 + random() * 80)  -- Pattern shift
        ELSE (1500 + 100 * SIN(i * 0.5) + i * 25 + random() * 60)                -- New trend
    END::DOUBLE as actual_demand,
    -- Predictors: lagged demand and trend
    LAG(CASE
        WHEN i <= 20 THEN (1000 + 200 * SIN(i * 0.5) + i * 10 + random() * 50)
        WHEN i <= 40 THEN (1200 + 300 * SIN(i * 0.5) + i * 15 + random() * 80)
        ELSE (1500 + 100 * SIN(i * 0.5) + i * 25 + random() * 60)
    END, 1) OVER (ORDER BY i) as lagged_demand,
    i::DOUBLE as time_trend
FROM generate_series(1, 80) as t(i);

-- Compare static OLS vs adaptive RLS forecasting
SELECT
    product_id,
    -- Static OLS model (fixed coefficients)
    ols.coefficients[1] as ols_lag_coef,
    ols.coefficients[2] as ols_trend_coef,
    ols.r2 as ols_r2,
    -- Adaptive RLS model (adjusts to changes)
    rls_slow.coefficients[1] as rls_slow_lag_coef,
    rls_slow.coefficients[2] as rls_slow_trend_coef,
    rls_slow.r2 as rls_slow_r2,
    0.98 as ff_slow,
    -- Fast-adapting RLS
    rls_fast.coefficients[1] as rls_fast_lag_coef,
    rls_fast.coefficients[2] as rls_fast_trend_coef,
    rls_fast.r2 as rls_fast_r2,
    0.92 as ff_fast,
    -- Business insight
    CASE
        WHEN rls_fast.r2 > ols.r2 + 0.05 THEN 'RLS significantly better - demand pattern changing'
        WHEN rls_fast.r2 > ols.r2 THEN 'RLS slightly better - moderate changes'
        ELSE 'OLS sufficient - stable demand pattern'
    END as model_recommendation
FROM (
    SELECT
        product_id,
        anofox_statistics_ols_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'intercept': true}
        ) as ols,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.98, 'intercept': true}
        ) as rls_slow,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.92, 'intercept': true}
        ) as rls_fast
    FROM demand_history
    WHERE lagged_demand IS NOT NULL
    GROUP BY product_id
) sub;

-- Rolling window analysis: Track model adaptation
WITH recent_performance AS (
    SELECT
        product_id,
        week,
        actual_demand,
        anofox_statistics_rls_fit_agg(
            actual_demand,
            [lagged_demand, time_trend],
            {'forgetting_factor': 0.95, 'intercept': true}
        ) OVER (
            PARTITION BY product_id
            ORDER BY week
            ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) as rolling_model
    FROM demand_history
    WHERE lagged_demand IS NOT NULL
)
SELECT
    product_id,
    week,
    rolling_model.coefficients[1] as adaptive_lag_coefficient,
    rolling_model.r2 as rolling_r2,
    CASE
        WHEN rolling_model.r2 > 0.8 THEN 'High forecast confidence'
        WHEN rolling_model.r2 > 0.6 THEN 'Moderate forecast confidence'
        ELSE 'Low confidence - model recalibration needed'
    END as forecast_confidence
FROM recent_performance
WHERE week >= 60
ORDER BY product_id, week
LIMIT 15;

-- Business recommendation
SELECT
    'Forecasting Strategy' as topic,
    'Use RLS for products with evolving demand patterns, seasonal shifts, or trend changes. Lower forgetting factors (0.90-0.95) adapt faster but may be more volatile.' as guidance
UNION ALL
SELECT
    'When to Use',
    'RLS is ideal for: (1) Fast-moving consumer goods with changing preferences, (2) Tech products with rapid adoption curves, (3) Markets with frequent promotions or competitive dynamics.' as use_cases;
```

**Business Interpretation**:

- **Adaptive coefficients**: Final model emphasizes recent demand patterns, automatically adjusting to changes
- **forgetting_factor**: Controls adaptation speed - lower values = faster response to changes
- **Per-product models**: Each product gets its own adaptive forecast model
- **Recent emphasis**: Old data is exponentially down-weighted

**Why use RLS here**:

- **Non-stationary demand**: Product demand patterns change over time (seasonality, trends, market shifts)
- **Fast adaptation**: Detects and responds to demand regime changes quickly
- **Automatic updates**: No need to retrain models - they adapt as new data arrives
- **Relevant patterns**: Recent data often predicts better than old data for operational decisions

**Business Decision**:

- **Inventory planning**: Use adaptive forecasts for ordering decisions
- **Volatility assessment**: Products with high forgetting_factor needs indicate unstable demand
- **Promotional impact**: RLS automatically adjusts to post-promotion demand changes
- **New product launches**: Adapts quickly as demand patterns stabilize

**Choosing forgetting_factor**:

- **λ = 0.98-0.99**: Slow adaptation - stable products with gradual changes
- **λ = 0.95-0.97**: Moderate adaptation - standard operational forecasting
- **λ = 0.90-0.94**: Fast adaptation - volatile products or rapid market changes
- **λ < 0.90**: Very fast - new products or post-disruption recovery

**Example Insights**:

```
Product A: coef = 1.2, λ = 0.95 → Growing demand, moderate volatility
Product B: coef = 0.8, λ = 0.90 → Declining demand, high volatility - needs attention
Product C: coef = 1.0, λ = 0.98 → Stable demand, predictable pattern
```

**Operational Actions**:

- **High adaptation needs (λ < 0.93)**: Review demand drivers, increase safety stock
- **Stable patterns (λ > 0.97)**: Can use tighter inventory controls
- **Growing trends (coef > 1.0)**: Plan capacity increases
- **Declining trends (coef < 1.0)**: Review product positioning, consider promotions

### Use Case 8: Quality Control Process Optimization

**Business Question**: Identify which process parameters affect product defects.


```sql

-- Create sample manufacturing batch data
CREATE OR REPLACE TABLE production_batches AS
SELECT
    i as batch_id,
    CURRENT_DATE - ((RANDOM() * 90)::INT) as batch_date,
    temp::DOUBLE as temperature,
    pressure::DOUBLE as pressure,
    humidity::DOUBLE as humidity,
    speed::DOUBLE as line_speed,
    -- Defects increase with high temp, high speed, decrease with optimal pressure
    (2.0 + temp * 0.05 + speed * 0.03 - pressure * 0.02 + humidity * 0.01 + RANDOM() * 1.5)::DOUBLE as defect_rate
FROM (
    SELECT
        i,
        (180 + RANDOM() * 40) as temp,  -- 180-220°F
        (25 + RANDOM() * 10) as pressure,  -- 25-35 PSI
        (40 + RANDOM() * 30) as humidity,  -- 40-70%
        (50 + RANDOM() * 30) as speed  -- 50-80 units/min
    FROM range(1, 101) t(i)
);

-- Analyze impact of each process parameter on defect rates
WITH recent_batches AS (
    SELECT * FROM production_batches
    WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
),
parameter_impacts AS (
    SELECT
        'Temperature' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [temperature], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [temperature], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Pressure' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [pressure], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [pressure], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Humidity' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [humidity], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [humidity], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
    UNION ALL
    SELECT
        'Line Speed' as variable,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [line_speed], {'intercept': true})).coefficients[1], 4) as impact_on_defects,
        ROUND((anofox_statistics_ols_fit_agg(defect_rate, [line_speed], {'intercept': true})).r2, 3) as model_fit
    FROM recent_batches
),
impacts_materialized AS (
    SELECT * FROM parameter_impacts
)
SELECT
    variable,
    impact_on_defects,
    model_fit,
    CASE
        WHEN impact_on_defects > 0 THEN 'Increases Defects'
        WHEN impact_on_defects < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN ABS(impact_on_defects) > 0.05 AND impact_on_defects > 0 THEN 'Critical - Reduce'
        WHEN ABS(impact_on_defects) > 0.05 AND impact_on_defects < 0 THEN 'Beneficial - Increase'
        ELSE 'Low Impact'
    END as action_recommendation
FROM impacts_materialized
ORDER BY ABS(impact_on_defects) DESC;
```

**Business Decision**: Adjust process parameters that significantly affect defect rates.

### Use Case 9: Employee Productivity Analysis

**Business Question**: Understand factors driving team productivity.


```sql

-- Create sample employee productivity data
CREATE OR REPLACE TABLE employee_productivity AS
SELECT
    i as employee_id,
    department,
    training_hours,
    experience_years,
    team_size,
    output_per_hour
FROM (
    SELECT
        i,
        CASE (i % 4)
            WHEN 0 THEN 'Manufacturing'
            WHEN 1 THEN 'Customer Service'
            WHEN 2 THEN 'IT'
            ELSE 'Sales'
        END as department,
        (10 + RANDOM() * 30)::DOUBLE as training_hours,  -- 10-40 hours training
        (1 + RANDOM() * 14)::DOUBLE as experience_years,  -- 1-15 years
        (3 + RANDOM() * 7)::INT::DOUBLE as team_size,  -- 3-10 people
        output as output_per_hour
    FROM (
        SELECT
            i,
            (50 + training * 2.5 + experience * 1.8 + team * 0.5 + RANDOM() * 10)::DOUBLE as output,
            training,
            experience,
            team
        FROM (
            SELECT
                i,
                (10 + RANDOM() * 30) as training,
                (1 + RANDOM() * 14) as experience,
                (3 + RANDOM() * 7) as team
            FROM range(1, 201) t(i)
        )
    )
);

-- Analyze productivity drivers by department - focus on training impact
SELECT
    department,
    ROUND((anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1], 2) as training_impact,
    ROUND((anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).r2, 3) as model_fit,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 5.0 THEN 'High Training ROI'
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 2.0 THEN 'Medium Training ROI'
        ELSE 'Low Training ROI'
    END as training_effectiveness,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(output_per_hour, training_hours)).coefficients[1] > 3.0 THEN 'Increase Training Budget'
        ELSE 'Maintain Current Level'
    END as budget_recommendation,
    COUNT(*) as sample_size
FROM employee_productivity
GROUP BY department
ORDER BY training_impact DESC;
```

**Business Decision**: Allocate training budget to departments with highest ROI.

[↑ Go to Top](#business-guide)

## Sales Analytics

### Use Case 10: Territory Performance Analysis

**Business Question**: Which territories are underperforming and why?


```sql

-- Create sample monthly sales data by territory
CREATE OR REPLACE TABLE monthly_sales AS
SELECT
    territory_id,
    month_date,
    month_index,
    sales_amount
FROM (
    SELECT
        territory,
        DATE_TRUNC('month', CURRENT_DATE) - (i * INTERVAL '1 month') as month_date,
        i as month_index,
        CASE territory
            WHEN 'NORTH' THEN (600000 + i * 5000 + RANDOM() * 50000)::DOUBLE  -- Growing high performer
            WHEN 'SOUTH' THEN (550000 - i * 2000 + RANDOM() * 40000)::DOUBLE  -- Declining high performer
            WHEN 'EAST' THEN (300000 + i * 8000 + RANDOM() * 30000)::DOUBLE   -- Rising star
            ELSE (250000 - i * 1000 + RANDOM() * 25000)::DOUBLE                -- Declining low (WEST)
        END as sales_amount,
        territory as territory_id
    FROM
        (SELECT unnest(['NORTH', 'SOUTH', 'EAST', 'WEST']) as territory) territories
        CROSS JOIN range(0, 12) t(i)
);

-- Rolling 6-month sales trend by territory
WITH territory_trends AS (
    SELECT
        territory_id,
        month_date,
        sales_amount,
        (anofox_statistics_ols_fit_agg(sales_amount::DOUBLE, month_index::DOUBLE) OVER (
            PARTITION BY territory_id
            ORDER BY month_date
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        )).coefficients[1] as trend_coefficient
    FROM monthly_sales
    WHERE month_date >= CURRENT_DATE - INTERVAL '12 months'
),
territory_classification AS (
    SELECT
        territory_id,
        ROUND(AVG(sales_amount), 0) as avg_sales,
        ROUND(AVG(trend_coefficient), 2) as avg_trend,
        CASE
            WHEN AVG(trend_coefficient) > 1000 THEN 'Growing'
            WHEN AVG(trend_coefficient) > -1000 THEN 'Stable'
            ELSE 'Declining'
        END as performance_status,
        CASE
            WHEN AVG(sales_amount) > 500000 AND AVG(trend_coefficient) > 0 THEN 'Star Territory'
            WHEN AVG(sales_amount) > 500000 AND AVG(trend_coefficient) < 0 THEN 'Cash Cow - Monitor'
            WHEN AVG(sales_amount) < 500000 AND AVG(trend_coefficient) > 0 THEN 'Rising Star'
            ELSE 'Needs Intervention'
        END as strategic_category
    FROM territory_trends
    GROUP BY territory_id
)
SELECT
    territory_id,
    avg_sales,
    avg_trend as monthly_growth,
    performance_status,
    strategic_category,
    CASE
        WHEN strategic_category = 'Star Territory' THEN 'Maintain & Expand'
        WHEN strategic_category = 'Rising Star' THEN 'Invest in Growth'
        WHEN strategic_category = 'Cash Cow - Monitor' THEN 'Investigate Decline'
        ELSE 'Urgent Action Needed'
    END as management_action
FROM territory_classification
ORDER BY avg_sales DESC;
```

**Business Decision**: Prioritize intervention in declining territories, invest in rising stars.

### Use Case 11: Regional Sales Analysis with OLS Aggregate

**Business Question**: How does pricing elasticity differ across regions? Which regions are most price-sensitive?

**Method**: Use `anofox_statistics_ols_agg` with GROUP BY to fit separate price-demand models per region in a single query.


```sql

-- Business Guide: Regional Sales Performance Analysis
-- Analyze how pricing and promotions affect sales across different regions

-- Sample data: Multi-region sales with pricing and promotion data
CREATE TEMP TABLE regional_sales AS
SELECT
    CASE i % 4
        WHEN 0 THEN 'north'
        WHEN 1 THEN 'south'
        WHEN 2 THEN 'east'
        ELSE 'west'
    END as region,
    i as week,
    (15.0 + random() * 5.0)::DOUBLE as price,
    (1000 + random() * 500)::DOUBLE as promo_spend,
    (500 + i * 10 - 50 * (15.0 + random() * 5.0) + 0.8 * (1000 + random() * 500) + random() * 200)::DOUBLE as units_sold
FROM generate_series(1, 80) as t(i);

-- Analyze price elasticity and promotion effectiveness per region
SELECT
    region,
    -- Key business metrics
    result.coefficients[1] as price_elasticity,
    result.coefficients[2] as promo_roi,
    result.intercept as baseline_demand,
    result.r2,
    result.n_obs as weeks_analyzed,
    -- Business interpretation
    CASE
        WHEN result.coefficients[1] < -30 THEN 'Highly price-sensitive'
        WHEN result.coefficients[1] < -20 THEN 'Moderately price-sensitive'
        ELSE 'Low price sensitivity'
    END as price_sensitivity_category,
    CASE
        WHEN result.coefficients[2] > 1.0 THEN 'Strong promotion response'
        WHEN result.coefficients[2] > 0.5 THEN 'Moderate promotion response'
        ELSE 'Weak promotion response'
    END as promo_effectiveness,
    -- Strategic recommendations
    CASE
        WHEN result.coefficients[1] < -30 AND result.coefficients[2] > 1.0
            THEN 'Focus on promotions over pricing'
        WHEN result.coefficients[1] > -20 AND result.coefficients[2] < 0.5
            THEN 'Consider premium pricing strategy'
        ELSE 'Balanced price and promotion strategy'
    END as strategy_recommendation
FROM (
    SELECT
        region,
        anofox_statistics_ols_fit_agg(
            units_sold,
            [price, promo_spend],
            {'intercept': true}
        ) as result
    FROM regional_sales
    GROUP BY region
) sub
ORDER BY result.r2 DESC;

-- Calculate revenue impact of $1 price change
SELECT
    region,
    result.coefficients[1] as unit_change_per_dollar,
    avg_price,
    result.coefficients[1] * avg_price as revenue_impact_per_dollar_increase,
    CASE
        WHEN result.coefficients[1] * avg_price < -1.0 THEN 'Price decrease would increase revenue'
        ELSE 'Current pricing may be optimal'
    END as pricing_insight
FROM (
    SELECT
        region,
        anofox_statistics_ols_fit_agg(units_sold, [price], {'intercept': true}) as result
    FROM regional_sales
    GROUP BY region
) sub
JOIN (
    SELECT region, AVG(price) as avg_price
    FROM regional_sales
    GROUP BY region
) price_stats USING (region);
```

**Business Interpretation**:

- **Price elasticity coefficient**: How quantity responds to price changes in each region
- **Negative values**: Normal demand curve - higher price → lower demand
- **R²**: How well price predicts demand in each region
- **Regional differences**: Some regions may be more price-sensitive than others

**Business Decision**:

- **High |elasticity| regions**: Use competitive pricing strategies, run promotions carefully
- **Low |elasticity| regions**: Can support premium pricing, focus on value messaging
- **Poor fit (low R²)**: Price isn't the main driver - investigate other factors (competition, demographics)

**Example Insights**:

```
North region: elasticity = -1.2, R² = 0.85 → Price-sensitive, good fit
South region: elasticity = -0.4, R² = 0.72 → Less price-sensitive, moderate fit
```

[↑ Go to Top](#business-guide)

## Key Business Metrics & Interpretation

### Understanding R² (R-Squared)

**What it means**: Percentage of variation in outcome explained by your model.

- **R² = 0.95**: Model explains 95% of variation (excellent for forecasting)
- **R² = 0.70**: Model explains 70% of variation (good for business decisions)
- **R² = 0.40**: Model explains 40% of variation (use with caution)

**Business Context**:

- **Marketing**: R² > 0.6 is strong (many external factors)
- **Finance**: R² > 0.7 expected (more predictable)
- **Operations**: R² > 0.8 desired (controlled environment)

### Understanding P-Values

**What it means**: Probability that relationship is due to chance.

- **p < 0.001**: Extremely strong evidence (***) - Very confident
- **p < 0.01**: Strong evidence (**) - Confident
- **p < 0.05**: Moderate evidence (*) - Reasonably confident
- **p > 0.05**: Weak evidence - Not confident

**Business Rule**: Only act on factors with p < 0.05 (95% confidence).

### Understanding Coefficients

**What it means**: How much Y changes when X increases by 1 unit.

**Example**: Marketing spend coefficient = 3.2

- For every $1 increase in marketing, revenue increases by $3.20
- ROI = 3.2 - 1 = 2.2, or 220% return

**Business Decision Framework**:

- If coefficient × volume > costs, invest
- If p-value < 0.05, trust the estimate
- If R² > 0.7, forecast with confidence

### Understanding Confidence Intervals

**What it means**: Range where true value likely falls (e.g., 90% or 95% confidence).

**Example**: Revenue forecast = $50M with 90% CI [$45M, $55M]

- Best estimate: $50M
- Conservative plan: $45M
- Optimistic plan: $55M
- Planning buffer: $5M variance

**Business Application**:

- **Budgeting**: Use lower bound (conservative)
- **Target Setting**: Use point estimate (realistic)
- **Capacity Planning**: Use upper bound (optimistic)

### Understanding Outliers & Influence

**Outliers**: Data points that don't fit the pattern.
**Influential**: Points that strongly affect the model.

**Business Decisions**:

- **Outlier + Influential**: Investigate (data error or special case?)
- **Outlier + Not Influential**: Monitor (unusual but not problematic)
- **Not Outlier + Influential**: Normal leverage point

**Example**:

- Q4 sales spike (outlier) due to holiday season → Keep (real pattern)
- Data entry error showing $10M sale → Remove (data quality issue)

[↑ Go to Top](#business-guide)

## ROI Analysis Framework

### Marketing Campaign ROI


```sql

-- Create sample campaign data
CREATE OR REPLACE TABLE campaigns AS
SELECT
    i as campaign_id,
    spend,
    (spend * 2.3 + RANDOM() * 500)::DOUBLE as revenue  -- 2.3x ROI with noise
FROM (
    SELECT
        i,
        (1000 + RANDOM() * 4000)::DOUBLE as spend  -- $1k-5k spend per campaign
    FROM range(1, 31) t(i)
);

-- Calculate marketing ROI with statistical confidence using aggregate functions
SELECT
    'Marketing ROI' as metric,
    ROUND((anofox_statistics_ols_fit_agg(revenue, spend)).coefficients[1] - 1, 2) as roi_multiplier,
    ROUND(((anofox_statistics_ols_fit_agg(revenue, spend)).coefficients[1] - 1) * 100, 1) || '%' as roi_percentage,
    CASE
        WHEN (anofox_statistics_ols_fit_agg(revenue, spend)).coefficients[1] > 1.5 THEN 'Strong - Scale Up'
        WHEN (anofox_statistics_ols_fit_agg(revenue, spend)).coefficients[1] > 1.0 THEN 'Positive - Continue'
        WHEN (anofox_statistics_ols_fit_agg(revenue, spend)).coefficients[1] < 1.0 THEN 'Negative - Stop Campaign'
        ELSE 'Inconclusive - Gather More Data'
    END as recommendation,
    ROUND((anofox_statistics_ols_fit_agg(revenue, spend)).residual_standard_error, 4) as std_error,
    ROUND((anofox_statistics_ols_fit_agg(revenue, spend)).r2, 3) as model_quality
FROM campaigns;
```

### Process Improvement ROI

**Formula**: (Defect Reduction × Cost per Defect) - Implementation Cost

Use regression to quantify defect reduction from process changes.

[↑ Go to Top](#business-guide)

## Best Practices for Business Users

### 1. Always Check Statistical Significance

Don't act on p-values > 0.05 unless you have strong business reasons.

### 2. Validate Model Quality

- Check R² before making forecasts
- Review residual diagnostics for unusual patterns
- Look for influential outliers

### 3. Use Confidence Intervals for Risk Management

- Plan for lower bound (pessimistic)
- Expect point estimate (realistic)
- Hope for upper bound (optimistic)

### 4. Consider Business Context

Statistical significance ≠ Business significance

**Example**:

- Coefficient = 0.001, p < 0.001 (statistically significant)
- But $1 increase → $0.001 revenue (not business significant)

### 5. Monitor Model Performance Over Time

Use rolling regressions to detect changes in relationships:


```sql

-- Create sample monthly data
CREATE OR REPLACE TABLE monthly_data AS
SELECT
    DATE_TRUNC('month', CURRENT_DATE) - (i * INTERVAL '1 month') as month,
    marketing,
    (marketing * 2.5 + RANDOM() * 500)::DOUBLE as sales
FROM (
    SELECT
        i,
        (1000 + i * 50 + RANDOM() * 300)::DOUBLE as marketing  -- Increasing marketing spend
    FROM range(0, 24) t(i)
);

-- Track rolling 12-month ROI to detect relationship changes over time
SELECT
    month,
    ROUND((anofox_statistics_ols_fit_agg(sales, marketing) OVER (
        ORDER BY month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )).coefficients[1], 2) as rolling_12mo_roi,
    ROUND((anofox_statistics_ols_fit_agg(sales, marketing) OVER (
        ORDER BY month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )).r2, 3) as rolling_model_quality
FROM monthly_data
ORDER BY month DESC
LIMIT 12;  -- Show last 12 months
```

### 6. Combine Multiple Models

Don't rely on single model - triangulate with multiple approaches:


```sql

-- Create sample data with multiple predictors
CREATE OR REPLACE TABLE model_comparison_data AS
WITH raw_data AS (
    SELECT
        i,
        (RANDOM() * 10)::DOUBLE as x1,
        (RANDOM() * 10)::DOUBLE as x2,
        (RANDOM() * 10)::DOUBLE as x3,  -- Noise
        (RANDOM() * 10)::DOUBLE as x4  -- Noise
    FROM range(1, 101) t(i)
)
SELECT
    i as obs_id,
    (10 + x1 * 2.5 + x2 * 0.5 + RANDOM() * 5)::DOUBLE as y,  -- x1 is strong, x2 is weak
    x1,
    x2,
    x3,
    x4
FROM raw_data;

-- Compare simple vs complex models using aggregate functions
-- Simple model: just x1
WITH simple_model AS (
    SELECT
        'Simple Model (x1 only)' as model_type,
        ROUND((anofox_statistics_ols_fit_agg(y, x1)).r2, 4) as r_squared,
        COUNT(*) as n_obs,
        1 as n_predictors
    FROM model_comparison_data
),
-- Complex model: analyze multiple predictors individually
complex_predictors AS (
    SELECT 'x1' as var, (anofox_statistics_ols_fit_agg(y, x1)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x2' as var, (anofox_statistics_ols_fit_agg(y, x2)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x3' as var, (anofox_statistics_ols_fit_agg(y, x3)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x4' as var, (anofox_statistics_ols_fit_agg(y, x4)).r2 as r2 FROM model_comparison_data
),
complex_summary AS (
    SELECT
        'Complex Model (all vars)' as model_type,
        ROUND(MAX(r2), 4) as r_squared,  -- Best predictor's R²
        (SELECT COUNT(*) FROM model_comparison_data) as n_obs,
        4 as n_predictors
    FROM complex_predictors
)
SELECT
    model_type,
    r_squared,
    n_predictors,
    CASE
        WHEN r_squared > 0.8 THEN 'Excellent'
        WHEN r_squared > 0.6 THEN 'Good'
        WHEN r_squared > 0.4 THEN 'Fair'
        ELSE 'Poor'
    END as model_quality
FROM simple_model
UNION ALL
SELECT model_type, r_squared, n_predictors,
    CASE
        WHEN r_squared > 0.8 THEN 'Excellent'
        WHEN r_squared > 0.6 THEN 'Good'
        WHEN r_squared > 0.4 THEN 'Fair'
        ELSE 'Poor'
    END as model_quality
FROM complex_summary
ORDER BY r_squared DESC;  -- Higher R² is better for comparison
```

[↑ Go to Top](#business-guide)
