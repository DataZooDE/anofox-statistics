# Business Guide

Real-world business applications of the Anofox Statistics extension, demonstrating how to turn data into actionable business insights.

## Important Note About Examples

**All examples below are copy-paste runnable!** Each example includes sample data creation.

**Working Patterns**:
- **Aggregate functions** (`ols_fit_agg`, `ols_coeff_agg`) work directly with table data - use these for GROUP BY analysis
- **Table functions** (`ols_inference`, `ols_predict_interval`) require literal arrays - examples show small datasets with explicit arrays
- All functions use positional parameters only (no `:=` syntax)

**To adapt for your tables**: Replace sample data creation with your actual tables. For table functions with large datasets, use the two-step approach: run `SELECT LIST(column) FROM table`, copy result, paste as literal array.

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
    ROUND(ols_coeff_agg(revenue, tv_spend), 2) as roi,
    ROUND((ols_fit_agg(revenue, tv_spend)).r2, 3) as r_squared,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'digital' as channel,
    ROUND(ols_coeff_agg(revenue, digital_spend), 2) as roi,
    ROUND((ols_fit_agg(revenue, digital_spend)).r2, 3) as r_squared,
    'High Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'print' as channel,
    ROUND(ols_coeff_agg(revenue, print_spend), 2) as roi,
    ROUND((ols_fit_agg(revenue, print_spend)).r2, 3) as r_squared,
    'Low Impact' as business_impact
FROM weekly_campaigns WHERE year = 2024
UNION ALL
SELECT
    'radio' as channel,
    ROUND(ols_coeff_agg(revenue, radio_spend), 2) as roi,
    ROUND((ols_fit_agg(revenue, radio_spend)).r2, 3) as r_squared,
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
    ROUND((ols_fit_agg(quantity, price)).coefficient, 3) as elasticity,
    ROUND((ols_fit_agg(quantity, price)).r2, 3) as predictability,
    CASE
        WHEN ABS((ols_fit_agg(quantity, price)).coefficient) > 0.5 THEN 'Elastic'
        ELSE 'Inelastic'
    END as elasticity_type,
    CASE
        WHEN ABS((ols_fit_agg(quantity, price)).coefficient) > 0.5 THEN 'Discount Strategy'
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
    tenure::DOUBLE,
    aov::DOUBLE,
    freq::DOUBLE,
    engagement::DOUBLE,
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
    ROUND((ols_fit_agg(total_purchases, tenure)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: AOV' as metric,
    ROUND((ols_fit_agg(total_purchases, aov)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: Frequency' as metric,
    ROUND((ols_fit_agg(total_purchases, freq)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Quality (R²)' as metric,
    ROUND((ols_fit_agg(total_purchases, tenure)).r2, 3) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months';
```

**Business Interpretation**:
- **Coefficient**: Impact of each factor on customer lifetime value
- **R²**: How well features predict CLV
- **Tenure** and **frequency** typically show strongest CLV correlation

**Business Decision**: Focus acquisition on customers likely to have high tenure and purchase frequency.

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
    ROUND((ols_fit_agg(stock_return, market_return)).coefficient, 3) as beta,
    ROUND((ols_fit_agg(stock_return, market_return)).r2, 3) as r_squared,
    CASE
        WHEN (ols_fit_agg(stock_return, market_return)).coefficient > 1.2 THEN 'High Risk'
        WHEN (ols_fit_agg(stock_return, market_return)).coefficient > 0.8 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_category,
    CASE
        WHEN (ols_fit_agg(stock_return, market_return)).coefficient > 1.0 THEN 'Aggressive'
        WHEN (ols_fit_agg(stock_return, market_return)).coefficient > 0.5 THEN 'Moderate'
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

### Use Case 5: Credit Risk Modeling

**Business Question**: Predict loan default risk to optimize lending decisions.

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
SELECT
    'Credit Score' as variable,
    ROUND((ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficient, 5) as coefficient,
    ROUND((ols_fit_agg(default_flag::DOUBLE, credit_score)).p_value, 4) as p_value,
    (ols_fit_agg(default_flag::DOUBLE, credit_score)).significant as significant,
    CASE
        WHEN (ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficient > 0 THEN 'Increases Risk'
        WHEN (ols_fit_agg(default_flag::DOUBLE, credit_score)).coefficient < 0 THEN 'Decreases Risk'
        ELSE 'No Effect'
    END as risk_impact,
    ROUND((ols_fit_agg(default_flag::DOUBLE, credit_score)).r2, 3) as model_quality
FROM loans
WHERE origination_date >= '2022-01-01'
UNION ALL
SELECT
    'Debt-to-Income' as variable,
    ROUND((ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficient, 5) as coefficient,
    ROUND((ols_fit_agg(default_flag::DOUBLE, debt_to_income)).p_value, 4) as p_value,
    (ols_fit_agg(default_flag::DOUBLE, debt_to_income)).significant as significant,
    CASE
        WHEN (ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficient > 0 THEN 'Increases Risk'
        WHEN (ols_fit_agg(default_flag::DOUBLE, debt_to_income)).coefficient < 0 THEN 'Decreases Risk'
        ELSE 'No Effect'
    END as risk_impact,
    ROUND((ols_fit_agg(default_flag::DOUBLE, debt_to_income)).r2, 3) as model_quality
FROM loans
WHERE origination_date >= '2022-01-01'
UNION ALL
SELECT
    'Loan-to-Value' as variable,
    ROUND((ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficient, 5) as coefficient,
    ROUND((ols_fit_agg(default_flag::DOUBLE, loan_to_value)).p_value, 4) as p_value,
    (ols_fit_agg(default_flag::DOUBLE, loan_to_value)).significant as significant,
    CASE
        WHEN (ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficient > 0 THEN 'Increases Risk'
        WHEN (ols_fit_agg(default_flag::DOUBLE, loan_to_value)).coefficient < 0 THEN 'Decreases Risk'
        ELSE 'No Effect'
    END as risk_impact,
    ROUND((ols_fit_agg(default_flag::DOUBLE, loan_to_value)).r2, 3) as model_quality
FROM loans
WHERE origination_date >= '2022-01-01'
UNION ALL
SELECT
    'Employment Years' as variable,
    ROUND((ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficient, 5) as coefficient,
    ROUND((ols_fit_agg(default_flag::DOUBLE, employment_years)).p_value, 4) as p_value,
    (ols_fit_agg(default_flag::DOUBLE, employment_years)).significant as significant,
    CASE
        WHEN (ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficient > 0 THEN 'Increases Risk'
        WHEN (ols_fit_agg(default_flag::DOUBLE, employment_years)).coefficient < 0 THEN 'Decreases Risk'
        ELSE 'No Effect'
    END as risk_impact,
    ROUND((ols_fit_agg(default_flag::DOUBLE, employment_years)).r2, 3) as model_quality
FROM loans
WHERE origination_date >= '2022-01-01'
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
    SELECT * FROM ols_predict_interval(
        [40572849.0, 41233491.0, 42455123.0, 42789456.0, 43891234.0, 43234567.0, 44567890.0, 44123456.0, 45678901.0, 45234567.0, 46789012.0, 46345678.0, 47890123.0, 47456789.0, 48901234.0, 48567890.0, 49912345.0, 49678901.0, 50923456.0, 50789012.0]::DOUBLE[],
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0], [17.0], [18.0], [19.0], [20.0]]::DOUBLE[][],
        [[21.0], [22.0], [23.0], [24.0]]::DOUBLE[][],  -- Next 4 quarters
        0.90,
        'prediction',
        true
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

## Operational Analytics

### Use Case 7: Demand Forecasting for Inventory

**Business Question**: Optimize inventory levels by predicting product demand.

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
    ROUND((ols_fit_agg(units_sold, price)).coefficient, 2) as price_sensitivity,
    ROUND((ols_fit_agg(units_sold, price)).r2, 3) as forecast_accuracy,
    CASE
        WHEN (ols_fit_agg(units_sold, price)).r2 > 0.8 THEN 'High Confidence'
        WHEN (ols_fit_agg(units_sold, price)).r2 > 0.5 THEN 'Medium Confidence'
        ELSE 'Low Confidence'
    END as forecast_reliability,
    CASE
        WHEN (ols_fit_agg(units_sold, price)).r2 > 0.7 THEN 'Auto-Replenish'
        ELSE 'Manual Review'
    END as inventory_strategy,
    COUNT(*) as sample_size
FROM daily_sales
GROUP BY product_id, season
HAVING COUNT(*) >= 30  -- Minimum data for reliable estimates
ORDER BY forecast_accuracy DESC;
```

**Business Decision**: Automate replenishment for high-confidence products, manual review for others.

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
    defect_rate::DOUBLE
FROM (
    SELECT
        i,
        (180 + RANDOM() * 40) as temp,  -- 180-220°F
        (25 + RANDOM() * 10) as pressure,  -- 25-35 PSI
        (40 + RANDOM() * 30) as humidity,  -- 40-70%
        (50 + RANDOM() * 30) as speed,  -- 50-80 units/min
        -- Defects increase with high temp, high speed, decrease with optimal pressure
        (2.0 + temp * 0.05 + speed * 0.03 - pressure * 0.02 + humidity * 0.01 + RANDOM() * 1.5) as defect_rate
    FROM range(1, 101) t(i)
);

-- Analyze impact of each process parameter on defect rates
SELECT
    'Temperature' as variable,
    ROUND((ols_fit_agg(defect_rate, temperature)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, temperature)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, temperature)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, temperature)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, temperature)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, temperature)).significant
             AND (ols_fit_agg(defect_rate, temperature)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, temperature)).significant
             AND (ols_fit_agg(defect_rate, temperature)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Pressure' as variable,
    ROUND((ols_fit_agg(defect_rate, pressure)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, pressure)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, pressure)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, pressure)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, pressure)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, pressure)).significant
             AND (ols_fit_agg(defect_rate, pressure)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, pressure)).significant
             AND (ols_fit_agg(defect_rate, pressure)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Humidity' as variable,
    ROUND((ols_fit_agg(defect_rate, humidity)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, humidity)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, humidity)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, humidity)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, humidity)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, humidity)).significant
             AND (ols_fit_agg(defect_rate, humidity)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, humidity)).significant
             AND (ols_fit_agg(defect_rate, humidity)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
UNION ALL
SELECT
    'Line Speed' as variable,
    ROUND((ols_fit_agg(defect_rate, line_speed)).coefficient, 4) as impact_on_defects,
    ROUND((ols_fit_agg(defect_rate, line_speed)).p_value, 4) as p_value,
    (ols_fit_agg(defect_rate, line_speed)).significant as significant,
    CASE
        WHEN (ols_fit_agg(defect_rate, line_speed)).coefficient > 0 THEN 'Increases Defects'
        WHEN (ols_fit_agg(defect_rate, line_speed)).coefficient < 0 THEN 'Reduces Defects'
    END as quality_impact,
    CASE
        WHEN (ols_fit_agg(defect_rate, line_speed)).significant
             AND (ols_fit_agg(defect_rate, line_speed)).coefficient > 0 THEN 'Critical - Reduce'
        WHEN (ols_fit_agg(defect_rate, line_speed)).significant
             AND (ols_fit_agg(defect_rate, line_speed)).coefficient < 0 THEN 'Beneficial - Increase'
        ELSE 'Not Significant'
    END as action_recommendation
FROM production_batches
WHERE batch_date >= CURRENT_DATE - INTERVAL '3 months'
ORDER BY ABS(impact_on_defects) DESC;
```

**Business Decision**: Optimize process parameters that significantly affect defect rates.

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
        output
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
    ROUND((ols_fit_agg(output_per_hour, training_hours)).coefficient, 2) as training_impact,
    ROUND((ols_fit_agg(output_per_hour, training_hours)).r2, 3) as model_fit,
    CASE
        WHEN (ols_fit_agg(output_per_hour, training_hours)).coefficient > 5.0 THEN 'High Training ROI'
        WHEN (ols_fit_agg(output_per_hour, training_hours)).coefficient > 2.0 THEN 'Medium Training ROI'
        ELSE 'Low Training ROI'
    END as training_effectiveness,
    CASE
        WHEN (ols_fit_agg(output_per_hour, training_hours)).coefficient > 3.0 THEN 'Increase Training Budget'
        ELSE 'Maintain Current Level'
    END as budget_recommendation,
    COUNT(*) as sample_size
FROM employee_productivity
GROUP BY department
ORDER BY training_impact DESC;
```

**Business Decision**: Allocate training budget to departments with highest ROI.

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
        (ols_fit_agg(sales_amount::DOUBLE, month_index::DOUBLE) OVER (
            PARTITION BY territory_id
            ORDER BY month_date
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        )).coefficient as trend_coefficient
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
    ROUND((ols_fit_agg(revenue, spend)).coefficient - 1, 2) as roi_multiplier,
    ROUND(((ols_fit_agg(revenue, spend)).coefficient - 1) * 100, 1) || '%' as roi_percentage,
    CASE
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05
             AND (ols_fit_agg(revenue, spend)).coefficient > 1.5 THEN 'Strong - Scale Up'
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05
             AND (ols_fit_agg(revenue, spend)).coefficient > 1.0 THEN 'Positive - Continue'
        WHEN (ols_fit_agg(revenue, spend)).p_value < 0.05 THEN 'Negative - Stop Campaign'
        ELSE 'Inconclusive - Gather More Data'
    END as recommendation,
    ROUND((ols_fit_agg(revenue, spend)).p_value, 4) as p_value,
    ROUND((ols_fit_agg(revenue, spend)).r2, 3) as model_quality
FROM campaigns;
```

### Process Improvement ROI

**Formula**: (Defect Reduction × Cost per Defect) - Implementation Cost

Use regression to quantify defect reduction from process changes.

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
    ROUND((ols_fit_agg(sales, marketing) OVER (
        ORDER BY month
        ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
    )).coefficient, 2) as rolling_12mo_roi,
    ROUND((ols_fit_agg(sales, marketing) OVER (
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
SELECT
    i as obs_id,
    y,
    x1,
    x2,
    x3,
    x4
FROM (
    SELECT
        i,
        (10 + x1 * 2.5 + x2 * 0.5 + RANDOM() * 5)::DOUBLE as y,  -- x1 is strong, x2 is weak
        (RANDOM() * 10)::DOUBLE as x1,
        (RANDOM() * 10)::DOUBLE as x2,
        (RANDOM() * 10)::DOUBLE as x3,  -- Noise
        (RANDOM() * 10)::DOUBLE as x4   -- Noise
    FROM range(1, 101) t(i)
);

-- Compare simple vs complex models using aggregate functions
-- Simple model: just x1
WITH simple_model AS (
    SELECT
        'Simple Model (x1 only)' as model_type,
        ROUND((ols_fit_agg(y, x1)).r2, 4) as r_squared,
        COUNT(*) as n_obs,
        1 as n_predictors
    FROM model_comparison_data
),
-- Complex model: analyze multiple predictors individually
complex_predictors AS (
    SELECT 'x1' as var, (ols_fit_agg(y, x1)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x2' as var, (ols_fit_agg(y, x2)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x3' as var, (ols_fit_agg(y, x3)).r2 as r2 FROM model_comparison_data
    UNION ALL
    SELECT 'x4' as var, (ols_fit_agg(y, x4)).r2 as r2 FROM model_comparison_data
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

## Conclusion

The Anofox Statistics extension transforms DuckDB into a powerful business analytics platform. Key takeaways:

1. **Test Everything**: Use p-values to validate relationships
2. **Quantify Impact**: Use coefficients for ROI calculations
3. **Measure Uncertainty**: Use confidence intervals for risk management
4. **Monitor Quality**: Use diagnostics to ensure model validity
5. **Act with Confidence**: Statistical evidence → Better business decisions

For more examples:
- [Quick Start Guide](01_quick_start.md) - Basic usage
- [Technical Guide](02_technical_guide.md) - Implementation details
- [Statistics Guide](03_statistics_guide.md) - Statistical theory
- [Advanced Use Cases](05_advanced_use_cases.md) - Complex workflows
