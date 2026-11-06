# Anofox Statistics - Advanced Use Cases

This guide demonstrates sophisticated analytical patterns using the Anofox Statistics extension.

## Table of Contents

1. [Multi-Stage Model Building](#multi-stage-model-building)
2. [Time-Series Analysis](#time-series-analysis)
3. [Hierarchical Analysis](#hierarchical-analysis)
4. [Cohort Analysis](#cohort-analysis)
5. [A/B Testing](#ab-testing)
6. [Causal Analysis](#causal-analysis)
7. [Production Patterns](#production-patterns)

---

## Multi-Stage Model Building

### Pattern: Fit → Diagnose → Predict Pipeline

Build complete analytical workflows where each stage uses results from previous stages.

```sql skip
-- Illustrative multi-stage pipeline (requires a historical_sales table with
-- columns: date, sales, price, advertising, seasonality)
-- Stage 1: Fit the model
WITH training_data AS (
    SELECT
        array_agg(sales::DOUBLE ORDER BY date) as y,
        [
            array_agg(price::DOUBLE ORDER BY date),
            array_agg(advertising::DOUBLE ORDER BY date),
            array_agg(seasonality::DOUBLE ORDER BY date)
        ] as x
    FROM historical_sales
),
model AS (
    SELECT ols_fit(y, x, {'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}) as fit
    FROM training_data
),

-- Stage 2: Generate predictions and residuals
predictions AS (
    SELECT
        training_data.y as actual,
        predict(training_data.x, model.fit.coefficients, model.fit.intercept) as predicted
    FROM training_data, model
),

-- Stage 3: Compute diagnostics
diagnostics AS (
    SELECT
        residuals_diagnostics(actual, predicted) as resid_diag,
        jarque_bera(
            (SELECT (residuals_diagnostics(actual, predicted)).raw FROM predictions)
        ) as normality_test
    FROM predictions
)

-- Final output
SELECT
    model.fit.r_squared as model_r_squared,
    model.fit.coefficients as coefficients,
    diagnostics.normality_test.p_value as normality_pvalue
FROM model, diagnostics;
```

### Pattern: Self-Contained Multi-Stage Pipeline

A runnable example using inline data.

```sql
-- Inline data: simulate sales driven by price + advertising + seasonality
CREATE OR REPLACE TABLE sales_demo AS
SELECT
    i as period,
    (100 - i * 0.5 + (random()-0.5)*5)::DOUBLE as price,
    (50 + i * 0.3 + (random()-0.5)*3)::DOUBLE as advertising,
    (CASE (i % 4) WHEN 0 THEN 1.2 WHEN 1 THEN 0.9 WHEN 2 THEN 1.1 ELSE 0.8 END)::DOUBLE as seasonality,
    (500 - i * 2 + i * 0.8 + seasonality_val * 50 + (random()-0.5)*20)::DOUBLE as sales
FROM (
    SELECT i,
        CASE (i % 4) WHEN 0 THEN 1.2 WHEN 1 THEN 0.9 WHEN 2 THEN 1.1 ELSE 0.8 END as seasonality_val
    FROM generate_series(1, 24) t(i)
);

-- Stage 1: Fit
WITH model AS (
    SELECT ols_fit(
        array_agg(sales ORDER BY period),
        [
            array_agg(price ORDER BY period),
            array_agg(advertising ORDER BY period),
            array_agg(seasonality ORDER BY period)
        ],
        {'compute_inference': true}
    ) as fit
    FROM sales_demo
)
-- Stage 2: Report
SELECT
    fit.r_squared as r_squared,
    fit.coefficients[1] as price_effect,
    fit.coefficients[2] as ad_effect,
    fit.coefficients[3] as seasonal_effect,
    fit.p_values[1] as price_pvalue
FROM model;
```

### Pattern: Model Selection with Information Criteria

Compare multiple model specifications.

```sql
-- Inline data for model selection
CREATE OR REPLACE TABLE analysis_data AS
SELECT
    i::DOUBLE as x1,
    (i * 0.7 + (random()-0.5)*2)::DOUBLE as x2,
    (i * 0.3 + (random()-0.5)*5)::DOUBLE as x3,
    (2.0*i + 1.5*(i*0.7) + (random()-0.5)*3)::DOUBLE as y
FROM generate_series(1, 30) t(i);

WITH data AS (
    SELECT
        array_agg(y::DOUBLE) as y_arr,
        array_agg(x1::DOUBLE) as x1_arr,
        array_agg(x2::DOUBLE) as x2_arr,
        array_agg(x3::DOUBLE) as x3_arr,
        COUNT(*) as n
    FROM analysis_data
),
models AS (
    SELECT
        'Model 1: x1 only' as model_name,
        ols_fit(y_arr, [x1_arr]) as fit,
        2 as k,
        n
    FROM data
    UNION ALL
    SELECT
        'Model 2: x1 + x2' as model_name,
        ols_fit(y_arr, [x1_arr, x2_arr]) as fit,
        3 as k,
        n
    FROM data
    UNION ALL
    SELECT
        'Model 3: x1 + x2 + x3' as model_name,
        ols_fit(y_arr, [x1_arr, x2_arr, x3_arr]) as fit,
        4 as k,
        n
    FROM data
)
SELECT
    model_name,
    ROUND(fit.r_squared, 4) as r_squared,
    ROUND(fit.adj_r_squared, 4) as adj_r_squared,
    ROUND(aic((1 - fit.r_squared) * n, n, k), 2) as aic,
    ROUND(bic((1 - fit.r_squared) * n, n, k), 2) as bic
FROM models
ORDER BY aic;
```

---

## Time-Series Analysis

### Pattern: Regime Detection with Rolling Regression

Detect structural breaks by monitoring coefficient stability.

```sql
-- Inline time-series data with a regime shift at t=50
CREATE OR REPLACE TABLE daily_prices AS
SELECT
    (DATE '2026-09-01' - (100 - i)::INTEGER)::DATE as date,
    CASE
        WHEN i <= 50 THEN (100.0 + i * 0.5 + (random()-0.5)*2)::DOUBLE
        ELSE (125.0 + (i-50) * 1.2 + (random()-0.5)*2)::DOUBLE
    END as value
FROM generate_series(1, 100) t(i);
```

```sql skip
-- Note: ols_fit_agg() as a window aggregate (OVER clause) triggers a DuckDB
-- INTERNAL Error in this build. Shown here as a pattern reference only.
WITH time_series AS (
    SELECT date, value, LAG(value, 1) OVER (ORDER BY date) as lag1
    FROM daily_prices
),
time_series_filtered AS (
    SELECT date, value, lag1 FROM time_series WHERE lag1 IS NOT NULL
),
rolling_betas AS (
    SELECT
        date,
        value,
        -- Short-term beta (10-day window)
        (ols_fit_agg(value, [lag1]) OVER (
            ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        )).coefficients[1] as beta_short,
        -- Long-term beta (30-day window)
        (ols_fit_agg(value, [lag1]) OVER (
            ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )).coefficients[1] as beta_long
    FROM time_series_filtered
)
SELECT
    date,
    value,
    ROUND(beta_short, 4) as beta_10d,
    ROUND(beta_long, 4) as beta_30d,
    ROUND(ABS(beta_short - beta_long), 4) as regime_indicator,
    CASE
        WHEN ABS(beta_short - beta_long) > 0.2 THEN 'REGIME SHIFT'
        ELSE 'STABLE'
    END as regime_status
FROM rolling_betas
WHERE beta_short IS NOT NULL AND beta_long IS NOT NULL
ORDER BY date;
```

### Pattern: Seasonality Decomposition

Separate trend from seasonal components.

```sql
-- Inline monthly data with trend + seasonality
CREATE OR REPLACE TABLE daily_data AS
SELECT
    (DATE '2022-01-01' + INTERVAL (i) MONTH)::DATE as date,
    (i % 12 + 1) as month_num,
    (100 + i * 2 + CASE (i%12)
        WHEN 0 THEN 20 WHEN 1 THEN -5 WHEN 2 THEN -10 WHEN 3 THEN 5
        WHEN 4 THEN 15 WHEN 5 THEN 25 WHEN 6 THEN 30 WHEN 7 THEN 25
        WHEN 8 THEN 10 WHEN 9 THEN -5 WHEN 10 THEN -10 ELSE 15
    END + (random()-0.5)*5)::DOUBLE as value
FROM generate_series(0, 35) t(i);

WITH monthly_data AS (
    SELECT
        date_trunc('month', date) as month,
        AVG(value) as value,
        EXTRACT(month FROM date)::INTEGER as month_num
    FROM daily_data
    GROUP BY 1, 3
),
-- Create seasonal dummies
with_dummies AS (
    SELECT
        month,
        value,
        month_num,
        ROW_NUMBER() OVER (ORDER BY month) as trend_idx,
        CASE WHEN month_num = 1 THEN 1.0 ELSE 0.0 END as jan,
        CASE WHEN month_num = 2 THEN 1.0 ELSE 0.0 END as feb,
        CASE WHEN month_num = 3 THEN 1.0 ELSE 0.0 END as mar,
        CASE WHEN month_num = 11 THEN 1.0 ELSE 0.0 END as nov
    FROM monthly_data
),
-- Fit trend + seasonal model
model AS (
    SELECT ols_fit(
        array_agg(value ORDER BY month),
        [
            array_agg(trend_idx::DOUBLE ORDER BY month),
            array_agg(jan ORDER BY month),
            array_agg(feb ORDER BY month),
            array_agg(mar ORDER BY month)
        ]
    ) as fit
    FROM with_dummies
)
SELECT
    fit.coefficients[1] as trend_coefficient,
    fit.coefficients[2] as jan_effect,
    fit.coefficients[3] as feb_effect,
    fit.coefficients[4] as mar_effect,
    fit.r_squared as model_fit
FROM model;
```

### Pattern: Adaptive Forecasting with RLS

Real-time coefficient adaptation using exponential forgetting.

```sql skip
-- Adaptive RLS forecasting (illustrative — requires a sensor_readings table with
-- columns: timestamp, target_value, feature_1, feature_2)
WITH streaming_data AS (
    SELECT
        timestamp,
        target_value,
        feature_1,
        feature_2,
        -- RLS with forgetting factor 0.98 (recent data weighted more)
        rls_fit_agg(
            target_value,
            [feature_1, feature_2],
            {'forgetting_factor': 0.98, 'fit_intercept': true}
        ) OVER (ORDER BY timestamp) as rls_model
    FROM sensor_readings
)
SELECT
    timestamp,
    target_value,
    rls_model.coefficients[1] as adaptive_coef_1,
    rls_model.coefficients[2] as adaptive_coef_2,
    rls_model.intercept as adaptive_intercept
FROM streaming_data;
```

---

## Hierarchical Analysis

### Pattern: Multi-Level Regression

Analyze data at multiple organizational levels.

```sql skip
-- Hierarchical analysis (illustrative — requires a retail_sales table with
-- columns: company, region, territory, store_id, sales, traffic, promotions)
WITH store_data AS (
    SELECT company, region, territory, store_id, sales, traffic, promotions
    FROM retail_sales
),
-- Level 1: Store-level analysis
store_models AS (
    SELECT
        store_id, territory, region, company,
        (ols_fit_agg(sales, [traffic, promotions])).coefficients[1] as traffic_coef,
        (ols_fit_agg(sales, [traffic, promotions])).coefficients[2] as promo_coef,
        (ols_fit_agg(sales, [traffic, promotions])).r_squared as r_squared,
        COUNT(*) as obs
    FROM store_data
    GROUP BY store_id, territory, region, company
),
-- Level 2: Territory aggregation
territory_summary AS (
    SELECT
        territory, region,
        AVG(traffic_coef) as avg_traffic_effect,
        STDDEV(traffic_coef) as std_traffic_effect,
        AVG(promo_coef) as avg_promo_effect,
        COUNT(*) as store_count
    FROM store_models
    GROUP BY territory, region
)
SELECT
    s.store_id,
    s.territory,
    s.traffic_coef,
    t.avg_traffic_effect as territory_avg,
    s.traffic_coef - t.avg_traffic_effect as vs_territory
FROM store_models s
JOIN territory_summary t ON s.territory = t.territory
ORDER BY s.traffic_coef DESC;
```

### Pattern: Fits and Tests at Every Level in One Query

`GROUP BY ROLLUP`, `CUBE` or `GROUPING SETS` fit one model per region and product, per
region, and overall in a single query. `GROUPING()` tells the levels apart. Each level's
result is identical to a separate `GROUP BY` at that level (bitwise, on DuckDB v1.5.6 and 2.0).

```sql
SELECT
    CASE GROUPING(region, product) WHEN 0 THEN 'region x product' WHEN 1 THEN 'region' ELSE 'overall' END AS level,
    region, product,
    round((ols_fit_agg(sales, [price])).coefficients[1], 3) AS price_effect,
    round((t_test_agg(sales, promo)).p_value, 4) AS promo_p
FROM (SELECT i % 3 AS region, i % 2 AS product, (i % 4 < 2)::INT AS promo, 10 + (i % 7) * 0.5 AS price,
             100 - 2.0 * (10 + (i % 7) * 0.5) + (i % 4 < 2)::INT * 3 + sin(i) * 4 AS sales
      FROM range(3000) r(i))
GROUP BY ROLLUP (region, product)
ORDER BY GROUPING(region, product), region, product;
```

Each level is still computed from the rows. DuckDB 2.0 builds coarser levels from finer
levels' states only for aggregates with compact, fixed-size states, which the fit and test
aggregates do not have yet (#266).

---

## Cohort Analysis

### Pattern: Lifetime Value Curves by Cohort

Model customer value trajectories for different acquisition cohorts.

```sql
-- Inline cohort data: simulate 4 cohorts x 12 months LTV curves
CREATE OR REPLACE TABLE cohort_data AS
SELECT
    cohort_month,
    months_since_acquisition,
    (EXP(base_rate + growth_rate * LN(months_since_acquisition + 1)) * (1 + (random()-0.5)*0.1))::DOUBLE as cumulative_revenue
FROM (
    SELECT
        DATE '2023-01-01' + INTERVAL (c) MONTH as cohort_month,
        m as months_since_acquisition,
        CASE c WHEN 0 THEN 4.5 WHEN 1 THEN 4.3 WHEN 2 THEN 4.6 ELSE 4.4 END as base_rate,
        CASE c WHEN 0 THEN 0.4 WHEN 1 THEN 0.35 WHEN 2 THEN 0.45 ELSE 0.38 END as growth_rate
    FROM generate_series(0, 3) t(c),
         generate_series(1, 12) u(m)
);

-- Fit growth model per cohort: revenue = a * months^b (log-linearized)
WITH cohort_models AS (
    SELECT
        cohort_month,
        (ols_fit_agg(
            LN(cumulative_revenue + 1),
            [LN(months_since_acquisition + 1)]
        )).coefficients[1] as growth_rate,
        (ols_fit_agg(
            LN(cumulative_revenue + 1),
            [LN(months_since_acquisition + 1)]
        )).intercept as initial_value,
        (ols_fit_agg(
            LN(cumulative_revenue + 1),
            [LN(months_since_acquisition + 1)]
        )).r_squared as model_fit,
        COUNT(*) as data_points
    FROM cohort_data
    GROUP BY cohort_month
)
SELECT
    cohort_month,
    ROUND(growth_rate, 3) as growth_rate,
    ROUND(EXP(initial_value), 2) as month_1_value,
    -- Project 12-month LTV
    ROUND(EXP(initial_value) * POWER(12, growth_rate), 2) as projected_12m_ltv,
    ROUND(model_fit, 3) as r_squared,
    CASE
        WHEN growth_rate > 0.5 THEN 'HIGH_GROWTH'
        WHEN growth_rate > 0.3 THEN 'MODERATE_GROWTH'
        ELSE 'LOW_GROWTH'
    END as cohort_classification
FROM cohort_models
ORDER BY cohort_month;
```

---

## A/B Testing

### Pattern: Regression-Based Test Analysis

Use regression for controlled experiment analysis.

```sql
-- Inline A/B test data
CREATE OR REPLACE TABLE ab_test_results AS
SELECT
    i as user_id,
    CASE WHEN random() < 0.5 THEN 'treatment' ELSE 'control' END as variant,
    (18 + random() * 50)::INTEGER as age,
    (1 + random() * 60)::INTEGER as tenure,
    CASE WHEN random() < 0.12 THEN 1 ELSE 0 END as conversion
FROM generate_series(1, 1000) t(i);

WITH experiment_data AS (
    SELECT
        user_id,
        CASE WHEN variant = 'treatment' THEN 1.0 ELSE 0.0 END as is_treatment,
        age::DOUBLE as age,
        tenure::DOUBLE as tenure,
        conversion::DOUBLE as converted
    FROM ab_test_results
),
-- OLS with control variables
model AS (
    SELECT ols_fit(
        array_agg(converted),
        [
            array_agg(is_treatment),
            array_agg(age),
            array_agg(tenure)
        ],
        {'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) as fit
    FROM experiment_data
)
SELECT
    'Treatment Effect' as metric,
    fit.coefficients[1] as point_estimate,
    fit.std_errors[1] as standard_error,
    fit.ci_lower[1] as ci_lower_95,
    fit.ci_upper[1] as ci_upper_95,
    fit.p_values[1] as p_value,
    CASE
        WHEN fit.p_values[1] < 0.05 AND fit.ci_lower[1] > 0 THEN 'SIGNIFICANT POSITIVE'
        WHEN fit.p_values[1] < 0.05 AND fit.ci_upper[1] < 0 THEN 'SIGNIFICANT NEGATIVE'
        WHEN fit.p_values[1] < 0.10 THEN 'MARGINALLY SIGNIFICANT'
        ELSE 'NOT SIGNIFICANT'
    END as conclusion
FROM model;
```

### Pattern: Heterogeneous Treatment Effects

Identify segments where treatment works differently.

```sql
-- Reuse ab_test_results created above (must run in same session)
WITH experiment_data AS (
    SELECT
        user_id,
        CASE WHEN age < 35 THEN 'young' ELSE 'mature' END as segment,
        CASE WHEN variant = 'treatment' THEN 1.0 ELSE 0.0 END as is_treatment,
        conversion::DOUBLE as outcome
    FROM ab_test_results
)
SELECT
    segment,
    COUNT(*) as sample_size,
    (ols_fit_agg(outcome, [is_treatment], {'compute_inference': true})).coefficients[1] as treatment_effect,
    (ols_fit_agg(outcome, [is_treatment], {'compute_inference': true})).p_values[1] as p_value,
    (ols_fit_agg(outcome, [is_treatment], {'compute_inference': true})).ci_lower[1] as ci_lower,
    (ols_fit_agg(outcome, [is_treatment], {'compute_inference': true})).ci_upper[1] as ci_upper
FROM experiment_data
GROUP BY segment
HAVING COUNT(*) >= 100
ORDER BY treatment_effect DESC;
```

---

## Causal Analysis

### Pattern: Difference-in-Differences

Estimate causal effects from observational data.

```sql
-- Inline DiD data: 2 periods, 2 groups
CREATE OR REPLACE TABLE panel_data AS
SELECT
    i as unit_id,
    period,
    CASE WHEN i <= 50 THEN true ELSE false END as is_treated,
    3 as treatment_start,
    -- Outcome: treated group gets +5 uplift post-treatment
    (10 + period * 0.5 + CASE WHEN i <= 50 AND period >= 3 THEN 5.0 ELSE 0.0 END
        + (random()-0.5)*2)::DOUBLE as outcome
FROM generate_series(1, 100) t(i),
     generate_series(1, 4) u(period);

WITH did_data AS (
    SELECT
        unit_id, time_period,
        CASE WHEN is_treated THEN 1.0 ELSE 0.0 END as treatment,
        CASE WHEN time_period >= treatment_start THEN 1.0 ELSE 0.0 END as post,
        CASE WHEN is_treated AND time_period >= treatment_start THEN 1.0 ELSE 0.0 END as treatment_x_post,
        outcome
    FROM (
        SELECT unit_id, period as time_period, is_treated, treatment_start, outcome
        FROM panel_data
    )
),
-- DiD regression: outcome = α + β₁*treatment + β₂*post + β₃*treatment×post + ε
did_model AS (
    SELECT ols_fit(
        array_agg(outcome),
        [
            array_agg(treatment),
            array_agg(post),
            array_agg(treatment_x_post)
        ],
        {'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}
    ) as fit
    FROM did_data
)
SELECT
    'Difference-in-Differences Analysis' as analysis,
    fit.intercept as control_pre_mean,
    fit.coefficients[1] as treatment_group_diff,
    fit.coefficients[2] as time_effect,
    fit.coefficients[3] as causal_effect_did,
    fit.p_values[3] as causal_effect_pvalue,
    fit.ci_lower[3] as effect_ci_lower,
    fit.ci_upper[3] as effect_ci_upper,
    fit.r_squared as model_r_squared
FROM did_model;
```

---

## Production Patterns

### Pattern: Materialized Model Cache

Pre-compute and cache model results for fast lookups.

```sql
-- Create inline training data
CREATE OR REPLACE TABLE training_data AS
SELECT
    CASE (i % 3) WHEN 0 THEN 'A' WHEN 1 THEN 'B' ELSE 'C' END as category,
    (i * 1.5 + (random()-0.5)*3)::DOUBLE as y,
    (i * 0.8 + (random()-0.5)*2)::DOUBLE as x1,
    (i * 0.5 + (random()-0.5)*1)::DOUBLE as x2,
    (DATE '2026-09-01' - INTERVAL (100 - i) DAY)::DATE as date
FROM generate_series(1, 90) t(i);

-- Create model cache table
CREATE OR REPLACE TABLE model_cache AS
WITH latest_data AS (
    SELECT
        category,
        array_agg(y::DOUBLE) as y_arr,
        array_agg(x1::DOUBLE) as x1_arr,
        array_agg(x2::DOUBLE) as x2_arr
    FROM training_data
    GROUP BY category
)
SELECT
    category,
    TIMESTAMP '2026-09-01 00:00:00' as trained_at,
    ols_fit(y_arr, [x1_arr, x2_arr], {'fit_intercept': true, 'compute_inference': true, 'confidence_level': 0.95}) as model
FROM latest_data;

-- Query cached model
SELECT
    category,
    model.r_squared,
    model.coefficients[1] as x1_coef,
    model.coefficients[2] as x2_coef,
    model.n_observations as n
FROM model_cache
ORDER BY category;
```

### Pattern: Model Drift Detection

Automate model retraining with drift detection.

```sql
-- Using model_cache and training_data from above
-- Check calibration: fit actual ~ predicted
WITH predictions AS (
    SELECT
        t.category,
        t.y as actual,
        predict([[t.x1], [t.x2]], c.model.coefficients, c.model.intercept)[1] as predicted
    FROM training_data t
    JOIN model_cache c ON t.category = c.category
    LIMIT 60  -- training set rows
),
current_performance AS (
    SELECT
        category,
        (ols_fit_agg(actual, [predicted])).r_squared as current_r2
    FROM predictions
    GROUP BY category
),
baseline_performance AS (
    SELECT category, model.r_squared as baseline_r2
    FROM model_cache
)
SELECT
    c.category,
    ROUND(c.current_r2, 3) as current_r2,
    ROUND(b.baseline_r2, 3) as baseline_r2,
    ROUND(c.current_r2 / b.baseline_r2, 3) as performance_ratio,
    CASE
        WHEN c.current_r2 / b.baseline_r2 < 0.9 THEN 'RETRAIN_NEEDED'
        ELSE 'OK'
    END as status
FROM current_performance c
JOIN baseline_performance b ON c.category = b.category;
```

### Pattern: Drift Alerts with a Trigger (DuckDB 2.0)

DuckDB 2.0 has triggers, so the database can check for drift itself whenever data arrives.
This trigger compares the latest 200 events with a reference sample after every `INSERT`
statement and records an alert when the PSI exceeds 0.25:

```sql skip
-- DuckDB 2.0 only (triggers do not exist on v1.5.6)
CREATE TABLE events(ts TIMESTAMP, v DOUBLE);
CREATE TABLE alerts(alert_ts TIMESTAMP, psi DOUBLE, n BIGINT);
CREATE TABLE ref AS SELECT sin(i * 0.37) AS v FROM range(500) t(i);

CREATE TRIGGER drift_check AFTER INSERT ON events FOR EACH STATEMENT
    INSERT INTO alerts
    SELECT max_ts, psi, n FROM (
        SELECT (psi_agg(v, src)).psi AS psi, count(*) FILTER (WHERE src = 1) AS n, max(ts) AS max_ts
        FROM (SELECT v, 0 AS src, NULL::TIMESTAMP AS ts FROM ref
              UNION ALL
              SELECT v, 1, ts FROM (SELECT v, ts FROM events ORDER BY ts DESC LIMIT 200)))
    WHERE n >= 200 AND psi > 0.25;

INSERT INTO events SELECT TIMESTAMP '2026-01-01' + INTERVAL (i) SECOND, sin(i * 0.37) FROM range(300) t(i);
-- no alert
INSERT INTO events SELECT TIMESTAMP '2026-01-02' + INTERVAL (i) SECOND, sin(i * 0.37) + 1.5 FROM range(300) t(i);
SELECT * FROM alerts;  -- one row, psi ≈ 8.8
```

Use `FOR EACH STATEMENT`. A `FOR EACH ROW` trigger fires once per inserted row, and each firing
sees the table after the whole statement. A 300-row `INSERT` would then write 300 identical
alerts and run the aggregate 300 times. Any drift aggregate (`ks_test_agg`, `wasserstein_agg`,
`jensen_shannon_agg`, ...) works the same way.

On DuckDB v1.5.6, run the same `INSERT INTO alerts SELECT ...` as a scheduled query (cron or
an orchestrator) instead of a trigger.

### Pattern: A Model That Stays Current (DuckDB 2.0)

A [least-squares state](../docs/api/regression/ls_state.md) summarizes rows in a few hundred
bytes, and states merge. A statement-level trigger with a transition table (`REFERENCING NEW
TABLE`) folds each insert's rows into a one-row model table, so the fit is always up to date
without rescanning the history:

```sql skip
-- DuckDB 2.0 only (triggers do not exist on v1.5.6)
CREATE TABLE sales(ts TIMESTAMP, price DOUBLE, units DOUBLE);
CREATE TABLE model_state(state BLOB);
INSERT INTO model_state VALUES (NULL);

CREATE TRIGGER refit AFTER INSERT ON sales REFERENCING NEW TABLE AS new_rows FOR EACH STATEMENT
    UPDATE model_state SET state = (
        SELECT ls_state_combine_agg(s) FROM (SELECT state AS s FROM model_state
                                             UNION ALL SELECT ls_state_agg(units, [price]) FROM new_rows));

INSERT INTO sales SELECT TIMESTAMP '2026-01-01' + INTERVAL (i) MINUTE, 10 + sin(i),
                         100 - 3 * (10 + sin(i)) + cos(i * 7) FROM range(1000) t(i);
INSERT INTO sales VALUES (TIMESTAMP '2026-01-02', 11.0, 67.2);

-- The current model; equals ols_fit_agg(units, [price]) over all of sales.
SELECT ols_fit_from_state(state, {'compute_inference': true}) AS model FROM model_state;
```

The trigger reads only the inserted rows (`new_rows`) and the stored state. Use
`ridge_fit_from_state` or `wls_fit_from_state` (with `ls_state_agg(y, x, weight)`) the same way.
Deletes and updates are not reflected: a state can only grow. Rebuild it from the table
after them.

On DuckDB v1.5.6, run the `UPDATE model_state ...` from a scheduler, with `new_rows`
replaced by a query for the rows since the last run.

### Pattern: Large-Scale Parallel Processing

Partition data for parallel regression.

```sql
-- Inline large dataset simulation
WITH large_dataset AS (
    SELECT
        i as id,
        (i * 1.2 + (random()-0.5)*5)::DOUBLE as y,
        (i * 0.8 + (random()-0.5)*3)::DOUBLE as x1,
        (i * 0.5 + (random()-0.5)*2)::DOUBLE as x2
    FROM generate_series(1, 500) t(i)
),
-- Partition large dataset
partitioned AS (
    SELECT
        NTILE(10) OVER (ORDER BY id) as partition_id,
        y, x1, x2
    FROM large_dataset
)
-- Process each partition (can be parallelized)
SELECT
    partition_id,
    (ols_fit_agg(y, [x1, x2])).coefficients as partition_coefs,
    (ols_fit_agg(y, [x1, x2])).r_squared as partition_r2,
    COUNT(*) as partition_size
FROM partitioned
GROUP BY partition_id;
```

### Pattern: Export for External Tools

Export model results for downstream systems.

```sql skip
-- Export model to JSON format (requires the json extension: LOAD json)
-- Run: INSTALL json; LOAD json; before executing.
SELECT
    json_object(
        'model_type', 'ols',
        'coefficients', model.coefficients,
        'intercept', model.intercept,
        'r_squared', model.r_squared,
        'n_observations', model.n_observations,
        'trained_at', TIMESTAMP '2026-09-01 00:00:00'
    ) as model_json
FROM model_cache
WHERE category = 'A';
```

---

## Best Practices for Production

### 1. Validate Before Deployment

```sql
-- Cross-validation pattern (using training_data from above)
WITH folds AS (
    -- hash() gives a reproducible pseudo-random fold (DuckDB 2.0 rejects random() in a window ORDER BY)
    SELECT *, hash(x1, x2, y) % 5 as fold FROM training_data
),
-- For each held-out fold, fit on all other folds
cv_results AS (
    SELECT
        held_out.fold as test_fold,
        (
            SELECT ols_fit_agg(tr.y, [tr.x1, tr.x2])
            FROM folds tr WHERE tr.fold != held_out.fold
        ).r_squared as train_r2
    FROM (SELECT DISTINCT fold FROM folds) held_out
)
SELECT
    AVG(train_r2) as mean_cv_r2,
    STDDEV(train_r2) as std_cv_r2
FROM cv_results;
```

### 2. Monitor Drift

```sql
-- Track calibration stability (using predictions from model_cache / training_data above)
WITH preds AS (
    SELECT
        t.date,
        t.y as actual,
        predict([[t.x1], [t.x2]], c.model.coefficients, c.model.intercept)[1] as predicted
    FROM training_data t
    JOIN model_cache c ON t.category = c.category
)
SELECT
    date_trunc('week', date) as week,
    (ols_fit_agg(actual, [predicted])).coefficients[1] as weekly_calibration
FROM preds
GROUP BY 1
ORDER BY 1;
```

### 3. Document Assumptions

```sql
-- Store model metadata (creates a simple metadata record)
CREATE OR REPLACE TABLE model_registry AS
SELECT
    gen_random_uuid() as model_id,
    'OLS' as model_type,
    ['x1', 'x2'] as features,
    'Linear relationship, homoscedastic errors' as assumptions,
    model.r_squared as validation_r2,
    TIMESTAMP '2026-09-01 00:00:00' as created_at
FROM model_cache
WHERE category = 'A';

SELECT * FROM model_registry;
```

---

## Summary

This guide covered advanced patterns for:
- **Multi-stage pipelines**: Fit → Diagnose → Predict workflows
- **Time-series**: Regime detection, seasonality, adaptive forecasting
- **Hierarchical analysis**: Multi-level organizational comparisons
- **Cohort analysis**: Lifetime value modeling
- **Experimentation**: A/B testing and causal inference
- **Production**: Caching, monitoring, and scaling

These patterns can be combined and adapted for specific business requirements.
