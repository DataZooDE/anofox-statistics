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

```sql
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
    SELECT anofox_stats_ols_fit(y, x, true, true, 0.95) as fit
    FROM training_data
),

-- Stage 2: Generate predictions and residuals
predictions AS (
    SELECT
        training_data.y as actual,
        anofox_stats_predict(training_data.x, model.fit.coefficients, model.fit.intercept) as predicted
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
),

-- Stage 4: Identify outliers
outliers AS (
    SELECT
        unnest(generate_series(1, list_count(resid_diag.raw))) as idx,
        unnest(resid_diag.raw) as residual
    FROM diagnostics
    WHERE ABS(unnest(resid_diag.raw)) > 2 * STDDEV(unnest(resid_diag.raw))
)

-- Final output
SELECT
    model.fit.r_squared as model_r_squared,
    model.fit.coefficients as coefficients,
    diagnostics.normality_test.p_value as normality_pvalue,
    (SELECT COUNT(*) FROM outliers) as outlier_count
FROM model, diagnostics;
```

### Pattern: Model Selection with Information Criteria

Compare multiple model specifications.

```sql
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
        anofox_stats_ols_fit(y_arr, [x1_arr]) as fit,
        2 as k,
        n
    FROM data
    UNION ALL
    SELECT
        'Model 2: x1 + x2' as model_name,
        anofox_stats_ols_fit(y_arr, [x1_arr, x2_arr]) as fit,
        3 as k,
        n
    FROM data
    UNION ALL
    SELECT
        'Model 3: x1 + x2 + x3' as model_name,
        anofox_stats_ols_fit(y_arr, [x1_arr, x2_arr, x3_arr]) as fit,
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
WITH time_series AS (
    SELECT
        date,
        value,
        LAG(value, 1) OVER (ORDER BY date) as lag1
    FROM daily_prices
    WHERE LAG(value, 1) OVER (ORDER BY date) IS NOT NULL
),
rolling_betas AS (
    SELECT
        date,
        value,
        -- Short-term beta (10-day window)
        (anofox_stats_ols_fit_agg(value, [lag1]) OVER (
            ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        )).coefficients[1] as beta_short,
        -- Long-term beta (30-day window)
        (anofox_stats_ols_fit_agg(value, [lag1]) OVER (
            ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )).coefficients[1] as beta_long
    FROM time_series
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
ORDER BY date;
```

### Pattern: Seasonality Decomposition

Separate trend from seasonal components.

```sql
WITH monthly_data AS (
    SELECT
        date_trunc('month', date) as month,
        AVG(value) as value,
        EXTRACT(month FROM date) as month_num
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
        -- ... (continue for all months, omit December as base)
        CASE WHEN month_num = 11 THEN 1.0 ELSE 0.0 END as nov
    FROM monthly_data
),
-- Fit trend + seasonal model
model AS (
    SELECT anofox_stats_ols_fit(
        array_agg(value ORDER BY month),
        [
            array_agg(trend_idx::DOUBLE ORDER BY month),
            array_agg(jan ORDER BY month),
            array_agg(feb ORDER BY month),
            array_agg(mar ORDER BY month)
            -- ... include all seasonal dummies
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

```sql
WITH streaming_data AS (
    SELECT
        timestamp,
        target_value,
        feature_1,
        feature_2,
        -- RLS with forgetting factor 0.98 (recent data weighted more)
        anofox_stats_rls_fit_agg(
            target_value,
            [feature_1, feature_2],
            0.98,    -- forgetting_factor
            true,    -- fit_intercept
            100.0    -- initial_p_diagonal
        ) OVER (ORDER BY timestamp) as rls_model
    FROM sensor_readings
)
SELECT
    timestamp,
    target_value,
    rls_model.coefficients[1] as adaptive_coef_1,
    rls_model.coefficients[2] as adaptive_coef_2,
    rls_model.intercept as adaptive_intercept,
    -- Generate forecast
    rls_model.intercept +
    rls_model.coefficients[1] * LEAD(feature_1) OVER (ORDER BY timestamp) +
    rls_model.coefficients[2] * LEAD(feature_2) OVER (ORDER BY timestamp) as next_forecast
FROM streaming_data;
```

---

## Hierarchical Analysis

### Pattern: Multi-Level Regression

Analyze data at multiple organizational levels.

```sql
WITH store_data AS (
    SELECT
        company,
        region,
        territory,
        store_id,
        sales,
        traffic,
        promotions
    FROM retail_sales
),
-- Level 1: Store-level analysis
store_models AS (
    SELECT
        store_id,
        territory,
        region,
        company,
        (anofox_stats_ols_fit_agg(sales, [traffic, promotions])).coefficients[1] as traffic_coef,
        (anofox_stats_ols_fit_agg(sales, [traffic, promotions])).coefficients[2] as promo_coef,
        (anofox_stats_ols_fit_agg(sales, [traffic, promotions])).r_squared as r_squared,
        COUNT(*) as obs
    FROM store_data
    GROUP BY store_id, territory, region, company
),
-- Level 2: Territory aggregation
territory_summary AS (
    SELECT
        territory,
        region,
        AVG(traffic_coef) as avg_traffic_effect,
        STDDEV(traffic_coef) as std_traffic_effect,
        AVG(promo_coef) as avg_promo_effect,
        COUNT(*) as store_count
    FROM store_models
    GROUP BY territory, region
),
-- Level 3: Regional benchmarks
region_benchmarks AS (
    SELECT
        region,
        AVG(avg_traffic_effect) as region_traffic_benchmark,
        AVG(avg_promo_effect) as region_promo_benchmark
    FROM territory_summary
    GROUP BY region
)
-- Compare stores to benchmarks
SELECT
    s.store_id,
    s.territory,
    s.region,
    s.traffic_coef,
    t.avg_traffic_effect as territory_avg,
    r.region_traffic_benchmark as region_avg,
    s.traffic_coef - t.avg_traffic_effect as vs_territory,
    s.traffic_coef - r.region_traffic_benchmark as vs_region,
    CASE
        WHEN s.traffic_coef > t.avg_traffic_effect * 1.1 THEN 'OUTPERFORMER'
        WHEN s.traffic_coef < t.avg_traffic_effect * 0.9 THEN 'UNDERPERFORMER'
        ELSE 'AVERAGE'
    END as performance_tier
FROM store_models s
JOIN territory_summary t ON s.territory = t.territory
JOIN region_benchmarks r ON s.region = r.region
ORDER BY s.traffic_coef DESC;
```

---

## Cohort Analysis

### Pattern: Lifetime Value Curves by Cohort

Model customer value trajectories for different acquisition cohorts.

```sql
WITH cohort_data AS (
    SELECT
        cohort_month,
        months_since_acquisition,
        cumulative_revenue
    FROM customer_revenue_by_cohort
),
-- Fit growth model per cohort: revenue = a * months^b (log-linearized)
cohort_models AS (
    SELECT
        cohort_month,
        (anofox_stats_ols_fit_agg(
            LN(cumulative_revenue + 1),
            [LN(months_since_acquisition + 1)]
        )).coefficients[1] as growth_rate,
        (anofox_stats_ols_fit_agg(
            LN(cumulative_revenue + 1),
            [LN(months_since_acquisition + 1)]
        )).intercept as initial_value,
        (anofox_stats_ols_fit_agg(
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
    SELECT anofox_stats_ols_fit(
        array_agg(converted),
        [
            array_agg(is_treatment),
            array_agg(age),
            array_agg(tenure)
        ],
        true, true, 0.95
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
WITH experiment_data AS (
    SELECT
        user_id,
        segment,
        is_treatment,
        outcome
    FROM ab_test_results
)
SELECT
    segment,
    COUNT(*) as sample_size,
    (anofox_stats_ols_fit_agg(outcome, [is_treatment], true, true, 0.95)).coefficients[1] as treatment_effect,
    (anofox_stats_ols_fit_agg(outcome, [is_treatment], true, true, 0.95)).p_values[1] as p_value,
    (anofox_stats_ols_fit_agg(outcome, [is_treatment], true, true, 0.95)).ci_lower[1] as ci_lower,
    (anofox_stats_ols_fit_agg(outcome, [is_treatment], true, true, 0.95)).ci_upper[1] as ci_upper
FROM experiment_data
GROUP BY segment
HAVING COUNT(*) >= 100  -- Minimum sample size
ORDER BY treatment_effect DESC;
```

---

## Causal Analysis

### Pattern: Difference-in-Differences

Estimate causal effects from observational data.

```sql
WITH did_data AS (
    SELECT
        unit_id,
        time_period,
        CASE WHEN is_treated THEN 1.0 ELSE 0.0 END as treatment,
        CASE WHEN time_period >= treatment_start THEN 1.0 ELSE 0.0 END as post,
        CASE WHEN is_treated AND time_period >= treatment_start THEN 1.0 ELSE 0.0 END as treatment_x_post,
        outcome
    FROM panel_data
),
-- DiD regression: outcome = α + β₁*treatment + β₂*post + β₃*treatment×post + ε
did_model AS (
    SELECT anofox_stats_ols_fit(
        array_agg(outcome),
        [
            array_agg(treatment),
            array_agg(post),
            array_agg(treatment_x_post)
        ],
        true, true, 0.95
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
-- Create model cache table
CREATE OR REPLACE TABLE model_cache AS
WITH latest_data AS (
    SELECT
        category,
        array_agg(y::DOUBLE) as y_arr,
        array_agg(x1::DOUBLE) as x1_arr,
        array_agg(x2::DOUBLE) as x2_arr
    FROM training_data
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY category
)
SELECT
    category,
    CURRENT_TIMESTAMP as trained_at,
    anofox_stats_ols_fit(y_arr, [x1_arr, x2_arr], true, true, 0.95) as model
FROM latest_data;

-- Query cached model for predictions
SELECT
    new_data.id,
    anofox_stats_predict(
        [[new_data.x1, new_data.x2]],
        cache.model.coefficients,
        cache.model.intercept
    )[1] as prediction
FROM new_data
JOIN model_cache cache ON new_data.category = cache.category;
```

### Pattern: Scheduled Model Refresh

Automate model retraining with drift detection.

```sql
-- Check if model needs retraining
WITH current_performance AS (
    SELECT
        category,
        (anofox_stats_ols_fit_agg(actual, [predicted])).r_squared as current_r2
    FROM recent_predictions
    GROUP BY category
),
baseline_performance AS (
    SELECT category, model.r_squared as baseline_r2
    FROM model_cache
)
SELECT
    c.category,
    c.current_r2,
    b.baseline_r2,
    c.current_r2 / b.baseline_r2 as performance_ratio,
    CASE
        WHEN c.current_r2 / b.baseline_r2 < 0.9 THEN 'RETRAIN_NEEDED'
        ELSE 'OK'
    END as status
FROM current_performance c
JOIN baseline_performance b ON c.category = b.category;
```

### Pattern: Large-Scale Parallel Processing

Partition data for parallel regression.

```sql
-- Partition large dataset
WITH partitioned AS (
    SELECT
        NTILE(10) OVER (ORDER BY id) as partition_id,
        y, x1, x2
    FROM large_dataset
)
-- Process each partition (can be parallelized)
SELECT
    partition_id,
    (anofox_stats_ols_fit_agg(y, [x1, x2])).coefficients as partition_coefs,
    (anofox_stats_ols_fit_agg(y, [x1, x2])).r_squared as partition_r2,
    COUNT(*) as partition_size
FROM partitioned
GROUP BY partition_id;
```

### Pattern: Export for External Tools

Export model results for downstream systems.

```sql
-- Export to JSON format
SELECT
    json_object(
        'model_type', 'ols',
        'coefficients', fit.coefficients,
        'intercept', fit.intercept,
        'r_squared', fit.r_squared,
        'n_observations', fit.n_observations,
        'trained_at', CURRENT_TIMESTAMP
    ) as model_json
FROM (
    SELECT anofox_stats_ols_fit(y, x) as fit FROM training_data
);

-- Export to CSV for reporting
COPY (
    SELECT
        category,
        model.coefficients[1] as coef_1,
        model.coefficients[2] as coef_2,
        model.intercept,
        model.r_squared,
        model.p_values[1] as coef_1_pvalue,
        model.p_values[2] as coef_2_pvalue
    FROM model_cache
) TO 'model_summary.csv' (HEADER, DELIMITER ',');
```

---

## Best Practices for Production

### 1. Validate Before Deployment

```sql
-- Cross-validation pattern
WITH folds AS (
    SELECT *, NTILE(5) OVER (ORDER BY random()) as fold FROM data
),
cv_results AS (
    SELECT
        fold as test_fold,
        (anofox_stats_ols_fit_agg(y, [x1, x2]) FILTER (WHERE fold != test_fold)).r_squared as train_r2
    FROM folds
    GROUP BY fold
)
SELECT
    AVG(train_r2) as mean_cv_r2,
    STDDEV(train_r2) as std_cv_r2
FROM cv_results;
```

### 2. Monitor Drift

```sql
-- Track coefficient stability over time
SELECT
    date_trunc('week', prediction_date) as week,
    (anofox_stats_ols_fit_agg(actual, [predicted])).coefficients[1] as weekly_calibration
FROM predictions
GROUP BY 1
ORDER BY 1;
```

### 3. Document Assumptions

```sql
-- Store model metadata
INSERT INTO model_registry (
    model_id,
    model_type,
    features,
    assumptions,
    validation_r2,
    created_at
)
SELECT
    gen_random_uuid(),
    'OLS',
    ['price', 'advertising'],
    'Linear relationship, homoscedastic errors, no multicollinearity',
    (SELECT r_squared FROM model_fit),
    CURRENT_TIMESTAMP;
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
