LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

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
    SELECT anofox_stats_ols_fit_agg(force_newtons, [mass_kg], {'intercept': true}) as result
    FROM physics_data
) sub
UNION ALL
SELECT
    'Physics: Without intercept (correct)' as model_type,
    result.intercept,
    result.coefficients[1] as acceleration_estimate,
    result.r2
FROM (
    SELECT anofox_stats_ols_fit_agg(force_newtons, [mass_kg], {'intercept': false}) as result
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
    SELECT anofox_stats_ols_fit_agg(revenue, [employees], {'intercept': true}) as result
    FROM business_data
) sub
UNION ALL
SELECT
    'Business: Without intercept (wrong)' as model_type,
    result.intercept,
    result.coefficients[1] as biased_estimate,
    result.r2
FROM (
    SELECT anofox_stats_ols_fit_agg(revenue, [employees], {'intercept': false}) as result
    FROM business_data
) sub;

-- Key insight: R² comparison
SELECT
    'R² comparison' as note,
    'intercept=true uses SS from mean, intercept=false uses SS from zero' as explanation;
