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
            {'intercept': true}
        ) as ols_model,
        -- WLS: Addresses heteroscedasticity
        anofox_statistics_wls_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            observation_weight,
            {'intercept': true}
        ) as wls_model,
        -- Ridge: Handles multicollinearity
        anofox_statistics_ridge_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            {'lambda': 1.0, 'intercept': true}
        ) as ridge_model,
        -- RLS: Adaptive to changes
        anofox_statistics_rls_agg(
            y,
            [x1_correlated, x2_correlated, x3_independent],
            {'forgetting_factor': 0.95, 'intercept': true}
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
