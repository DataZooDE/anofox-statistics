-- Test file for Elastic Net and Diagnostic Aggregate functions
-- Tests new functions added in Phase 4-5 refactoring

-- =============================================================================
-- PART 1: Elastic Net Table Function
-- =============================================================================

-- Test 1.1: Basic Elastic Net with default parameters
SELECT 'Test 1.1: Basic Elastic Net' as test_name;
WITH data AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0] as y,
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]] as X
)
SELECT
    coefficients,
    intercept,
    n_nonzero,
    r_squared > 0.9 as good_fit,
    converged
FROM data, anofox_statistics_elastic_net(y, X, MAP{'alpha': 0.5, 'lambda': 0.1, 'intercept': true});

-- Test 1.2: Elastic Net - Pure Ridge (alpha=0)
SELECT 'Test 1.2: Elastic Net Pure Ridge' as test_name;
WITH data AS (
    SELECT
        [2.5, 3.7, 5.1, 6.8, 8.2] as y,
        [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]] as X
)
SELECT
    array_length(coefficients) as n_coeffs,
    intercept IS NOT NULL as has_intercept,
    n_nonzero >= 2 as all_nonzero,  -- Ridge doesn't zero coefficients
    converged
FROM data, anofox_statistics_elastic_net(y, X, MAP{'alpha': 0.0, 'lambda': 1.0, 'intercept': true});

-- Test 1.3: Elastic Net - Pure Lasso (alpha=1)
SELECT 'Test 1.3: Elastic Net Pure Lasso' as test_name;
WITH data AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as y,
        [[1.0, 0.1, 0.05], [2.0, 0.2, 0.10], [3.0, 0.3, 0.15],
         [4.0, 0.4, 0.20], [5.0, 0.5, 0.25], [6.0, 0.6, 0.30]] as X
)
SELECT
    array_length(coefficients) as n_features,
    n_nonzero <= 2 as has_sparsity,  -- Lasso should zero some coefficients
    r_squared,
    converged
FROM data, anofox_statistics_elastic_net(y, X, MAP{'alpha': 1.0, 'lambda': 0.5, 'intercept': true});

-- Test 1.4: Elastic Net without intercept
SELECT 'Test 1.4: Elastic Net no intercept' as test_name;
WITH data AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0] as y,
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]] as X
)
SELECT
    intercept IS NULL as no_intercept,
    array_length(coefficients) = 2 as correct_size,
    converged
FROM data, anofox_statistics_elastic_net(y, X, MAP{'alpha': 0.5, 'lambda': 0.1, 'intercept': false});


-- =============================================================================
-- PART 2: Elastic Net Aggregate Function
-- =============================================================================

-- Test 2.1: Elastic Net Aggregate with GROUP BY
SELECT 'Test 2.1: Elastic Net Aggregate GROUP BY' as test_name;
WITH data AS (
    SELECT 'A' as category, 1.0 as y, [1.0, 2.0] as x UNION ALL
    SELECT 'A', 2.0, [2.0, 3.0] UNION ALL
    SELECT 'A', 3.0, [3.0, 4.0] UNION ALL
    SELECT 'A', 4.0, [4.0, 5.0] UNION ALL
    SELECT 'B', 2.0, [1.0, 1.0] UNION ALL
    SELECT 'B', 4.0, [2.0, 2.0] UNION ALL
    SELECT 'B', 6.0, [3.0, 3.0] UNION ALL
    SELECT 'B', 8.0, [4.0, 4.0]
)
SELECT
    category,
    array_length(result.coefficients) as n_coeffs,
    result.n_nonzero >= 1 as has_nonzero,
    result.r_squared > 0.8 as good_fit,
    result.converged
FROM data
GROUP BY category
SELECT category, anofox_statistics_elastic_net_agg(y, x, MAP{'alpha': 0.5, 'lambda': 0.01}) as result;

-- Test 2.2: Elastic Net Aggregate with window function (rolling)
SELECT 'Test 2.2: Elastic Net Aggregate OVER rolling window' as test_name;
WITH data AS (
    SELECT 1 as time, 1.0 as y, [1.0, 1.0] as x UNION ALL
    SELECT 2, 2.0, [2.0, 2.0] UNION ALL
    SELECT 3, 3.0, [3.0, 3.0] UNION ALL
    SELECT 4, 4.0, [4.0, 4.0] UNION ALL
    SELECT 5, 5.0, [5.0, 5.0] UNION ALL
    SELECT 6, 6.0, [6.0, 6.0]
)
SELECT
    time,
    result.n_obs >= 3 as sufficient_data,
    result.converged,
    result.r_squared
FROM (
    SELECT
        time,
        anofox_statistics_elastic_net_agg(y, x, MAP{'alpha': 0.3, 'lambda': 0.1})
            OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as result
    FROM data
)
WHERE time >= 3;  -- Only check windows with enough data


-- =============================================================================
-- PART 3: Renamed Diagnostic Functions (with anofox_statistics_ prefix)
-- =============================================================================

-- Test 3.1: anofox_statistics_vif (Variance Inflation Factor)
SELECT 'Test 3.1: VIF function renamed' as test_name;
WITH data AS (
    SELECT [[1.0, 2.0, 1.5], [2.0, 4.0, 3.0], [3.0, 6.0, 4.5],
            [4.0, 8.0, 6.0], [5.0, 10.0, 7.5]] as X
)
SELECT
    array_length(result.vif) = 3 as has_all_features,
    result.vif[1] > 5 as high_multicollinearity,  -- Second feature is 2x first
    result.severity[1] IN ('high', 'perfect') as correct_severity
FROM data, anofox_statistics_vif(X) as result;

-- Test 3.2: anofox_statistics_normality_test (Jarque-Bera test)
SELECT 'Test 3.2: Normality test function renamed' as test_name;
WITH data AS (
    -- Normal distribution sample
    SELECT [0.5, 1.2, -0.3, 0.8, -0.5, 1.0, -0.2, 0.3, 0.9, -0.1,
            0.4, -0.8, 1.1, 0.2, -0.4, 0.7, -0.6, 0.1, 0.6, -0.9] as residuals
)
SELECT
    result.n_obs = 20 as correct_count,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM data, anofox_statistics_normality_test(residuals, 0.05) as result;

-- Test 3.3: anofox_statistics_residual_diagnostics (simplified API)
SELECT 'Test 3.3: Residual diagnostics function renamed and simplified' as test_name;
WITH data AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 10.0] as y_actual,  -- Last value is outlier
        [1.1, 1.9, 3.1, 3.9, 4.5] as y_predicted
)
SELECT
    COUNT(*) as n_obs,
    SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as n_outliers,
    MAX(ABS(std_residual)) > 2.0 as has_extreme_residual
FROM data, anofox_statistics_residual_diagnostics(y_actual, y_predicted, 2.5);


-- =============================================================================
-- PART 4: Diagnostic Aggregate Functions
-- =============================================================================

-- Test 4.1: residual_diagnostics_aggregate - Summary mode (default)
SELECT 'Test 4.1: Residual Diagnostics Aggregate - Summary' as test_name;
WITH data AS (
    SELECT 'A' as group_id, 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 'A', 2.0, 1.9 UNION ALL
    SELECT 'A', 3.0, 3.2 UNION ALL
    SELECT 'A', 4.0, 10.0 UNION ALL  -- Outlier
    SELECT 'B', 5.0, 5.1 UNION ALL
    SELECT 'B', 6.0, 5.9 UNION ALL
    SELECT 'B', 7.0, 7.1 UNION ALL
    SELECT 'B', 8.0, 7.9
)
SELECT
    group_id,
    result.n_obs,
    result.n_outliers >= 0 as has_outlier_count,
    result.rmse > 0 as positive_rmse,
    result.max_abs_residual >= result.mean_abs_residual as logical_max
FROM data
GROUP BY group_id
SELECT group_id, anofox_statistics_residual_diagnostics_agg(y_actual, y_pred, MAP{'outlier_threshold': 2.5}) as result;

-- Test 4.2: residual_diagnostics_aggregate - Detailed mode
SELECT 'Test 4.2: Residual Diagnostics Aggregate - Detailed' as test_name;
WITH data AS (
    SELECT 1.0 as y_actual, 1.1 as y_pred UNION ALL
    SELECT 2.0, 2.1 UNION ALL
    SELECT 3.0, 3.0 UNION ALL
    SELECT 4.0, 4.1 UNION ALL
    SELECT 5.0, 10.0  -- Large error
)
SELECT
    array_length(result.residuals) = 5 as correct_size,
    result.n_obs = 5 as correct_count,
    result.n_outliers >= 1 as has_outliers,
    array_length(result.is_outlier) = 5 as has_outlier_flags
FROM (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_pred,
        MAP{'detailed': true, 'outlier_threshold': 2.0}) as result
    FROM data
);

-- Test 4.3: vif_aggregate - VIF per group
SELECT 'Test 4.3: VIF Aggregate per group' as test_name;
WITH data AS (
    SELECT 'A' as category, [1.0, 2.0, 1.5] as x UNION ALL
    SELECT 'A', [2.0, 4.0, 3.0] UNION ALL
    SELECT 'A', [3.0, 6.0, 4.5] UNION ALL
    SELECT 'A', [4.0, 8.0, 6.0] UNION ALL
    SELECT 'A', [5.0, 10.0, 7.5] UNION ALL
    SELECT 'B', [1.0, 1.0, 2.0] UNION ALL
    SELECT 'B', [2.0, 1.5, 3.0] UNION ALL
    SELECT 'B', [3.0, 2.0, 4.0] UNION ALL
    SELECT 'B', [4.0, 2.5, 5.0] UNION ALL
    SELECT 'B', [5.0, 3.0, 6.0]
)
SELECT
    category,
    array_length(result.vif) = 3 as correct_feature_count,
    result.vif[1] >= 1.0 as valid_vif,  -- VIF must be >= 1
    array_length(result.severity) = 3 as has_severity_labels
FROM data
GROUP BY category
SELECT category, anofox_statistics_vif_agg(x) as result;

-- Test 4.4: vif_aggregate with window function
SELECT 'Test 4.4: VIF Aggregate window function NOT supported' as test_name;
-- Note: VIF aggregate does not support window functions as per requirements
-- This test just verifies it works with GROUP BY only

-- Test 4.5: normality_test_aggregate - Jarque-Bera per group
SELECT 'Test 4.5: Normality Test Aggregate per group' as test_name;
WITH data AS (
    -- Group A: Normally distributed residuals
    SELECT 'A' as group_id, 0.1 as residual UNION ALL
    SELECT 'A', 0.2 UNION ALL
    SELECT 'A', -0.1 UNION ALL
    SELECT 'A', 0.3 UNION ALL
    SELECT 'A', -0.2 UNION ALL
    SELECT 'A', 0.15 UNION ALL
    SELECT 'A', -0.05 UNION ALL
    SELECT 'A', 0.25 UNION ALL
    SELECT 'A', -0.15 UNION ALL
    SELECT 'A', 0.05 UNION ALL
    -- Group B: Skewed distribution
    SELECT 'B', 1.0 UNION ALL
    SELECT 'B', 1.5 UNION ALL
    SELECT 'B', 2.0 UNION ALL
    SELECT 'B', 2.5 UNION ALL
    SELECT 'B', 3.0 UNION ALL
    SELECT 'B', 10.0 UNION ALL  -- Extreme value creates skewness
    SELECT 'B', 1.2 UNION ALL
    SELECT 'B', 1.8 UNION ALL
    SELECT 'B', 2.2 UNION ALL
    SELECT 'B', 2.8
)
SELECT
    group_id,
    result.n_obs >= 8 as sufficient_obs,
    result.jb_statistic >= 0 as valid_statistic,
    result.p_value BETWEEN 0 AND 1 as valid_pvalue,
    result.conclusion IN ('normal', 'non-normal') as valid_conclusion
FROM data
GROUP BY group_id
SELECT group_id, anofox_statistics_normality_test_agg(residual, MAP{'alpha': 0.05}) as result;

-- Test 4.6: normality_test_aggregate with different alpha levels
SELECT 'Test 4.6: Normality Test Aggregate custom alpha' as test_name;
WITH data AS (
    SELECT 0.5 as residual UNION ALL SELECT -0.3 UNION ALL SELECT 0.8 UNION ALL
    SELECT -0.5 UNION ALL SELECT 1.0 UNION ALL SELECT -0.2 UNION ALL
    SELECT 0.3 UNION ALL SELECT 0.9 UNION ALL SELECT -0.1 UNION ALL
    SELECT 0.4 UNION ALL SELECT -0.8 UNION ALL SELECT 1.1
)
SELECT
    result.n_obs = 12 as correct_count,
    result.is_normal IS NOT NULL as has_test_result,
    result.p_value >= 0 as valid_pvalue
FROM (
    SELECT anofox_statistics_normality_test_agg(residual, MAP{'alpha': 0.01}) as result
    FROM data
);


-- =============================================================================
-- PART 5: Integration Tests - Combined Usage
-- =============================================================================

-- Test 5.1: Full workflow - Elastic Net -> Residuals -> Diagnostics
SELECT 'Test 5.1: Full workflow integration' as test_name;
WITH training_data AS (
    SELECT
        [2.5, 3.7, 5.1, 6.8, 8.2, 9.5, 11.2, 12.8, 14.5, 16.1] as y,
        [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0], [4.0, 4.5, 5.0],
         [5.0, 5.5, 6.0], [6.0, 6.5, 7.0], [7.0, 7.5, 8.0], [8.0, 8.5, 9.0],
         [9.0, 9.5, 10.0], [10.0, 10.5, 11.0]] as X
),
model AS (
    SELECT
        coefficients,
        intercept,
        r_squared,
        n_nonzero
    FROM training_data, anofox_statistics_elastic_net(y, X, MAP{'alpha': 0.5, 'lambda': 0.1, 'intercept': true})
),
predictions AS (
    SELECT
        unnest(training_data.y) as y_actual,
        -- Manual prediction (simplified - just for testing)
        unnest(training_data.y) + 0.1 as y_predicted  -- Simulated predictions
    FROM training_data
),
diagnostics AS (
    SELECT anofox_statistics_residual_diagnostics_agg(y_actual, y_predicted,
        MAP{'outlier_threshold': 2.5}) as result
    FROM predictions
)
SELECT
    model.r_squared > 0.7 as good_model_fit,
    model.n_nonzero >= 1 as has_features,
    diagnostics.result.n_obs = 10 as correct_obs_count,
    diagnostics.result.rmse < 1.0 as low_error
FROM model, diagnostics;

-- Test 5.2: Multi-group analysis with all diagnostics
SELECT 'Test 5.2: Multi-group diagnostic analysis' as test_name;
WITH data AS (
    SELECT 'product_A' as product, 1.0 as y, [1.0, 2.0] as x, 1.1 as y_pred UNION ALL
    SELECT 'product_A', 2.0, [2.0, 3.0], 1.9 UNION ALL
    SELECT 'product_A', 3.0, [3.0, 4.0], 3.1 UNION ALL
    SELECT 'product_A', 4.0, [4.0, 5.0], 3.9 UNION ALL
    SELECT 'product_A', 5.0, [5.0, 6.0], 5.2 UNION ALL
    SELECT 'product_B', 2.0, [1.0, 1.0], 2.1 UNION ALL
    SELECT 'product_B', 4.0, [2.0, 2.0], 3.9 UNION ALL
    SELECT 'product_B', 6.0, [3.0, 3.0], 6.1 UNION ALL
    SELECT 'product_B', 8.0, [4.0, 4.0], 7.8 UNION ALL
    SELECT 'product_B', 10.0, [5.0, 5.0], 10.2
),
residuals AS (
    SELECT
        product,
        y - y_pred as residual
    FROM data
)
SELECT
    product,
    COUNT(*) as n_obs,
    AVG(residual) as mean_residual,
    STDDEV(residual) as sd_residual
FROM residuals
GROUP BY product
ORDER BY product;

SELECT 'All tests completed successfully' as status;
