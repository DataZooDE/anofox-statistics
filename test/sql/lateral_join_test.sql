LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- ============================================================================
-- Lateral Join Test Suite
-- Tests that all regression functions (OLS, Ridge, RLS, WLS) support both:
-- 1. Literal parameters (backward compatibility)
-- 2. Lateral joins with column references (new functionality)
-- ============================================================================

-- Test 1: OLS with literal parameters (backward compatibility)
-- This should work exactly as before
SELECT 'Test 1: OLS with literal parameters' as test_name;
SELECT * FROM anofox_statistics_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
    MAP{'intercept': true}
);

-- Test 2: OLS with lateral join
-- This tests the new lateral join functionality
SELECT 'Test 2: OLS with lateral join' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
)
SELECT result.* FROM input,
LATERAL anofox_statistics_ols_fit(
    input.y,
    input.X,
    MAP{'intercept': true}
) as result;

-- Test 3: Ridge with literal parameters (backward compatibility)
SELECT 'Test 3: Ridge with literal parameters' as test_name;
SELECT * FROM anofox_statistics_ridge_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
    MAP(['lambda', 'intercept'], [0.1::DOUBLE, 1.0::DOUBLE])
);

-- Test 4: Ridge with lateral join
SELECT 'Test 4: Ridge with lateral join' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
)
SELECT result.* FROM input,
LATERAL anofox_statistics_ridge_fit(
    input.y,
    input.X,
    MAP(['lambda', 'intercept'], [0.1::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 5: RLS with literal parameters (backward compatibility)
SELECT 'Test 5: RLS with literal parameters' as test_name;
SELECT * FROM anofox_statistics_rls_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
    MAP(['forgetting_factor', 'intercept'], [0.95::DOUBLE, 1.0::DOUBLE])
);

-- Test 6: RLS with lateral join
SELECT 'Test 6: RLS with lateral join' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
)
SELECT result.* FROM input,
LATERAL anofox_statistics_rls_fit(
    input.y,
    input.X,
    MAP(['forgetting_factor', 'intercept'], [0.95::DOUBLE, 1.0::DOUBLE])
) as result;

-- Test 7: WLS with literal parameters (backward compatibility)
SELECT 'Test 7: WLS with literal parameters' as test_name;
SELECT * FROM anofox_statistics_wls_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
    [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
    [1.0, 1.0, 1.0, 1.0, 1.0]::DOUBLE[],
    MAP{'intercept': true}
);

-- Test 8: WLS with lateral join
SELECT 'Test 8: WLS with lateral join' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
        [1.0, 1.0, 1.0, 1.0, 1.0]::DOUBLE[] as weights
)
SELECT result.* FROM input,
LATERAL anofox_statistics_wls_fit(
    input.y,
    input.X,
    input.weights,
    MAP{'intercept': true}
) as result;

-- Test 9: Multiple rows with lateral join (OLS)
-- This tests that lateral joins work correctly when processing multiple input rows
SELECT 'Test 9: Multiple rows with lateral join (OLS)' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
        'dataset1' as name
    UNION ALL
    SELECT
        [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[] as y,
        [[2.0, 4.0, 6.0, 8.0, 10.0]]::DOUBLE[][] as X,
        'dataset2' as name
)
SELECT input.name, result.* FROM input,
LATERAL anofox_statistics_ols_fit(
    input.y,
    input.X,
    MAP{'intercept': true}
) as result
ORDER BY input.name;

-- Test 10: Multiple rows with lateral join (Ridge)
SELECT 'Test 10: Multiple rows with lateral join (Ridge)' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
        'dataset1' as name
    UNION ALL
    SELECT
        [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[] as y,
        [[2.0, 4.0, 6.0, 8.0, 10.0]]::DOUBLE[][] as X,
        'dataset2' as name
)
SELECT input.name, result.* FROM input,
LATERAL anofox_statistics_ridge_fit(
    input.y,
    input.X,
    MAP(['lambda', 'intercept'], [0.1::DOUBLE, 1.0::DOUBLE])
) as result
ORDER BY input.name;

-- Test 11: Verify results match between literal and lateral join modes (OLS)
-- Both queries should produce identical results
SELECT 'Test 11: Verify OLS literal vs lateral results match' as test_name;
WITH literal_result AS (
    SELECT * FROM anofox_statistics_ols_fit(
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
        MAP{'intercept': true}
    )
),
lateral_result AS (
    SELECT result.* FROM (
        SELECT
            [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
            [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
    ) as input,
    LATERAL anofox_statistics_ols_fit(
        input.y,
        input.X,
        MAP{'intercept': true}
    ) as result
)
SELECT
    'Literal coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM literal_result
UNION ALL
SELECT
    'Lateral coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM lateral_result;

-- Test 12: Verify results match between literal and lateral join modes (Ridge)
SELECT 'Test 12: Verify Ridge literal vs lateral results match' as test_name;
WITH literal_result AS (
    SELECT * FROM anofox_statistics_ridge_fit(
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
        MAP(['lambda', 'intercept'], [0.5::DOUBLE, 1.0::DOUBLE])
    )
),
lateral_result AS (
    SELECT result.* FROM (
        SELECT
            [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
            [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
    ) as input,
    LATERAL anofox_statistics_ridge_fit(
        input.y,
        input.X,
        MAP(['lambda', 'intercept'], [0.5::DOUBLE, 1.0::DOUBLE])
    ) as result
)
SELECT
    'Literal coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM literal_result
UNION ALL
SELECT
    'Lateral coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM lateral_result;

-- Test 13: Multiple rows with lateral join (RLS)
SELECT 'Test 13: Multiple rows with lateral join (RLS)' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
        'dataset1' as name
    UNION ALL
    SELECT
        [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[] as y,
        [[2.0, 4.0, 6.0, 8.0, 10.0]]::DOUBLE[][] as X,
        'dataset2' as name
)
SELECT input.name, result.* FROM input,
LATERAL anofox_statistics_rls_fit(
    input.y,
    input.X,
    MAP(['forgetting_factor', 'intercept'], [0.95::DOUBLE, 1.0::DOUBLE])
) as result
ORDER BY input.name;

-- Test 14: Multiple rows with lateral join (WLS)
SELECT 'Test 14: Multiple rows with lateral join (WLS)' as test_name;
WITH input AS (
    SELECT
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
        [1.0, 1.0, 1.0, 1.0, 1.0]::DOUBLE[] as weights,
        'dataset1' as name
    UNION ALL
    SELECT
        [2.0, 4.0, 6.0, 8.0, 10.0]::DOUBLE[] as y,
        [[2.0, 4.0, 6.0, 8.0, 10.0]]::DOUBLE[][] as X,
        [1.0, 1.5, 2.0, 1.5, 1.0]::DOUBLE[] as weights,
        'dataset2' as name
)
SELECT input.name, result.* FROM input,
LATERAL anofox_statistics_wls_fit(
    input.y,
    input.X,
    input.weights,
    MAP{'intercept': true}
) as result
ORDER BY input.name;

-- Test 15: Verify RLS literal vs lateral results match
SELECT 'Test 15: Verify RLS literal vs lateral results match' as test_name;
WITH literal_result AS (
    SELECT * FROM anofox_statistics_rls_fit(
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
        MAP(['forgetting_factor', 'intercept'], [0.98::DOUBLE, 1.0::DOUBLE])
    )
),
lateral_result AS (
    SELECT result.* FROM (
        SELECT
            [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
            [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X
    ) as input,
    LATERAL anofox_statistics_rls_fit(
        input.y,
        input.X,
        MAP(['forgetting_factor', 'intercept'], [0.98::DOUBLE, 1.0::DOUBLE])
    ) as result
)
SELECT
    'Literal coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM literal_result
UNION ALL
SELECT
    'Lateral coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM lateral_result;

-- Test 16: Verify WLS literal vs lateral results match
SELECT 'Test 16: Verify WLS literal vs lateral results match' as test_name;
WITH literal_result AS (
    SELECT * FROM anofox_statistics_wls_fit(
        [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],
        [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][],
        [1.0, 1.0, 1.0, 1.0, 1.0]::DOUBLE[],
        MAP{'intercept': true}
    )
),
lateral_result AS (
    SELECT result.* FROM (
        SELECT
            [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[] as y,
            [[1.1, 2.1, 2.9, 4.2, 4.8]]::DOUBLE[][] as X,
            [1.0, 1.0, 1.0, 1.0, 1.0]::DOUBLE[] as weights
    ) as input,
    LATERAL anofox_statistics_wls_fit(
        input.y,
        input.X,
        input.weights,
        MAP{'intercept': true}
    ) as result
)
SELECT
    'Literal coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM literal_result
UNION ALL
SELECT
    'Lateral coefficients' as mode,
    coefficients,
    intercept,
    r_squared,
    adj_r_squared,
    mse,
    rmse,
    n_obs,
    n_features
FROM lateral_result;

-- Test 17: Complex lateral join scenario - per-group regression
-- This demonstrates a real-world use case where lateral joins are essential
SELECT 'Test 17: Complex per-group regression with lateral join' as test_name;
WITH sales_data AS (
    SELECT 'ProductA' as product,
           [100.0, 120.0, 140.0, 160.0, 180.0]::DOUBLE[] as sales,
           [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][] as time_periods
    UNION ALL
    SELECT 'ProductB' as product,
           [50.0, 55.0, 60.0, 65.0, 70.0]::DOUBLE[] as sales,
           [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][] as time_periods
    UNION ALL
    SELECT 'ProductC' as product,
           [200.0, 190.0, 180.0, 170.0, 160.0]::DOUBLE[] as sales,
           [[1.0, 2.0, 3.0, 4.0, 5.0]]::DOUBLE[][] as time_periods
)
SELECT
    sales_data.product,
    result.coefficients[1] as slope,
    result.intercept,
    result.r_squared,
    result.n_obs
FROM sales_data,
LATERAL anofox_statistics_ols_fit(
    sales_data.sales,
    sales_data.time_periods,
    MAP{'intercept': true}
) as result
ORDER BY sales_data.product;

SELECT 'All lateral join tests completed successfully!' as status;
