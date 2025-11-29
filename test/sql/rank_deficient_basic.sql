-- Test rank-deficiency handling: constant features should get NULL coefficients

-- Test 1: Single constant feature
-- x2 is constant (all values = 5.0), should return NULL coefficient
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [                                       -- X: 2D array (5 rows x 2 columns)
        [1.0, 5.0],                       -- Row 1: x1=1.0, x2=5.0 (constant)
        [2.0, 5.0],                       -- Row 2
        [3.0, 5.0],                       -- Row 3
        [4.0, 5.0],                       -- Row 4
        [5.0, 5.0]                        -- Row 5
    ]::DOUBLE[][],
    {'intercept': true}
);

-- Expected output:
-- coefficients: [valid_value, NULL] where x1 has a coefficient, x2 is NULL

-- Test 2: All valid features (regression test)
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [
        [1.0, 2.0],                       -- Row 1: x1=1.0, x2=2.0
        [2.0, 3.0],                       -- Row 2: x1=2.0, x2=3.0
        [3.0, 5.0],                       -- Row 3
        [4.0, 7.0],                       -- Row 4
        [5.0, 11.0]                       -- Row 5
    ]::DOUBLE[][],
    {'intercept': true}
);

-- Expected: Both coefficients should be valid (not NULL)

-- Test 3: Perfect collinearity (x2 = 2 * x1)
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [
        [1.0, 2.0],                       -- Row 1: x1=1.0, x2=2.0 (x2 = 2*x1)
        [2.0, 4.0],                       -- Row 2: x1=2.0, x2=4.0
        [3.0, 6.0],                       -- Row 3
        [4.0, 8.0],                       -- Row 4
        [5.0, 10.0]                       -- Row 5
    ]::DOUBLE[][],
    {'intercept': true}
);

-- Expected: One of x1 or x2 should be NULL (aliased due to perfect correlation)

-- Test 4: Multiple constant features
SELECT * FROM anofox_stats_ols_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],  -- y
    [
        [7.0, 5.0],                       -- Row 1: x1=7.0 (constant), x2=5.0 (constant)
        [7.0, 5.0],                       -- Row 2
        [7.0, 5.0],                       -- Row 3
        [7.0, 5.0],                       -- Row 4
        [7.0, 5.0]                        -- Row 5
    ]::DOUBLE[][],
    {'intercept': true}
);

-- Expected: Both x1 and x2 should be NULL
