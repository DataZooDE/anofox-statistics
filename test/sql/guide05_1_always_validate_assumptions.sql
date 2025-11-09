LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Create sample data for validation checks
CREATE TEMP TABLE data AS
SELECT
    (i + random() * 5)::DOUBLE as y,
    (i * 2)::DOUBLE as x
FROM generate_series(1, 50) t(i);

-- Check list before deploying model (using literal array examples)
WITH validation AS (
    SELECT
        'Sample Size' as check_name,
        CAST((SELECT COUNT(*) FROM data) AS VARCHAR) as value,
        CASE WHEN (SELECT COUNT(*) FROM data) >= 30 THEN 'PASS' ELSE 'FAIL' END as status
    UNION ALL
    SELECT
        'Multicollinearity',
        CAST(MAX(vif) AS VARCHAR),
        CASE WHEN MAX(vif) < 10 THEN 'PASS' ELSE 'FAIL' END
    FROM anofox_statistics_vif([[1.0::DOUBLE, 2.0, 3.0], [1.1::DOUBLE, 2.1, 3.1], [1.2::DOUBLE, 2.2, 3.2], [1.3::DOUBLE, 2.3, 3.3], [1.4::DOUBLE, 2.4, 3.4]])
    UNION ALL
    SELECT
        'Normality',
        CAST(p_value AS VARCHAR),
        CASE WHEN p_value > 0.05 THEN 'PASS' ELSE 'FAIL' END
    FROM anofox_statistics_normality_test([0.1::DOUBLE, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25], 0.05)
    UNION ALL
    SELECT
        'Outliers',
        CAST(COUNT(*) AS VARCHAR),
        CASE WHEN COUNT(*) < 0.05 * (SELECT COUNT(*) FROM data) THEN 'PASS' ELSE 'WARN' END
    FROM anofox_statistics_residual_diagnostics(
        [100.0::DOUBLE, 110.0, 120.0, 130.0, 140.0],
        [102.0::DOUBLE, 108.0, 118.0, 132.0, 138.0],
        outlier_threshold := 2.5
    ) WHERE is_outlier
)
SELECT * FROM validation;
