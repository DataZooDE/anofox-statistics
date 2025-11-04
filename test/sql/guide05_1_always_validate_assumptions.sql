-- Check list before deploying model (using literal array examples)
WITH validation AS (
    SELECT
        'Sample Size' as check_name,
        CAST((SELECT COUNT(*) FROM data) AS VARCHAR) as value,
        CASE WHEN (SELECT COUNT(*) FROM data) >= 30 THEN 'PASS' ELSE 'FAIL' END as status
    UNION ALL
    SELECT
        'Multicollinearity',
        CAST(MAX(vif_value) AS VARCHAR),
        CASE WHEN MAX(vif_value) < 10 THEN 'PASS' ELSE 'FAIL' END
    FROM vif([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2], [1.3, 2.3, 3.3], [1.4, 2.4, 3.4]])
    UNION ALL
    SELECT
        'Normality',
        CAST(p_value AS VARCHAR),
        CASE WHEN p_value > 0.05 THEN 'PASS' ELSE 'FAIL' END
    FROM normality_test([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.15, -0.25])
    UNION ALL
    SELECT
        'Outliers',
        CAST(COUNT(*) AS VARCHAR),
        CASE WHEN COUNT(*) < 0.05 * (SELECT COUNT(*) FROM data) THEN 'PASS' ELSE 'WARN' END
    FROM residual_diagnostics(
        [100.0, 110.0, 120.0, 130.0, 140.0],
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]],
        true, 2.5, 0.5
    ) WHERE is_influential
)
SELECT * FROM validation;
