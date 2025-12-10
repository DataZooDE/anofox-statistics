LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Get coefficient statistics using fit with full_output
WITH model AS (
    SELECT * FROM anofox_stats_ols_fit(
        [65.0, 72.0, 78.0, 85.0, 92.0, 88.0]::DOUBLE[],                          -- y: exam_score
        [[3.0, 7.0], [4.0, 8.0], [5.0, 7.0], [6.0, 8.0], [7.0, 9.0], [6.5, 7.5]]::DOUBLE[][], -- x: study_hours, sleep_hours
        {'intercept': true, 'full_output': true, 'confidence_level': 0.95}
    )
)
SELECT
    'x1' as variable,
    coefficients[1] as estimate,
    coefficient_std_errors[1] as std_error,
    coefficient_t_statistics[1] as t_statistic,
    coefficient_p_values[1] as p_value,
    coefficient_p_values[1] < 0.05 as significant
FROM model
UNION ALL
SELECT
    'x2' as variable,
    coefficients[2] as estimate,
    coefficient_std_errors[2] as std_error,
    coefficient_t_statistics[2] as t_statistic,
    coefficient_p_values[2] as p_value,
    coefficient_p_values[2] < 0.05 as significant
FROM model
UNION ALL
SELECT
    'intercept' as variable,
    intercept as estimate,
    intercept_std_error as std_error,
    intercept_t_statistic as t_statistic,
    intercept_p_value as p_value,
    intercept_p_value < 0.05 as significant
FROM model;
