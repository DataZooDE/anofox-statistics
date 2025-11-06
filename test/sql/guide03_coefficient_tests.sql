LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

SELECT
    variable,
    estimate,
    std_error,
    t_statistic,
    p_value,
    significant  -- TRUE if p < 0.05
FROM ols_inference(
    [65.0, 72.0, 78.0, 85.0, 92.0, 88.0]::DOUBLE[],                          -- y: exam_score
    [[3.0, 7.0], [4.0, 8.0], [5.0, 7.0], [6.0, 8.0], [7.0, 9.0], [6.5, 7.5]]::DOUBLE[][], -- x: study_hours, sleep_hours
    0.95,                                                                      -- confidence_level
    true                                                                       -- add_intercept
);
