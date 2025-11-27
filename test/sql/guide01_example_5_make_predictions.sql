LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Predict for new values using the predict function
SELECT * FROM anofox_statistics_predict_ols(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],          -- y_train
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x_train
    [[6.0], [7.0], [8.0]]::DOUBLE[][],             -- x_new
    0.95,                                           -- confidence_level
    'prediction',                                   -- interval_type
    true                                            -- add_intercept
);
