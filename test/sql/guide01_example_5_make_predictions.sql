LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Predict for new values (use positional parameters and literal arrays)
SELECT * FROM ols_predict_interval(
    [1.0, 2.0, 3.0, 4.0, 5.0]::DOUBLE[],          -- y_train
    [[1.0], [2.0], [3.0], [4.0], [5.0]]::DOUBLE[][], -- x_train
    [[6.0], [7.0], [8.0]]::DOUBLE[][],             -- x_new: values to predict
    0.95,                                           -- confidence_level
    'prediction',                                   -- interval_type
    true                                            -- add_intercept
);
