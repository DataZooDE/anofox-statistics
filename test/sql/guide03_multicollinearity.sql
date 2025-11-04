SELECT
    variable_name,
    vif,
    severity
FROM vif(
    [[10.0, 9.9, 10.1], [20.0, 19.8, 20.2], [30.0, 29.9, 30.1], [40.0, 39.7, 40.3]]::DOUBLE[][]
    -- x matrix: price, competitors_price, industry_avg_price (highly correlated columns)
);
