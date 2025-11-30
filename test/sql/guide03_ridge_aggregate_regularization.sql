LOAD 'build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension';

-- Statistics Guide: Ridge Regression - Handling Multicollinearity
-- Demonstrates lambda tuning and coefficient shrinkage

-- Create data with severe multicollinearity
CREATE TEMP TABLE collinear_data AS
SELECT
    'product_line_a' as product,
    i::DOUBLE as advertising,
    (i + random() * 0.5)::DOUBLE as social_media,  -- Nearly identical to advertising
    (i + random() * 0.5)::DOUBLE as influencer,     -- Nearly identical to advertising
    (1000 + 50 * i + random() * 100)::DOUBLE as sales
FROM generate_series(1, 25) as t(i);

-- Compare different lambda values
SELECT
    product,
    'OLS (lambda=0)' as method,
    result.lambda,
    result.coefficients[1] as adv_coef,
    result.coefficients[2] as social_coef,
    result.coefficients[3] as influencer_coef,
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_stats_ridge_fit_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 0.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'Ridge (lambda=1)' as method,
    result.lambda,
    result.coefficients[1],
    result.coefficients[2],
    result.coefficients[3],
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_stats_ridge_fit_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 1.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub
UNION ALL
SELECT
    product,
    'Ridge (lambda=10)' as method,
    result.lambda,
    result.coefficients[1],
    result.coefficients[2],
    result.coefficients[3],
    result.r2,
    result.adj_r2
FROM (
    SELECT
        product,
        anofox_stats_ridge_fit_agg(
            sales,
            [advertising, social_media, influencer],
            {'lambda': 10.0, 'intercept': true}
        ) as result
    FROM collinear_data
    GROUP BY product
) sub;

-- Interpretation note
SELECT
    'Key Insight' as note,
    'Ridge shrinks coefficients towards zero, reducing variance at the cost of small bias. Higher lambda = more shrinkage.' as interpretation;
