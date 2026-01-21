-- ============================================================================
-- OLS Multiple Series Examples (GROUP BY)
-- ============================================================================
-- Demonstrates per-group regression using aggregate functions.
-- Use case: Fit separate models for each category, region, or entity.
--
-- Run: ./build/release/duckdb < examples/ols_multiple_series.sql

LOAD 'anofox_statistics';

-- ============================================================================
-- Create Sample Dataset
-- ============================================================================
-- Sales data with different categories, each having its own price-sales relationship

CREATE OR REPLACE TABLE sales_data AS
SELECT
    category,
    month,
    price,
    -- Different relationship per category:
    -- Electronics: sales = 1000 - 8*price (price sensitive)
    -- Clothing: sales = 500 - 3*price (less price sensitive)
    -- Food: sales = 800 - 2*price (staple items)
    CASE category
        WHEN 'Electronics' THEN 1000.0 - 8.0 * price + (RANDOM() * 40 - 20)
        WHEN 'Clothing' THEN 500.0 - 3.0 * price + (RANDOM() * 30 - 15)
        WHEN 'Food' THEN 800.0 - 2.0 * price + (RANDOM() * 20 - 10)
    END AS sales
FROM (
    SELECT
        category,
        month,
        -- Different price ranges per category
        CASE category
            WHEN 'Electronics' THEN 50.0 + (month - 1) * 5.0
            WHEN 'Clothing' THEN 30.0 + (month - 1) * 3.0
            WHEN 'Food' THEN 10.0 + (month - 1) * 1.0
        END AS price
    FROM (VALUES ('Electronics'), ('Clothing'), ('Food')) AS c(category),
         generate_series(1, 12) AS t(month)
);

-- ============================================================================
-- Example 1: Basic Per-Group Regression
-- ============================================================================

SELECT '=== Example 1: Basic Per-Group Regression ===' AS section;

SELECT
    category,
    ROUND(result.intercept, 2) AS intercept,
    ROUND(result.coefficients[1], 2) AS price_effect,
    ROUND(result.r_squared, 4) AS r_squared,
    result.n_observations AS observations
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 2: Extract Coefficients for Interpretation
-- ============================================================================

SELECT '=== Example 2: Coefficient Interpretation ===' AS section;

SELECT
    category,
    'For every $1 increase in price, sales change by ' ||
        ROUND(result.coefficients[1], 1)::VARCHAR || ' units' AS interpretation,
    CASE
        WHEN result.coefficients[1] < -5 THEN 'High'
        WHEN result.coefficients[1] < -2 THEN 'Medium'
        ELSE 'Low'
    END AS price_sensitivity
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
ORDER BY result.coefficients[1];

-- ============================================================================
-- Example 3: Compare Model Fit Across Groups
-- ============================================================================

SELECT '=== Example 3: Model Comparison Across Groups ===' AS section;

SELECT
    category,
    ROUND(result.r_squared, 4) AS r_squared,
    ROUND(result.adj_r_squared, 4) AS adj_r_squared,
    ROUND(result.residual_std_error * result.residual_std_error, 2) AS mse,
    ROUND(result.residual_std_error, 2) AS rmse,
    CASE
        WHEN result.r_squared > 0.95 THEN 'Excellent fit'
        WHEN result.r_squared > 0.80 THEN 'Good fit'
        WHEN result.r_squared > 0.60 THEN 'Moderate fit'
        ELSE 'Poor fit'
    END AS model_quality
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
ORDER BY result.r_squared DESC;

-- ============================================================================
-- Example 4: Full Inference Per Group
-- ============================================================================

SELECT '=== Example 4: Full Inference Per Group ===' AS section;

SELECT
    category,
    ROUND(result.coefficients[1], 4) AS price_coef,
    ROUND(result.std_errors[1], 4) AS std_error,
    ROUND(result.t_values[1], 4) AS t_statistic,
    ROUND(result.p_values[1], 6) AS p_value,
    CASE
        WHEN result.p_values[1] < 0.001 THEN '***'
        WHEN result.p_values[1] < 0.01 THEN '**'
        WHEN result.p_values[1] < 0.05 THEN '*'
        ELSE 'ns'
    END AS significance
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true, 'compute_inference': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 5: Multiple Predictors Per Group
-- ============================================================================

SELECT '=== Example 5: Multiple Predictors Per Group ===' AS section;

-- Add advertising spend as second predictor
CREATE OR REPLACE TABLE sales_with_ads AS
SELECT
    *,
    -- Advertising effect varies by category
    CASE category
        WHEN 'Electronics' THEN 10.0 + month * 2.0
        WHEN 'Clothing' THEN 5.0 + month * 1.5
        WHEN 'Food' THEN 2.0 + month * 0.5
    END AS ad_spend,
    -- Re-calculate sales including ad effect
    CASE category
        WHEN 'Electronics' THEN 1000.0 - 8.0 * price + 5.0 * (10.0 + month * 2.0) + (RANDOM() * 30 - 15)
        WHEN 'Clothing' THEN 500.0 - 3.0 * price + 3.0 * (5.0 + month * 1.5) + (RANDOM() * 20 - 10)
        WHEN 'Food' THEN 800.0 - 2.0 * price + 2.0 * (2.0 + month * 0.5) + (RANDOM() * 15 - 7.5)
    END AS sales_v2
FROM sales_data;

SELECT
    category,
    ROUND(result.intercept, 2) AS intercept,
    ROUND(result.coefficients[1], 2) AS price_effect,
    ROUND(result.coefficients[2], 2) AS ad_effect,
    ROUND(result.r_squared, 4) AS r_squared
FROM (
    SELECT
        category,
        ols_fit_agg(sales_v2, [price, ad_spend], {'intercept': true}) AS result
    FROM sales_with_ads
    GROUP BY category
) sub
ORDER BY category;

-- ============================================================================
-- Example 6: Hierarchical Grouping (Region + Category)
-- ============================================================================

SELECT '=== Example 6: Hierarchical Grouping ===' AS section;

CREATE OR REPLACE TABLE regional_sales AS
SELECT
    region,
    category,
    month,
    price * region_factor AS price,
    sales * region_factor + (RANDOM() * 20 - 10) AS sales
FROM (
    SELECT
        region,
        CASE region
            WHEN 'North' THEN 1.2
            WHEN 'South' THEN 0.9
            WHEN 'East' THEN 1.1
            WHEN 'West' THEN 1.0
        END AS region_factor,
        category,
        month,
        price,
        sales
    FROM sales_data,
         (VALUES ('North'), ('South'), ('East'), ('West')) AS r(region)
);

-- Fit model for each region-category combination
SELECT
    region,
    category,
    ROUND(result.coefficients[1], 2) AS price_effect,
    ROUND(result.r_squared, 4) AS r_squared,
    result.n_observations AS n
FROM (
    SELECT
        region,
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM regional_sales
    GROUP BY region, category
) sub
ORDER BY region, category;

-- ============================================================================
-- Example 7: Predictions Using Group Models
-- ============================================================================

SELECT '=== Example 7: Predictions Using Group Models ===' AS section;

-- Fit models and store coefficients
CREATE OR REPLACE TABLE category_models AS
SELECT
    category,
    result.intercept AS intercept,
    result.coefficients[1] AS slope
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub;

-- New price scenarios to predict
CREATE OR REPLACE TABLE new_prices AS
SELECT category, new_price
FROM (VALUES ('Electronics'), ('Clothing'), ('Food')) AS c(category),
     (VALUES (40.0), (60.0), (80.0), (100.0)) AS p(new_price);

-- Predict using each category's model
SELECT
    np.category,
    np.new_price,
    ROUND(cm.intercept + cm.slope * np.new_price, 2) AS predicted_sales
FROM new_prices np
JOIN category_models cm ON np.category = cm.category
ORDER BY np.category, np.new_price;

-- ============================================================================
-- Example 8: Rank Groups by Model Performance
-- ============================================================================

SELECT '=== Example 8: Rank Groups by Performance ===' AS section;

SELECT
    category,
    ROUND(result.r_squared, 4) AS r_squared,
    ROUND(result.residual_std_error, 2) AS rmse,
    RANK() OVER (ORDER BY result.r_squared DESC) AS r2_rank,
    RANK() OVER (ORDER BY result.residual_std_error ASC) AS rmse_rank
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
ORDER BY r2_rank;

-- ============================================================================
-- Example 9: Filter Groups by Model Quality
-- ============================================================================

SELECT '=== Example 9: Filter by Model Quality ===' AS section;

-- Only show groups where R² > 0.90
SELECT
    category,
    ROUND(result.r_squared, 4) AS r_squared,
    ROUND(result.coefficients[1], 2) AS price_effect
FROM (
    SELECT
        category,
        ols_fit_agg(sales, [price], {'intercept': true}) AS result
    FROM sales_data
    GROUP BY category
) sub
WHERE result.r_squared > 0.90
ORDER BY result.r_squared DESC;

-- Cleanup
DROP TABLE IF EXISTS sales_data;
DROP TABLE IF EXISTS sales_with_ads;
DROP TABLE IF EXISTS regional_sales;
DROP TABLE IF EXISTS category_models;
DROP TABLE IF EXISTS new_prices;
