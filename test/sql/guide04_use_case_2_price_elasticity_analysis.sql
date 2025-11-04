-- Create sample sales transaction data
CREATE OR REPLACE TABLE sales_transactions AS
SELECT
    i as transaction_id,
    CURRENT_DATE - (RANDOM() * 180)::INT as transaction_date,
    category,
    price,
    quantity
FROM (
    SELECT i,
           CASE (i % 4)
               WHEN 0 THEN 'Electronics'
               WHEN 1 THEN 'Clothing'
               WHEN 2 THEN 'Groceries'
               ELSE 'Luxury Goods'
           END as category,
           CASE (i % 4)
               WHEN 0 THEN (100 + RANDOM() * 50)::DOUBLE  -- Electronics
               WHEN 1 THEN (30 + RANDOM() * 20)::DOUBLE   -- Clothing
               WHEN 2 THEN (5 + RANDOM() * 3)::DOUBLE     -- Groceries
               ELSE (500 + RANDOM() * 200)::DOUBLE        -- Luxury
           END as price,
           CASE (i % 4)
               WHEN 0 THEN (50 - (100 + RANDOM() * 50) * 0.25 + RANDOM() * 10)::DOUBLE  -- Electronics: elastic
               WHEN 1 THEN (80 - (30 + RANDOM() * 20) * 0.8 + RANDOM() * 15)::DOUBLE    -- Clothing: elastic
               WHEN 2 THEN (100 - (5 + RANDOM() * 3) * 0.5 + RANDOM() * 20)::DOUBLE     -- Groceries: inelastic
               ELSE (20 - (500 + RANDOM() * 200) * 0.01 + RANDOM() * 5)::DOUBLE         -- Luxury: inelastic
           END as quantity
    FROM range(1, 401) t(i)
);

-- Calculate price elasticity by product category
SELECT
    category,
    ROUND((ols_fit_agg(quantity, price)).coefficient, 3) as elasticity,
    ROUND((ols_fit_agg(quantity, price)).r2, 3) as predictability,
    CASE
        WHEN ABS((ols_fit_agg(quantity, price)).coefficient) > 0.5 THEN 'Elastic'
        ELSE 'Inelastic'
    END as elasticity_type,
    CASE
        WHEN ABS((ols_fit_agg(quantity, price)).coefficient) > 0.5 THEN 'Discount Strategy'
        ELSE 'Premium Pricing'
    END as pricing_recommendation
FROM sales_transactions
WHERE transaction_date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY category
ORDER BY ABS(elasticity) DESC;
