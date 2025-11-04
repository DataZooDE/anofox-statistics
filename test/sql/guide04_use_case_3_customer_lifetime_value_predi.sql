-- Create sample customer data
CREATE OR REPLACE TABLE customer_summary AS
SELECT
    i as customer_id,
    (tenure * 45 + aov * 3 + freq * 80 + engagement * 500 + RANDOM() * 200)::DOUBLE as total_purchases,
    tenure::DOUBLE,
    aov::DOUBLE,
    freq::DOUBLE,
    engagement::DOUBLE,
    CURRENT_DATE - (12 + (RANDOM() * 24)::INT) * INTERVAL '1 month' as cohort_month
FROM (
    SELECT
        i,
        (6 + RANDOM() * 18)::DOUBLE as tenure,         -- months active: 6-24
        (30 + RANDOM() * 70)::DOUBLE as aov,           -- avg order value: $30-100
        (1 + RANDOM() * 5)::DOUBLE as freq,            -- purchases per month: 1-6
        (RANDOM() * 0.8)::DOUBLE as engagement         -- email engagement: 0-80%
    FROM range(1, 101) t(i)
);

-- Build CLV model using aggregate functions (works directly with table data)
SELECT
    'Model Coefficient: Tenure' as metric,
    ROUND((ols_fit_agg(total_purchases, tenure)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: AOV' as metric,
    ROUND((ols_fit_agg(total_purchases, aov)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Coefficient: Frequency' as metric,
    ROUND((ols_fit_agg(total_purchases, freq)).coefficient, 2) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months'
UNION ALL
SELECT
    'Model Quality (R²)' as metric,
    ROUND((ols_fit_agg(total_purchases, tenure)).r2, 3) as value
FROM customer_summary
WHERE cohort_month <= CURRENT_DATE - INTERVAL '12 months';
