SELECT category, ols_fit_agg(y, x) as model
FROM data GROUP BY category;
