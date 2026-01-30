# Quantile Regression

Quantile regression for estimating conditional quantiles of the response distribution. Robust to outliers.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `quantile_fit` | Scalar | Process complete arrays in a single call |
| `quantile_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `quantile_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |

## anofox_stats_quantile_fit / quantile_fit

Quantile regression estimates conditional quantiles of the response variable distribution, rather than the conditional mean.

**Signature:**
```sql
anofox_stats_quantile_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| tau | DOUBLE | 0.5 | Quantile to estimate (0 < tau < 1) |
| fit_intercept | BOOLEAN | true | Include intercept term |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-6 | Convergence tolerance |

**Returns:**
```
STRUCT(
    coefficients LIST(DOUBLE),  -- Regression coefficients
    intercept DOUBLE,           -- Intercept term (if fitted)
    tau DOUBLE,                 -- Quantile estimated
    n_observations BIGINT,      -- Number of observations
    n_features INTEGER          -- Number of features
)
```

**Example:**
```sql
-- Median regression (tau = 0.5) - robust to outliers
SELECT quantile_fit(
    [y1, y2, y3, y4, y5],
    [[x1, x2, x3, x4, x5]],
    {'tau': 0.5}
);

-- 90th percentile regression (upper bound estimation)
SELECT quantile_fit(
    prices,
    [size, location_score],
    {'tau': 0.9}
);

-- Compare different quantiles
SELECT
    0.25 as quantile, (quantile_fit(y, [x], {'tau': 0.25})).coefficients[1] as coef
UNION ALL
SELECT
    0.50 as quantile, (quantile_fit(y, [x], {'tau': 0.50})).coefficients[1] as coef
UNION ALL
SELECT
    0.75 as quantile, (quantile_fit(y, [x], {'tau': 0.75})).coefficients[1] as coef;
```

## anofox_stats_quantile_fit_agg / quantile_fit_agg

Streaming quantile regression aggregate function.

```sql
-- Per-group median regression
SELECT
    region,
    (quantile_fit_agg(price, [sqft, bedrooms], {'tau': 0.5})).coefficients
FROM housing
GROUP BY region;
```

## Common Tau Values

| Tau | Description |
|-----|-------------|
| 0.10 | 10th percentile (lower tail) |
| 0.25 | First quartile |
| 0.50 | Median (robust central tendency) |
| 0.75 | Third quartile |
| 0.90 | 90th percentile (upper tail) |
| 0.95 | 95th percentile (risk analysis) |

## Use Cases

- **Robust regression**: Median regression is outlier-resistant
- **Full response distribution**: Understand effects across quantiles
- **Risk analysis**: VaR, conditional tail expectations
- **Heteroscedastic data**: Effects that vary across the distribution
- **Asymmetric distributions**: When mean doesn't represent typical values

## See Also

- [OLS](ols.md) - Mean regression
- [ALM](../glm/alm.md) - Asymmetric Laplace for quantile regression
- [Isotonic](isotonic.md) - Monotonic regression
