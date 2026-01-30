# PLS (Partial Least Squares)

Partial Least Squares regression for high-dimensional data and multicollinearity using the SIMPLS algorithm.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pls_fit` | Scalar | Process complete arrays in a single call |
| `pls_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `pls_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |

## anofox_stats_pls_fit / pls_fit

PLS regression using the SIMPLS algorithm to find latent components that maximize covariance between X scores and y.

**Signature:**
```sql
anofox_stats_pls_fit(
    y LIST(DOUBLE),
    x LIST(LIST(DOUBLE)),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| n_components | INTEGER | 2 | Number of latent components to extract |
| fit_intercept | BOOLEAN | true | Include intercept term |

**Returns:**
```
STRUCT(
    coefficients LIST(DOUBLE),  -- Regression coefficients
    intercept DOUBLE,           -- Intercept term (if fitted)
    r_squared DOUBLE,           -- Coefficient of determination
    n_components INTEGER,       -- Number of components used
    n_observations BIGINT,      -- Number of observations
    n_features INTEGER          -- Number of features
)
```

**Example:**
```sql
-- PLS with 3 components for high-dimensional data
SELECT pls_fit(
    [y1, y2, y3, y4, y5],
    [[x1_1, x1_2, x1_3, x1_4, x1_5],
     [x2_1, x2_2, x2_3, x2_4, x2_5],
     [x3_1, x3_2, x3_3, x3_4, x3_5]],
    {'n_components': 2}
);

-- Per-group PLS regression
SELECT
    category,
    (pls_fit_agg(y, [x1, x2, x3, x4, x5], {'n_components': 2})).r_squared
FROM high_dim_data
GROUP BY category;
```

## anofox_stats_pls_fit_agg / pls_fit_agg

Streaming PLS regression aggregate function.

```sql
SELECT pls_fit_agg(y, [x1, x2, x3], {'n_components': 2}) FROM data;
```

## Choosing n_components

- **n_components = 1**: Maximum explained variance in one direction
- **n_components = 2-3**: Typical for moderate-dimensional data
- **n_components = min(n, p)**: Maximum extractable components

Use cross-validation to select optimal number of components.

## Use Cases

- **High-dimensional data**: More features than observations
- **Multicollinearity**: Correlated predictors
- **Chemometrics and spectroscopy**: NIR, Raman spectral analysis
- **Genomics and bioinformatics**: Gene expression data
- **Dimension reduction**: When features outnumber samples

## See Also

- [OLS](ols.md) - Standard regression
- [Ridge](ridge.md) - L2 regularization alternative
- [Elastic Net](elasticnet.md) - L1+L2 regularization
