# BLS/NNLS (Bounded/Non-Negative Least Squares)

Bounded Least Squares and Non-Negative Least Squares for constrained optimization.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `bls_fit_agg` | Aggregate | Bounded Least Squares with box constraints |
| `nnls_fit_agg` | Aggregate | Non-Negative Least Squares (coefficients >= 0) |
| `bls_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |
| `bls_fit_predict_by` | Table Macro | Per-group regression with long-format output |

## anofox_stats_bls_fit_agg

Bounded Least Squares with box constraints on coefficients.

**Signature:**
```sql
anofox_stats_bls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | false | Include intercept term |
| lower_bound | DOUBLE | - | Lower bound for all coefficients |
| upper_bound | DOUBLE | - | Upper bound for all coefficients |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-10 | Convergence tolerance |

**Returns:** [BlsFitResult](../reference/return_types.md#blsfitresult-structure) STRUCT

**Example:**
```sql
-- Coefficients bounded between 0 and 1
SELECT bls_fit_agg(
    y,
    [x1, x2, x3],
    {'lower_bound': 0.0, 'upper_bound': 1.0}
)
FROM portfolio_data;

-- Only lower bound (coefficients >= 0)
SELECT bls_fit_agg(
    y,
    [x1, x2],
    {'lower_bound': 0.0}
)
FROM data;
```

## anofox_stats_nnls_fit_agg

Non-Negative Least Squares - all coefficients constrained to be >= 0.

**Signature:**
```sql
anofox_stats_nnls_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | false | Include intercept term |
| max_iterations | INTEGER | 1000 | Maximum iterations |
| tolerance | DOUBLE | 1e-10 | Convergence tolerance |

**Returns:** [BlsFitResult](../reference/return_types.md#blsfitresult-structure) STRUCT

**Example:**
```sql
-- Non-negative coefficients (e.g., mixture models)
SELECT nnls_fit_agg(spectrum, [component1, component2, component3])
FROM spectral_data;

-- Portfolio weights (no short selling)
SELECT nnls_fit_agg(returns, [stock1, stock2, stock3])
FROM portfolio_data;

-- Per-group NNLS
SELECT
    category,
    (nnls_fit_agg(y, [x1, x2])).coefficients
FROM data
GROUP BY category;
```

## Use Cases

- **Spectral unmixing / mixture models**: Component proportions must be non-negative
- **Portfolio optimization**: No short selling constraint
- **Physical constraints**: Concentrations, weights must be positive
- **Image processing**: Non-negative matrix factorization
- **Signal processing**: Source separation

## See Also

- [OLS](ols.md) - Unconstrained regression
- [Ridge](ridge.md) - Regularized regression
- [Table Macros](../macros/table_macros.md#bls_fit_predict_by) - Per-group predictions
