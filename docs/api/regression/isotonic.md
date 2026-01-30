# Isotonic Regression

Isotonic regression for monotonic constraints using the Pool Adjacent Violators Algorithm (PAVA).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `isotonic_fit` | Scalar | Process complete arrays in a single call |
| `isotonic_fit_agg` | Aggregate | Streaming row-by-row accumulation |
| `isotonic_fit_predict_agg` | Aggregate | Fit and predict with GROUP BY support |

## anofox_stats_isotonic_fit / isotonic_fit

Fits a monotonic (non-decreasing or non-increasing) function to the data.

**Signature:**
```sql
anofox_stats_isotonic_fit(
    x LIST(DOUBLE),
    y LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| increasing | BOOLEAN | true | Fit increasing (true) or decreasing (false) function |

**Returns:**
```
STRUCT(
    fitted_values LIST(DOUBLE),  -- Monotonic fitted values
    r_squared DOUBLE,            -- Coefficient of determination
    n_observations BIGINT,       -- Number of observations
    increasing BOOLEAN           -- Direction of monotonicity
)
```

**Example:**
```sql
-- Fit increasing monotonic function (e.g., dose-response curve)
SELECT isotonic_fit(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.5, 2.0, 1.8, 3.5, 4.0],  -- Noisy but generally increasing
    {'increasing': true}
);

-- Decreasing isotonic regression (e.g., decay curve)
SELECT isotonic_fit(
    dose_levels,
    response_values,
    {'increasing': false}
);
```

## anofox_stats_isotonic_fit_agg / isotonic_fit_agg

Streaming isotonic regression aggregate function.

```sql
SELECT isotonic_fit_agg(x, y, {'increasing': true}) FROM calibration_data;
```

## How It Works

The PAVA algorithm:
1. Starts with raw y values
2. Scans for violations of monotonicity
3. Replaces violating adjacent values with their weighted average
4. Repeats until all values satisfy the constraint

## Use Cases

- **Dose-response modeling**: Response increases with dose
- **Calibration curves**: Probability should increase with score
- **Monotonic trend estimation**: When domain knowledge implies monotonicity
- **Quality control thresholds**: Performance should improve with investment
- **ROC curve smoothing**: True positive rate must be monotonic

## See Also

- [OLS](ols.md) - Unconstrained linear regression
- [Quantile](quantile.md) - Distribution quantile estimation
