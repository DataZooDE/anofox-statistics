# BLS / NNLS (Bounded and Non-Negative Least Squares)

Least squares with box constraints on the coefficients. NNLS is the special
case where every coefficient must be >= 0.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `bls_fit_agg` | Aggregate | Bounded least squares with box constraints |
| `nnls_fit_agg` | Aggregate | Non-negative least squares (coefficients >= 0) |
| `bls_fit_predict_agg` | Aggregate | Fit and predict every row of a group, see [Fit-predict aggregates](fit_predict_agg.md) |
| `bls_fit_predict_by` | Table macro | Per-group fit and predict in long format, see [Table macros](../macros/table_macros.md#bls_fit_predict_by) |

## bls_fit_agg

**Signature:**

```text
bls_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

The same bound applies to every coefficient. If neither `lower_bound` nor
`upper_bound` is given, the fit is non-negative least squares (lower bound 0
for all coefficients), identical to `nnls_fit_agg`.

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lower_bound` (alias `lower`) | DOUBLE | unset | Lower bound for all coefficients |
| `upper_bound` (alias `upper`) | DOUBLE | unset | Upper bound for all coefficients |
| `fit_intercept` (alias `intercept`) | BOOLEAN | `false` | Include an (unconstrained) intercept |
| `max_iterations` (alias `max_iter`) | INTEGER | `1000` | Maximum iterations |
| `tolerance` (alias `tol`) | DOUBLE | `1e-10` | Convergence tolerance; also used to flag coefficients at a bound |

Option keys the function does not support raise an error.

**Example:**

```sql
CREATE OR REPLACE TABLE bls_demo AS
SELECT
    i AS t,
    sin(i)::DOUBLE AS f1,
    cos(i)::DOUBLE AS f2,
    ((i % 5) - 2)::DOUBLE AS f3,
    0.6 * sin(i) + 0.4 * cos(i) - 0.1 * ((i % 5) - 2) AS y
FROM range(1, 41) t(i);

-- Coefficients bounded to [0, 1]
SELECT bls_fit_agg(y, [f1, f2, f3], {'lower_bound': 0.0, 'upper_bound': 1.0}) AS fit
FROM bls_demo;
```

## nnls_fit_agg

**Signature:**

```text
nnls_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

**Options:** `fit_intercept` (default `false`), `max_iterations` (default
`1000`) and `tolerance` (default `1e-10`), as above.

**Example:**

```sql
-- f3 has a negative true effect, so NNLS pins it at 0
SELECT
    (nnls_fit_agg(y, [f1, f2, f3])).coefficients AS coefficients,
    (nnls_fit_agg(y, [f1, f2, f3])).at_lower_bound AS at_lower_bound
FROM bls_demo;
```

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Constrained coefficient estimates |
| `intercept` | `DOUBLE` | Intercept (NaN when `fit_intercept` is false) |
| `ssr` | `DOUBLE` | Sum of squared residuals |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `n_observations` | `BIGINT` | Rows used in the fit |
| `n_features` | `BIGINT` | Number of features |
| `n_active_constraints` | `BIGINT` | Number of coefficients sitting on a bound |
| `at_lower_bound` | `BOOLEAN[]` | Per coefficient: at the lower bound |
| `at_upper_bound` | `BOOLEAN[]` | Per coefficient: at the upper bound |

## NULL handling

Rows where `y` or `x` is NULL, or where `x` contains a NULL element, are
skipped.

## Use Cases

- **Mixture models / spectral unmixing**: component weights must be non-negative
- **Portfolio weights**: no short selling, capped positions
- **Physical constraints**: concentrations, rates or weights that must be positive

## See Also

- [OLS](ols.md) - Unconstrained regression
- [Ridge](ridge.md) - Regularized regression
- [Table macros](../macros/table_macros.md#bls_fit_predict_by) - Per-group predictions
