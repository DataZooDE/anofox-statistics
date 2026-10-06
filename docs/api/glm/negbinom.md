# Negative Binomial GLM

Regression for overdispersed count data — counts whose variance exceeds their
mean, which a Poisson GLM cannot represent.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `negbinom_fit_agg` | Aggregate | Fit a Negative Binomial GLM (log link) |

## negbinom_fit_agg

**Signature:**

```text
negbinom_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Response; non-negative counts |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` | BOOLEAN | `true` | Include an intercept |
| `theta` (aliases `nb_theta`, `dispersion`) | DOUBLE | estimated | Dispersion `theta`, must be finite and positive. Unset means estimate it from the data. |
| `max_iterations` | INTEGER | `100` | Maximum IRLS iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance |
| `compute_inference` | BOOLEAN | `false` | Standard errors, z-tests, p-values, intervals |
| `confidence_level` | DOUBLE | `0.95` | Interval level |
| `glm_lambda` | DOUBLE | `0.0` | L2 (ridge) penalty strength |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the design) |
| `feature_names`, `prior`, `vcov` | | | See [Explicit priors](priors.md) |

**Returns:** the standard GLM struct (see [Poisson](poisson.md#poisson_fit_agg)
for the field descriptions):

| Field | Type |
|-------|------|
| `coefficients` | DOUBLE[] |
| `intercept` | DOUBLE |
| `deviance` | DOUBLE |
| `null_deviance` | DOUBLE |
| `pseudo_r_squared` | DOUBLE |
| `aic` | DOUBLE |
| `dispersion` | DOUBLE — the `theta` used (given or estimated) |
| `n_observations` | BIGINT |
| `n_features` | BIGINT |
| `iterations` | INTEGER |
| `converged` | BOOLEAN |
| `std_errors`, `z_values`, `p_values`, `ci_lower`, `ci_upper` | DOUBLE[] — only with `compute_inference` |

**Example:**

```sql
CREATE OR REPLACE TABLE demand AS
SELECT i AS week,
       (i % 2)::DOUBLE AS promo,
       ((i % 3) + 1)::DOUBLE AS shelf_facings,
       (((i * 7) % 11) + 3 * (i % 2) + (i % 3))::DOUBLE AS qty
FROM range(200) r(i);

-- theta estimated from the data
SELECT negbinom_fit_agg(qty, [promo, shelf_facings]) AS fit FROM demand;

-- With a known dispersion and inference
SELECT negbinom_fit_agg(qty, [promo],
                        {'theta': 2.5, 'compute_inference': true}) AS fit
FROM demand;
```

## Dispersion

The variance is `mu + mu^2 / theta`. Small `theta` means heavy overdispersion;
as `theta` grows the model approaches Poisson.

When `theta` is not supplied it is estimated by alternating an IRLS fit at the
current `theta` with a method-of-moments update, which is how `MASS::glm.nb`
proceeds. The estimate is clamped to `[1e-6, 1e6]`; `1e6` means no
overdispersion was detected. `theta` already enters the IRLS weights, so it
does **not** additionally scale the coefficient covariance.

## When to use it over Poisson

Fit Poisson first and look at its `dispersion` field. Materially above 1 means
the Poisson variance assumption is violated. Negative Binomial models the extra
variation explicitly instead of only inflating the standard errors.

## NULL and Invalid Input Handling

- Rows where `y` or the `x` list is NULL are skipped; a NULL element inside `x`
  or any non-finite value drops the row. `n_observations` counts the rows used.
- Fewer than two usable rows, a negative `y`, a non-positive `theta`, or a
  failed fit returns `NULL`.

## Use Cases

- **Intermittent demand** — many zeros and occasional spikes
- **Claim or defect counts** — clustered rather than uniformly random
- **Any count where the Poisson dispersion comes out well above 1**

## See Also

- [Poisson GLM](poisson.md)
- [Mixed-effects GLMs](glmm.md)
- [ALM](alm.md) — a broader distribution set
