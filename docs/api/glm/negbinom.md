# Negative Binomial GLM

Regression for overdispersed count data — counts whose variance exceeds their
mean, which a Poisson GLM cannot represent.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `negbinom_fit_agg` | Aggregate | Fit a Negative Binomial GLM |

## anofox_stats_negbinom_fit_agg / negbinom_fit_agg

**Signature:**

```sql
anofox_stats_negbinom_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| fit_intercept | BOOLEAN | true | Include an intercept |
| theta | DOUBLE | — | Dispersion. Unset means estimate it from the data. |
| max_iterations | INTEGER | 100 | Maximum IRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Standard errors, z-tests, intervals |
| confidence_level | DOUBLE | 0.95 | Interval level |
| glm_lambda | DOUBLE | 0.0 | L2 regularization |
| feature_names, prior, vcov | | | See [Explicit priors](priors.md) |

**Returns:** the standard GLM STRUCT. `dispersion` carries `theta`.

**Example:**

```sql
-- Overdispersed weekly demand
SELECT negbinom_fit_agg(qty, [promo, week_of_year]) FROM demand;

-- With a known dispersion
SELECT negbinom_fit_agg(qty, [promo], {'theta': 2.5}) FROM demand;
```

## Dispersion

The variance is `mu + mu^2 / theta`. Small `theta` means heavy overdispersion;
as `theta` grows the model approaches Poisson.

When `theta` is not supplied it is estimated by alternating an IRLS fit at the
current `theta` with a method-of-moments update, which is how `MASS::glm.nb`
proceeds. `theta` is a shape parameter that already enters the IRLS weights, so
it does **not** additionally scale the coefficient covariance.

## When to use it over Poisson

Fit Poisson first and look at its `dispersion` field. Materially above 1 means
the Poisson variance assumption is being violated and the Poisson standard errors
are too small. Negative Binomial models the extra variation explicitly rather
than papering over it.

## Use Cases

- **Intermittent demand** — many zeros and occasional spikes
- **Claim or defect counts** — clustered rather than uniformly random
- **Any count where the Poisson dispersion comes out well above 1**

## See Also

- [Poisson GLM](poisson.md)
- [Mixed-effects GLMs](glmm.md) — `family := 'negbinomial'` with a random intercept
- [ALM](alm.md) — a broader distribution set
