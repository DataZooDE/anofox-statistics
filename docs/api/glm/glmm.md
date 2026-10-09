# Mixed-Effects GLMs

Partial pooling done properly: one model fitted jointly across all groups, with a
random intercept that lets each group deviate from the population while borrowing
strength from the rest.

This is "lme4 in SQL" for a random intercept over one grouping factor.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `glmm_fit_agg` | Aggregate | Fit a mixed-effects GLM |
| `glmm_fit_by` | Table Macro | Same, returning one row per group with its BLUP |

The examples on this page use this table:

```sql
-- 12 SKUs, 30 weeks each
CREATE OR REPLACE TABLE demand AS
SELECT i AS week,
       'SKU-' || (i % 12) AS sku,
       (i % 2)::DOUBLE AS promo,
       (i % 3)::DOUBLE AS region,
       (((i * 7) % 11) + 3 * (i % 2) + (i % 12) % 4)::DOUBLE AS qty
FROM range(360) r(i);
```

## glmm_fit_agg

**Signature:**

```text
glmm_fit_agg(y DOUBLE, x DOUBLE[], group ANY [, options MAP]) -> STRUCT
```

`group` is the grouping key and may be of any type.

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| family | VARCHAR | 'gaussian' | `gaussian` (aliases `normal`, `lmm`), `poisson`, or `binomial` (alias `logistic`). `negbinomial`, `gamma` and `tweedie` are recognised but not yet supported and return `NULL`. |
| fit_intercept | BOOLEAN | true | Include a fixed intercept |
| reml | BOOLEAN | true | REML rather than ML for the Gaussian variance components |
| random | INTEGER[] | — | 1-based indices into `x` of feature columns that also get a **random slope** (unstructured covariance with the intercept) |
| groups | INTEGER[] | — | 1-based indices into `x` of additional **crossed** grouping-factor columns; each becomes an independent random intercept and is removed from the design |
| max_iterations | INTEGER | 100 | Inner PIRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Fixed-effect standard errors, z-tests, intervals |
| confidence_level | DOUBLE | 0.95 | Interval level |

> The solver lives upstream in `anofox-regression`; this extension is a wrapper.
> Not yet available (tracked upstream, [anofox-regression#29](https://github.com/sipemu/anofox-regression/issues/29)):
> NegBinomial/Gamma/Tweedie mixed-effects families, an `offset`, per-group BLUP
> standard errors, and random slopes combined with multiple grouping factors.
> Requesting an unsupported combination returns `NULL`. The `offset`, `theta`
> and `power` keys are parsed but currently lead to a `NULL` result.

**Returns:**

```text
STRUCT(coefficients DOUBLE[], intercept DOUBLE,
       var_group DOUBLE, var_residual DOUBLE, icc DOUBLE,
       log_likelihood DOUBLE, aic DOUBLE, bic DOUBLE, deviance DOUBLE,
       n_observations BIGINT, n_groups BIGINT, n_features BIGINT,
       iterations INTEGER, converged BOOLEAN,
       random_cov DOUBLE[],           -- Sigma, flattened row-major (random_dim x random_dim)
       random_dim INTEGER,            -- q = 1 + number of random slopes
       factors LIST(STRUCT(n_levels BIGINT, var DOUBLE))  -- per-factor variances (crossed fits)
     [, std_errors DOUBLE[], z_values DOUBLE[], p_values DOUBLE[],
        ci_lower DOUBLE[], ci_upper DOUBLE[], intercept_std_error DOUBLE],
       ranef LIST(STRUCT(group VARCHAR, intercept DOUBLE, se DOUBLE, n BIGINT)))
```

**Example:**

```sql
-- Random intercept over SKUs (Poisson counts)
SELECT (fit).coefficients, (fit).var_group, (fit).icc, (fit).p_values
FROM (
    SELECT glmm_fit_agg(qty, [promo], sku, {
        'family': 'poisson',
        'compute_inference': true
    }) AS fit
    FROM demand
);

-- Random intercept + random slope on promo (x-column 1); read Sigma from random_cov
SELECT (glmm_fit_agg(qty, [promo], sku, {'random': [1]})).random_cov FROM demand;

-- Crossed factors: sku (positional) and region (x-column 2, named in 'groups')
SELECT (glmm_fit_agg(qty, [promo, region], sku, {'groups': [2]})).factors FROM demand;
```

## glmm_fit_by

Table macro that fits **one** mixed model across all groups of a table and
returns one row per group with its random effect (BLUP) and the shared fixed
effects. This differs from the `*_fit_predict_by` macros, which fit independent
models per group.

**Signature:**

```text
glmm_fit_by(source VARCHAR, group_col, y_col, x_cols [, options MAP]) -> TABLE
```

Options are the `glmm_fit_agg` options.

**Returns:**

| Column | Type | Description |
|--------|------|-------------|
| `group` | VARCHAR | Group key (rendered as text) |
| `ranef` | DOUBLE | The group's random-intercept BLUP |
| `ranef_se` | DOUBLE | Its conditional SE (currently `NaN`, see below) |
| `n` | BIGINT | Observations in the group |
| `fixed_intercept` | DOUBLE | Shared fixed intercept |
| `fixed_coefficients` | DOUBLE[] | Shared fixed-effect coefficients |
| `var_group` | DOUBLE | Between-group variance |
| `var_residual` | DOUBLE | Residual variance |
| `icc` | DOUBLE | Intra-class correlation |

Rows are ordered by `group`.

```sql
-- One row per SKU, with its shrunken effect
SELECT "group", round(ranef, 3) AS ranef, n, round(icc, 3) AS icc
FROM glmm_fit_by('demand', sku, qty, [promo], {'family': 'poisson'});
```

## The model

```
g(mu_ij) = x_ij'beta + b_j ,    b_j ~ N(0, sigma_b^2)
```

Each group `j` gets its own intercept deviation `b_j`, drawn from a common
distribution rather than estimated freely. That shared distribution is what makes
a sparse group's estimate sensible: it is pulled toward zero in proportion to how
little the group has to say.

## Reading the output

| Field | Meaning |
|-------|---------|
| `var_group` | `sigma_b^2`, the between-group variance |
| `var_residual` | Residual variance; 1.0 for Poisson and Binomial, where the dispersion is fixed |
| `icc` | `var_group / (var_group + var_residual)`. Near 1 means group identity explains most of the variation; near 0 means the groups are interchangeable and a pooled fit would do. |
| `random_cov` / `random_dim` | The random-effects covariance Σ, flattened row-major as a length-`random_dim²` list. For a plain random intercept `random_dim = 1` and `random_cov = [var_group]`; with a random slope `random_dim = 2` and Σ is `[σ²_int, cov, cov, σ²_slope]`. |
| `factors` | For **crossed** fits, one `{n_levels, var}` per grouping factor. Empty for a single-factor fit (use `var_group` / `random_cov` instead). |
| `ranef` | One entry per group: the intercept BLUP, its conditional SE, and the observation count. (The SE is currently `NaN` — not yet exposed by the upstream solver.) |

The BLUPs are labelled with the original grouping key, whatever its type.

## Random slopes and crossed factors

```sql
-- Random slope on x-column 1 (unstructured covariance with the intercept).
-- random_dim = 2 and random_cov holds the 2x2 Sigma.
SELECT (fit).random_dim, (fit).random_cov
FROM (SELECT glmm_fit_agg(qty, [promo], sku, {'random': [1]}) AS fit FROM demand);

-- Crossed factors: sku is the positional factor; the second x-column is a
-- second, independent random intercept named by its 1-based index in 'groups'.
-- It is dictionary-encoded and removed from the design; per-factor variances
-- come back in `factors`.
SELECT (glmm_fit_agg(qty, [promo, region], sku, {'groups': [2]})).factors FROM demand;
```

Nesting `(1|a/b)` is expressed by passing the composed `a:b` key as a factor
column. Random slopes combined with multiple grouping factors is not yet
supported (tracked upstream, anofox-regression#29).

## Degenerate inputs and NULL handling

| Situation | Result |
|-----------|--------|
| Fewer than two groups | `NULL` — the random intercept is not separable from the fixed one |
| Every group a singleton | `NULL` — the between-group variance is not identified |
| Fewer observations than parameters | `NULL` |

Rows with a `NULL` grouping key, or a non-finite value in `y` or any feature, are
dropped; `n_observations` reports how many were used.

## Scope

Random intercept and random slopes over one grouping factor, or several crossed /
nested random-intercept factors, for the gaussian, poisson and binomial families.
Not yet available (tracked upstream, anofox-regression#29): NegBinomial/Gamma/
Tweedie mixed-effects families, an offset, per-group BLUP standard errors, and
random slopes combined with multiple grouping factors.

If you already have per-group estimates and only want them shrunk, the cheaper
[empirical-Bayes helper](eb_shrink.md) gets most of the way there.

## Use Cases

- **Intermittent demand across many SKUs** — the motivating case: thousands of
  groups, few observations each.
- **Store or region effects** — small stores borrow strength from the chain.
- **Repeated measures** — several observations per subject.
- **Anywhere independent per-group fits are too noisy and a pooled fit is too blunt.**

## See Also

- [Empirical-Bayes shrinkage](eb_shrink.md) — the post-hoc approximation
- [Explicit priors](priors.md) — a Gaussian prior is the same object as a random effect
- [Poisson GLM](poisson.md)
