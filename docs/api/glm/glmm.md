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

## anofox_stats_glmm_fit_agg / glmm_fit_agg

**Signature:**

```sql
anofox_stats_glmm_fit_agg(
    y DOUBLE,
    x LIST(DOUBLE),
    group ANY,          -- grouping key; any type
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| family | VARCHAR | 'gaussian' | `gaussian`, `poisson`, `binomial`, `negbinomial`, `gamma`, `tweedie` |
| fit_intercept | BOOLEAN | true | Include a fixed intercept |
| reml | BOOLEAN | true | REML rather than ML for the Gaussian variance components |
| theta | DOUBLE | 1.0 | Negative Binomial dispersion |
| power | DOUBLE | 1.5 | Tweedie variance power |
| offset | INTEGER | — | 1-based index into `x` of an offset column (see below) |
| max_iterations | INTEGER | 100 | Inner PIRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Fixed-effect standard errors, z-tests, intervals |
| confidence_level | DOUBLE | 0.95 | Interval level |

**Returns:**

```
STRUCT(coefficients DOUBLE[], intercept DOUBLE,
       var_group DOUBLE, var_residual DOUBLE, icc DOUBLE,
       log_likelihood DOUBLE, aic DOUBLE, bic DOUBLE, deviance DOUBLE,
       n_observations BIGINT, n_groups BIGINT, n_features BIGINT,
       iterations INTEGER, converged BOOLEAN
     [, std_errors DOUBLE[], z_values DOUBLE[], p_values DOUBLE[],
        ci_lower DOUBLE[], ci_upper DOUBLE[], intercept_std_error DOUBLE],
       ranef LIST(STRUCT(group VARCHAR, intercept DOUBLE, se DOUBLE, n BIGINT)))
```

**Example:**

```sql
-- Intermittent demand across thousands of SKUs
SELECT glmm_fit_agg(qty, [promo], sku, {
    'family': 'negbinomial',
    'compute_inference': true
}) FROM demand;

-- One row per SKU, with its shrunken effect
SELECT * FROM glmm_fit_by('demand', sku, qty, [promo]);
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
| `ranef` | One entry per group: the BLUP, its conditional standard error, and the group's observation count |

The BLUPs are labelled with the original grouping key, whatever its type.

## Offsets

The issue's motivating example needs an exposure offset. Since an offset is
per-row it cannot be a scalar option, and the overload space is already taken, so
it is addressed by position:

```sql
-- x = [promo, log_exposure]; the second column is the offset
SELECT glmm_fit_agg(qty, [promo, LN(exposure)], sku, {
    'family': 'negbinomial',
    'offset': 2
}) FROM demand;
```

That column is removed from the design and added to the linear predictor with
coefficient fixed at 1. Take logs yourself if the link needs it.

## Degenerate inputs

| Situation | Result |
|-----------|--------|
| Fewer than two groups | `NULL` — the random intercept is not separable from the fixed one |
| Every group a singleton | `NULL` — the between-group variance is not identified |
| Fewer observations than parameters | `NULL` |

Rows with a `NULL` grouping key, or a non-finite value in `y` or any feature, are
dropped; `n_observations` reports how many were used.

## Scope

One random intercept over one grouping factor. Random slopes and crossed or
nested grouping factors are not supported yet.

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
