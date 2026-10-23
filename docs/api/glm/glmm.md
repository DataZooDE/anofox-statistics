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
| `glmm_tidy_by` | Table Macro | Fixed effects, variance components and BLUPs as `terms` rows (broom.mixed `tidy`) |
| `glmm_glance_by` | Table Macro | Fit statistics as `summary` rows (broom.mixed `glance`) |
| `glmm_augment_by` | Table Macro | Per-row fitted values and residuals as `obs` rows (broom.mixed `augment`) |
| `glmm_augment_agg` | Window Aggregate | The per-row values behind `glmm_augment_by` |

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
| family | VARCHAR | 'gaussian' | `gaussian` (aliases `normal`, `lmm`), `poisson`, `binomial` (alias `logistic`), `negbinom` (aliases `negbinomial`, `negative_binomial`, `negbin`; NB2, log link), `gamma` (log link, `y > 0`) or `tweedie` (log link, `y >= 0`) |
| theta | DOUBLE | estimated | `negbinom` only: fixes the size θ (`Var = μ + μ²/θ`), as `glmer(family = MASS::negative.binomial(θ))`. Unset, θ is estimated as `lme4::glmer.nb` |
| power | DOUBLE | 1.5 | `tweedie` only: variance power, `1 < power < 2` |
| offset | INTEGER | — | 1-based index into `x` of a column added to the linear predictor with coefficient 1 (`offset()` in an lme4 formula) and removed from the design |
| fit_intercept | BOOLEAN | true | Include a fixed intercept |
| reml | BOOLEAN | true | REML rather than ML for the Gaussian variance components |
| random | INTEGER[] | — | 1-based indices into `x` of feature columns that also get a **random slope** (unstructured covariance with the intercept) on the positional `group` factor; with `groups` this is `(1 + x | group) + (1 | other)` |
| groups | INTEGER[] | — | 1-based indices into `x` of additional **crossed** grouping-factor columns; each becomes an independent random intercept and is removed from the design |
| max_iterations | INTEGER | 100 | Inner PIRLS iterations |
| tolerance | DOUBLE | 1e-8 | Convergence tolerance |
| compute_inference | BOOLEAN | false | Fixed-effect standard errors, tests and intervals |
| ddf | VARCHAR | family default | Degrees of freedom of the fixed-effect tests: `'satterthwaite'` (t tests with lmerTest's Satterthwaite df; gaussian only, the gaussian default) or `'wald'` (z tests; the default for every other family) |
| confidence_level | DOUBLE | 0.95 | Interval level (fixed effects and BLUPs) |

> The solver lives upstream in `anofox-regression`; this extension is a wrapper.
> Non-gaussian families maximise the Laplace approximation with the fixed effects
> inside the penalised conditional mode, so they match lme4 with `nAGQ = 0`
> (`glmer(..., nAGQ = 0)`, `glmer.nb(..., nAGQ = 0)`), not lme4's default
> `nAGQ = 1` (tracked upstream in
> [anofox-regression#67](https://github.com/sipemu/anofox-regression/issues/67)).
> For `gamma` and `tweedie`, `var_residual` is lme4's `sigma()^2 = pwrss / n`, which
> also scales the fixed-effect standard errors, and the log-likelihood uses the
> dispersion `deviance / n` (R's `Gamma()$aic`). `aic`/`bic` count the fixed effects,
> the random-effect covariance parameters and, except for poisson and binomial, one
> scale parameter (θ for `negbinom`), as the df of lme4's `logLik`.
> Not yet available: random slopes on the additional `groups` factors, and the BLUPs
> of the additional `groups` factors (`ranef` holds the positional `group` factor's).
> Invalid `theta` / `power` / `ddf` values, `ddf := 'satterthwaite'` with a
> non-gaussian family, and an `offset` or `random` column that is also listed in
> `groups`, raise an error.

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
       ranef LIST(STRUCT(group VARCHAR, intercept DOUBLE, se DOUBLE, n BIGINT,
                         effects DOUBLE[], effects_se DOUBLE[],
                         effects_conf_low DOUBLE[], effects_conf_high DOUBLE[])),
       nb_theta DOUBLE,               -- negbinom size theta (estimated or fixed); NULL otherwise
       df DOUBLE[],                   -- Satterthwaite df of the coefficients; NULL for Wald
       intercept_df DOUBLE,           -- Satterthwaite df of the intercept; NULL for Wald
       df_residual DOUBLE,            -- n - number of parameters (lme4 df.residual)
       t_values DOUBLE[],             -- t statistics under Satterthwaite (z_values is then NULL)
       ...)                           -- plus the shared summary fields (conf_low, model_type, ...)
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

-- Overdispersed counts: negative binomial with theta estimated (as glmer.nb)
SELECT (glmm_fit_agg(qty, [promo], sku, {'family': 'negbinom'})).nb_theta FROM demand;

-- Gaussian LMM: Satterthwaite t tests (lmerTest), or Wald z tests on request
SELECT (fit).df, (fit).p_values, (fit).intercept_df
FROM (SELECT glmm_fit_agg(qty, [promo], sku, {'compute_inference': true}) AS fit FROM demand);
SELECT (glmm_fit_agg(qty, [promo], sku, {'compute_inference': true, 'ddf': 'wald'})).p_values FROM demand;
```

### Inference

With `compute_inference := true` the fixed effects get standard errors from
`vcov(β) = σ² M⁻¹` (lme4's `vcov()`), and tests and intervals by `ddf`:

| `ddf` | Statistic | p-value and interval | `df` | Matches |
|-------|-----------|----------------------|------|---------|
| `'satterthwaite'` (gaussian default) | `t_values` = β̂ / SE (`z_values` NULL) | t distribution with Satterthwaite df, `β̂ ∓ t·SE` | per coefficient | `lmerTest::summary(lmer(...))`, `contest1D(..., confint = TRUE)` |
| `'wald'` (default otherwise) | `z_values` = β̂ / SE (`t_values` NULL) | standard normal, `β̂ ∓ z·SE` | NULL | `summary(glmer(...))`, `confint(m, method = "Wald")` |

Exactly one of `t_values` / `z_values` is set; a non-NULL `df` also marks t tests.
`intercept_statistic` holds the intercept's t or z. The Satterthwaite degrees of freedom
follow lmerTest: the Hessian of the REML (or ML) deviance in the variance
parameters and the Jacobian of `vcov(β)`, both by Richardson-extrapolated finite
differences. `compute_inference` stays off by default, as before; the
`glmm_tidy_by` macro turns it on.

The BLUPs always come with their **conditional standard errors** (`ranef[].se`,
and `effects_se` for every random effect), the square roots of the diagonal of
`σ² Λ (ΛᵀZᵀWZΛ + I)⁻¹ Λᵀ` as `lme4::ranef(condVar = TRUE)` / `arm::se.ranef`, and
a normal interval `effects ∓ z·effects_se` at `confidence_level`
(`broom.mixed::tidy(effects = "ran_vals", conf.int = TRUE)`). They are
conditional on the estimated fixed effects and variance components.

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
| `ranef_se` | DOUBLE | Its conditional standard error (lme4 `ranef(condVar = TRUE)`) |
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

## glmm_tidy_by, glmm_glance_by, glmm_augment_by

The integration-contract outputs (`anofox_contract_version() = '1'`) of a mixed
model, in the shapes of `broom.mixed::tidy`, `glance` and `augment`. They fit
**one model per `group_col`** (the outer group; pass a constant column to fit one
model over the whole table) with random effects over `re_group`, and accept the
`glmm_fit_agg` options. `model_id` and `model_name` are `'glmm'`; `id` is the outer
group as text (not in `glmm_augment_by`, which keeps every source column).

**Signatures:**

```text
glmm_tidy_by(source, group_col, y_col, x_cols, re_group
             [, options := NULL, names := NULL, re_names := NULL]) -> TABLE
glmm_glance_by(source, group_col, y_col, x_cols, re_group [, options := NULL]) -> TABLE
glmm_augment_by(source, group_col, y_col, x_cols, re_group
                [, options := NULL, order_by := NULL]) -> TABLE
```

**glmm_tidy_by** returns the `terms` columns (group column, `id`, `model_id`,
`model_name`, `term`, `estimate`, `std_error`, `statistic`, `p_value`,
`conf_low`, `conf_high`, `conf_level`, `index_name`, `index_value`) followed by
`effect`, `re_group`, `re_level` and `df`. Inference is always computed.

| `effect` | Rows | Columns filled |
|----------|------|----------------|
| `fixed` | one per fixed effect, intercept first | all; `statistic` is `t_values` (Satterthwaite, with `df`) or `z_values` (Wald), whichever is set |
| `ran_pars` | `sd__<term>` per random effect, `cor__<term>.<term>` per correlation (random slopes), `sd__(Intercept)` of each further `groups` factor, and `sd__Observation` (`re_group = 'Residual'`) for the families with a dispersion | `estimate` |
| `ran_vals` | one per level (`re_level`) and random effect of `re_group` | `estimate` (BLUP), `std_error` (conditional SE), `conf_low` / `conf_high`, `conf_level` |

`names := [...]` labels the fixed-effect columns (as in the other `_tidy_by`
macros; random-slope terms use the same labels), `re_names := [...]` the random
effects' grouping factors: `re_group` first (default `'group'`), then each `groups`
column (default `x<k>`).

**glmm_glance_by** returns the `summary` columns, one row per metric:
`n_observations`, `n_groups`, `n_features`, `sigma`, `log_likelihood`, `aic`,
`bic`, `deviance` (the REML criterion for a REML fit), `df_residual`, `var_group`,
`var_residual`, `icc`, `nb_theta` (negative binomial), `converged` (1/0) and
`iterations`.

**glmm_augment_by** returns every source row followed by the `obs` columns of the
other `_augment_by` macros (`model_id`, `model_name`, `row_id`, `fitted`,
`residual`, `resid_type`, …) and `fitted_population` and `pearson_residual`.
`fitted` is conditional on the BLUPs and on the response scale (lme4 `fitted()`),
`fitted_population` sets the random effects to zero (`predict(re.form = NA, type =
"response")`), `residual` is `y − fitted` (`resid_type = 'response'`) and
`pearson_residual` is lme4's `residuals(type = "pearson")`. `std_residual`,
`stud_residual`, `leverage`, `cooks_d`, `dffits` and `n_params` are NULL. A row
the fit could not use (a NULL or non-finite value, a NULL `re_group`) has NULL
values and `is_training = false`.

```sql
CREATE OR REPLACE TABLE demand_all AS SELECT 'all' AS scope, * FROM demand;

-- Fixed effects with Satterthwaite df, variance components and BLUPs
SELECT effect, re_group, re_level, term, round(estimate, 3) AS estimate, round(std_error, 3) AS se, round(df, 1) AS df
FROM glmm_tidy_by('demand_all', scope, qty, [promo], sku, names := ['promo'], re_names := ['sku'])
WHERE effect <> 'ran_vals';

-- Fit statistics
SELECT metric, round(value, 3) AS value FROM glmm_glance_by('demand_all', scope, qty, [promo], sku);

-- Per-row fitted values and residuals of a Poisson GLMM
SELECT week, sku, qty, round(fitted, 3) AS fitted, round(pearson_residual, 3) AS pearson
FROM glmm_augment_by('demand_all', scope, qty, [promo], sku, options := {'family': 'poisson'})
ORDER BY week
LIMIT 5;
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
| `ranef` | One entry per level of the positional `group` factor: the intercept BLUP, its conditional SE, the observation count, and the full random-effect vector `effects` (`[intercept, slope_1, …]`) with its conditional SEs and normal interval. |
| `df` / `intercept_df` | Satterthwaite degrees of freedom of the fixed effects (gaussian with `compute_inference`); NULL for Wald inference. |
| `df_residual` | `n_observations` minus the number of parameters (fixed effects, covariance parameters, dispersion), lme4's `df.residual()`. |

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
| `ddf := 'satterthwaite'` with a non-gaussian family | Error |
| A singular gaussian fit (a variance at zero) | Satterthwaite `df` can be NaN or very large, as lmerTest's |

Rows with a `NULL` grouping key, or a non-finite value in `y` or any feature, are
dropped; `n_observations` reports how many were used.

## Scope

Random intercept and random slopes over one grouping factor, or several crossed /
nested random-intercept factors (random slopes on the first one), for the gaussian,
poisson, binomial, negative binomial, Gamma and Tweedie families, with an optional
offset. Validated against `lme4::lmer` and `lme4::glmer` / `glmer.nb` with
`nAGQ = 0` (`test/sql/reference/glm_glmm.test`), and the inference and macros
against lmerTest and broom.mixed (`test/sql/reference/glmm_inference.test`);
Tweedie has no lme4 family and is validated upstream. Not yet available: random
slopes on the additional `groups` factors and their BLUPs.

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
