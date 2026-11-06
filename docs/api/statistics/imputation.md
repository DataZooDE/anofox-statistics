# Multiple Imputation and Rubin Pooling

Multiple imputation of missing values by additive regression, bootstrapping
and predictive mean matching (`Hmisc::aregImpute`, Harrell's *Regression
Modeling Strategies* §3.7), and Rubin's rules to combine the fits to the
imputed data sets (`mice::pool`, `mice::D1`).

The workflow is: impute `m` completed copies of the rows with
`areg_impute_agg` (or `areg_impute_by`), fit the same model to each copy with
any `*_fit_agg` grouped by `imputation`, and pool the `m` model STRUCTs with
`pool_rubin_agg`.

The computation is upstream anofox-regression `imputation::areg_impute` and
`inference::pool` (`pool_rubin`, `pooled_wald_d1`). The imputations use R's
random number generator: with the same seed they replay
`set.seed(seed); aregImpute(...)` in R value for value
(`test/sql/reference/imputation_reference.test`, which also checks the pooled
results against `mice::pool` and `mice::D1`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `areg_impute_agg` | Aggregate | `n_impute` completed copies of the rows: `LIST(STRUCT(imputation, id, x))` |
| `areg_impute_by` | Table macro | The completed copies per group as rows (`group_col, imputation, id, x`) |
| `pool_rubin_agg` | Aggregate | Pools one model STRUCT per imputation by Rubin's rules |

## areg_impute_agg

**Signature:**

```text
areg_impute_agg(id ANY, x DOUBLE[] [, options MAP]) -> LIST(STRUCT(imputation INTEGER, id ANY, x DOUBLE[]))
```

`x` holds the variables of one row; a NULL or NaN element is a missing value.
Every variable is imputed in turn from all the others (chained equations),
so `x` should contain every variable of the analysis model, including the
outcome. The rows are taken in `id` order (the order of the data frame in
R): the random draws, and therefore the imputations, depend on it, so the
result does not depend on the order the rows arrive in. `id` must be unique
within a group.

Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_impute` | INTEGER | 5 | Number of imputations (`n.impute`) |
| `nk` | INTEGER | 3 | Spline knots (`nk`); 0 makes every variable linear |
| `tlinear` | BOOLEAN | true | Model only the linear part of a spline variable being imputed (`tlinear`) |
| `burnin` | INTEGER | Hmisc's | Iterations before the first kept imputation (`burnin`; Hmisc's default is 3, or 0 when only one variable has missing values) |
| `match` | VARCHAR | `'weighted'` | Donor selection: `'weighted'` (tricube weights on the distance in predicted values) or `'closest'` |
| `fweighted` | DOUBLE | 0.2 | Tricube scale of weighted matching (`fweighted`) |
| `seed` | INTEGER | 1 | Seed of R's random number generator (`set.seed`) |
| `categorical` | INTEGER[] | none | 1-based positions in `x` of categorical variables (level codes; `factor` in R) |
| `linear` | INTEGER[] | none | 1-based positions in `x` of variables that always enter linearly (`I(x)` in R's formula) |

A variable with only two distinct values always enters linearly.

**Returns:** one STRUCT per imputation and row, ordered by `imputation`
(1 .. `n_impute`) and `id`; `x` is the row with its missing values filled in
(observed values are kept). Each imputed value is an observed value of a
donor row (predictive mean matching), so a categorical variable gets an
existing level code.

**NULL handling.** Rows whose `id` or `x` is NULL are left out. A group without
rows returns NULL; so does a group whose data cannot be imputed (a variable
that is constant or entirely missing, or a bootstrap resample that lost a
level of a categorical variable). `x` must have at least two variables and
the same number in every row; infinite values are an error.

**Example:**

```sql
CREATE OR REPLACE TABLE imp_demo AS
SELECT
    i AS id,
    1 + 0.5 * (i % 7) + 0.3 * (i % 5) + ((i * 37) % 11) * 0.1 AS y,
    CASE WHEN i % 6 = 0 THEN NULL ELSE (i % 7)::DOUBLE + ((i * 13) % 5) * 0.2 END AS x1,
    CASE WHEN i % 8 = 3 THEN NULL ELSE (i % 5)::DOUBLE END AS x2,
    (i % 3 + 1)::DOUBLE AS grp
FROM range(1, 81) t(i);

-- 5 completed copies of the 80 rows
SELECT r.imputation, count(*) AS n_rows
FROM (SELECT unnest(areg_impute_agg(id, [y, x1, x2, grp], {'categorical': [4], 'seed': 2026})) AS r FROM imp_demo)
GROUP BY r.imputation
ORDER BY r.imputation;
```

## areg_impute_by

**Signature:**

```text
areg_impute_by(source VARCHAR, group_col, id_col, x_cols [, options := MAP]) -> TABLE
```

Runs `areg_impute_agg(id_col, x_cols, options)` per group of `source` and
returns one row per group, imputation and id: `group_col`, `imputation`,
`id`, `x` (the completed row, `DOUBLE[]`), ordered by group, imputation and
id. Use a constant `group_col` to impute the whole table at once.

**Example:**

```sql
SELECT imputation, id, x
FROM areg_impute_by('imp_demo', 1, id, [y, x1, x2, grp],
                    options := {'categorical': [4], 'n_impute': 3, 'seed': 2026})
WHERE id IN (6, 11)
ORDER BY imputation, id;
```

## pool_rubin_agg

**Signature:**

```text
pool_rubin_agg(model STRUCT [, options MAP]) -> STRUCT
```

Pools the fits of the same model to `m >= 2` imputed data sets (one model
STRUCT per row, NULL models skipped) by Rubin's rules: the pooled
coefficients `Q̄` are the mean of the estimates, the total covariance is
`T = Ū + (1 + 1/m) B` (mean within-imputation covariance plus the inflated
between-imputation covariance). Each model must have `coefficients`,
`intercept` (NULL or absent without one) and either `vcov` or `std_errors`
(+ `intercept_std_error`): fit with `{'compute_inference': true}`. Any model
STRUCT of this extension with these fields works (OLS, WLS, Ridge, the GLMs,
...). With only standard errors the coefficients are pooled one at a time and
the D1 test assumes them uncorrelated.

The degrees of freedom are Barnard and Rubin's small-sample d.f. given the
complete-data d.f. (`mice`'s `dfcom`), taken from the models' `df_residual`
(the smallest, if they differ) unless `df_complete` is given; without one,
Rubin's (1987) large-sample d.f. `(m − 1)(1 + 1/r)²`.

Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `confidence_level` | DOUBLE | 0.95 | Level of the confidence intervals |
| `df_complete` (alias `dfcom`) | DOUBLE | models' `df_residual` | Complete-data residual d.f. |
| `small_sample` | BOOLEAN | true | false: Rubin's large-sample d.f. whatever the complete-data d.f. (`mice::pool(dfcom = Inf)`) |

**Returns** a model-like STRUCT (`tidy()` lists its terms), slopes in the
lists and the intercept in the `intercept_*` fields:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Pooled estimates `Q̄` |
| `std_errors` | `DOUBLE[]` | `√T`, total standard errors |
| `t_values` | `DOUBLE[]` | `Q̄ / √T` |
| `p_values` | `DOUBLE[]` | Two-sided p-values on `df` |
| `conf_low`, `conf_high` | `DOUBLE[]` | Confidence interval `Q̄ ± t(df) √T` |
| `df` | `DOUBLE[]` | Pooled d.f. per coefficient |
| `riv` | `DOUBLE[]` | Relative increase in variance `(1 + 1/m) B / Ū` |
| `lambda` | `DOUBLE[]` | Proportion of variance due to missingness `(1 + 1/m) B / T` |
| `fmi` | `DOUBLE[]` | Fraction of missing information |
| `intercept`, `intercept_std_error`, `intercept_statistic`, `intercept_p_value`, `intercept_conf_low`, `intercept_conf_high`, `intercept_df`, `intercept_riv`, `intercept_lambda`, `intercept_fmi` | `DOUBLE` | The same for the intercept (NULL without one) |
| `conf_level` | `DOUBLE` | Confidence level |
| `vcov` | `DOUBLE[]` | Pooled covariance `T`, row-major, intercept first |
| `f_statistic`, `f_df1`, `f_df2`, `f_pvalue`, `f_riv` | `DOUBLE` | D1 pooled Wald test that all slopes are zero (Li, Raghunathan and Rubin 1991, `mice::D1`; Reiter's denominator d.f. with a complete-data d.f.), and its average relative increase in variance |
| `df_complete` | `DOUBLE` | Complete-data d.f. used (NULL: Rubin's d.f.) |
| `n_imputations` | `BIGINT` | `m` |
| `model_type` | `VARCHAR` | `'rubin_pooled'` |

Fewer than two models give NULL. Values that are not finite are NULL: when
the imputations agree exactly on a coefficient (`B = 0`, e.g. no missing
values), Rubin's d.f. is infinite (NULL) and its p-value and interval use the
standard normal, as `mice` (`df = Inf`); with the small-sample d.f. the
Barnard–Rubin d.f. is finite there (mice's form `(m − 1)·tmp / ((ν + 3)(m − 1) + λ²·tmp)`,
`tmp = (1 − λ)(ν + 1)ν`). Reiter's d.f. of the D1 test needs
`k (m − 1) > 4` (`k` slopes, `m` imputations); with fewer it is unreliable or
NULL, as in `mitml`. The result does not depend on the order of the models.

**Example:**

```sql
-- impute, fit one OLS per completed data set, pool
WITH imp AS (
    SELECT * FROM areg_impute_by('imp_demo', 1, id, [y, x1, x2, grp],
                                 options := {'categorical': [4], 'n_impute': 5, 'seed': 2026})
),
fits AS (
    SELECT imputation, ols_fit_agg(x[1], [x[2], x[3]], {'compute_inference': true}) AS fit
    FROM imp GROUP BY imputation
)
SELECT
    round(p.intercept, 4) AS intercept,
    list_transform(p.coefficients, lambda c: round(c, 4)) AS coefficients,
    list_transform(p.std_errors, lambda s: round(s, 4)) AS std_errors,
    list_transform(p.fmi, lambda f: round(f, 3)) AS fmi,
    round(p.f_pvalue, 6) AS d1_p_value,
    p.n_imputations
FROM (SELECT pool_rubin_agg(fit) AS p FROM fits);
```

The pooled terms as a table:

```sql
WITH imp AS (
    SELECT * FROM areg_impute_by('imp_demo', 1, id, [y, x1, x2, grp],
                                 options := {'categorical': [4], 'seed': 2026})
)
SELECT t.term, round(t.estimate, 4) AS estimate, round(t.std_error, 4) AS std_error,
       round(t.p_value, 5) AS p_value
FROM (
    SELECT unnest(tidy(pool_rubin_agg(fit), ['x1', 'x2'])) AS t
    FROM (SELECT ols_fit_agg(x[1], [x[2], x[3]], {'compute_inference': true}) AS fit
          FROM imp GROUP BY imputation)
);
```
