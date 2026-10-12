# Regularisation Paths and Cross-Validation

Ridge, Elastic Net and LARS fits return one solution. Choosing the penalty needs
the coefficient path over a grid of penalties and a cross-validation curve
(`glmnet`'s `plot(fit)` and `plot(cv.glmnet(...))`). These functions return both
in the long shapes of the integration contract: the path as `terms` rows with
`index_name = 'lambda'`, the CV curve as `curve` rows with
`curve_type = 'lambda_cv'`.

| Function | Type | Returns |
|----------|------|---------|
| `ridge_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per lambda and term |
| `elasticnet_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per lambda and term |
| `lars_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per knot and term |
| `ridge_cv_agg(y, x [, fold_id], options)` | Aggregate | `curve` rows, one per lambda |
| `elasticnet_cv_agg(y, x [, fold_id], options)` | Aggregate | `curve` rows, one per lambda |
| `ridge_path_by`, `elasticnet_path_by`, `lars_path_by` | Table macro | the path per group, with `model_id` |
| `ridge_cv_by`, `elasticnet_cv_by` | Table macro | the CV curve per group, with `model_id` |

The paths and CV curves are computed by anofox-regression
(`solvers::path::{fit_path, lambda_grid, cv_path, kfold_ids}` and
`FittedLars::path`). Rows with a NULL `y`, a NULL `x` or a NULL list element are
skipped; a group without usable rows gives NULL. A constant column is dropped
and its estimates are NULL.

## The glmnet objective

For a decreasing grid of penalties `λ` the ridge and elastic-net paths solve
`glmnet(family = "gaussian")`'s problem

```text
1/(2n) · ||y − β₀ − Xβ||² + λ · [ (1 − l1_ratio)/2 · ||β||² + l1_ratio · ||β||₁ ]
```

with glmnet's conventions: `λ` is the per-observation penalty, the intercept is
not penalised, `standardize` penalises the coefficients of unit-variance columns
(coefficients are always reported on the original scale), and `y` is scaled to
unit variance internally, so the ridge part of the penalty is divided by
`sd(y)` exactly as glmnet does. `l1_ratio` is glmnet's `alpha` (0 = ridge,
1 = lasso). This `λ` is **not** the `alpha` / `lambda` option of
`ridge_fit_agg` / `elasticnet_fit_agg`, which use the raw scaling by default.

Without `lambdas` the path uses glmnet's default grid: `n_lambda` values
log-spaced from `λ_max` (the smallest `λ` with all coefficients 0; for ridge
glmnet uses `l1_ratio = 0.001` in that formula) down to
`λ_max · lambda_min_ratio` (default 1e-4 when n > p, else 0.01).

## ridge_path_agg / elasticnet_path_agg

```text
ridge_path_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT(...)[]
elasticnet_path_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT(...)[]
```

Each element is a `terms` row: `term` (`'(Intercept)'`, then `x1`, `x2`, ... or
the `feature_names`), `estimate`, `std_error`, `statistic`, `p_value`,
`conf_low`, `conf_high`, `conf_level` (NULL: a penalised path has no
inference), `index_name` (`'lambda'`) and `index_value` (the penalty). The
lambdas keep the order they were given in (the default grid is decreasing).

| Option | Default | Description |
|--------|---------|-------------|
| `lambdas` | glmnet's grid | Penalties on glmnet's scale, `>= 0` (`0` is least squares) |
| `n_lambda` (alias `nlambda`) | 100 | Size of the default grid |
| `lambda_min_ratio` | 1e-4 (n > p) / 0.01 | Smallest / largest lambda of the default grid |
| `l1_ratio` | 0.5 | Elastic net only: mixing in [0, 1] |
| `standardize` | false | Penalise standardised coefficients (glmnet's default is true) |
| `fit_intercept` | true | Fit an unpenalised intercept |
| `feature_names` | x1, x2, ... | Term labels |

```sql
CREATE OR REPLACE TABLE path_demo AS
SELECT i AS id, (i % 2)::VARCHAR AS grp, i::DOUBLE AS x1, ((i * 3) % 7)::DOUBLE AS x2, sin(i) AS x3,
       1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) + cos(3 * i) AS y, (i % 5)::BIGINT AS fold
FROM range(1, 81) t(i);

-- The lasso path at three penalties
SELECT e.index_value AS penalty, e.term AS term, round(e.estimate, 4) AS estimate
FROM (SELECT unnest(elasticnet_path_agg(y, [x1, x2, x3], {'l1_ratio': 1.0, 'lambdas': [1.0, 0.3, 0.01]})) AS e
      FROM path_demo)
ORDER BY penalty DESC, term;
```

## lars_path_agg

```text
lars_path_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT(...)[]
```

The LARS (`'method': 'lar'`) or Lasso-LARS (`'lasso'`, the default) path, one
entry per knot, from `FittedLars::path`. The first knot has all slopes 0, the
last is the least-squares fit. `index_value` is upstream's `alpha` at the knot:
`max_j |x_j'(y − ŷ)|` on the centred design (columns scaled to unit norm with
`standardize`). It is `n · λ` of the glmnet objective above without
`standardize` and `sqrt(n) · λ` with it. Options: `method`, `standardize`
(default true, as `lars_fit_agg`), `n_nonzero_coefs`, `fit_intercept`,
`feature_names`.

```sql
SELECT e.index_value AS alpha, e.term AS term, round(e.estimate, 4) AS estimate
FROM (SELECT unnest(lars_path_agg(y, [x1, x2, x3], {'standardize': false})) AS e FROM path_demo)
WHERE e.term <> '(Intercept)'
ORDER BY alpha DESC, term;
```

## ridge_cv_agg / elasticnet_cv_agg

```text
ridge_cv_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT(...)[]
ridge_cv_agg(y DOUBLE, x DOUBLE[], fold_id BIGINT, options) -> STRUCT(...)[]
elasticnet_cv_agg(...)  -- same forms
```

k-fold cross-validation of the path, as `cv.glmnet(type.measure = "mse")`: for
each fold the path is fitted on the other folds (with the same lambdas) and the
fold's mean squared prediction error recorded. Each element is a `curve` row:

| Field | Description |
|-------|-------------|
| `curve_type` | `'lambda_cv'` |
| `x`, `y` | lambda and `cv_mean` |
| `y_low`, `y_high` | `cv_mean ∓ cv_se` |
| `threshold` | NULL |
| `lambda`, `cv_mean`, `cv_se` | glmnet's `lambda`, `cvm`, `cvsd` |
| `n_nonzero` | non-zero coefficients of the full-data fit at lambda |
| `is_min`, `is_1se` | the row of `lambda.min` and of `lambda.1se` |
| `n_folds` | number of folds |

`lambda` is a keyword in DuckDB's SQL; quote the field as `"lambda"` (or use `x`).

Folds: with a `fold_id` column each row's fold is given (any labels, at least
three distinct values); otherwise `folds` folds (default 10, at least 3) are
drawn with the seed `seed` (alias `random_state`, default 42). The seeded folds
use upstream's own generator, so they differ from R's `sample()`; rows are put
in a canonical order first, so the result does not depend on the order in which
DuckDB's threads deliver them. All path options apply.

```sql
-- lambda.min and lambda.1se of a 5-fold CV with the folds from a column
SELECT c."lambda", round(c.cv_mean, 4) AS cv_mean, c.is_min, c.is_1se
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2, x3], fold, {'n_lambda': 30})) AS c FROM path_demo)
WHERE c.is_min OR c.is_1se;
```

## Table macros

`ridge_path_by`, `elasticnet_path_by`, `lars_path_by(source, group_col, y_col, x_cols [, options := ..., names := [...]])`
return the group column, `model_id` and the `terms` columns (the same columns as
`<model>_tidy_by`). `ridge_cv_by` and `elasticnet_cv_by(source, group_col, y_col, x_cols [, options := ...])`
return the group column, `model_id` and the `curve` columns `curve_type, series,
x, y, y_low, y_high, threshold, weight, n_censor` (the optional ones NULL) plus
`lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds`. All five are
tagged as integration-contract producers in `duckdb_functions()`.

```sql
SELECT grp, term, index_value AS penalty, round(estimate, 4) AS estimate
FROM elasticnet_path_by('path_demo', grp, y, [x1, x2, x3], options := {'n_lambda': 5}, names := ['trend', 'cycle', 'wave'])
ORDER BY grp, penalty DESC, term;

SELECT grp, x AS penalty, y AS cv_mean, y_low, y_high, is_min
FROM ridge_cv_by('path_demo', grp, y, [x1, x2, x3], options := {'n_lambda': 10, 'folds': 5});
```

## Validation

`test/sql/reference/regularization_path.test` compares the paths and CV
aggregates with glmnet's documented objective, default grid and `cv.glmnet`
aggregates (coordinate descent to 1e-26, ridge in closed form), with grpreg as
an independent lasso solver, and the LARS knots with the lasso solution and
`lm` (generator: `validation/generators/make_reference_paths_inference.R`;
glmnet itself was not installed).

## See Also

- [Ridge](ridge.md), [Elastic Net](elasticnet.md), [LARS](lars.md) - single fits
- [Model tools](model_tools.md) - `tidy_by` uses the same `terms` columns
