# Regularisation Paths and Cross-Validation

Ridge, Elastic Net and LARS fits return one solution. Choosing the penalty needs
the coefficient path over a grid of penalties and a cross-validation curve
(`glmnet`'s `plot(fit)` and `plot(cv.glmnet(...))`). These functions return both
in the long shapes of the integration contract: the path as `terms` rows with
`index_name = 'lambda'` (`'alpha'` for LARS, see below), the CV curve as `curve` rows with
`curve_type = 'lambda_cv'`.

| Function | Type | Returns |
|----------|------|---------|
| `ridge_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per lambda and term |
| `elasticnet_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per lambda and term |
| `lars_path_agg(y, x [, options])` | Aggregate | `terms` rows, one per knot and term |
| `ridge_cv_agg(y, x [, cv_key], options)` | Aggregate | `curve` rows, one per lambda |
| `elasticnet_cv_agg(y, x [, cv_key], options)` | Aggregate | `curve` rows, one per lambda |
| `ridge_nested_cv_agg(y, x [, cv_key], options)` | Aggregate | `curve` rows, one per outer fold plus a summary row |
| `elasticnet_nested_cv_agg(y, x [, cv_key], options)` | Aggregate | `curve` rows, one per outer fold plus a summary row |
| `ridge_path_by`, `elasticnet_path_by`, `lars_path_by` | Table macro | the path per group, with `id` (group) and `model_id` (method) |
| `ridge_cv_by`, `elasticnet_cv_by` | Table macro | the CV curve per group, with `id` (group) and `model_id` (method) |
| `ridge_nested_cv_by`, `elasticnet_nested_cv_by` | Table macro | the nested CV per group |

The paths and CV curves are computed by anofox-regression
(`solvers::path::{fit_path, lambda_grid}`, `solvers::cv` for the folds, the
CV curve and nested CV, and `FittedLars::path`). Rows with a NULL `y`, a NULL `x` or a NULL list element are
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
last is the least-squares fit. `index_name` is `'alpha'` and `index_value` is
upstream's `alpha` at the knot (not glmnet's `lambda`, so do not overlay it on a
ridge / elastic-net path without rescaling):
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
ridge_cv_agg(y DOUBLE, x DOUBLE[], cv_key ANY, options) -> STRUCT(...)[]
elasticnet_cv_agg(...)  -- same forms
```

k-fold cross-validation of the path, as `cv.glmnet(type.measure = "mse")`: for
each fold the path is fitted on the other folds (with the same lambdas) and the
fold's mean squared prediction error recorded. `cv_key` and `cv_strategy`
choose other splits (see [CV strategies](#cv-strategies)). Each element is a
`curve` row:

| Field | Description |
|-------|-------------|
| `curve_type` | `'lambda_cv'` |
| `x`, `y` | lambda and `cv_mean` |
| `y_low`, `y_high` | `cv_mean ∓ cv_se` |
| `threshold` | NULL |
| `lambda`, `cv_mean`, `cv_se` | glmnet's `lambda`, `cvm`, `cvsd` |
| `n_nonzero` | non-zero coefficients of the full-data fit at lambda |
| `is_min`, `is_1se` | the row of `lambda.min` and of `lambda.1se` |
| `n_folds` | number of folds (splits; with `repeats`, folds × repeats) |

With `'metric': 'mae'` the fold errors are mean absolute errors
(`type.measure = "mae"`); `cv_mean` and `cv_se` are then on that scale.

`lambda` is a keyword in DuckDB's SQL; quote the field as `"lambda"` (or use `x`).

Folds: without a `cv_key`, `folds` folds (default 10, at least 3) are drawn
with the seed `seed` (alias `random_state`, default 42); `repeats` repeats
this with the seeds `seed`, `seed + 1`, ... and pools all folds into one
curve. With a `cv_key` column and no `cv_strategy`, the key is each row's fold
(any labels, at least three distinct values). The seeded folds use upstream's
own generator, so they differ from R's `sample()`; rows are put in a canonical
order first, so the result does not depend on the order in which DuckDB's
threads deliver them. All path options apply.

```sql
-- lambda.min and lambda.1se of a 5-fold CV with the folds from a column
SELECT c."lambda", round(c.cv_mean, 4) AS cv_mean, c.is_min, c.is_1se
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2, x3], fold, {'n_lambda': 30})) AS c FROM path_demo)
WHERE c.is_min OR c.is_1se;
```

## CV strategies

`cv_strategy` decides how the rows are split; the `cv_key` argument holds the
fold label, group, stratum or time of each row, depending on the strategy.
The fold logic is upstream's (`anofox_regression::solvers::cv`), which matches
scikit-learn's splitters.

| `cv_strategy` | `cv_key` | Splits | scikit-learn |
|---------------|----------|--------|--------------|
| `'kfold'` (default without `cv_key`) | none | `folds` seeded random folds, `repeats` times | `KFold(shuffle=True)` |
| `'fold_id'` (default with `cv_key`) | the fold | one fold per distinct key (at least 3) | `PredefinedSplit` |
| `'group'` | the group | `folds` folds; a group never spans two folds, groups placed largest first into the fold with the fewest rows | `GroupKFold` |
| `'logo'` | the group | one fold per group | `LeaveOneGroupOut` |
| `'stratified'` | the stratum (optional) | `folds` folds; each stratum's count per fold differs by at most one. Without `cv_key`, the strata are `n_bins` (5) quantile bins of `y` | `StratifiedKFold(shuffle=True)` |
| `'time_series'` | the time | `folds` (default 5) rolling-origin splits: each test block of `test_size` time points is predicted from the time points before it, minus `gap` | `TimeSeriesSplit` |

Several group variables form one key as a STRUCT: `{'store': store, 'region':
region}`. Integer keys are used as they are, text, STRUCT and other keys by
their hash (two distinct keys share a hash with probability about
`n_groups² / 2⁵⁴`). Rows with a NULL key are skipped.

Time-series options (`cv_strategy = 'time_series'` only):

| Option | Default | Description |
|--------|---------|-------------|
| `test_size` (alias `horizon`) | `n_times / (folds + 1)` | Time points per test block |
| `gap` | 0 | Time points left out between the training and the test rows |
| `max_train_size` | none | Use only the last `max_train_size` time points before the gap (a sliding window) |
| `window` | `'expanding'` | `'sliding'` needs `max_train_size`; setting `max_train_size` alone already slides |

The `cv_key` of `'time_series'` must be a number, date, time or timestamp. The
splits are over its distinct values in increasing order, so in a panel all
series of a date are on the same side of every split, and `test_size`, `gap`
and `max_train_size` count time points, not rows. Rows outside every test
block are not scored; `cv_mean` averages the test blocks weighted by their
rows and `cv_se` is computed over the splits.

```sql
CREATE OR REPLACE TABLE cv_panel AS
SELECT i AS id, i % 12 AS store, CASE WHEN i % 12 < 6 THEN 'north' ELSE 'south' END AS region,
       DATE '2024-01-01' + (i // 12)::INTEGER AS day,
       ((i * 7) % 13)::DOUBLE AS x1, ((i * 5) % 11)::DOUBLE AS x2,
       1.0 + 0.8 * ((i * 7) % 13) - 0.4 * ((i * 5) % 11) + (i % 12) / 4.0 + sin(i) AS y
FROM range(0, 240) t(i);

-- Group k-fold: a store never appears in two folds
SELECT c."lambda", round(c.cv_mean, 4) AS cv_mean, c.n_folds
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2], store, {'cv_strategy': 'group', 'folds': 4, 'n_lambda': 20})) AS c
      FROM cv_panel)
WHERE c.is_min;

-- Two group variables as one key; leave one (store, region) group out
SELECT c.n_folds
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2], {'store': store, 'region': region},
                                  {'cv_strategy': 'logo', 'n_lambda': 10})) AS c FROM cv_panel)
WHERE c.is_min;

-- Rolling origin over the 20 days: 4 test blocks of 3 days, 1 day embargo, at most 10 days of training
SELECT c."lambda", round(c.cv_mean, 4) AS cv_mean, c.is_1se
FROM (SELECT unnest(elasticnet_cv_agg(y, [x1, x2], day,
        {'cv_strategy': 'time_series', 'folds': 4, 'test_size': 3, 'gap': 1, 'max_train_size': 10,
         'n_lambda': 10})) AS c FROM cv_panel)
WHERE c.is_min OR c.is_1se;

-- Stratified on quantile bins of y, repeated 3 times
SELECT c.n_folds
FROM (SELECT unnest(ridge_cv_agg(y, [x1, x2], {'cv_strategy': 'stratified', 'folds': 5, 'repeats': 3,
                                               'n_lambda': 10})) AS c FROM cv_panel)
WHERE c.is_min;
```

The low-level form stays available: compute any fold column in SQL and pass it
as `cv_key` (`cv_strategy` `'fold_id'`).

## ridge_nested_cv_agg / elasticnet_nested_cv_agg

```text
ridge_nested_cv_agg(y DOUBLE, x DOUBLE[] [, options]) -> STRUCT(...)[]
ridge_nested_cv_agg(y DOUBLE, x DOUBLE[], cv_key ANY, options) -> STRUCT(...)[]
elasticnet_nested_cv_agg(...)  -- same forms
```

Nested cross-validation estimates the prediction error of the whole procedure
"choose lambda by CV, then refit", which the minimum of a CV curve
underestimates. The outer splits come from `cv_strategy` and `cv_key` as
above (`folds` outer folds, default 5). For each outer split:

1. the inner folds are built from the outer training rows only, with the same
   key: `inner_folds` (default 5) seeded random folds for `'kfold'` and
   `'fold_id'`, group k-fold on the same groups for `'group'` and `'logo'`,
   stratified folds on the same strata for `'stratified'` and rolling-origin
   splits over the training time points (same `gap` and window) for
   `'time_series'`;
2. without `lambdas`, the lambda grid is glmnet's default grid of the outer
   training rows;
3. lambda is chosen on the inner CV curve by `rule`: `'min'` (default,
   `lambda.min`) or `'1se'` (`lambda.1se`);
4. the path fitted on all outer training rows is scored at that lambda on the
   outer test rows.

No outer test row enters an inner fit, an inner score, the grid or the refit.
Each element is a `curve` row with `curve_type = 'nested_cv'`:

| Field | Outer fold rows (`series = 'fold'`) | Summary row (`series = 'mean'`) |
|-------|-------------------------------------|---------------------------------|
| `x`, `fold` | outer fold number (1, 2, ...) | NULL |
| `y`, `test_error` | error at the chosen lambda on the fold's test rows | mean outer error, weighted by test rows |
| `y_low`, `y_high` | NULL | mean ∓ `test_se` |
| `test_se` | NULL | standard error of the mean (`cv.glmnet`'s formula over the outer folds) |
| `lambda` | the chosen lambda | NULL |
| `inner_cv_mean`, `inner_cv_se` | inner CV error at the chosen lambda | NULL |
| `n_train`, `n_test` | outer training / test rows | NULL / rows tested |
| `n_inner_folds` | inner splits | NULL |
| `n_folds` | outer splits | outer splits |
| `metric` | `'mse'` or `'mae'` | same |
| `threshold` | NULL | NULL |

Options: those of the CV aggregates (`cv_strategy`, `folds`, `seed`, `metric`,
the time-series options, `n_bins` and the path options) plus `inner_folds`
(default 5, at least 2) and `rule` (`'min'` or `'1se'`). `repeats` is not
supported.

```sql
-- Outer: 4 store-group folds; inner: 3 store-group folds on each outer training set
SELECT n.series, n.fold, round(n."lambda", 5) AS "lambda", round(n.test_error, 4) AS test_error,
       round(n.test_se, 4) AS test_se
FROM (SELECT unnest(ridge_nested_cv_agg(y, [x1, x2], store,
        {'cv_strategy': 'group', 'folds': 4, 'inner_folds': 3, 'n_lambda': 20})) AS n FROM cv_panel)
ORDER BY n.series, n.fold;

-- Rolling origin with the 1se rule
SELECT n.fold, n.n_train, n.n_test, round(n.test_error, 4) AS test_error
FROM (SELECT unnest(elasticnet_nested_cv_agg(y, [x1, x2], day,
        {'cv_strategy': 'time_series', 'folds': 3, 'inner_folds': 3, 'rule': '1se', 'n_lambda': 10})) AS n
      FROM cv_panel)
WHERE n.series = 'fold';
```

## Table macros

`ridge_path_by`, `elasticnet_path_by`, `lars_path_by(source, group_col, y_col, x_cols [, options := ..., names := [...]])`
return the group column, `id` (the group as text), `model_id` and `model_name` (the method) and the `terms` columns (the same columns as
`<model>_tidy_by`; `index_name` is `'lambda'`, `'alpha'` for LARS). `ridge_cv_by` and `elasticnet_cv_by(source, group_col, y_col, x_cols [, options := ..., cv_key := ...])`
return the group column, `id`, `model_id`, `model_name` and the `curve` columns `curve_type, series,
x, y, y_low, y_high, threshold, weight, n_censor` (the optional ones NULL) plus
`lambda, cv_mean, cv_se, n_nonzero, is_min, is_1se, n_folds`.
`ridge_nested_cv_by` and `elasticnet_nested_cv_by` take the same arguments and
return the group column, `id`, `model_id`, `model_name`, the `curve` columns
and the nested fields `fold, lambda, test_error, test_se, inner_cv_mean,
inner_cv_se, n_train, n_test, n_inner_folds, n_folds, metric`. `cv_key :=` is
a column or a STRUCT of columns (the CV is per group of `group_col`, the folds
follow `cv_key` within it). All seven are tagged as integration-contract
producers in `duckdb_functions()`.

```sql
SELECT grp, term, index_value AS penalty, round(estimate, 4) AS estimate
FROM elasticnet_path_by('path_demo', grp, y, [x1, x2, x3], options := {'n_lambda': 5}, names := ['trend', 'cycle', 'wave'])
ORDER BY grp, penalty DESC, term;

SELECT grp, x AS penalty, y AS cv_mean, y_low, y_high, is_min
FROM ridge_cv_by('path_demo', grp, y, [x1, x2, x3], options := {'n_lambda': 10, 'folds': 5});

-- Per region, group folds by store
SELECT region, "lambda", round(cv_mean, 4) AS cv_mean
FROM ridge_cv_by('cv_panel', region, y, [x1, x2], options := {'cv_strategy': 'group', 'folds': 3, 'n_lambda': 10},
                 cv_key := store)
WHERE is_min
ORDER BY region;

SELECT region, series, fold, round(test_error, 4) AS test_error
FROM ridge_nested_cv_by('cv_panel', region, y, [x1, x2],
                        options := {'cv_strategy': 'group', 'folds': 3, 'inner_folds': 2, 'n_lambda': 10},
                        cv_key := store)
ORDER BY region, series, fold;
```

## Validation

The fold generators are upstream's `solvers::cv`, validated there against
scikit-learn 1.9.1 (`GroupKFold`, `LeaveOneGroupOut`, `StratifiedKFold`,
`StratifiedGroupKFold`, `TimeSeriesSplit`) with property tests (no group in two
folds, stratum counts per fold within one, no outer test row in an inner
split). `test/sql/paths/test_cv_strategies.test` checks the SQL side: group
integrity, the time-series splits against the scikit-learn fixture, the
equivalence of every strategy with the same folds passed as `fold_id`, and
nested CV against a hand-written outer loop and under perturbed outer test
responses.

`test/sql/reference/regularization_path.test` compares the paths and CV
aggregates with glmnet's documented objective, default grid and `cv.glmnet`
aggregates (coordinate descent to 1e-26, ridge in closed form), with grpreg as
an independent lasso solver, and the LARS knots with the lasso solution and
`lm` (generator: `validation/generators/make_reference_paths_inference.R`;
glmnet itself was not installed).

## See Also

- [Ridge](ridge.md), [Elastic Net](elasticnet.md), [LARS](lars.md) - single fits
- [Model tools](model_tools.md) - `tidy_by` uses the same `terms` columns
