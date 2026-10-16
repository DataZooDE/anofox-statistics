# Bayesian Ridge and ARD Regression

Two empirical-Bayes linear models, scikit-learn's `BayesianRidge` and
`ARDRegression`. Both put a Gaussian prior on the coefficients and estimate the
noise precision `alpha` and the prior (weight) precision `lambda` from the data
by maximising the evidence (type-II maximum likelihood), with gamma
hyperpriors on both.

- **Bayesian Ridge** uses one weight precision for all coefficients: a ridge
  regression whose penalty is chosen by the data instead of by
  cross-validation.
- **ARD** (automatic relevance determination) gives every coefficient its own
  precision. Precisions of irrelevant features grow without bound; once one
  reaches `threshold_lambda` the feature is pruned (coefficient exactly 0), so
  ARD selects features on wide, sparse problems.

Both return the posterior mean coefficients and the posterior standard
deviation of each one. The fit-predict forms bound every prediction with the
**posterior predictive interval**, `yhat ± z · sd` with
`sd = sqrt((x − x̄)' Σ (x − x̄) + 1/alpha)` and `Σ` the posterior covariance of
the coefficients (scikit-learn's `predict(X, return_std=True)`), so no
bootstrap is needed. The results agree with scikit-learn to about 1e-8
(`test/sql/reference/regression_bayesian.test`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `bayesian_ridge_fit_agg`, `ard_fit_agg` | Aggregate | Fit per group (`GROUP BY`) or over a window (`OVER`) |
| `bayesian_ridge_fit_predict_agg`, `ard_fit_predict_agg` | Aggregate | Fit on the training rows of a partition and return one prediction with its interval per row |
| `bayesian_ridge_fit_predict_by`, `ard_fit_predict_by` | Table macro | Fit and predict per group, one row per source row |
| `bayesian_ridge_tidy_by`, `ard_tidy_by` | Table macro | Coefficients per group (`terms`) |
| `bayesian_ridge_glance_by`, `ard_glance_by` | Table macro | Fit statistics per group (`summary`) |
| `bayesian_ridge_prediction_by`, `ard_prediction_by` | Table macro | Fitted values with the posterior predictive band (`prediction`) |

## bayesian_ridge_fit_agg / ard_fit_agg

**Signature:**

```text
bayesian_ridge_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
ard_fit_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `DOUBLE` | Response value |
| `x` | `DOUBLE[]` | Feature values for the row |
| `options` | `MAP` / `STRUCT` | Optional settings, see [Options](#options) |

**Example:**

```sql
CREATE OR REPLACE TABLE bayes_demo AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END AS grp,
    sin(i * 0.37) * 2 AS x1,
    cos(i * 0.11) AS x2,
    ((i * 13) % 17) / 17.0 AS x3,
    1.0 + 2.0 * sin(i * 0.37) * 2 - cos(i * 0.11) + 0.3 * sin(i * 1.7) AS y
FROM range(1, 201) t(i);

SELECT
    grp,
    list_transform(m.coefficients, lambda c: round(c, 3)) AS coefficients,
    list_transform(m.posterior_sd, lambda s: round(s, 4)) AS posterior_sd,
    round(1 / sqrt(m.noise_precision), 3) AS noise_sd
FROM (SELECT grp, bayesian_ridge_fit_agg(y, [x1, x2, x3]) AS m FROM bayes_demo GROUP BY grp)
ORDER BY grp;
```

ARD prunes the feature that does not enter `y` (`x3`): its coefficient and
posterior standard deviation are exactly 0 and its precision is at or above
`threshold_lambda`.

```sql
SELECT
    list_transform(m.coefficients, lambda c: round(c, 3)) AS coefficients,
    list_transform(m.weight_precisions, lambda l: l >= 10000) AS pruned
FROM (SELECT ard_fit_agg(y, [x1, x2, x3]) AS m FROM bayes_demo);
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_intercept` (alias `intercept`) | BOOLEAN | `true` | Include an (unpenalised) intercept; `x` and `y` are centred |
| `max_iterations` (alias `max_iter`) | INTEGER | `300` | Maximum number of evidence iterations |
| `tolerance` (alias `tol`) | DOUBLE | `0.001` | Stop when the summed absolute change of the coefficients is below this; > 0 |
| `alpha_1`, `alpha_2` | DOUBLE | `1e-6` | Shape and rate of the gamma prior on the noise precision; >= 0 |
| `lambda_1`, `lambda_2` | DOUBLE | `1e-6` | Shape and rate of the gamma prior on the weight precision(s); >= 0 |
| `threshold_lambda` | DOUBLE | `10000` | ARD only: prune a feature once its precision reaches this; > 0 |
| `confidence_level` (alias `confidence`) | DOUBLE | `0.95` | `*_fit_predict_agg` / `_by` only: level of the posterior predictive interval |
| `null_policy` | VARCHAR | `'drop'` | `*_fit_predict_agg` / `_by` only: `'drop_y_zero_x'` also leaves rows with a feature equal to 0 out of the fit, `'zero'` fits a NULL `y` or feature as 0, `'ignore'` = `'drop'`; see [null_policy](fit_predict_agg.md#null_policy) |

The defaults are scikit-learn's. Option keys a function does not support raise
an error.

## Returns

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `DOUBLE[]` | Posterior mean of each coefficient (ARD: 0 for pruned features) |
| `intercept` | `DOUBLE` | Intercept (0 with `fit_intercept` false) |
| `r_squared` | `DOUBLE` | Coefficient of determination |
| `n_observations`, `n_features` | `BIGINT` | Rows used, number of features |
| `noise_precision` | `DOUBLE` | Estimated noise precision `alpha` (scikit-learn `alpha_`); the noise standard deviation is `1 / sqrt(noise_precision)` |
| `weight_precision` | `DOUBLE` | Bayesian Ridge: the weight precision `lambda` (scikit-learn `lambda_`) |
| `weight_precisions` | `DOUBLE[]` | ARD: one precision per feature (scikit-learn `lambda_`) |
| `posterior_sd` | `DOUBLE[]` | Posterior standard deviation of each coefficient, `sqrt(diag(sigma_))` (ARD: 0 for pruned features) |
| `iterations` | `BIGINT` | Evidence iterations run (scikit-learn `n_iter_`) |
| `log_marginal_likelihood` | `DOUBLE` | Bayesian Ridge: log evidence at the estimated precisions (scikit-learn's last `scores_` entry) |

Like every model STRUCT (see [model tools](model_tools.md)) the result also has
the inference fields (`std_errors`, `t_values`, `p_values`, `conf_low`,
`conf_high`, `conf_level`, `intercept_*`) and the shared tail (`log_likelihood`,
`aic`, `bic`, `model_type`, `family`, `link`). The models have no frequentist
inference or likelihood-based information criteria, so these are NULL; the
coefficient uncertainty is in `posterior_sd`. `model_type` is
`'bayesian_ridge'` or `'ard'`.

## bayesian_ridge_fit_predict_agg / ard_fit_predict_agg

```text
<model>_fit_predict_agg(y DOUBLE, x DOUBLE[] [, options MAP]) -> STRUCT[]
<model>_fit_predict_agg(y DOUBLE, x DOUBLE[], split VARCHAR [, options MAP]) -> STRUCT[]
```

Fits on the training rows (y not NULL; with `split`, only rows whose split is
`'train'`) and returns one STRUCT per input row, in input order:
`y`, `yhat`, `yhat_lower`, `yhat_upper`, `is_training`. The bounds are the
posterior predictive interval at `confidence_level`.

```sql
SELECT
    round(p.yhat, 3) AS yhat,
    round(p.yhat_lower, 3) AS lower,
    round(p.yhat_upper, 3) AS upper
FROM (
    SELECT unnest(ard_fit_predict_agg(y, [x1, x2, x3], {'confidence_level': 0.9})) AS p
    FROM bayes_demo
)
LIMIT 3;
```

## Table macros

```sql
-- Every row with yhat and its 95% posterior predictive band
SELECT grp, round(yhat, 3) AS yhat, round(yhat_lower, 3) AS lower, round(yhat_upper, 3) AS upper
FROM bayesian_ridge_fit_predict_by('bayes_demo', grp, y, [x1, x2, x3])
LIMIT 3;

-- Coefficients (integration contract `terms`; std_error etc. are NULL)
SELECT grp, term, round(estimate, 3) AS estimate
FROM ard_tidy_by('bayes_demo', grp, y, [x1, x2, x3], names := ['x1', 'x2', 'x3']);

-- Fit statistics (`summary`): n_observations, n_features, r_squared, iterations
SELECT grp, metric, round(value, 4) AS value FROM ard_glance_by('bayes_demo', grp, y, [x1, x2, x3]);

-- Fitted values with the band (`prediction`), x = the plot position
SELECT grp, x, round(yhat, 3) AS yhat, round(yhat_lower, 3) AS lower, round(yhat_upper, 3) AS upper, level
FROM bayesian_ridge_prediction_by('bayes_demo', grp, x1, y, [x1, x2, x3], level := 0.9)
LIMIT 3;
```

The `_fit_predict_by` macros take `options`, `split` and `order_by` as the
other [fit-predict table macros](../macros/table_macros.md).

## NULL handling

Rows where `y` or `x` is NULL, or `x` has a NULL element, are not used for
fitting; non-finite values are dropped too. Fewer than two complete rows give
NULL. In the fit-predict forms such rows are still returned: a row with a NULL
`y` and complete `x` is predicted, a row with an incomplete `x` gets NULL
predictions.

## See Also

- [Ridge](ridge.md) - fixed L2 penalty, with frequentist inference
- [LARS](lars.md) and [Elastic Net](elasticnet.md) - sparse fits with an L1 penalty
- [Regularisation paths](paths.md) - ridge / elastic-net paths and cross-validation
