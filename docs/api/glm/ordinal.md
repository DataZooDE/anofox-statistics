# Ordinal Regression

Regression models for an ordered response: ratings, disease stages, pain
scores, or a continuous response treated semiparametrically. Two model
families are available:

- **Cumulative link models** (`ordinal_fit_agg`), as R's `rms::orm` and
  `MASS::polr`. For the ordered levels `y_1 < ... < y_k` (the distinct values
  of `y`):

  ```text
  P(Y >= y_{j+1} | x) = F(alpha_j + x'beta),   j = 1, ..., k - 1
  ```

  with decreasing intercepts `alpha_1 > ... > alpha_{k-1}` and one slope
  vector `beta` for all cut-points. The logistic link `F` is the proportional
  odds model; probit, log-log, complementary log-log and Cauchit links are
  also available. A continuous response gets one intercept per distinct value
  (`orm`'s semiparametric model).
- **Continuation-ratio model** (`continuation_ratio_fit_agg`), RMS section
  13.4 (`rms::cr.setup` + `lrm`):

  ```text
  logit P(Y = y_j | Y >= y_j, x) = theta_j + x'gamma   (+ x'gamma_j for cohort-specific columns)
  ```

Both fits are upstream anofox-regression (`OrdinalRegressor`,
`ContinuationRatioRegressor`): maximum likelihood by Newton-Raphson, and
logistic regression on the expanded data. They match `rms::orm`, `MASS::polr`
and `rms::cr.setup` + `lrm`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `ordinal_fit_agg` | Aggregate | Fit a cumulative link model |
| `continuation_ratio_fit_agg` | Aggregate | Fit a continuation-ratio model |
| `ordinal_predict` | Scalar | Most probable level, level probabilities, exceedance probabilities or mean of one row (also `predict(model, x)`) |
| `ordinal_fit_predict_by` | Table macro | Per-group cumulative link fit and predictions for every row |
| `continuation_ratio_fit_predict_by` | Table macro | Per-group continuation-ratio fit and predictions for every row |

## ordinal_fit_agg

**Signature:**

```text
ordinal_fit_agg(y DOUBLE, x DOUBLE[] [, weight DOUBLE] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | DOUBLE | Ordered response; its sorted distinct values are the levels (at least two) |
| `x` | DOUBLE[] | Predictor values for the row (no intercept column), same length on every row |
| `weight` | DOUBLE | Optional non-negative case (frequency) weight |
| `options` | MAP/STRUCT | Optional; must be a constant |

A row with a NULL `y`, a NULL `x`, a NULL or non-finite predictor value or a
NULL weight is left out. A group with fewer than two rows or fewer than two
levels returns NULL. A fit that does not converge within `max_iterations` is
returned with `converged = false`.

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `link` | VARCHAR | `'logistic'` | `logistic` (alias `logit`), `probit`, `loglog`, `cloglog`, `cauchit` (`rms::orm`'s `family`) |
| `penalty` | DOUBLE | `0` | Quadratic penalty `lambda * diag(var(x_j))` on the slopes (`orm`'s `penalty = lambda`) |
| `offset` | INTEGER | none | 1-based index into `x` of an offset column (coefficient fixed at 1, removed from the slopes) |
| `max_iterations` | INTEGER | `100` | Maximum Newton iterations |
| `tolerance` | DOUBLE | `1e-10` | Convergence tolerance on the largest score |
| `po_test` | BOOLEAN | `true` up to 50 levels | Compute the proportional-odds score test |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `levels` | DOUBLE[] | The sorted distinct response values `y_1 < ... < y_k` |
| `counts` | DOUBLE[] | Total weight of each level |
| `intercepts` | DOUBLE[] | `alpha_j` of `P(Y >= y_{j+1})`, `k - 1` values (decreasing). `MASS::polr`'s cut-points are `zeta_j = -alpha_j` |
| `intercept_std_errors` | DOUBLE[] | Standard errors of the intercepts (NULL above 1000 levels) |
| `coefficients` | DOUBLE[] | Slopes `beta` (same sign as `orm` and `polr`: a positive slope shifts the response upwards) |
| `std_errors`, `z_values`, `p_values` | DOUBLE[] | Wald inference for the slopes (two-sided normal p-values) |
| `log_likelihood` | DOUBLE | Log-likelihood at the estimate (unpenalized, also for a penalized fit) |
| `null_log_likelihood` | DOUBLE | Log-likelihood of the intercept-only model (with the offset, if any) |
| `penalized_log_likelihood` | DOUBLE | The maximised penalized objective (equals `log_likelihood` without a penalty) |
| `lr_chisq`, `lr_df`, `lr_p_value` | DOUBLE | Model likelihood-ratio chi-square against the intercept-only model, its degrees of freedom (the effective d.f. of the slopes for a penalized fit) and p-value |
| `aic` | DOUBLE | `-2 log_likelihood + 2 (k - 1 + lr_df)` |
| `po_test_statistic`, `po_test_df`, `po_test_p_value` | DOUBLE | Score test of proportional odds (parallel slopes) for all slopes against the partial proportional-odds model, RMS section 13.3; `(k - 2) * n_features` d.f. NULL when not computed, with two levels or without slopes |
| `n_levels`, `n_observations`, `n_features` | BIGINT | Levels, rows used, slopes |
| `offset_column` | BIGINT | The `offset` option (NULL without an offset) |
| `penalty` | DOUBLE | The penalty used |
| `iterations` | INTEGER | Newton iterations performed |
| `converged` | BOOLEAN | Whether the score reached the tolerance |
| `model_type`, `link` | VARCHAR | `'ordinal'` and the link |

**Example:**

```sql
CREATE OR REPLACE TABLE ratings AS
SELECT i AS id, i % 2 AS site,
       round(30 + ((i * 37) % 41), 1) AS age,
       ((i * 7) % 3)::DOUBLE AS dose,
       CASE WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 1.5 THEN 1
            WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 3.0 THEN 2
            WHEN ((i * 37) % 41) / 10.0 + ((i * 7) % 3) + sin(i * 1.7) * 1.5 < 4.5 THEN 3
            ELSE 4 END::DOUBLE AS rating
FROM range(1, 201) t(i);

-- Proportional odds model: slopes, Wald p-values and the test of proportional odds
SELECT m.levels, m.coefficients, m.p_values,
       round(m.po_test_statistic, 3) AS po_chisq, m.po_test_df, round(m.po_test_p_value, 4) AS po_p
FROM (SELECT ordinal_fit_agg(rating, [age, dose]) AS m FROM ratings);

-- Probit link; the intercepts are orm's alpha_j (polr's zeta_j = -alpha_j)
SELECT m.intercepts, m.coefficients, round(m.aic, 2) AS aic
FROM (SELECT ordinal_fit_agg(rating, [age, dose], {'link': 'probit'}) AS m FROM ratings);
```

## continuation_ratio_fit_agg

**Signature:**

```text
continuation_ratio_fit_agg(y DOUBLE, x DOUBLE[] [, weight DOUBLE] [, options MAP]) -> STRUCT
```

Parameters and NULL handling as `ordinal_fit_agg`; the response needs at least
three levels (a group with fewer returns NULL). The fit is the logistic
regression of `rms::cr.setup`'s expanded data: each row becomes one binary
record per cohort `j` it reaches (`Y >= y_j`), with response 1 in the cohort
where it stops.

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cohort_specific` | INTEGER or INTEGER[] | none | 1-based `x` columns whose slope varies by cohort (a cohort x column interaction) |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `levels` | DOUBLE[] | The sorted distinct response values |
| `coefficients` | DOUBLE[] | In the order of `lrm(y ~ cohort + x + cohort:x_j)` on `cr.setup` data: the intercept `theta_1`, the cohort effects `theta_j - theta_1` (`j = 2, ..., k - 1`), the slopes `gamma`, then for each cohort-specific column its slope differences in cohorts `2, ..., k - 1` |
| `std_errors`, `z_values`, `p_values` | DOUBLE[] | Wald inference, same order |
| `cohort_specific` | BIGINT[] | The cohort-specific columns (1-based) |
| `log_likelihood`, `aic` | DOUBLE | Log-likelihood (equal to that of the expanded data) and AIC |
| `n_levels`, `n_observations`, `n_records`, `n_features` | BIGINT | Levels, rows used, expanded records, predictors |
| `iterations`, `converged` | INTEGER, BOOLEAN | IRLS iterations of the logistic fit and convergence |
| `model_type`, `link` | VARCHAR | `'continuation_ratio'`, `'logit'` |

```sql
SELECT m.coefficients, round(m.log_likelihood, 3) AS loglik, m.n_records
FROM (SELECT continuation_ratio_fit_agg(rating, [age, dose]) AS m FROM ratings);

-- dose with a slope per cohort
SELECT len(m.coefficients) AS n_coefficients, m.cohort_specific
FROM (SELECT continuation_ratio_fit_agg(rating, [age, dose], {'cohort_specific': [2]}) AS m FROM ratings);
```

## ordinal_predict

**Signature:**

```text
ordinal_predict(model STRUCT, x DOUBLE[] [, options]) -> DOUBLE | DOUBLE[]
predict(model STRUCT, x DOUBLE[] [, options])
```

Predicts one row `x` (the same columns as the fit, including an offset column)
from an `ordinal_fit_agg` or `continuation_ratio_fit_agg` model.
[`predict(model, x)`](../regression/model_tools.md#predict) does the same.

| `type` | Returns | Description |
|--------|---------|-------------|
| `'class'` (default) | DOUBLE | The most probable level (`predict.polr(type = "class")`) |
| `'prob'` | DOUBLE[] | `P(Y = y_j \| x)` in the order of `levels` (`predict.orm(type = "fitted.ind")`) |
| `'exceedance'` | DOUBLE[] | `P(Y >= y_{j+1} \| x)` for `j = 1, ..., k - 1` (`predict.orm(type = "fitted")`) |
| `'mean'` | DOUBLE | The predicted mean `sum_j y_j P(Y = y_j \| x)` (`rms::Mean.orm`) |

An offset column is added to the linear predictor (as `predict.glm`;
`predict.orm` ignores an offset in new data). NULL when the model or `x` is
NULL or `x` holds a NULL. Predicted quantiles (`rms::Quantile.orm`) are not
available from a stored model.

```sql
WITH fit AS (SELECT ordinal_fit_agg(rating, [age, dose]) AS m FROM ratings)
SELECT predict(m, [50, 1]) AS most_probable,
       ordinal_predict(m, [50, 1], {'type': 'prob'}) AS probabilities,
       ordinal_predict(m, [50, 1], {'type': 'exceedance'}) AS exceedance,
       round(ordinal_predict(m, [50, 1], {'type': 'mean'}), 4) AS mean
FROM fit;
```

## ordinal_fit_predict_by and continuation_ratio_fit_predict_by

```text
ordinal_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
continuation_ratio_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fit `ordinal_fit_agg` / `continuation_ratio_fit_agg` (options as there) on each
group's training rows (`y` not NULL and `split` omitted or `'train'` /
`'training'` in any case) and predict every row. Return all source columns
plus `yhat` (the most probable level), `yhat_mean` (the predicted mean),
`probabilities` (`DOUBLE[]` in the order of the model's levels) and
`is_training`, ordered by the group column. A group whose fit is NULL gets
NULL predictions.

```sql
SELECT site, rating, yhat, round(yhat_mean, 3) AS yhat_mean, is_training
FROM ordinal_fit_predict_by('ratings', site, rating, [age, dose], options := {'link': 'logistic'})
ORDER BY site, id
LIMIT 5;

SELECT count(*) AS n, count(yhat) AS predicted
FROM continuation_ratio_fit_predict_by('ratings', site, rating, [age, dose],
                                       split := CASE WHEN id % 4 = 0 THEN 'test' ELSE 'train' END);
```
