# Multinomial Logistic Regression

Multinomial (softmax) logistic regression for a class label with two or more
classes, with an optional ridge, lasso or elastic-net penalty. The objective is
scikit-learn's `LogisticRegression`:

```text
sum_i s_i * l_i(W, b) + (1 / C) * (l1_ratio * ||W||_1 + (1 - l1_ratio) / 2 * ||W||_2^2)
```

where `l_i` is the softmax cross-entropy of row `i`, `s_i` its sample weight
(1 by default), `W` the coefficient matrix and `b` the intercepts (not
penalized). The fit is upstream anofox-regression's
`MultinomialLogisticRegressor` (a proximal Newton method), so lasso and
elastic-net fits give exact zeros. The penalized fits match scikit-learn and
the unpenalized fit (`lambda` 0) matches R's `nnet::multinom`.

For a binary 0/1 response with GLM inference (standard errors, p-values), use
[`logistic_fit_agg`](logistic.md).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `multinomial_fit_agg` | Aggregate | Fit a multinomial logistic regression |
| `multinomial_predict` | Scalar | Most probable class or class probabilities of one row (also `predict(model, x)`) |
| `multinomial_fit_predict_by` | Table macro | Per-group fit, class and probabilities for every row |

## multinomial_fit_agg

**Signature:**

```text
multinomial_fit_agg(y VARCHAR | BIGINT, x DOUBLE[] [, weight DOUBLE] [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | VARCHAR or an integer type | Class label. The classes are the sorted distinct labels (at least two) |
| `x` | DOUBLE[] | Feature values for the row, same length on every row |
| `weight` | DOUBLE | Optional non-negative sample weight (scikit-learn's `sample_weight`) |
| `options` | MAP/STRUCT | Optional; must be a constant |

A row with a NULL label, a NULL `x`, a NULL or non-finite feature value or a
NULL weight is left out. A group with fewer than two rows or fewer than two
classes returns NULL. A fit that does not converge within `max_iterations`
(for example separable classes without a penalty) is returned with
`converged = false`. A `DOUBLE` label is rejected; cast it to `INTEGER` or
`VARCHAR`.

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `C` | DOUBLE | `1.0` | Inverse penalty strength, as scikit-learn (`'inf'::DOUBLE`: no penalty) |
| `lambda` | DOUBLE | `1.0` | Penalty strength `1 / C` (0: no penalty); give `C` or `lambda`, not both |
| `l1_ratio` | DOUBLE | `0.0` | Elastic-net mixing in `[0, 1]`: 0 ridge, 1 lasso |
| `fit_intercept` | BOOLEAN | `true` | Fit (unpenalized) intercepts |
| `max_iterations` | INTEGER | `100` | Maximum proximal Newton iterations |
| `tolerance` | DOUBLE | `1e-8` | Convergence tolerance on the largest parameter change |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `classes` | VARCHAR[] or BIGINT[] | The sorted distinct labels (scikit-learn `classes_`) |
| `coefficients` | DOUBLE[][] | One list of `n_features` coefficients per class (`coef_`). With two classes a single row: the log-odds of the second class against the first |
| `intercepts` | DOUBLE[] | One intercept per coefficient row (`intercept_`); with three or more classes they sum to 0 |
| `n_classes` | BIGINT | Number of classes |
| `accuracy` | DOUBLE | Share of training rows whose most probable class is their class |
| `log_likelihood` | DOUBLE | Weighted log-likelihood `sum_i s_i log p_i(y_i)` |
| `objective` | DOUBLE | The minimised objective: `-log_likelihood` plus the penalty |
| `lambda`, `l1_ratio` | DOUBLE | The penalty used |
| `n_observations` | BIGINT | Rows used in the fit |
| `n_features` | BIGINT | Number of features |
| `iterations` | INTEGER | Proximal Newton iterations performed |
| `converged` | BOOLEAN | Whether the fit reached the tolerance |
| `model_type`, `family`, `link` | VARCHAR | `'multinomial_logistic'`, `'multinomial'`, `'softmax'` (`'logit'` for two classes) |

With three or more classes the softmax coefficients are only defined up to a
common shift; without a penalty they are reported centred to sum to zero over
the classes, as scikit-learn does. `coefficients[k] - coefficients[1]` are the
baseline-category log-odds that R's `nnet::multinom` reports.

**Example:**

```sql
CREATE OR REPLACE TABLE flowers AS
SELECT i AS id, i % 2 AS site,
       round(4.5 + ((i * 37) % 101) / 40.0, 2) AS sepal_length,
       round(1.0 + ((i * 53) % 97) / 16.0, 2) AS petal_length,
       CASE WHEN ((i * 53) % 97) / 16.0 + sin(i) < 1.8 THEN 'setosa'
            WHEN ((i * 53) % 97) / 16.0 + sin(i) < 4.0 THEN 'versicolor'
            ELSE 'virginica' END AS species
FROM range(1, 151) t(i);

SELECT m.classes, m.n_classes, m.converged, round(m.accuracy, 3) AS accuracy
FROM (SELECT multinomial_fit_agg(species, [sepal_length, petal_length]) AS m FROM flowers);

-- Elastic net with scikit-learn's C, the per-class coefficient rows
SELECT unnest(m.classes) AS class,
       unnest(m.coefficients) AS coefficients,
       unnest(m.intercepts) AS intercept
FROM (SELECT multinomial_fit_agg(species, [sepal_length, petal_length],
                                 {'C': 0.5, 'l1_ratio': 0.5}) AS m FROM flowers);
```

## multinomial_predict

**Signature:**

```text
multinomial_predict(model STRUCT, x DOUBLE[] [, options]) -> VARCHAR | BIGINT | DOUBLE[]
predict(model STRUCT, x DOUBLE[] [, options])
```

Returns the most probable class of the row `x` (the type of the model's
`classes`; scikit-learn's `predict`), or with `{'type': 'prob'}` the class
probabilities as a `DOUBLE[]` in the order of `classes` (`predict_proba`).
[`predict(model, x)`](../regression/model_tools.md#predict) does the same for a
`multinomial_fit_agg` model. NULL when the model or `x` is NULL or `x` holds a
NULL.

```sql
WITH fit AS (
    SELECT multinomial_fit_agg(species, [sepal_length, petal_length]) AS m FROM flowers
)
SELECT predict(m, [5.5, 4.2]) AS predicted,
       multinomial_predict(m, [5.5, 4.2], {'type': 'prob'}) AS probabilities,
       m.classes
FROM fit;
```

## multinomial_fit_predict_by

```text
multinomial_fit_predict_by(source VARCHAR, group_col, y_col, x_cols,
    options := NULL, split := NULL) -> TABLE
```

Fits `multinomial_fit_agg` (options as there) on each group's training rows
(`y` not NULL and `split` omitted or `'train'` / `'training'` in any case) and
predicts every row. Returns all source columns plus `yhat` (the most probable
class), `probabilities` (`DOUBLE[]` in the order of the model's classes) and
`is_training`, ordered by the group column. A group whose fit is NULL (for
example a single class) gets NULL predictions.

```sql
SELECT site, species, yhat, is_training
FROM multinomial_fit_predict_by('flowers', site, species, [sepal_length, petal_length],
                                options := {'C': 1.0})
ORDER BY site, id
LIMIT 5;
```
