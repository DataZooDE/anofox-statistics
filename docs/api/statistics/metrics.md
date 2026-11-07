# Performance Metrics

Standard performance metrics of labelled predictions, as evidently and NannyML
report them per chunk (#368, #369): classification metrics with the confusion
matrix, regression error metrics and ranking (recommender) metrics, plus a
data-quality summary of a table against a reference table. Each metric
function is an aggregate, so it works per group with `GROUP BY`, per chunk
over a window, and per group as a `_by` table macro.

The metrics come from upstream anofox-statistics 0.10 `metrics` and follow
scikit-learn (`sklearn.metrics`) and R yardstick; the ranking metrics other
than NDCG follow evidently. They are validated against scikit-learn
(`test/sql/reference/metrics_reference.test`).

## Functions

| Function | Type | Returns |
|----------|------|---------|
| `classification_metrics_agg(y_true, y_pred [, options])` | Aggregate | `STRUCT(n, accuracy, precision, recall, f1, specificity, balanced_accuracy, mcc, cohen_kappa, average, labels, confusion)` |
| `regression_metrics_agg(y_true, y_pred)` | Aggregate | `STRUCT(n, mae, mse, rmse, mape, smape, median_ae, r2, explained_variance, max_error)` |
| `ranking_metrics_agg(query_id, relevance, score [, options])` | Aggregate | `STRUCT(n_queries, k, ndcg, map, mrr, precision, recall, hit_rate)` |
| `classification_metrics_by`, `regression_metrics_by`, `ranking_metrics_by` | Table macro | one row per group, the fields above as columns |
| `data_quality_by(source, columns, reference := NULL)` | Table macro | one row per column |

Rows with a NULL argument are skipped; no rows left gives NULL.

## classification_metrics_agg

```text
classification_metrics_agg(y_true, y_pred [, options])
```

| Argument | Type | Description |
|----------|------|-------------|
| `y_true` | BIGINT, VARCHAR or BOOLEAN | The true label (any number of classes) |
| `y_pred` | the type of `y_true`, or DOUBLE | The predicted label, or a score with `options.threshold` |

| Option | Default | Description |
|--------|---------|-------------|
| `average` | `'macro'` | How the per-class precision, recall, F1 and specificity are combined: `'macro'` (unweighted mean over the labels), `'micro'` (from the summed confusion counts; precision, recall and F1 then equal the accuracy), `'weighted'` (mean weighted by the true count of each label) or `'binary'` (the positive class only) |
| `pos_label` | 1 / `true` | The positive class of `average = 'binary'`; required for VARCHAR labels |
| `zero_division` | 0 | Value of a ratio whose denominator is zero (precision without predicted positives, recall without true positives), before averaging; scikit-learn's `zero_division` |
| `threshold` | none | Required when `y_pred` is a DOUBLE score: the predicted label is 1 / `true` when `score >= threshold`, else 0 / `false`. `y_true` must then be BOOLEAN or 0/1 |

- **Labels.** `labels` lists the distinct labels of `y_true` and `y_pred`
  together, sorted; `confusion[i][j]` counts the rows with true label
  `labels[i]` predicted as `labels[j]` (scikit-learn's `confusion_matrix`
  layout).
- **Accuracy, balanced accuracy, MCC and Cohen's kappa** do not depend on
  `average`. Balanced accuracy is the mean recall over the labels that occur in
  `y_true`; MCC is the multiclass R_K statistic (0 when undefined); Cohen's
  kappa is unweighted and NULL when undefined.
- **Specificity** is `tn / (tn + fp)` per class (R `yardstick::spec`), averaged
  as the other per-class metrics; micro specificity is `sum tn / sum (tn + fp)`.
- `average = 'binary'` with more than two labels, or with two labels of which
  neither is `pos_label`, is an error (as scikit-learn). A NaN score gives NULL.

```sql
CREATE TABLE scored AS SELECT * FROM (VALUES
  ('a', 1, 0.9), ('a', 0, 0.1), ('a', 1, 0.8), ('a', 1, 0.35), ('a', 0, 0.6),
  ('b', 1, 0.7), ('b', 0, 0.2), ('b', 0, 0.55), ('b', 1, 0.45), ('b', 0, 0.3)
) t(model, label, score);

-- Binary metrics of the positive class from a score.
SELECT model, m.precision, m.recall, m.f1, m.confusion
FROM (SELECT model, classification_metrics_agg(label, score, {'threshold': 0.5, 'average': 'binary'}) AS m
      FROM scored GROUP BY model)
ORDER BY model;

-- Multiclass labels, macro and weighted averages.
CREATE TABLE pets AS SELECT * FROM (VALUES
  ('cat', 'cat'), ('dog', 'dog'), ('fish', 'dog'), ('cat', 'cat'), ('dog', 'cat'), ('fish', 'fish')
) t(y, p);

SELECT classification_metrics_agg(y, p) AS macro,
       (classification_metrics_agg(y, p, {'average': 'weighted'})).f1 AS weighted_f1
FROM pets;
```

## regression_metrics_agg

```text
regression_metrics_agg(y_true DOUBLE, y_pred DOUBLE)
```

| Field | Definition |
|-------|------------|
| `mae`, `mse`, `rmse` | Mean absolute error, mean squared error and its root |
| `mape` | `mean(abs(y - yhat) / max(abs(y), eps))` as a fraction (not percent), `eps` the machine epsilon, as scikit-learn: a zero `y` gives a huge but finite term |
| `smape` | `mean(abs(y - yhat) / ((abs(y) + abs(yhat)) / 2))` as a fraction (`yardstick::smape / 100`); a row with `y = yhat = 0` counts 0 |
| `median_ae`, `max_error` | Median and largest absolute error |
| `r2` | `1 - SS_res / SS_tot`; for a constant `y_true` 1 if the predictions are perfect, else 0 (scikit-learn `force_finite`); NULL for a single row |
| `explained_variance` | `1 - var(y - yhat) / var(y)`, constant `y_true` as for `r2` |

A NaN or infinite value gives NULL.

```sql
CREATE TABLE forecasts AS SELECT * FROM (VALUES
  ('s1', 3.0, 2.5), ('s1', -0.5, 0.0), ('s1', 2.0, 2.0), ('s1', 7.0, 8.0),
  ('s2', 10.0, 11.0), ('s2', 12.0, 12.5), ('s2', 9.0, 8.0)
) t(series, actual, forecast);

SELECT series, m.mae, m.rmse, m.smape, m.r2
FROM (SELECT series, regression_metrics_agg(actual, forecast) AS m FROM forecasts GROUP BY series)
ORDER BY series;
```

## ranking_metrics_agg

```text
ranking_metrics_agg(query_id, relevance DOUBLE, score DOUBLE [, options])
```

One row per (query, item): `query_id` (BIGINT or VARCHAR) groups the items of
a query (a user, a search), `relevance` is the true relevance (0 = not
relevant, graded values allowed) and `score` the model's score; the items of a
query are ranked by decreasing score. Each metric is computed per query at the
cut-off `k` and averaged over the queries.

| Option | Default | Description |
|--------|---------|-------------|
| `k` | 10 | The cut-off, at least 1 |
| `min_relevance` | none | An item is relevant when `relevance >= min_relevance`, and every metric, NDCG included, uses the binary relevance (evidently's `min_rel_score`). Without it an item is relevant when `relevance > 0` and NDCG uses the graded relevance as gain |
| `include_no_feedback` | `false` | Average over all queries (`true`, scikit-learn's `ndcg_score`), or only over the queries with at least one relevant item (evidently's default). A query without relevant items scores 0 |

| Field | Definition |
|-------|------------|
| `ndcg` | NDCG@k as scikit-learn `ndcg_score(k = k)`: gain = relevance, discount `1 / log2(rank + 1)`, tied scores share the mean gain of their group |
| `map` | MAP@k (evidently): `sum over i <= min(k, m) of precision@i * rel_i / R`, `R` the relevant items of the query, `m` its item count |
| `mrr` | Reciprocal rank of the first relevant item within the top k, 0 if none |
| `precision`, `recall` | Relevant items in the top k over `min(k, m)` / over `R` |
| `hit_rate` | Share of queries with a relevant item in the top k |
| `n_queries` | Queries averaged over |

Apart from NDCG, tied scores are ranked with the less relevant item first, so
the result does not depend on the row order. A table without any relevant
item gives NULL; negative relevance without `min_relevance` is an error.

```sql
CREATE TABLE recs AS SELECT * FROM (VALUES
  (1, 3, 0.9), (1, 0, 0.8), (1, 1, 0.5), (1, 0, 0.2),
  (2, 0, 0.7), (2, 0, 0.6), (2, 2, 0.3),
  (3, 1, 0.9), (3, 0, 0.1)
) t(user_id, relevance, score);

SELECT ranking_metrics_agg(user_id, relevance, score, {'k': 2}) AS at_2 FROM recs;
```

## Table macros

```text
classification_metrics_by(source, group_col, y_true_col, y_pred_col, options := NULL)
regression_metrics_by(source, group_col, y_true_col, y_pred_col)
ranking_metrics_by(source, group_col, query_col, relevance_col, score_col, options := NULL)
```

Each runs the aggregate per group of `source` and returns one row per group:
the group column followed by the fields of the aggregate's result.

```sql
SELECT * FROM classification_metrics_by('scored', model, label, score,
                                        options := {'threshold': 0.5, 'average': 'binary'});

SELECT series, mae, rmse FROM regression_metrics_by('forecasts', series, actual, forecast);
```

## data_quality_by

```text
data_quality_by(source, columns, reference := NULL)
```

A data-quality summary of the columns of `source` named in `columns` (a list
of column names), one row per column in that order. With `reference :=
'<table>'` (a table or view with the same columns, e.g. the training data) the
current data is also compared with the reference. Numeric columns (integers,
floats, decimals) are profiled as numbers; every other type as categories (its
text).

| Column | Description |
|--------|-------------|
| `column_name`, `column_type` | The column and its type in `source` |
| `n`, `n_missing`, `missing_share` | Rows, NULL values and their share |
| `n_distinct` | Distinct non-NULL values |
| `min`, `max`, `mean`, `sd` | Numeric columns: basic statistics (`sd` the sample standard deviation); NULL for categories |
| `reference_min`, `reference_max` | Numeric columns: the range of the reference |
| `out_of_range_share` | Numeric columns: share of the non-missing values outside `[reference_min, reference_max]` |
| `new_categories`, `dropped_categories` | Categorical columns: sorted categories seen only in `source` / only in the reference |

The comparison columns are NULL without a reference.

```sql
CREATE TABLE train AS SELECT * FROM (VALUES
  (25, 'berlin'), (31, 'munich'), (47, 'berlin'), (52, 'hamburg'), (38, NULL)
) t(age, city);
CREATE TABLE today AS SELECT * FROM (VALUES
  (29, 'berlin'), (61, 'cologne'), (NULL, 'munich'), (19, 'berlin')
) t(age, city);

SELECT column_name, missing_share, n_distinct, out_of_range_share, new_categories, dropped_categories
FROM data_quality_by('today', ['age', 'city'], reference := 'train');
```
