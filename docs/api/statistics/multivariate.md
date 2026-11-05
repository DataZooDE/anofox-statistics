# Multivariate: Principal Component Analysis

Principal component analysis (PCA) of a column of feature vectors, as R's
`prcomp(x, center = TRUE, scale. = FALSE)`. The components are the
eigenvectors of the sample covariance matrix (divisor `n - 1`), ordered by
decreasing variance. Fit a model per group with `pca_agg`, score rows with
`pca_transform` (R's `predict.prcomp`), or get every row's scores per group
with the `pca_by` table macro.

The computation is upstream anofox-statistics `multivariate::pca` and
`Pca::transform`, validated against `prcomp`
(`test/sql/reference/pca_reference.test`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `pca_agg` | Aggregate | Fit a PCA per group: component standard deviations, proportions of variance, loadings, center |
| `pca_transform` | Scalar | Scores of one feature vector on the components of a `pca_agg` model |
| `pca_by` | Table macro | Every source row with its scores on its group's components |

**Sign convention.** The sign of a principal component is arbitrary; R's
`prcomp` takes whatever LAPACK returns. Here each component's sign is fixed so
that its loading of largest absolute value is positive (scikit-learn's
`svd_flip` on the components). Results are therefore deterministic, and equal
to `prcomp`'s up to the sign of each component (and of its scores).

**Number of components.** With `n` rows and `d` features there are
`min(n - 1, d)` components. `prcomp` returns `min(n, d)`: when `n <= d` its
last component has a standard deviation of 0 and is not returned here.

**Scaling.** The data are centred, not scaled (`scale. = FALSE`). The
`scale` option of `prcomp(scale. = TRUE)` is not available yet (the upstream
`pca` has no scaling); standardise the columns yourself if the features are on
different scales, and apply the same standardisation before `pca_transform`.

## pca_agg

**Signature:**

```text
pca_agg(x DOUBLE[] [, options MAP]) -> STRUCT
```

Options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `n_components` (alias `components`) | INTEGER | all | Keep the first `n_components` components (at least 1). More than the data support keeps all of them, like `prcomp`'s `rank.` |

**Returns:**

| Field | Type | Description |
|-------|------|-------------|
| `sdev` | `DOUBLE[]` | Standard deviations of the components (divisor `n - 1`), decreasing (`prcomp()$sdev`) |
| `proportion_of_variance` | `DOUBLE[]` | Share of the total variance per component (`summary(prcomp())$importance[2, ]`); with `n_components` they remain shares of the total and need not sum to 1 |
| `loadings` | `DOUBLE[][]` | One list per component with one loading per feature: `loadings[j]` is `prcomp()$rotation[, j]` (unit length) |
| `center` | `DOUBLE[]` | Column means of the fitted rows |
| `n_components` | `BIGINT` | Number of components returned |
| `n_features` | `BIGINT` | Number of features |
| `n_observations` | `BIGINT` | Rows used in the fit |

Rows whose `x` is NULL or holds a NULL, NaN or infinite element are skipped.
All rows must have the same number of features. A group with fewer than two
usable rows returns NULL. The result does not depend on the row order.

**Example:**

```sql
CREATE OR REPLACE TABLE pca_demo AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS region,
    i AS id,
    i * 0.5 + (i % 3) AS x1,
    i * 0.3 - (i % 4) * 0.2 AS x2,
    (i % 5)::DOUBLE AS x3
FROM range(1, 41) t(i);

SELECT
    region,
    m.n_components,
    list_transform(m.sdev, lambda s: round(s, 4)) AS sdev,
    list_transform(m.proportion_of_variance, lambda p: round(p, 4)) AS proportion,
    list_transform(m.loadings[1], lambda l: round(l, 4)) AS pc1_loadings
FROM (SELECT region, pca_agg([x1, x2, x3]) AS m FROM pca_demo GROUP BY region)
ORDER BY region;
```

Keep two components:

```sql
SELECT pca_agg([x1, x2, x3], {'n_components': 2}).n_components AS k FROM pca_demo;
```

## pca_transform

**Signature:**

```text
pca_transform(model STRUCT, x DOUBLE[]) -> DOUBLE[]
```

Scores of the feature vector `x` on every component of `model` (the STRUCT
returned by `pca_agg`): `x` minus the model's `center`, times each component's
loadings, as R's `predict(prcomp_fit, newdata)`. `x` must have `n_features`
values. The result is NULL when the model or `x` is NULL, or `x` holds a NULL.
Any STRUCT with the fields `center`, `sdev` and `loadings` is accepted.

**Example:**

```sql
WITH m AS (SELECT pca_agg([x1, x2, x3], {'n_components': 2}) AS m FROM pca_demo)
SELECT list_transform(pca_transform(m, [10.0, 6.0, 2.0]), lambda s: round(s, 4)) AS scores
FROM m;
```

## pca_by

**Signature:**

```text
pca_by(source VARCHAR, group_col, x_cols [, options := MAP]) -> TABLE
```

Fits `pca_agg(x_cols, options)` per group of `source` and returns every source
row, in source order within each group, with `scores DOUBLE[]` appended: the
row's scores on its group's components (`pca_transform`). Rows with a NULL
element get NULL scores (they are not used in the fit), and so do the rows of
a group with fewer than two usable rows.

**Example:**

```sql
SELECT region, id, list_transform(scores, lambda s: round(s, 3)) AS scores
FROM pca_by('pca_demo', region, [x1, x2, x3], options := {'n_components': 2})
WHERE id <= 4
ORDER BY region, id;
```
