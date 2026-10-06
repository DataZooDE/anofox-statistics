# Empirical-Bayes Shrinkage

Shrink per-group estimates toward their common mean by an amount the data itself
determines. Partial pooling without fitting a hierarchical model.

Independent per-group fits are unusable when groups are sparse: a SKU with three
observations gets a wild coefficient. A fully pooled fit erases the differences
that matter. Shrinkage sits between the two — precisely measured groups keep
their estimate, noisy ones are pulled toward the mean.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `eb_shrink_agg` | Aggregate | Shrink a set of estimates toward their pooled mean |
| `eb_shrink_by` | Table Macro | Same, returning one row per input |

## eb_shrink_agg

**Signature:**

```text
eb_shrink_agg(estimate DOUBLE, se DOUBLE [, options MAP]) -> STRUCT
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimate` | DOUBLE | One per-group estimate per row |
| `se` | DOUBLE | Its standard error (must be positive to contribute) |
| `options` | MAP/STRUCT | Optional; must be a constant |

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| tau_squared (alias `tau2`) | DOUBLE | — | Fix the between-group variance instead of estimating it |
| tau_method (alias `shrinkage`) | VARCHAR | 'dl' | `dl` (DerSimonian-Laird) or `none` (complete pooling) |

**Returns:**

```text
STRUCT(mu DOUBLE, mu_se DOUBLE, tau_squared DOUBLE, i_squared DOUBLE,
       q DOUBLE, n_groups BIGINT,
       shrunken LIST(STRUCT(estimate DOUBLE, se DOUBLE, shrunken DOUBLE,
                            shrunken_se DOUBLE, weight DOUBLE)))
```

The `shrunken` list is in **input order**, matching the convention the
`*_fit_predict_agg` functions use, so it can be `UNNEST`ed or indexed by
`ROW_NUMBER()`.

## The intended workflow

Fit per group first, then shrink the group estimates:

```sql
-- 12 SKUs, 30 weeks each, promotion in alternating 12-week blocks
CREATE OR REPLACE TABLE demand AS
SELECT i AS week,
       'SKU-' || (i % 12) AS sku,
       ((i // 12) % 2)::DOUBLE AS promo,
       (((i * 7) % 11) + (1 + (i % 12) % 4) * ((i // 12) % 2) + (i % 12) % 4)::DOUBLE AS qty
FROM range(360) r(i);

-- 1. Fit per group
CREATE OR REPLACE TABLE per_sku AS
SELECT sku, (fit).coefficients[1] AS est, (fit).std_errors[1] AS se
FROM (
    SELECT sku, poisson_fit_agg(qty, [promo], {'compute_inference': true}) AS fit
    FROM demand
    GROUP BY sku
);

-- 2. Shrink the group estimates
SELECT r.mu, r.tau_squared, r.i_squared, r.q, r.n_groups
FROM (SELECT eb_shrink_agg(est, se) AS r FROM per_sku);

-- With a fixed between-group variance
SELECT (eb_shrink_agg(est, se, {'tau_squared': 0.01})).mu FROM per_sku;
```

Because the inputs are estimates rather than data, this composes with **any**
per-group fit — not just GLMs.

## eb_shrink_by

Table macro returning every source row with its shrunken estimate.

**Signature:**

```text
eb_shrink_by(source VARCHAR, estimate_col, se_col [, options MAP]) -> TABLE
```

**Returns:** all source columns plus

| Column | Type | Description |
|--------|------|-------------|
| `shrunken` | DOUBLE | Posterior mean for the row |
| `shrunken_se` | DOUBLE | Posterior standard deviation |
| `weight` | DOUBLE | Share of its own estimate the row keeps |
| `mu` | DOUBLE | Pooled mean (same on every row) |
| `tau_squared` | DOUBLE | Between-group variance (same on every row) |

```sql
SELECT sku, round(est, 3) AS est, round(shrunken, 3) AS shrunken, round(weight, 3) AS weight
FROM eb_shrink_by('per_sku', est, se)
ORDER BY sku;
```

## The model

```
theta_g ~ N(mu, tau^2)          between-group variation
est_g   ~ N(theta_g, se_g^2)    within-group sampling error
```

`tau^2` is the DerSimonian-Laird moment estimator, so the numbers line up with
`metafor::rma(yi, sei, method = "DL")`. Each group's posterior mean is the
precision-weighted blend:

```
shrunken_g = w_g * est_g + (1 - w_g) * mu ,   w_g = (1/se_g^2) / (1/se_g^2 + 1/tau^2)
```

`weight` is that `w_g`: the share of its own estimate the group keeps. 1 means
untouched, 0 means fully pooled.

## Reading the output

| Field | Meaning |
|-------|---------|
| `mu` | Precision-weighted pooled mean |
| `tau_squared` | Estimated between-group variance. Zero means the groups are indistinguishable and everything collapses onto `mu`. |
| `i_squared` | Share of total variance that is between-group. High means the groups really do differ. |
| `q` | Cochran's Q heterogeneity statistic |
| `shrunken_se` | Posterior standard deviation, always at most the input `se` |

## Degenerate inputs and NULL handling

Fewer than two usable groups returns `NULL` — with one group there is nothing to
shrink toward.

Rows with a non-finite estimate, or a non-positive standard error, are excluded
from `mu` and `tau^2` but still appear in `shrunken` as `NaN`, so the list stays
aligned with the input.

## Use Cases

- **Thousands of SKUs, few observations each** — the case the DuckDB ecosystem
  has no answer for.
- **Regional or store-level effects** — small regions borrow strength from large ones.
- **A/B tests across many segments** — stops small segments producing spurious winners.
- **Any per-group estimate with a standard error** — the input is deliberately generic.

## See Also

- [Mixed-effects GLMs](glmm.md) — the fully specified version, fitting one model
  jointly instead of shrinking after the fact
- [Explicit priors](priors.md) — shrinkage toward a value you choose in advance
