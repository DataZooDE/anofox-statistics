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

## anofox_stats_eb_shrink_agg / eb_shrink_agg

**Signature:**

```sql
anofox_stats_eb_shrink_agg(
    estimate DOUBLE,
    se DOUBLE,
    [options MAP]
) -> STRUCT
```

**Options MAP:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| tau_squared | DOUBLE | — | Fix the between-group variance instead of estimating it |
| tau_method | VARCHAR | 'dl' | `dl` (DerSimonian-Laird) or `none` (complete pooling) |

**Returns:**

```
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
CREATE TABLE per_sku AS
SELECT sku,
       (poisson_fit_agg(qty, [promo], {'compute_inference': true})).coefficients[1] AS est,
       (poisson_fit_agg(qty, [promo], {'compute_inference': true})).std_errors[1]   AS se
FROM demand
GROUP BY sku;

SELECT * FROM eb_shrink_by('per_sku', est, se);
```

Because the inputs are estimates rather than data, this composes with **any**
per-group fit — not just GLMs.

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

## Degenerate inputs

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
