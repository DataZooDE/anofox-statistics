# Causal Mediation Analysis

How much of the effect of a treatment `X` on an outcome `Y` runs through a
mediator `M`? `mediation_by` and `mediation_agg` answer this per group with the
linear models of R's `mediation::mediate`.

Two OLS models are fitted on the same rows of each group:

- the **mediator model** `M ~ X + C`, giving `a`, the effect of `X` on `M`;
- the **outcome model** `Y ~ X + M + C`, giving `b`, the effect of `M` on `Y`, and
  `c'`, the direct effect of `X` on `Y`.

`C` are optional covariates. For a change of the treatment from `control_value`
to `treat_value` (`Δ = treat_value − control_value`, by default 0 → 1) the
effects are:

| `term` | Value | Meaning |
|--------|-------|---------|
| `acme` | `a · b · Δ` | average causal mediation effect (the indirect effect) |
| `ade` | `c' · Δ` | average direct effect |
| `total` | `(a · b + c') · Δ` | total effect; equals `c · Δ` of `Y ~ X + C` |
| `prop_mediated` | `acme / total` | proportion mediated; NULL when the total effect is 0, unstable near 0 |

The scope is the linear, no-interaction case: a continuous mediator and
outcome, a single binary or continuous treatment, optional covariates. Rows
with a NULL in the treatment, mediator, outcome or any covariate are dropped
from **both** models, so they use the same rows. The usual causal reading needs
the sequential-ignorability assumptions of Imai, Keele and Tingley (2010): no
unmeasured confounding of treatment–mediator, treatment–outcome and
mediator–outcome, given the covariates.

## Inference

| `method` | Standard error | Interval | p-value |
|----------|----------------|----------|---------|
| `'bootstrap'` (default) | SD of the bootstrap draws | percentile (`ci_type 'percentile'`, R `quantile` type 7) or `mediate`'s BCa (`ci_type 'bca'`) | `2 · min(#draws > 0, #draws < 0) / #draws`, as `mediate` |
| `'sobel'` | delta method: `sqrt(a² se_b² + b² se_a²)` for `acme` (Sobel), OLS SE of `c'` for `ade` and of `c` in `Y ~ X + C` for `total`, delta method for `prop_mediated` | `estimate ± z·SE` (`acme`, `prop_mediated`) or `± t·SE` (`ade`, `total`, residual df of the OLS fit) | normal (`acme`, `prop_mediated`) or t |
| `'quasi_bayesian'` | SD of the Monte Carlo draws | percentile | as for the bootstrap |

The **bootstrap** resamples whole rows (case resampling) `n_bootstrap` times
(default 1000) and refits both models on each resample; this is the
recommended inference for `a · b`, whose sampling distribution is not normal
(Preacher and Hayes). The point estimates are those of the original data. The
draws are seeded (`seed`, default 42) and the rows of a group are put in a
canonical order first, so a result is reproducible across calls and thread
counts; pass another `seed` for other draws.

The **quasi-Bayesian** Monte Carlo is `mediate`'s default: the coefficients are
drawn from their asymptotic normal distributions and, as in `mediate`, the
point estimates are the means of the draws (the median for `prop_mediated`).

`statistic` is the z or t statistic for `'sobel'` and NULL for the simulation
methods. The draws come from this extension's generator, not R's, so simulated
intervals match `mediate`'s within simulation error, not to the digit; the
point estimates match exactly.

## mediation_by

```sql skip
mediation_by(source VARCHAR, group_col, treatment, mediator, outcome
             [, covariates := [c1, c2, ...]] [, options := {...}]) -> TABLE
```

All three model columns can be given by name (`treatment := treat`). Returns one
row per group and effect, in the integration contract's `terms` schema plus
three columns:

| Column | Type | Meaning |
|--------|------|---------|
| *group_col* | as in `source` | the group |
| `id` | VARCHAR | the group as text |
| `model_id`, `model_name` | VARCHAR | `'mediation'` |
| `term` | VARCHAR | `acme`, `ade`, `total`, `prop_mediated` (in this order) |
| `estimate`, `std_error`, `statistic`, `p_value` | DOUBLE | see [Inference](#inference) |
| `conf_low`, `conf_high`, `conf_level` | DOUBLE | the interval and its level |
| `index_name`, `index_value` | VARCHAR, DOUBLE | NULL |
| `method` | VARCHAR | `bootstrap_percentile`, `bootstrap_bca`, `sobel` or `quasi_bayesian` |
| `n_obs` | BIGINT | complete rows used by both models |
| `n_sims` | BIGINT | bootstrap resamples or Monte Carlo draws that entered the summaries (0 for Sobel) |

A group with too few complete rows (fewer than the number of covariates plus
4) or a treatment or mediator that is constant or collinear with the covariates
yields no rows.

**Options:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `method` | VARCHAR | `'bootstrap'` | `'bootstrap'`, `'sobel'` or `'quasi_bayesian'` |
| `ci_type` (alias `boot_ci_type`) | VARCHAR | `'percentile'` | `'percentile'` or `'bca'` (bootstrap only) |
| `n_bootstrap` (aliases `n_sims`, `sims`) | INTEGER | 1000 | resamples / Monte Carlo draws, at least 2 |
| `seed` (alias `random_state`) | INTEGER | 42 | seed of the draws |
| `confidence_level` (alias `conf_level`) | DOUBLE | 0.95 | level of the intervals |
| `treat_value` | DOUBLE | 1 | treatment value of the contrast |
| `control_value` | DOUBLE | 0 | control value of the contrast |

```sql
CREATE OR REPLACE TABLE trial AS
SELECT i, 'site_' || (i % 2) AS site,
       ((i // 2) % 2)::DOUBLE AS treat,
       20 + (i * 7) % 40 AS age,
       sin(i * 0.7) AS baseline,
       0.5 + 0.8 * ((i // 2) % 2) + 0.01 * ((i * 7) % 40) + sin(i * 2.1 + 0.4) * 0.9 AS m
FROM range(200) t(i);

CREATE OR REPLACE TABLE trial AS
SELECT *, 1 + 0.4 * treat + 0.6 * m + 0.2 * baseline + cos(i * 1.7 + 0.2) * 0.8 AS y FROM trial;

SELECT site, term, round(estimate, 3) AS estimate, round(conf_low, 3) AS conf_low,
       round(conf_high, 3) AS conf_high, p_value
FROM mediation_by('trial', site, treatment := treat, mediator := m, outcome := y,
                  covariates := [age, baseline], options := {'n_bootstrap': 1000, 'seed': 42});
```

Sobel / delta-method inference, a continuous treatment contrast and a 90% level:

```sql
SELECT site, term, round(estimate, 3) AS estimate, round(std_error, 3) AS std_error, round(p_value, 4) AS p_value
FROM mediation_by('trial', site, treat, m, y, covariates := [age, baseline],
                  options := {'method': 'sobel', 'treat_value': 1, 'control_value': 0,
                              'confidence_level': 0.9})
WHERE term IN ('acme', 'ade');
```

## mediation_agg

```sql skip
mediation_agg(y DOUBLE, m DOUBLE, x DOUBLE [, covariates DOUBLE[] [, options MAP]]) -> LIST(STRUCT)
```

The aggregate behind `mediation_by`: outcome `y`, mediator `m`, treatment `x`.
It returns a list of four rows (`acme`, `ade`, `total`, `prop_mediated`), each
`STRUCT(term, estimate, std_error, statistic, p_value, conf_low, conf_high,
conf_level, method, n_obs, n_sims)`, or NULL for a degenerate group. Use it
with `GROUP BY` and `UNNEST`, or to pick out one effect:

```sql
SELECT site, (r[1]).estimate AS acme, (r[1]).conf_low AS acme_low, (r[1]).conf_high AS acme_high
FROM (
    SELECT site, mediation_agg(y, m, treat, [age, baseline], {'n_bootstrap': 500}) AS r
    FROM trial GROUP BY site
)
ORDER BY site;
```

## Validation

The upstream implementation (`anofox_regression::solvers::mediation`) is
validated against R: the point estimates of `mediate` (`boot = TRUE`, percentile
and BCa, a continuous treatment), `lavaan` (`a*b`) and `lm` match to 1e-10;
`mediate`'s intervals and p-values are reproduced exactly from its own bootstrap
draws and within simulation error from the extension's draws; the Sobel z and p
match `bda::mediation.test`. The SQL tests (`test/sql/mediation/`) check the
point estimates and Sobel standard errors against R on the same rows.

## Not yet supported

Binary outcome or mediator (logit/probit models), a treatment × mediator
interaction (separate ACME(0)/ACME(1)), multiple mediators and the sensitivity
analysis of `medsens`.
