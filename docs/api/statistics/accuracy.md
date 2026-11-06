# Predictive Accuracy Indexes

Discrimination, calibration and explained-variation indexes of a prediction
(Harrell, *Regression Modeling Strategies*, §4.5 and §10.8), as in R `rms` and
`Hmisc` (#308). Each aggregate takes the outcome and the prediction of every
row, so it works for any model (fitted in this extension or elsewhere), per group
with `GROUP BY`, and over windows.

The statistics come from `anofox_regression::diagnostics::accuracy`; they are
validated against `rms::lrm`, `rms::val.prob`, `Hmisc::somers2`,
`Hmisc::rcorr.cens`, `Hmisc::GiniMd`, `Hmisc::R2Measures` and
`survival::concordance` (`test/sql/reference/accuracy_reference.test`).

## Functions

| Function | Type | Returns | R |
|----------|------|---------|---|
| `concordance_agg(y, x [, weight])` | Aggregate | `STRUCT(concordant, discordant, tied_x, tied_y, tied_xy, c, dxy, gamma, tau_a)` | `survival::concordance(y ~ x)` |
| `somers2_agg(y_true, prob [, weight])` | Aggregate | `STRUCT(c, dxy, n)` | `Hmisc::somers2(prob, y)` |
| `rcorr_cens_agg(time, event, x)` | Aggregate | `STRUCT(c, dxy, sd, n, missing, uncensored, relevant_pairs, concordant, uncertain)` | `Hmisc::rcorr.cens(x, Surv(time, event))` |
| `lrm_stats_agg(y_true, lp, df [, weight])` | Aggregate | `STRUCT(n, sum_weights, model_lr, df, p_value, c, dxy, gamma, tau_a, r2, r2_n, r2_adj_n, ess, r2_ess, r2_adj_ess, brier, g, gr, gp)` | `lrm(...)$stats` |
| `val_prob_agg(y_true, prob)` | Aggregate | `STRUCT(dxy, c, r2, d, d_chisq, d_p, u, u_chisq, u_p, q, brier, intercept, slope, emax, e90, eavg, s_z, s_p)` | `rms::val.prob(prob, y, pl = FALSE)` |
| `gini_md_agg(x)` | Aggregate | `DOUBLE` | `Hmisc::GiniMd(x)` |
| `r2_measures(lr, p, n [, ess])` | Scalar | `STRUCT(r2, r2_adj, r2_ess, r2_adj_ess)` | `Hmisc::R2Measures(lr, p, n, ess)` |

The ROC area alone is also `auc_agg`, the Brier score alone `brier_agg`
([Classification](classification.md)).

## Inputs

| Argument | Type | Description |
|----------|------|-------------|
| `y_true` | BOOLEAN, or numeric 0/1 | The binary outcome; any other value is an error |
| `y` | DOUBLE | An ordered (or continuous) outcome |
| `x`, `prob`, `lp` | DOUBLE | The prediction: any score where larger means a larger outcome (`x`), a predicted probability (`prob`), or a logistic linear predictor (`lp`) |
| `weight` | DOUBLE | Case weights (finite, >= 0; a negative weight is an error) |
| `time`, `event` | DOUBLE; BOOLEAN or numeric | Right-censored survival time and its event indicator (`true` / non-zero = observed) |
| `df` | DOUBLE constant | Model degrees of freedom (number of non-intercept coefficients) |

## Semantics

- **NULLs.** A row with a NULL in any argument is skipped; no rows left gives
  NULL. A NaN prediction is dropped by the rank indexes (`c`, `dxy`, ...) and by
  `val_prob_agg`; `gini_md_agg` is NULL then. An index that is undefined (e.g.
  `c` when only one class is present) is NULL.
- **Concordance.** Pairs with different `y` are compared; a pair is concordant
  when the larger `y` has the larger `x`. `c = (concordant + tied_x / 2) /
  (concordant + discordant + tied_x)`, `dxy = 2 (c - 1/2)`, `gamma = (concordant
  - discordant) / (concordant + discordant)` and `tau_a = (concordant -
  discordant) / all pairs`. With weights a pair counts `w_i w_j`. For a binary
  `y` `c` is the ROC area.
- **`somers2_agg`** is the ROC area `c` and `dxy` for a 0/1 outcome with `n` (the
  sum of the weights when weighted), as `Hmisc::somers2` with case weights.
- **`rcorr_cens_agg`** is Harrell's `C` for right-censored data: a larger `x`
  predicts a longer survival time (pass the negated risk score or linear
  predictor of a hazard model). `sd` is the standard deviation of `dxy`; the pair
  counts are over ordered pairs, as `rcorr.cens` prints them.
- **`lrm_stats_agg`** reproduces the `stats` vector of a binary `rms::lrm` fit
  from its linear predictor at the maximum likelihood estimate (with intercept):
  the likelihood-ratio χ² `model_lr` on `df` d.f. (`p_value` is its upper tail),
  the rank indexes of `lp`, the Nagelkerke `r2`, `r2_n` / `r2_adj_n` = R2(n) /
  R2(p,n) of `Hmisc::R2Measures` with `n` the sum of weights, `r2_ess` /
  `r2_adj_ess` = R2(m) / R2(p,m) with the effective sample size `ess = 3 n1 n0 /
  n`, the Brier score, the g-index `g` (Gini mean difference of `lp`), `gr =
  exp(g)` and `gp` (Gini mean difference of the probabilities). `rms` prints `P`
  as `1 - pchisq()`, which loses digits below about 1e-12; `p_value` keeps them.
- **`val_prob_agg`** validates predicted probabilities against outcomes, as
  `rms::val.prob(pl = FALSE)`: the discrimination index `d = (χ²_D - 1) / n`, the
  unreliability index `u = (χ²_U - 2) / n`, `q = d - u`, the logistic calibration
  `logit P(y = 1) = intercept + slope · logit(prob)`, `emax` / `e90` / `eavg` of
  `|prob - lowess calibration curve|` and Spiegelhalter's `s_z` / `s_p`.
  Probabilities of exactly 0 or 1 are left out of the logistic calibration (as
  R) and kept in the deviance of `U` as R does: a probability of 0 for an event
  gives `u = inf`, `u_chisq = inf`, `u_p = 0` and `q = -inf`, while a
  probability of 1 for an event and 0 or 1 for a non-event drop out.
  Constant probabilities give `c = 0.5` and `slope = 0`, as R.
- **`r2_measures`** is `1 - exp(-lr / n)` (Maddala / Cox-Snell) and the version
  adjusted for the `p` model d.f., `1 - exp(-max(lr - p, 0) / n)`; with `ess`,
  the same two with `ess` in place of `n` (NULL without it).

## Examples

```sql
CREATE TABLE risk AS
SELECT i, x1, x2, y,
       1 + i % 3 AS w,
       round(sin(0.9 * i) + 0.5 * cos(1.7 * i)) AS grade,
       round(1 / (1 + exp(-(0.2 + 1.1 * sin(0.9 * i)))), 2) AS prob,
       round(exp(1 + 0.5 * cos(0.8 * i) + 0.3 * sin(2.1 * i)), 1) AS time,
       i % 4 <> 0 AS event
FROM (SELECT i, round(1.5 * sin(0.9 * i), 2) AS x1, round(cos(0.4 * i), 2) AS x2,
             (sin(0.9 * i) + 0.6 * cos(1.7 * i) + 0.3 > 0)::INTEGER AS y
      FROM range(40) r(i));

-- ROC area and Somers' Dxy of an external prediction, unweighted and weighted
SELECT somers2_agg(y, prob) AS s, somers2_agg(y, prob, w) AS s_weighted FROM risk;

-- Rank concordance with an ordered outcome
SELECT concordance_agg(grade, x1) AS conc FROM risk;

-- Harrell's C for censored survival times
SELECT rcorr_cens_agg(time, event, x1) AS cindex FROM risk;

-- External validation of predicted probabilities
SELECT unnest(val_prob_agg(y, prob)) FROM risk;
```

The `lrm` statistics of a logistic model fitted with `logistic_fit_agg`: compute
the linear predictor with `predict(..., {'type': 'link'})` and aggregate it with
the outcome.

```sql
WITH m AS (SELECT logistic_fit_agg(y, [x1, x2]) AS fit FROM risk),
     scored AS (SELECT y, predict(m.fit, [x1, x2], {'type': 'link'}) AS lp FROM risk, m)
SELECT r.n, round(r.model_lr, 4) AS model_lr, round(r.c, 4) AS c, round(r.dxy, 4) AS dxy,
       round(r.r2, 4) AS r2, round(r.brier, 4) AS brier, round(r.g, 4) AS g
FROM (SELECT lrm_stats_agg(y, lp, 2) AS r FROM scored);

-- R2 measures from a likelihood-ratio chi-square of 32.8 on 2 d.f., n = 40
SELECT r2_measures(32.8, 2, 40) AS r2, r2_measures(32.8, 2, 40, 28.1) AS r2_ess;

-- The g-index of the linear predictor alone
WITH m AS (SELECT logistic_fit_agg(y, [x1, x2]) AS fit FROM risk)
SELECT round(gini_md_agg(predict(m.fit, [x1, x2], {'type': 'link'})), 4) AS g FROM risk, m;
```
