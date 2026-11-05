# Wald ANOVA (chunk tests) and fast backward step-down

Regression Modeling Strategies style inference from one fitted model: the
`rms::anova.rms` table of multi-degree-of-freedom Wald tests (every predictor
with all of its spline and interaction columns, its nonlinear part, the
interaction chunks, the pooled totals) and `rms::fastbw`, the fast backward
step-down of Lawless and Singhal. Both work from the model's coefficients and
coefficient covariance `vcov` only, so any fit that carries `vcov` qualifies:
`ols_fit_agg`, `wls_fit_agg`, `ridge_fit_agg` and the GLM aggregates
(`logistic_fit_agg`, `poisson_fit_agg`, `binomial_fit_agg`, `negbinom_fit_agg`,
`gamma_fit_agg`, `tweedie_fit_agg`) fitted with `{'compute_inference': true}`,
including a cluster-robust `vcov`. The statistics come from anofox-regression's
`inference::anova` and `inference::fastbw`.

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `wald_anova(model [, terms [, options]])` | Scalar | `anova.rms` table of Wald chunk tests, one list element per row |
| `fastbw(model [, terms [, options]])` | Scalar | Fast backward step-down: deletion order, kept terms, approximate reduced-model coefficients |

The examples on this page use this table; the columns `age1 .. age3` play the
part of a restricted cubic spline basis of age (`rcs(age, 4)` in rms: the
linear column and two nonlinear columns).

```sql
CREATE OR REPLACE TABLE patients AS
SELECT i AS id,
       'site' || (i % 2) AS site,
       20 + (i * 37) % 50 AS age,
       (20 + (i * 37) % 50) / 10.0 AS age1,
       greatest((20 + (i * 37) % 50) / 10.0 - 3.5, 0) ^ 3 / 10 AS age2,
       greatest((20 + (i * 37) % 50) / 10.0 - 5.0, 0) ^ 3 / 10 AS age3,
       100 + (i * 13) % 40 AS bp,
       (i % 3 = 0)::INTEGER::DOUBLE AS treated,
       sin(i * 0.7) AS noise,
       0.4 * (20 + (i * 37) % 50) / 10.0 + 0.03 * (100 + (i * 13) % 40) + 0.8 * (i % 3 = 0)::INTEGER
           + sin(i * 1.3) AS outcome,
       ((i * 7919) % 100 < (20 + (i * 37) % 50) * 1.2 - 15)::INTEGER::DOUBLE AS event
FROM range(200) r(i);
```

## Terms

Both functions take `terms`, one label per coefficient of the model (in the
order of `x`), as `ols_anova_table_agg`'s `terms` option does:

- Coefficients with the same label form one term, for example the columns of
  a spline basis (`['age', 'age', 'age', 'bp', 'treated']`). Terms are ordered
  by first appearance.
- A NULL label keeps the coefficient out of every term: `wald_anova` leaves
  it out of all tests, `fastbw` never deletes it (rms `fastbw(force = ...)`).
- Without `terms` (or with `NULL`) every coefficient is its own term, labelled
  `x1 .. xk`.
- `terms` (and `options`) must be constants.

## wald_anova

**Signature:**

```text
wald_anova(model STRUCT [, terms VARCHAR[] [, options MAP | STRUCT]]) -> STRUCT[]
```

The rows follow `anova.rms` (with `main.effect = FALSE`): for every predictor
the test of all its columns including the interactions it is part of
(`'age  (Factor+Higher Order Factors)'`), the test of its interactions
(`' All Interactions'`) and of its nonlinear columns (`' Nonlinear'`); for an
interaction the test of its columns and the nonlinear-interaction chunks
(`' Nonlinear Interaction : f(A,B) vs. AB'`, `' f(A,B) vs. Af(B) + Bg(A)'`,
`' Nonlinear Interaction in A vs. Af(B)'`, ...); then `TOTAL NONLINEAR`,
`TOTAL INTERACTION`, `TOTAL NONLINEAR + INTERACTION` (each when it applies)
and `TOTAL` (rms prints `REGRESSION` for an `ols` fit). Sub-rows keep rms's
leading space; use `trim(label)` to match them. Without interactions the
`(Factor+Higher Order Factors)` suffix is dropped, as in rms.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nonlinear` | `BOOLEAN[]` | all `false` | One flag per coefficient: the nonlinear columns of a term (the `'` and `''` columns of an rcs basis). Flags of interaction columns are ignored (derived from the factors) |
| `interactions` | `MAP(VARCHAR, VARCHAR[])` | none | Interaction label to its two main-effect factor labels, e.g. `MAP {'age * treated': ['age', 'treated']}` (a STRUCT literal works too) |
| `test` | `VARCHAR` | `'auto'` | `'chisq'` (Wald chi-square), `'F'` (`F = chi-square / df` on the model's `df_residual`), `'auto'`: F for the linear models (OLS, WLS, Ridge), chi-square otherwise, as rms |

- **Interactions.** The columns of an interaction term are the products of
  its factors' columns, in rms order: the full product `a * b` with the first
  factor varying fastest (`a1*b1, a2*b1, ..., a1*b2, ...`), or the restricted
  `a %ia% b`, which drops the products of two nonlinear columns and runs over
  the first factor (`a1*b1, a1*b2, ..., a2*b1, a3*b1, ...`). Which one is
  recognised from the number of columns. Interaction rows are labelled
  `'a * b'` (the factor labels), whatever the interaction's own label.
- **Partial sums of squares.** For the linear models `partial_ss = chi-square *
  sigma^2` and `mean_sq = partial_ss / df1` (rms `ss = TRUE`); an F test ends
  with an `ERROR` row holding the residual degrees of freedom, sum of squares
  and mean square.
- **NULL.** A NULL model or NULL coefficients give NULL. A model whose `vcov`
  is NULL (fitted without `compute_inference`) raises an error. A coefficient
  that was not estimated (aliased, NaN) in one of the terms gives NULL; label
  it NULL to leave it out.

Each list element:

| Field | Type | Description |
|-------|------|-------------|
| `label` | `VARCHAR` | Row label as `anova.rms` prints it |
| `kind` | `VARCHAR` | `main_effect`, `combined_effect`, `interaction`, `all_interactions`, `nonlinear`, `nonlinear_interaction`, `doubly_nonlinear_interaction`, `nonlinear_interaction_first`, `nonlinear_interaction_second`, `total_nonlinear`, `total_interaction`, `total_nonlinear_interaction`, `total`, `error` |
| `term` | `VARCHAR` | The term (its label) the row belongs to; NULL for the totals |
| `columns` | `BIGINT[]` | 1-based positions in `x` of the coefficients tested |
| `method` | `VARCHAR` | `'Wald chi-square'` or `'Wald F'` |
| `statistic` | `DOUBLE` | Chi-square, or F for an F test |
| `df1` | `DOUBLE` | Degrees of freedom (number of coefficients tested) |
| `df2` | `DOUBLE` | Residual degrees of freedom of an F test, else NULL |
| `p_value` | `DOUBLE` | Upper tail of chi-square(df1) or F(df1, df2) |
| `chi_square` | `DOUBLE` | Wald chi-square (also for an F test) |
| `chi_square_minus_df` | `DOUBLE` | `chi_square - df1`, the chance-corrected importance measure of RMS section 4 |
| `partial_ss`, `mean_sq` | `DOUBLE` | Partial sum of squares and `partial_ss / df1` (linear models) |

**Examples:**

```sql
-- OLS with a spline in age and a treatment interaction: F tests, partial SS
SELECT trim(r.label) AS factor, r.df1, round(r.statistic, 3) AS f, round(r.p_value, 4) AS p
FROM (SELECT unnest(wald_anova(
          ols_fit_agg(outcome, [age1, age2, age3, bp, treated, age1 * treated, age2 * treated, age3 * treated],
                      {'compute_inference': true}),
          ['age', 'age', 'age', 'bp', 'treated', 'age * treated', 'age * treated', 'age * treated'],
          {'nonlinear': [false, true, true, false, false, false, false, false],
           'interactions': MAP {'age * treated': ['age', 'treated']}})) AS r
      FROM patients);

-- A logistic model: Wald chi-square tests and chi-square - d.f. as importance
SELECT trim(r.label) AS factor, r.df1, round(r.chi_square, 3) AS chi2, round(r.chi_square_minus_df, 3) AS importance
FROM (SELECT unnest(wald_anova(
          logistic_fit_agg(event, [age1, age2, age3, bp, noise], {'compute_inference': true}),
          ['age', 'age', 'age', 'bp', 'noise'], {'nonlinear': [false, true, true, false, false]})) AS r
      FROM patients);

-- Per group, the overall test of age (all three columns)
SELECT site, round(t[1].statistic, 3) AS f_age, round(t[1].p_value, 4) AS p_age
FROM (SELECT site, wald_anova(ols_fit_agg(outcome, [age1, age2, age3, bp], {'compute_inference': true}),
                              ['age', 'age', 'age', 'bp']) AS t
      FROM patients GROUP BY site)
ORDER BY site;
```

## fastbw

**Signature:**

```text
fastbw(model STRUCT [, terms VARCHAR[] [, options MAP | STRUCT]]) -> STRUCT
```

At each step the remaining term with the smallest Wald criterion
(`chi-square - k_aic * df` for the AIC rule, the largest p-value for the P
rule) is a candidate; it is deleted while the residual chi-square of all
deleted coefficients (`type = 'residual'`, the default) or its own chi-square
(`type = 'individual'`) satisfies the rule. The tests of later steps and the
coefficients of the reduced model are approximated from the full fit
(`b_in - V_io V_oo^-1 b_out`) without refitting. Like rms, the step-down
ignores the term hierarchy: an interaction may stay after one of its main
effects is deleted. The intercept and NULL-labelled coefficients are never
deleted.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rule` | `VARCHAR` | `'aic'` | `'aic'`: delete while `chi-square - k_aic * df <= aics`; `'p'`: delete while the p-value is above `sls` |
| `type` | `VARCHAR` | `'residual'` | Apply the rule to the residual test of all deleted terms (`'residual'`) or to the term's own test (`'individual'`) |
| `sls` | `DOUBLE` | `0.05` | Significance level to stay, for `rule = 'p'` (in (0, 1)) |
| `aics` | `DOUBLE` | `0` | AIC threshold |
| `k_aic` | `DOUBLE` | `2` | Penalty per degree of freedom (rms `k.aic`) |

NULL handling and aliased coefficients as for `wald_anova` (an aliased
coefficient anywhere in the model gives NULL).

| Field | Type | Description |
|-------|------|-------------|
| `deleted` | `STRUCT[]` | The deletions in order: `term`, `chi_square`, `df`, `p_value` (the term's test at that step), `residual_chi_square`, `residual_df`, `residual_p_value` (all deleted so far), `aic` (`residual_chi_square - k_aic * residual_df`) |
| `kept_terms` | `VARCHAR[]` | Terms in the final model, in term order |
| `coefficients` | `DOUBLE[]` | Approximate coefficients of the reduced model, in the order of `x`; 0 for deleted ones |
| `intercept` | `DOUBLE` | Approximate intercept (NULL without one) |
| `rule`, `type` | `VARCHAR` | The rule and type used |

**Examples:**

```sql
-- Which factors does the AIC rule drop from the logistic model?
SELECT list_transform(b.deleted, lambda d: d.term) AS deleted, b.kept_terms,
       list_transform(b.coefficients, lambda c: round(c, 4)) AS approx_coefficients
FROM (SELECT fastbw(logistic_fit_agg(event, [age1, age2, age3, bp, noise], {'compute_inference': true}),
                    ['age', 'age', 'age', 'bp', 'noise']) AS b
      FROM patients);

-- The deletion steps under the P rule (sls = 0.2), keeping treatment in the model
SELECT d.term, round(d.chi_square, 3) AS chi2, round(d.residual_p_value, 4) AS residual_p
FROM (SELECT unnest((fastbw(ols_fit_agg(outcome, [age1, age2, age3, bp, treated, noise], {'compute_inference': true}),
                            ['age', 'age', 'age', 'bp', NULL, 'noise'], {'rule': 'p', 'sls': 0.2})).deleted) AS d
      FROM patients);
```

## See Also

- [OLS](ols.md) - `ols_anova_agg` (nested-model F test) and `ols_anova_table_agg` (type I / II sums of squares)
- [Likelihood tests](../glm/likelihood.md) - `lr_test`, the likelihood-ratio alternative to a Wald chunk test
- [Model tools](model_tools.md) - `predict(model, x, {'se_fit': true})` reads the same `vcov`
