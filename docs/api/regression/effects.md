# Effects, Contrasts and Nomograms

Effects of predictors on the covariate scale (Harrell, *Regression Modeling
Strategies* §5.1), as `rms::summary`, `rms::contrast`, `rms::Predict` and
`rms::nomogram`. A predictor that enters a model through several columns (a
restricted cubic spline, a linear spline, the dummies of a factor, an
interaction) has no single coefficient to read; its effect is a contrast of the
linear predictor between two settings of the predictor, with the other
predictors held at typical values. These functions build those contrasts from
a fitted model STRUCT and a description of how each predictor enters the
design. The statistics come from `anofox-regression`
(`effects::ModelSpec`: `summary`, `contrast`, `double_contrast`,
`predict_grid`, `predict_grid_simultaneous`, `nomogram`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `effects_term(name, kind [, levels])`, `effects_term(name, spline_spec)` | Scalar | A predictor term: linear, categorical, or an rcs / lsp spline from `spline_spec` |
| `effects_interaction(a, b [, restricted])` | Scalar | An interaction term (`a * b`, or `a %ia% b` with `restricted`) |
| `effects_design(x, terms)` | Scalar | The design row of raw predictor values, to fit the model on |
| `effects_summary_agg(x, model, terms [, options])` | Aggregate | `summary.rms`: inter-quartile-range effects and factor-level effects |
| `contrast_agg(x, model, terms, a, b [, options])` | Aggregate | `contrast.rms(fit, a, b)` |
| `double_contrast_agg(x, model, terms, a, b, a2, b2 [, options])` | Aggregate | `contrast.rms(fit, a, b, a2, b2)`: `(a − b) − (a2 − b2)` |
| `predict_grid_agg(x, model, terms, term [, options])` | Aggregate | `Predict(fit, term)`: partial effect over a grid, pointwise or simultaneous band |
| `nomogram_agg(x, lp, model, terms [, options])` | Aggregate | `nomogram(fit)`: points, total-points, linear-predictor and risk axes |
| `predict_grid_by(source, group_col, x_cols, model, terms, term [, ...])` | Table macro | `predict_grid_agg` per group, in the `prediction` schema |

## The workflow

1. Describe the predictors as a list of terms. Spline terms take their knots
   from a [`spline_spec`](splines.md#spline_spec--spline_basis) (for example
   `spline_spec('rcs', rcs_knots_agg(age, 4))`); a categorical term is coded
   with treatment contrasts, a dummy for every level but the first (the
   numeric levels sorted, as R's factor coding).
2. Fit any `*_fit_agg` with `{'compute_inference': true}` on
   `effects_design(x, terms)`, where `x` lists the raw predictor values in
   term order. The design is the main-effect bases in term order, then the
   interactions, exactly the column order of the `rms` formula
   `rcs(age, 4) + sex + rcs(age, 4) %ia% sex`.
3. Pass the same `x`, the model and the terms to the effects aggregates. They
   run over the data rows because, as `rms::datadist`, the reference values
   come from the data: the lower and upper quartiles (the effect range), the
   median (the adjust-to value; the most frequent level for a categorical
   predictor) and the `10/max(n, 200)` quantiles (the default
   `Predict` range), with rms's rules for fewer than four distinct values.

```sql
CREATE OR REPLACE TABLE fx AS
SELECT i,
       round(30 + 40 * ((i * 37) % 101) / 100.0, 1) AS age,
       (i % 3)::DOUBLE AS grp,
       ((i * 7) % 11 < 5)::DOUBLE AS sex
FROM range(240) t(i);

ALTER TABLE fx ADD COLUMN y DOUBLE;
UPDATE fx SET y = 1 + 0.05 * age + 0.002 * (age - 50) ^ 2 + 0.4 * sex - 0.3 * grp
                  + ((i * 13) % 17 - 8) / 8.0;
ALTER TABLE fx ADD COLUMN event DOUBLE;
UPDATE fx SET event = (((i * 29) % 23) / 23.0 < 1 / (1 + exp(-(-4 + 0.07 * age + 0.5 * sex))))::DOUBLE;

-- Terms: rcs(age, 4 knots), sex (0/1), grp (a 3-level factor)
CREATE OR REPLACE TABLE fx_terms AS
SELECT [effects_term('age', spline_spec('rcs', rcs_knots_agg(age, 4))),
        effects_term('sex', 'categorical', [0, 1]),
        effects_term('grp', 'categorical', [0, 1, 2])] AS terms
FROM fx;

-- OLS on the design, with the covariance the effects need
CREATE OR REPLACE TABLE fx_ols AS
SELECT ols_fit_agg(y, effects_design([age, sex, grp], t.terms), {'compute_inference': true}) AS m,
       any_value(t.terms) AS terms
FROM fx, fx_terms t;

SELECT len(effects_design([45.0, 1, 2], terms)) AS n_columns FROM fx_terms;
```

## effects_term / effects_interaction

```text
effects_term(name VARCHAR, kind VARCHAR [, levels DOUBLE[]]) -> STRUCT
effects_term(name VARCHAR, spec STRUCT) -> STRUCT
effects_interaction(a VARCHAR, b VARCHAR [, restricted BOOLEAN]) -> STRUCT
```

The term STRUCT is `{name, kind, knots, norm, levels, a, b}`; every term has
the same type, so a list of terms is a plain `LIST`. `kind` is `'linear'` or
`'categorical'` (alias `'factor'`), or the kind of the `spline_spec`:
`'rcs'` (with its `norm`) or `'lsp'`. B-splines are not supported (upstream's
`ModelSpec` has no B-spline predictor). `levels` lists the levels of a
categorical predictor; `effects_design` needs them, and the aggregates check
that the levels in the data are exactly these (otherwise the model's dummy
columns would not line up). Without `levels` the aggregates take the levels
from the data.

`effects_interaction(a, b)` is the full tensor-product interaction of the two
named predictors (`rms` `a * b`); with `restricted = true` it is the
restricted interaction `a %ia% b`, which leaves out the doubly nonlinear
products. Interactions can be listed anywhere in the term list; their columns
always follow the main effects. A term STRUCT literal with these fields (or a
`spline` field holding a `spline_spec`) works too.

## effects_design

```text
effects_design(x DOUBLE[], terms LIST) -> DOUBLE[]
```

The design row (no intercept column) of the raw predictor values `x`, one value
per predictor term in term order (interaction terms take no value). A NULL `x`,
or a NULL / NaN / infinite element, gives NULL. A value that is not one of a
categorical term's levels, or an `x` of the wrong length, raises.

## effects_summary_agg

```text
effects_summary_agg(x DOUBLE[], model STRUCT, terms LIST [, options]) -> LIST(STRUCT)
```

`summary.rms`: for every numeric predictor the effect of moving it from its
lower to its upper quartile, for every categorical predictor the effect of each
level against the most frequent one, the other predictors at their adjust-to
values. One element per effect:

| Field | Description |
|-------|-------------|
| `term` | The predictor |
| `label` | `term`, or `term - level:reference` for a categorical predictor (rms's row names) |
| `low`, `high`, `diff` | The two settings and their difference |
| `estimate`, `std_error` | The effect (difference of the linear predictor) and its standard error |
| `conf_low`, `conf_high` | Confidence limits at `conf_level` |
| `statistic`, `p_value` | `estimate / std_error` and its two-sided p-value |
| `ratio`, `ratio_low`, `ratio_high` | `exp` of the effect and limits (odds ratio for a logit link, rate / mean ratio for a log link); NULL for other links unless `ratio := true` |
| `conf_level` | The level |

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `level` | `0.95` | Confidence level |
| `critical` | `'auto'` | Reference distribution: `'auto'` uses Student t with the model's `df_residual` for linear models (OLS, WLS, Ridge: the models with a `residual_std_error`) and the standard normal otherwise, as `rms`; `'normal'` or `'t'` force one |
| `ratio` | auto | `true` / `false` to add or drop the ratio columns |

```sql
SELECT r.label, round(r.low, 2) AS low, round(r.high, 2) AS high,
       round(r.estimate, 3) AS effect, round(r.conf_low, 3) AS lo, round(r.conf_high, 3) AS hi
FROM (SELECT unnest(effects_summary_agg([age, sex, grp], f.m, f.terms)) AS r FROM fx, fx_ols f);
```

## contrast_agg / double_contrast_agg

```text
contrast_agg(x DOUBLE[], model STRUCT, terms LIST, a, b [, options]) -> STRUCT
double_contrast_agg(x DOUBLE[], model STRUCT, terms LIST, a, b, a2, b2 [, options]) -> STRUCT
```

The difference of the linear predictor between the settings `a` and `b`
(`contrast.rms(fit, a, b)`), or the double difference
`(a − b) − (a2 − b2)` (`contrast.rms(fit, a, b, a2, b2)`, for example how an
age effect differs between the sexes). A setting is a constant STRUCT or MAP
`{predictor: value}`; predictors it leaves out are at their adjust-to values
(median, or most frequent level). A categorical value must be one of its
levels. The result is `{estimate, std_error, conf_low, conf_high, statistic,
p_value, conf_level}`. Options `level` and `critical`, as above. For a GLM the
contrast is on the link scale: `exp(estimate)` of a logistic model is the odds
ratio between the two settings.

```sql
-- Age 60 vs 40, everything else at its typical value
SELECT round(c.estimate, 3) AS estimate, round(c.std_error, 3) AS se, round(c.p_value, 4) AS p
FROM (SELECT contrast_agg([age, sex, grp], f.m, f.terms, {'age': 60}, {'age': 40}) AS c FROM fx, fx_ols f);
```

```sql
-- Odds ratio of an event for sex 1 vs 0 at age 50, from a logistic fit
WITH fit AS (
    SELECT logistic_fit_agg(event, effects_design([age, sex, grp], t.terms), {'compute_inference': true}) AS m,
           any_value(t.terms) AS terms
    FROM fx, fx_terms t
)
SELECT round(exp(c.estimate), 3) AS odds_ratio, round(exp(c.conf_low), 3) AS lo, round(exp(c.conf_high), 3) AS hi
FROM (SELECT contrast_agg([age, sex, grp], f.m, f.terms, {'age': 50, 'sex': 1}, {'age': 50, 'sex': 0}) AS c
      FROM fx, fit f);
```

## predict_grid_agg

```text
predict_grid_agg(x DOUBLE[], model STRUCT, terms LIST, term VARCHAR [, options]) -> LIST(STRUCT)
```

`Predict(fit, term)`: the linear predictor (with the intercept) over a grid
of the predictor `term`, the other predictors at their adjust-to values. One
element per grid point `{x, yhat, se_fit, yhat_lower, yhat_upper, critical}`.
The default grid is 200 equally spaced points between the `datadist`
prediction limits (the levels for a categorical predictor). For a GLM the
values are on the link scale (`Predict`'s default); map them through the
inverse link for the response scale.

| Option | Default | Description |
|--------|---------|-------------|
| `values` | | The grid values instead of the default grid |
| `simultaneous` | `false` | Bands simultaneous over the grid (`Predict(conf.type = "simultaneous")`, `multcomp`'s single-step max-|t| critical value, given in `critical`) |
| `level`, `critical` | | As above |

```sql
SELECT p.x, round(p.yhat, 3) AS yhat, round(p.yhat_lower, 3) AS lower, round(p.yhat_upper, 3) AS upper
FROM (SELECT unnest(predict_grid_agg([age, sex, grp], f.m, f.terms, 'age', {'values': [35, 45, 55, 65]})) AS p
      FROM fx, fx_ols f);
```

The simultaneous critical value comes from a deterministic quadrature
(upstream `simultaneous_critical`); `multcomp`'s randomised Genz–Bretz
algorithm agrees to about 1e-3.

## predict_grid_by

```text
predict_grid_by(source, group_col, x_cols, model, terms, term [, grid_values := NULL, simultaneous := false, level := 0.95])
```

`predict_grid_agg` per group of `source`, one row per grid point in the
integration contract's `prediction` schema (the same columns as
[`<model>_effects_by`](model_tools.md#prediction_by-and-effects_by)): the group
column, `id`, `model_id` and `model_name` (`'predict_grid'`), `focal` (the
predictor's 1-based position among the predictors), `x`, `y` (NULL), `yhat`,
`yhat_lower`, `yhat_upper`, `level`, `split` (NULL), `se_fit`, `focal_name`
(`term`) and `row_order`. `model` and `terms` are expressions over `source`,
for example columns of a view that joins each group's data with its model.

```sql
-- One model per site, joined back to the site's rows
CREATE OR REPLACE VIEW fx_by_site AS
WITH fits AS (
    SELECT i % 2 AS site,
           ols_fit_agg(y, effects_design([age, sex, grp], t.terms), {'compute_inference': true}) AS m,
           any_value(t.terms) AS terms
    FROM fx, fx_terms t
    GROUP BY site
)
SELECT fx.*, fits.site, fits.m, fits.terms FROM fx JOIN fits ON fx.i % 2 = fits.site;

SELECT site, focal_name, x, round(yhat, 3) AS yhat, round(yhat_lower, 3) AS lower
FROM predict_grid_by('fx_by_site', site, [age, sex, grp], m, terms, 'age', grid_values := [40, 60])
ORDER BY site, x;
```

A group is one `datadist`: grouping by a categorical predictor of the model
itself leaves one level per group, which raises (see below).

## nomogram_agg

```text
nomogram_agg(x DOUBLE[], lp DOUBLE, model STRUCT, terms LIST [, options]) -> STRUCT
```

`rms::nomogram` of a main-effects model (no interaction terms): every
predictor gets an axis whose tick values (its levels, its distinct values
when there are at most 10, or `pretty` values over its range) carry their term
of the linear predictor (`xbeta`) and their points, `(xbeta − min) · scale`,
with `scale = maxscale / (widest range of a term)`. Adding the points of a
patient's values and reading the total on the linear-predictor axis gives the
linear predictor. `lp` is the fit's linear predictor of each row (with the
intercept, for example `predict(m, effects_design(x, terms), {'type': 'link'})`),
which places the `lp` axis.

| Field | Description |
|-------|-------------|
| `axes` | `LIST(STRUCT(term, values, xbeta, points))`, one per predictor |
| `scale` | Points per unit of the linear predictor |
| `intercept` | Linear predictor at zero total points |
| `total_points` | Ticks of the total-points axis |
| `lp_at`, `lp_points` | Ticks of the linear-predictor axis and their positions in points |
| `fun_at`, `fun_points` | Ticks of the function axis (`fun`) and their positions; NULL without `fun` |

Options: `maxscale` (default 100, points of the widest axis), `nint` (default
10, the number of intervals for `pretty`), `fun`: `'none'` (default),
`'plogis'` (a risk axis for a logistic model, `nomogram(fun = plogis)`) or
`'exp'`. A nomogram is a picture: this STRUCT holds every number needed to draw
it (or to score a patient without it), not the drawing.

```sql
WITH fit AS (
    SELECT logistic_fit_agg(event, effects_design([age, sex, grp], t.terms), {'compute_inference': true}) AS m,
           any_value(t.terms) AS terms
    FROM fx, fx_terms t
),
nom AS (
    SELECT nomogram_agg([age, sex, grp], predict(f.m, effects_design([age, sex, grp], f.terms), {'type': 'link'}),
                        f.m, f.terms, {'fun': 'plogis'}) AS n
    FROM fx, fit f
)
SELECT a.term, a.values, list_transform(a.points, lambda p: round(p, 1)) AS points
FROM (SELECT unnest((n).axes) AS a FROM nom);
```

## NULL handling and errors

- Rows with a NULL `x`, a NULL / NaN / infinite element of `x` (or, for
  `nomogram_agg`, a NULL or non-finite `lp`) are skipped. A group without rows
  gives NULL.
- `model` and `terms` are taken from the first row of the group where they are
  not NULL (they are meant to be constant within a group); a group where
  either is always NULL gives NULL.
- A model without a covariance (`vcov` NULL: fit with
  `{'compute_inference': true}`), a model whose number of coefficients does
  not match the terms' design, an unknown predictor in a setting or `term`,
  categorical levels in the data that differ from the term's `levels` (for
  example a group in which a level does not occur: upstream builds the dummy
  columns from the levels in the data), a
  setting value that is not a level, a NULL setting value and unknown or
  invalid options raise errors. Settings, `term` and options must be
  constants. Coefficients that are NULL / NaN (aliased columns) give NULL
  estimates.

## References

- Harrell FE (2015). *Regression Modeling Strategies*, 2nd ed., §5.1, Springer.
- `rms::datadist`, `rms::summary.rms`, `rms::contrast.rms`, `rms::Predict`, `rms::nomogram`.
- Reference test: `test/sql/reference/effects_reference.test` (R `rms` and
  `multcomp`, `validation/generators/make_reference_effects.R`).
