# Regression Splines

Spline basis expansions for ordinary design matrices, as `rms::rcs`,
`rms::lsp`, `rms::%ia%` (Harrell, *Regression Modeling Strategies* §2.4) and
scikit-learn's `SplineTransformer`. Each scalar expands one value into a
`DOUBLE[]` row of basis columns given the knots, so a nonlinear term can be
built in SQL and fed to any `*_fit_agg` that takes `x DOUBLE[]`. The knots are
chosen from the data by an aggregate. The bases and knot rules come from
`anofox-regression` (`splines::{Rcs, Lsp, BSpline, rcs_default_knots,
restricted_interaction}`).

## Functions

| Function | Type | Description |
|----------|------|-------------|
| `rcs(x, knots [, options])` | Scalar | Restricted cubic spline basis `[x, x', x'', …]` (`k − 1` columns) |
| `lsp(x, knots)` | Scalar | Linear spline basis `[x, (x − t₁)₊, …]` |
| `bspline(x, knots [, degree [, options]])` | Scalar | B-spline basis (`n_knots + degree − 1` columns) |
| `spline_spec(kind, knots [, options])` | Scalar | The validated definition of a basis as a STRUCT |
| `spline_basis(x, spec)` | Scalar | The basis of a `spline_spec` STRUCT |
| `restricted_interaction(a, b [, a_nonlinear, b_nonlinear])` | Scalar | `rms` `%ia%`: products of two basis rows without the doubly nonlinear terms |
| `default_knot_quantiles(nk)` | Scalar | Knot quantiles of RMS Table 2.3 |
| `rcs_knots_agg(x [, nk [, options]])` | Aggregate | Default restricted cubic spline knots (`rms::rcs`) |
| `bspline_knots_agg(x [, n_knots [, options]])` | Aggregate | B-spline base knots placed on the data (sklearn) |

**NULL handling.** A NULL, NaN or infinite `x`, or NULL knots / spec, give a
NULL row (upstream treats a non-finite value as missing), which the fit
aggregates then skip. A NULL element inside `knots` is an error. The knot
aggregates skip NULL / NaN / infinite values and return NULL when the data
cannot place the knots (fewer than 6 values, too few distinct values).
Invalid knots or options raise.

## rcs

```text
rcs(x DOUBLE, knots DOUBLE[] [, options]) -> DOUBLE[]
```

The restricted (natural) cubic spline in Harrell's truncated-power form,
`Hmisc::rcspline.eval(x, knots, inclx = TRUE)` / `rms::rcs(x, knots)`: with `k`
knots (at least 3; sorted and de-duplicated) there are `k − 1` columns, `x`
itself and `k − 2` nonlinear terms, and the function is linear beyond the outer
knots. Options: `norm` `0` (none) | `1` (divide by `(t_k − t_{k−1})³`) | `2`
(default, divide by `(t_k − t₁)²`, which puts the terms on the scale of `x`).

## lsp

```text
lsp(x DOUBLE, knots DOUBLE[]) -> DOUBLE[]
```

The linear spline `[x, (x − t₁)₊, (x − t₂)₊, …]` (`rms::lsp`), one column more
than knots.

## bspline

```text
bspline(x DOUBLE, knots DOUBLE[] [, degree BIGINT [, options]]) -> DOUBLE[]
```

The B-spline basis of `degree` (default 3) on the strictly increasing base
knots, as `sklearn.preprocessing.SplineTransformer` for one feature: `degree`
further knots are added on each side at the spacing of the outermost interval,
giving `n_knots + degree − 1` columns (`n_knots − 1` when periodic) that sum to
one inside the knots. Options:

| Option | Default | Description |
|--------|---------|-------------|
| `extrapolation` | `'constant'` | Outside the base knots: `'constant'` (the boundary values), `'linear'`, `'continue'` (the boundary polynomials), `'periodic'` (period = last − first knot, needs `degree < n_knots`) or `'error'` |
| `include_bias` | `true` | `false` drops the last column, so a model with an intercept stays identifiable |

## rcs_knots_agg / bspline_knots_agg / default_knot_quantiles

```text
rcs_knots_agg(x DOUBLE [, nk BIGINT [, options]]) -> DOUBLE[]
bspline_knots_agg(x DOUBLE [, n_knots BIGINT [, options]]) -> DOUBLE[]
default_knot_quantiles(nk BIGINT) -> DOUBLE[]
```

`rcs_knots_agg` places `nk` knots (default 5, at least 3) exactly as
`Hmisc::rcspline.eval(x, nk, knots.only = TRUE)`, i.e. the default knots of
`rms::rcs(x, nk)`: at the quantiles of RMS Table 2.3
(`default_knot_quantiles(nk)`: 0.1–0.9 for 3 knots, 0.05–0.95 for 4–6,
0.025–0.975 for 7 or more), with the outer knots moved to the 5th smallest /
largest value when there are fewer than 100 values, the distinct values
without the extremes when there are fewer than `nk + 3` of them, and an
extreme tied in at least `fractied` of the rows (option, default 0.05) handled
separately. It may return fewer knots than requested when they coincide.

`bspline_knots_agg` places `n_knots` (default 5, at least 2) base knots on the
data as `SplineTransformer(n_knots, knots = ...)`: option `knots` `'uniform'`
(default, equally spaced from the minimum to the maximum) or `'quantile'`
(equally spaced type-7 quantiles).

## spline_spec / spline_basis

```text
spline_spec(kind VARCHAR, knots DOUBLE[] [, options]) -> STRUCT
spline_basis(x DOUBLE, spec STRUCT) -> DOUBLE[]
```

`spline_spec` validates a basis and returns its definition, the stable
representation of a spline term that downstream functions (effects and
contrasts, #319) consume. `kind` is a constant `'rcs'`, `'lsp'` or
`'bspline'`; options `norm` (rcs), `degree`, `extrapolation`, `include_bias`
(bspline).

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `VARCHAR` | `rcs`, `lsp` or `bspline` |
| `knots` | `DOUBLE[]` | The knots as used (rcs / lsp sorted and de-duplicated; bspline the base knots) |
| `norm` | `INTEGER` | rcs normalisation; NULL for the other kinds |
| `degree` | `INTEGER` | bspline degree; NULL otherwise |
| `extrapolation` | `VARCHAR` | bspline extrapolation; NULL otherwise |
| `include_bias` | `BOOLEAN` | bspline bias column; NULL otherwise |
| `n_columns` | `BIGINT` | Number of basis columns |
| `nonlinear` | `BOOLEAN[]` | Per column, whether it is a nonlinear term (`rms` `nonlinear` attribute; the chunk tested for linearity). All B-spline columns are nonlinear |

`spline_basis(x, spec)` expands `x` with such a STRUCT (only `kind` and
`knots` are required; missing fields take their defaults). Keep the spec next
to the fitted model and expand new data with the same knots for prediction.

## restricted_interaction

```text
restricted_interaction(a DOUBLE[], b DOUBLE[] [, a_nonlinear BOOLEAN[], b_nonlinear BOOLEAN[]]) -> DOUBLE[]
```

The restricted interaction `a %ia% b` of `rms` (RMS §2.7.2): the products
`a_i · b_j` in the order `i`, then `j`, stopping the inner loop at the first
pair where both columns are nonlinear, so no doubly nonlinear product enters.
For `rcs(x1, 4 knots)` and `rcs(x2, 3 knots)` this is `x1*x2, x1*x2', x1'*x2,
x1''*x2`. Without flags the first element of each basis is linear and the rest
nonlinear (the `rcs` / `lsp` layout); pass `spline_spec(...).nonlinear` for a
B-spline basis.

## Example

```sql
CREATE OR REPLACE TABLE spline_demo AS
SELECT i, (i * 37 % 200) / 4.0 AS age,
       2.0 + 0.05 * (i * 37 % 200) / 4.0 + sin((i * 37 % 200) / 40.0) + ((i * 13) % 7 - 3) / 10.0 AS y
FROM range(200) t(i);

-- Default knots (rms::rcs(age, 4)), kept as a spline spec
CREATE OR REPLACE TABLE age_spec AS
SELECT spline_spec('rcs', rcs_knots_agg(age, 4)) AS spec FROM spline_demo;

SELECT spec.knots, spec.n_columns, spec.nonlinear FROM age_spec;

-- Fit y ~ rcs(age, 4) and predict at new ages with the same knots
CREATE OR REPLACE TABLE age_fit AS
SELECT ols_fit_agg(y, spline_basis(age, spec)) AS model
FROM spline_demo, age_spec;

SELECT model.r_squared, model.coefficients FROM age_fit;

SELECT a AS age,
       model.intercept + list_sum(list_transform(range(1, 4), lambda j: model.coefficients[j] * spline_basis(a, spec)[j])) AS fitted
FROM age_fit, age_spec, (VALUES (10.0), (25.0), (40.0)) t(a);
```

```sql
-- The bases directly
SELECT rcs(30.0, [10.0, 20.0, 35.0, 45.0]) AS rcs_row,
       lsp(30.0, [20.0, 40.0]) AS lsp_row,
       bspline(30.0, [0.0, 25.0, 50.0], 2, {'extrapolation': 'linear'}) AS bspline_row,
       default_knot_quantiles(5) AS quantiles;

-- B-spline knots from data (sklearn SplineTransformer(n_knots = 5, knots = 'quantile'))
SELECT bspline_knots_agg(age, 5, {'knots': 'quantile'}) FROM spline_demo;

-- rcs(age, 4) %ia% rcs(z, 3)
SELECT restricted_interaction(rcs(30.0, [10.0, 20.0, 35.0, 45.0]), rcs(1.5, [0.0, 1.0, 3.0])) AS ia;
```

## References

- Harrell FE (2015). *Regression Modeling Strategies*, 2nd ed., §2.4 (Table 2.3), §2.7.2.
- `Hmisc::rcspline.eval`, `rms::rcs`, `rms::lsp`, `rms::%ia%`; `splines::splineDesign`.
- scikit-learn `SplineTransformer`.
- Reference tests: `test/sql/reference/splines_reference.test` (R,
  `validation/generators/make_reference_splines.R`) and
  `test/sql/reference/splines_sklearn.test` (scikit-learn,
  `validation/generators/make_reference_splines.py`).
