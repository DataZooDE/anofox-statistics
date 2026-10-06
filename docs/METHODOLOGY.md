# Methodology

This page describes how the extension computes its standard errors, p-values,
confidence intervals and prediction intervals, and how its iterative solvers
decide they have converged. It gives the formulas behind the numbers, so you can
reproduce them or compare them with R, statsmodels or scikit-learn.

Notation: `n` is the number of rows used in the fit (`n_observations`). `p` is the
number of estimated parameters: the non-aliased slope coefficients, plus one
when `fit_intercept` is true. `X` is the design matrix (with a leading column of
ones when an intercept is fitted), `e` is the residual vector, and `α = 1 −
confidence_level`. `confidence_level` defaults to `0.95`.

## Linear models (OLS and WLS)

### Estimation

The coefficients minimise the residual sum of squares, weighted for WLS. The
linear system is solved by the decomposition chosen with the `solver` option:

| `solver` | Behaviour |
|----------|-----------|
| `'svd'` (default) | Singular value decomposition. Rank-deficient designs get the minimum-norm solution |
| `'qr'` | QR with column pivoting. Perfectly collinear columns are flagged as aliased, and their coefficient is `NaN` |
| `'cholesky'` | Cholesky factorisation of `XᵀX`. Fastest. Aliased columns are reported as `NaN` |

Zero-variance (constant) feature columns are detected before solving. They get a
`NaN` coefficient whichever solver is used. See [NULL_SEMANTICS.md](NULL_SEMANTICS.md).

### Goodness of fit

- `r_squared = 1 − RSS / TSS`, where `TSS` is the sum of squares around the mean
  of `y` (around zero when there is no intercept).
- `adj_r_squared = 1 − (1 − R²)(n − 1)/(n − p)`.
- `residual_std_error = s = sqrt(RSS / (n − p))`.

### Classical inference

With `compute_inference: true`:

- Covariance: `Var(β̂) = s² (XᵀX)⁻¹`. For WLS, `X` and `y` are scaled by `sqrt(w)`.
- Standard errors: `std_errors[j] = sqrt(Var(β̂)[j, j])`.
- `t_values[j] = β̂[j] / std_errors[j]`.
- `p_values[j] = 2 · (1 − F_t(|t_j|; n − p))`, a two-sided test against Student's
  *t* with `n − p` degrees of freedom.
- Confidence intervals: `β̂[j] ± t(1 − α/2; n − p) · std_errors[j]`.
- Overall F-test: `f_statistic = (ESS / df_model) / (RSS / (n − p))`, where `df_model`
  is the number of slope parameters. `f_pvalue` is the upper tail of
  `F(df_model, n − p)`.

The inference arrays are aligned with `coefficients`, so they do not include the
intercept.

### Heteroscedasticity-consistent (robust) standard errors

Setting `hc_type` replaces the classical covariance with the sandwich estimator.
`hc_type` is read by the OLS and WLS functions: `ols_fit_agg`, `ols_fit`,
`ols_fit_predict`, `ols_fit_predict_agg` and the WLS equivalents.

```
Var_HC(β̂) = (XᵀX)⁻¹ · Xᵀ diag(ω) X · (XᵀX)⁻¹
```

The weights `ω_i` depend on the type. `h_ii` is the leverage, the i-th diagonal
element of the hat matrix `X(XᵀX)⁻¹Xᵀ`.

| `hc_type` | `ω_i` | Notes |
|-----------|-------|-------|
| `'none'` (default) | — | Classical `s²(XᵀX)⁻¹` |
| `'hc0'` | `e_i²` | White (1980) |
| `'hc1'` | `n/(n − p) · e_i²` | Degrees-of-freedom correction. Matches Stata's `robust` |
| `'hc2'` | `e_i² / (1 − h_ii)` | Leverage-adjusted |
| `'hc3'` | `e_i² / (1 − h_ii)²` | Jackknife-like. The most conservative of the four, and a good choice for small samples |

For HC2 and HC3, an observation with `h_ii ≥ 1` falls back to the HC0 weight, for
numerical safety. Robust *t* statistics, p-values and confidence intervals still
use Student's *t* with `n − p` degrees of freedom. If the robust computation
fails, the function falls back to the classical standard errors.

```sql
SELECT (ols_fit_agg(y, [x], {'compute_inference': true, 'hc_type': 'hc3'})).std_errors AS hc3_se
FROM (VALUES (1.0, 1.0), (2.3, 2.0), (2.8, 3.0), (4.6, 4.0), (4.9, 5.0), (6.8, 6.0)) t(y, x);
```

## Prediction intervals

The `*_fit_predict`, `*_fit_predict_agg` and `*_fit_predict_by` functions return
`yhat`, `yhat_lower` and `yhat_upper`. For a new row with features `x₀`, written
`x̃₀ = [1, x₀]` when an intercept is fitted:

```
yhat           = x̃₀ᵀ β̂
se_pred(x₀)    = s · sqrt(1 + x̃₀ᵀ (XᵀX)⁻¹ x̃₀)
[lower, upper] = yhat ∓ t(1 − α/2; n − p) · se_pred(x₀)
```

This is the textbook interval for a **new observation**. The `1` accounts for
the noise in the new observation. The quadratic form `x̃₀ᵀ(XᵀX)⁻¹x̃₀` is the
leverage of the new point, and accounts for uncertainty in the estimated
coefficients. As a result, intervals widen as `x₀` moves away from the centre of
the training data, and are narrowest near the mean of the features.
`confidence_level` (default `0.95`) sets the coverage.

<!-- TODO(lead): confirm which model families use the leverage-aware formula
     (OLS/WLS certainly; state what regularised/robust/GLM fit-predict functions do)
     once the prediction-interval change lands, and fill in the release that
     introduced it. -->

Earlier releases (up to and including 0.10.0) used the approximation
`se_pred = s · sqrt(1 + 1/n)` for every row. That approximation ignores leverage,
so it is too narrow far from the data and slightly too wide near the centre.

When there are no residual degrees of freedom (`n − p = 0`) or `s` is not
finite, the interval collapses to `yhat_lower = yhat_upper = yhat`. Coefficients
that are `NaN` (constant or aliased columns) contribute nothing to `yhat`.

## Generalized linear models: IRLS

`poisson_fit_agg`, `binomial_fit_agg`, `logistic_fit_agg`, `negbinom_fit_agg`,
`gamma_fit_agg` and `tweedie_fit_agg` are fitted by iteratively reweighted least
squares (Fisher scoring):

1. Start from `μ` initialised from `y`. Compute the linear predictor `η = g(μ)`.
2. Form the working response `z = η + (y − μ) g'(μ)` and the working weights
   `W = 1 / (V(μ) g'(μ)²)`.
3. Solve the weighted least-squares problem for `β`, then update `η = Xβ` and
   `μ = g⁻¹(η)`.
4. If the deviance went up, halve the step (step-halving, as in R's `glm.fit`).

**Convergence.** Iteration stops when either criterion holds:

- relative deviance change: `|D − D_old| / (0.1 + |D|) < tolerance`; or
- maximum absolute coefficient change: `max_j |β_j − β_j,old| < tolerance`.

The defaults are `tolerance = 1e-8` and `max_iterations = 100`. Both can be
changed with options. The result reports `iterations` and `converged`. If a fit
fails, including when it does not converge within `max_iterations`, the
aggregate returns NULL for that group. In that case, try raising `max_iterations`
or rescaling the features.

**Inference.** GLM standard errors come from the inverse Fisher information,
`Var(β̂) = φ · (XᵀWX)⁻¹`, evaluated at convergence. The tests are Wald tests
against the standard normal distribution. That is why GLM results carry
`z_values` rather than `t_values`:

- `z_j = β̂_j / SE_j`, `p_j = 2 · (1 − Φ(|z_j|))`.
- Confidence intervals: `β̂_j ± z(1 − α/2) · SE_j`.

For Poisson, the dispersion `φ` is `max(1, X²_Pearson / (n − p))`. When the data
are overdispersed, the standard errors are inflated quasi-Poisson-style. Otherwise
`φ = 1`, the standard Poisson value. The `dispersion` field reports the value
used.

**Fit statistics.** `deviance` and `null_deviance` are the model and
intercept-only deviances. `pseudo_r_squared = 1 − deviance / null_deviance`, the
deviance-based (McFadden-type) R². `aic` is the Akaike information criterion.

## Hypothesis tests

### p-values

Unless noted otherwise on the function's page, test aggregates compute exact or
asymptotic p-values from the test statistic's reference distribution. They take an
`alternative` option: `'two_sided'` (default), `'less'` or `'greater'`.

### Two-sample t-test: Welch is the default

`t_test_agg(value, group)` performs **Welch's** unequal-variance t-test by default.
The Welch–Satterthwaite degrees of freedom are

```
df = (s₁²/n₁ + s₂²/n₂)² / ( (s₁²/n₁)²/(n₁ − 1) + (s₂²/n₂)²/(n₂ − 1) )
```

Pass `{'kind': 'student'}` (or `{'var_equal': true}`) for the pooled-variance
Student t-test, which has `n₁ + n₂ − 2` degrees of freedom. Pass
`{'paired': true}` for a paired test. The `method` field of the result names the
variant that was used.

```sql
SELECT (t_test_agg(v, g)).method AS default_method,
       (t_test_agg(v, g, {'kind': 'student'})).method AS student_method
FROM (VALUES (1.0, 0), (2.0, 0), (3.5, 0), (4.0, 0),
             (4.0, 1), (5.5, 1), (6.0, 1), (8.0, 1)) t(v, g);
```

The same Welch default applies to `tost_t_test_agg`.
