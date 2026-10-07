# Reference-value generators

The scripts in this directory compute reference values with R (and scipy) and
write them into the sqllogictests under `test/sql/reference/`. The tests
inline small fixed datasets as `INSERT ... VALUES` and compare each extension
result with the reference using a mixed absolute/relative tolerance
(`abs(a - b) <= tol * (1 + abs(b))`). There are no intermediate CSV/JSON
fixtures: the generated `.test` files are the source of truth and are
committed.

Only re-run a generator when you change a dataset, add a check, or a reference
implementation changes. Run from the repository root.

## `make_reference_regression.R`

```bash
Rscript validation/generators/make_reference_regression.R
```

Requires R >= 4.0 with `MASS`, `survival`, `lme4` and `quantreg` (glmnet is
not needed). It (re)writes:

| File | Extension functions | R reference |
|------|---------------------|-------------|
| `regression_ols.test` | `ols_fit_agg` (coefficients, inference, HC0-HC3, solvers, no intercept, rank deficiency), `ols_fit_predict_agg` prediction intervals | `lm`, `summary.lm`, `confint`, `predict.lm(interval = "prediction")`, sandwich estimators coded by hand |
| `regression_wls.test` | `wls_fit_agg` | `lm(weights = w)` |
| `regression_ridge.test` | `ridge_fit_agg` (raw and glmnet lambda scaling) | closed-form ridge on centred data |
| `regression_quantile.test` | `quantile_fit_predict_agg` | `quantreg::rq` |
| `glm_fit.test` | `poisson_fit_agg`, `binomial_fit_agg` (logit/probit/cloglog), `logistic_fit_agg`, `negbinom_fit_agg`, `gamma_fit_agg` (log link) | `glm`, `MASS::glm.nb`, `negative.binomial()` |
| `glm_glmm.test` | `glmm_fit_agg` (gaussian REML/ML, poisson, binomial) | `lme4::lmer`, `lme4::glmer(nAGQ = 0)` |
| `survival_aft.test` | `aft_fit_agg` (weibull, lognormal, loglogistic, exponential) | `survival::survreg` |

Where the extension follows a different convention from R's default (e.g. the
floored quasi-Poisson covariance scaling, centred R^2 without an intercept,
moment-estimated negative-binomial theta, Pearson dispersion in the Gamma AIC),
the generator computes the reference by hand with the extension's convention
and the generated test explains the difference in a comment.

The datasets are literal constants in the script (drawn once with a fixed seed
and rounded), so the output does not depend on R's RNG implementation.

## Other generators

`make_reference_tests.R` / `make_reference_tests.py` produce the
hypothesis-test, correlation, categorical and normality reference tests in the
same directory; see the header of each script.

## `generate_issue107_tests.R`

An older, never-executed cross-check for explicit priors (`arm::bayesglm`) and
empirical-Bayes shrinkage (`metafor`). It writes JSON to `test/data/issue107/`,
which no test consumes. Its AFT and GLMM parts are superseded by
`make_reference_regression.R`; the priors and shrinkage parts remain until
they are ported to the generated-test format.
