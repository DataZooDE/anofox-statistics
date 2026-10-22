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
| `glmm_inference.test` (generator `make_reference_glmm_inference.R`) | `glmm_fit_agg` Satterthwaite / Wald inference and BLUP SEs, `glmm_tidy_by`, `glmm_glance_by`, `glmm_augment_by` (#218) | `lmerTest::contest1D`, `confint(method = "Wald")`, `ranef(condVar = TRUE)`, `broom.mixed::tidy` / `glance` / `augment` |
| `survival_aft.test` | `aft_fit_agg` (weibull, lognormal, loglogistic, exponential) | `survival::survreg` |

Where the extension follows a different convention from R's default (e.g. centred
R^2 without an intercept, moment-estimated negative-binomial theta, z instead
of t p-values for the Gamma family),
the generator computes the reference by hand with the extension's convention
and the generated test explains the difference in a comment.

The datasets are literal constants in the script (drawn once with a fixed seed
and rounded), so the output does not depend on R's RNG implementation.

## `make_reference_paths_inference.R`

```bash
Rscript validation/generators/make_reference_paths_inference.R
```

Requires `quantreg`, `grpreg`, `MASS` and a built extension
(`build/release`, for the LARS knots). glmnet is not needed: the path and CV
references implement glmnet's documented Gaussian objective, default lambda
grid and `cv.glmnet` aggregates (as anofox-regression's own validation does),
with grpreg as an independent lasso solver. It (re)writes:

| File | Extension functions | R reference |
|------|---------------------|-------------|
| `regularization_path.test` | `ridge_path_agg`, `elasticnet_path_agg`, `lars_path_agg`, `ridge_cv_agg`, `elasticnet_cv_agg` | glmnet objective (closed form / coordinate descent), `cv.glmnet` aggregates with fold ids, `grpreg`, `lm` |
| `quantile_inference.test` | `quantile_fit_agg` (`se` iid / nid / boot), `quantile_process_agg`, `quantile_tidy_by` | `quantreg::rq`, `summary.rq(se = ...)` |
| `predict_se.test` | `predict(model, x, {'se_fit', 'interval', 'level'})` for OLS, WLS, Poisson, binomial, Gamma | `predict.lm(se.fit = TRUE, interval = ...)`, `predict.glm(se.fit = TRUE)` |
| `partial_residuals.test` | `ols_` / `wls_` / `poisson_partial_residuals_by` | `residuals(type = "partial")` (`car::crPlots`), `lsfit` residuals (`car::avPlots`) |

## `make_reference_ts_survival.R`

```bash
Rscript validation/generators/make_reference_ts_survival.R
```

Requires `survival` and `lmtest`. It (re)writes:

| File | Extension functions | R reference |
|------|---------------------|-------------|
| `timeseries_diagnostics.test` | `acf_agg`, `pacf_agg`, `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg`, `breusch_pagan_agg`, `white_test_agg` | `acf`, `pacf`, `Box.test`, `lmtest::dwtest` (statistic), `lmtest::bptest` |
| `survival_curves.test` | `kaplan_meier_agg` (log / log-log / plain), `kaplan_meier_by`, `log_rank_test_agg`, `aft_survival` | `survival::survfit`, `survdiff`, `survreg` + `psurvreg` |

## `make_reference_bayesian.py`

```bash
uv run --no-project --with scikit-learn python3 validation/generators/make_reference_bayesian.py
```

Requires scikit-learn >= 1.8 (earlier releases did not centre `x` in
`predict(return_std=True)`). It (re)writes `regression_bayesian.test`:
`bayesian_ridge_fit_agg` / `ard_fit_agg` (coefficients, intercept, noise and
weight precisions, posterior SDs, iterations, log marginal likelihood; default,
no-intercept, informative-prior and `threshold_lambda` fits) against
`sklearn.linear_model.BayesianRidge` / `ARDRegression`, and the
`*_fit_predict_agg` posterior predictive bands against `predict(X,
return_std=True)` with normal quantiles.

## `make_reference_passive_aggressive.py` and `make_reference_smoothing.R`

```bash
uv run --no-project --with scikit-learn python3 validation/generators/make_reference_passive_aggressive.py
Rscript validation/generators/make_reference_smoothing.R
```

The first (re)writes `regression_passive_aggressive.test`: `passive_aggressive_fit_agg`
(one online pass, PA-I and PA-II, and a multi-pass fit), the window
`passive_aggressive_fit_predict` over cumulative frames and
`passive_aggressive_fit_predict_agg` against scikit-learn's
`PassiveAggressiveRegressor` (`shuffle=False`, `partial_fit` / `fit`). The second
needs `greybox` and (re)writes `lowess_reference.test` (`lowess_agg`, `lowess_by`
against `stats::lowess`) and `lm_dynamic_reference.test` (`lm_dynamic_fit_agg`
against `greybox::lmDynamic(bruteforce = TRUE)` with an explicit lowess `f`).

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
