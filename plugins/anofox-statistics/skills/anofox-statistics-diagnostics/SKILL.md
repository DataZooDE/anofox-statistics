---
name: anofox-statistics-diagnostics
description: >
  Model diagnostics, residual analysis, influence, multicollinearity,
  separation, predictive-accuracy indexes, data reduction, model-selection
  criteria, residual autocorrelation / heteroskedasticity checks, and
  demand-pattern classification in the anofox_statistics DuckDB extension.
  Covers per-row influence tables (ols_augment_by, wls_augment_by,
  poisson/binomial/logistic/negbinom/gamma/tweedie_augment_by: fitted,
  residual, std / studentized residuals, leverage, Cook's distance, DFFITS),
  partial residuals and added-variable data (<model>_partial_residuals_by,
  car::crPlots / avPlots), Variance Inflation Factor (vif / vif_agg) and
  generalized VIF for multi-column terms (gvif_agg, car::vif GVIF^(1/(2df))),
  Belsley-Kuh-Welsch condition indexes and variance-decomposition proportions
  (condition_diagnostic_agg, perturb::colldiag), separation of logistic /
  count fits (detect_separation_agg, separation_check_agg,
  count_sparsity_agg), rms / Hmisc accuracy indexes (somers2_agg,
  concordance_agg, rcorr_cens_agg, lrm_stats_agg, val_prob_agg, gini_md_agg,
  r2_measures), PCA (pca_agg with scale, pca_transform, pca_by), Harrell-style
  data reduction (spearman2_agg, hoeffding_d_agg, varclus_agg, redun_agg,
  box_cox_agg / box_cox_transform / box_cox_inverse), survival and GLS
  residuals (cox_zph_agg, cox_augment_by, aft_augment_by, gls_augment_by),
  AIC / BIC, residual diagnostics (residuals_diagnostics /
  residuals_diagnostics_agg), residual ACF / PACF with the white-noise band
  (acf_agg, pacf_agg, acf_by, pacf_by), Ljung-Box / Box-Pierce, Durbin-Watson
  (with p-value), Breusch-Pagan and White heteroskedasticity tests, AID
  demand-pattern and anomaly classification (aid_agg, aid_anomaly_agg, aid_by,
  aid_anomaly_by), plus tidy / glance and model-aware predict for fitted
  values. Also leave-one-out diagnostics: exact loo_residual / loo_fitted in
  ols / wls / ridge_augment_by (ridge leverage), PRESS, press_rmse and
  loo_r_squared in glance_by (compute_loo), one-step approx_loo_fitted for GLMs.
  Use when checking whether a fitted regression is sound — influential points,
  outliers, nonlinearity, multicollinearity, separation, autocorrelated or
  heteroskedastic residuals, proportional hazards — when measuring
  discrimination / calibration of a prediction, when screening or clustering
  predictors before modelling, or classifying a series' demand regime before
  choosing a model.
version: 0.17.0
user-invocable: false
---

# Anofox Statistics — Diagnostics Cheat Sheet

**Extension:** `anofox_statistics` v0.17.0 | **DuckDB:** v1.5.6 / v2.0

Diagnostics answer "is this fit trustworthy?" — run them after fitting a model (see **anofox-statistics-regression**).

> **New in 0.16.0:** `gvif_agg`, `condition_diagnostic_agg`, the separation checks (`detect_separation_agg`, `separation_check_agg`, `count_sparsity_agg`), the accuracy indexes (`somers2_agg`, `concordance_agg`, `rcorr_cens_agg`, `lrm_stats_agg`, `val_prob_agg`, `gini_md_agg`, `r2_measures`), `pca_agg` with `{'scale': true}`, data reduction (`spearman2_agg`, `hoeffding_d_agg`, `varclus_agg`, `redun_agg`, `box_cox_agg`), Cox / AFT / GLS residual tables. **Behaviour changes:** `vif` / `vif_agg` now come from anofox-regression — a VIF above 10^4 is its finite value (was `inf`), and `inf` marks only the *aliased* columns (constant column, exact linear dependency; every column when rows <= columns) while the others keep a finite VIF (`car::vif` on the design without them); `vif_agg` drops a row with a NaN as a whole (was a dimension-mismatch error). `val_prob_agg` with probabilities of exactly 0 / 1 now gives `u`, `u_p`, `q` as `rms::val.prob` (e.g. `u = inf`) instead of NULL.

## Functions

| Function | Type | Purpose |
|---|---|---|
| `vif`, `vif_agg` | scalar / aggregate | Variance Inflation Factor per predictor — detect multicollinearity |
| `gvif_agg(x[, {'terms': [...]}])` | aggregate | Generalized VIF per term (`car::vif` for factor / spline terms): `LIST<STRUCT(term, df, gvif, gvif_adj, n_observations)>`; columns with the same `terms` label form one term, `gvif_adj = GVIF^(1/(2·df))` |
| `condition_diagnostic_agg(x[, {'fit_intercept'}])` | aggregate | Belsley-Kuh-Welsch (`perturb::colldiag`): `condition_indices`, `variance_proportions` (one list per component, intercept first), `scaled_condition_number`, `condition_number` (`kappa(X, exact = TRUE)`, unscaled), `severity`, `warning` |
| `detect_separation_agg(y, x[, {'fit_intercept'}])` | aggregate | Exact LP check (`detectseparation`): `separated`, `infinite` (±inf per diverging coefficient, intercept first), `direction`; `y` must be 0/1 |
| `separation_check_agg(y, x)`, `count_sparsity_agg(y, x)` | aggregate | Per-predictor heuristics (binary `y`; count `y` with > 90% zeros): `has_separation`, `separated_predictors` (1-based), `separation_types` (`complete` / `quasi` / `monotonic_response`), `warning` |
| `aic(rss, n, k)` | scalar | Akaike Information Criterion for model selection |
| `bic(rss, n, k)` | scalar | Bayesian Information Criterion for model selection |
| `jarque_bera(values[])`, `jarque_bera_agg(value)` | scalar / aggregate | Jarque-Bera normality test (`statistic`, `p_value`, `skewness`, `kurtosis`, `n`) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | scalar / aggregate | Residual analysis: `raw`, `standardized`, `studentized`, `leverage` arrays |
| `ols_augment_by`, `wls_augment_by`, `<glm>_augment_by` | table macro | broom `augment`: every source row + `model_id` / `model_name` (the method; no `id`, the group column is among the source columns), `row_id`, `fitted`, `residual`, `resid_type`, `std_residual` (R `rstandard`), `stud_residual` (`rstudent`), `leverage` (`hatvalues`), `cooks_d`, `dffits`, `weight`, `is_training`, `n_params` (`obs` schema); OLS / WLS also `loo_residual` (`residual / (1 − leverage)`, the PRESS residual) and `loo_fitted` (prediction without the row), NULL where leverage = 1; GLMs instead `approx_loo_fitted` / `approx_loo_residual` (one-step approximation, R `influence()$coefficients`) |
| `ridge_augment_by` | table macro | Same `obs` columns for Ridge: `leverage` = ridge hat diagonal (sums to effective df), exact `loo_residual` / `loo_fitted` at the same `alpha`; `std_residual`, `stud_residual`, `cooks_d`, `dffits` NULL |
| `cox_augment_by(source, group_col, time_col, x_cols, event_col)` | table macro | `obs` rows: `residual` = martingale (`resid_type = 'martingale'`), then `deviance_residual`, `score_residuals`, `dfbeta`, `dfbetas`, `schoenfeld_residuals`, `scaled_schoenfeld_residuals` (`residuals.coxph`) |
| `cox_zph_agg(time, x, event[, weight][, options])` | aggregate | Grambsch-Therneau proportional-hazards test (`cox.zph`): `LIST<STRUCT(term, statistic, df, p_value)>`, per covariate and `GLOBAL` |
| `aft_augment_by(source, group_col, time_col, x_cols, event_col)`, `gls_augment_by(...)` | table macro | `obs` rows for AFT (`standardized_residual`, `cox_snell_residual`, score contributions) and GLS (Pearson `std_residual`, `normalized_residual`, `se_fit`) — signatures in **anofox-statistics-regression** |
| `<model>_partial_residuals_by` | table macro | `ols`, `wls`, six GLMs: one row per source row and term — `term_index`, `term`, `x`, `component`, `partial_residual` (crPlots), `x_residual`, `y_residual` (avPlots; NULL for GLMs) |
| `ols_augment_agg`, `wls_augment_agg(y, x, weight)`, `ridge_augment_agg`, `binomial_augment_agg`, `logistic_augment_agg`, `poisson_augment_agg`, `negbinom_augment_agg`, `gamma_augment_agg`, `tweedie_augment_agg`, `glmm_augment_agg` | aggregate | The aggregates behind the `*_augment_by` macros: `(y, x[, options] ORDER BY …)` returns a LIST with one STRUCT of the `obs` columns per row, in `ORDER BY` order. Prefer the macro; use these inline (`GROUP BY`, or a window over the whole partition) |
| `ols_partial_residuals_agg`, `wls_partial_residuals_agg`, `binomial_partial_residuals_agg`, `logistic_partial_residuals_agg`, `poisson_partial_residuals_agg`, `negbinom_partial_residuals_agg`, `gamma_partial_residuals_agg`, `tweedie_partial_residuals_agg` | aggregate | The aggregates behind `<model>_partial_residuals_by`: a LIST per row of a LIST with one STRUCT per term (`term_index`, `x`, `partial_residual`, `component`, `x_residual`, `y_residual`) |
| `acf_agg(x[, max_lag] ORDER BY t)`, `pacf_agg(…)` | aggregate | `LIST<STRUCT(lag, acf\|pacf, conf_low, conf_high)>`; band = ±qnorm(0.975)/√n; `max_lag` default `floor(10·log10(n))` |
| `acf_by`, `pacf_by` `(source, group_col, value_col, order_col[, max_lag := …])` | table macro | One ACF / PACF curve per group (`curve` schema, `curve_type` `'acf'` / `'pacf'`, `x` = lag, `y_low`/`y_high` = band) |
| `ljung_box_agg`, `box_pierce_agg`, `durbin_watson_agg` | aggregate | Residual autocorrelation tests (unified test fields; see **anofox-statistics-tests**); `durbin_watson_agg(resid, x ORDER BY t)` adds the `dwtest` p-value |
| `breusch_pagan_agg(resid, x[, studentize])`, `white_test_agg(resid, x)` | aggregate | Heteroskedasticity tests (`lmtest::bptest`) |
| `tidy(model[, names])` | scalar on a model STRUCT | Per-term table: `term`, `estimate`, `std_error`, `statistic`, `p_value`, `conf_low`, `conf_high` (intercept row first, `'(Intercept)'`) — `unnest(…, recursive := true)` |
| `glance(model)` | scalar on a model STRUCT | One-row summary of the model's scalar fields (`r_squared`, `aic`, `n_observations`, GLM `family`/`link`, …) — `unnest(glance(m))` |
| `predict(model, x)` | scalar on a model STRUCT | Fitted value for one row; use it to build `y_hat` for the residual diagnostics |
| `aid_agg`, `aid_by` | aggregate / table macro | Demand classification: `demand_type` is `regular` or `intermittent`, plus distribution, zero proportion, stockout / new / obsolete product and outlier counts |
| `aid_anomaly_agg`, `aid_anomaly_by` | aggregate / table macro | Per-observation flags: `stockout`, `new_product`, `obsolete_product`, `high_outlier`, `low_outlier` |

## Accuracy indexes (rms / Hmisc)

Aggregates over each row's outcome and prediction, so they score **any** model (this extension's or an external one), per `GROUP BY` group or window. For resampling-corrected (optimism) versions use `validate_agg` / `calibrate_agg` (**anofox-statistics-regression**); ROC / PR curves and `auc_agg` / `brier_agg` are in **anofox-statistics-tests**.

| Function | Returns | R |
|---|---|---|
| `somers2_agg(y01, prob[, weight])` | `c` (ROC area), `dxy`, `n` | `Hmisc::somers2` |
| `concordance_agg(y, x[, weight])` | pair counts, `c`, `dxy`, `gamma`, `tau_a` (ordinal / continuous `y`) | `survival::concordance` |
| `rcorr_cens_agg(time, event, x)` | Harrell's `c`, `dxy`, `sd`, pair counts — larger `x` = longer survival, so pass the **negated** risk score / Cox `lp` | `Hmisc::rcorr.cens` |
| `lrm_stats_agg(y01, lp, df[, weight])` | the `lrm()$stats` vector from a logistic linear predictor: `model_lr`, `p_value`, `c`, `dxy`, `r2` (Nagelkerke), `r2_n`, `r2_adj_n`, `ess`, `brier`, `g`, `gr`, `gp` | `rms::lrm` |
| `val_prob_agg(y01, prob)` | external validation: `c`, `dxy`, `d`, `u` (+ chi-square / p), `q`, `brier`, calibration `intercept` / `slope`, `emax`, `e90`, `eavg`, Spiegelhalter `s_z` / `s_p` | `rms::val.prob` |
| `gini_md_agg(x)`, `r2_measures(lr, p, n[, ess])` | Gini mean difference (the g-index of an `lp`); Maddala R² and its d.f.-adjusted versions | `Hmisc::GiniMd`, `R2Measures` |

`y01` is BOOLEAN or numeric 0/1 (anything else is an error); `df` is the constant number of non-intercept coefficients. Get `lp` with `predict(model, x, {'type': 'link'})`.

## Data reduction, transforms and PCA (before looking at the outcome)

| Function | Purpose | Key options / fields |
|---|---|---|
| `spearman2_agg(y, x[, {'p': 1\|2}])` | Generalized Spearman ρ² of `y` with one predictor (`p = 2` adds squared ranks: non-monotone; VARCHAR `x` = categorical) — screen where to spend d.f. (`Hmisc::spearman2`) | `rho2`, `adjusted_rho2`, `df1`, `p_value` + unified test fields |
| `hoeffding_d_agg(x, y)` | Hoeffding's D (×30, any form of dependence) with p-value (`Hmisc::hoeffd`); O(n²) | `d`, `p_value` |
| `varclus_agg(x[], options)` | Variable clustering (`Hmisc::varclus`, R `hclust` conventions) | `similarity` (`'spearman'` default, `'pearson'`, `'hoeffding'`), `trans` (`'square'` / `'abs'` / `'none'`), `method` (`'complete'`, `'average'`, `'ward.d2'`, …), `cut_height`, `feature_names` → `variables`, `similarity_matrix`, `merges`, `height`, `dendrogram_order`, `clusters` (NULL without `cut_height`) |
| `redun_agg(x[], options)` | Redundancy analysis: drop features predictable (spline / dummy expansions) from the others (`Hmisc::redun`) | `r2` (0.9), `type` (`'ordinary'` / `'adjusted'`), `nk` (3), `types` per feature (`'spline'`, `'linear'`, `'categorical'`), `feature_names` → `kept`, `removed`, `r_squared`, `r_squared_first` |
| `box_cox_agg(y[, x[][, options]])` | Box-Cox profile likelihood of λ for positive `y` on `x` (`MASS::boxcox`) | `lambdas` grid (`seq(-2,2,0.1)`), `confidence_level` → `lambda_hat`, `ci_lower`, `ci_upper`, `log_likelihood`; apply with `box_cox_transform(y, λ)` / back with `box_cox_inverse(z, λ)` (a back-transformed fit is a median) |
| `pca_agg(x[][, {'n_components', 'scale'}])`, `pca_transform(model, x)`, `pca_by(source, group_col, x_cols[, options := …])` | PCA as `prcomp` (centred; `'scale': true` = correlation PCA, `prcomp(scale. = TRUE)`, stores column sds in `scale` and `pca_transform` applies them) | `sdev`, `proportion_of_variance`, `loadings[k]` (component k), `center`, `scale`; sign fixed so the largest loading is positive |

All skip rows with NULL / non-finite values and return NULL when the data cannot support a result (too few rows, constant variable). `varclus_agg` uses complete rows (Hmisc uses pairwise-complete correlations). The STRUCT field is `similarity_matrix` (`similarity` is a keyword on DuckDB 2.0).

## Interpretation guidance

- **VIF:** > 5 warns of multicollinearity, > 10 is severe — consider dropping/combining collinear predictors or switching to Ridge/Elastic Net.
- **GVIF:** compare `gvif_adj²` with the VIF thresholds (a 1-df term has `gvif_adj² = VIF`); pass `terms` so the dummies of a factor or the columns of an `rcs` basis form one term — their per-column VIFs are high by construction.
- **Condition indexes:** a component with condition index > 30 on which **two or more** coefficients have `variance_proportions` > 0.5 is a near dependency among those coefficients (intercept counts: index 1). `severity` buckets `scaled_condition_number` (< 30 well conditioned, < 100 moderate, < 1000 high). `condition_number` is in the data's units — use the scaled one.
- **Separation:** check `detect_separation_agg(...).separated` before reading a logistic fit; huge coefficients / SEs are the symptom. `separation_check_agg` only sees single predictors. Remedy: drop / merge the level, or a penalized / prior fit (`glm_lambda`, `prior`; see **anofox-statistics-regression**). For Poisson / NB use `count_sparsity_agg`.
- **Accuracy:** `c` = 0.5 is chance, `dxy = 2(c − 0.5)`. Calibration `slope` < 1 ⇒ overfitting (predictions too extreme); `intercept` ≠ 0 ⇒ miscalibration in the large; small `u_p` ⇒ unreliable probabilities. Apparent indexes on the training rows are optimistic — validate.
- **Data reduction:** screen and cluster predictors with `spearman2_agg` / `varclus_agg` / `redun_agg` **blind to the outcome's form**, then combine a cluster into one score (e.g. its first PC with `pca_agg`, scaled when units differ). Box-Cox: round λ to an interpretable value (0 = log, 0.5, 1) inside the interval.
- **Cox PH:** small `cox_zph_agg` p-value (per covariate or `GLOBAL`) ⇒ proportional hazards violated (stratify or add a time interaction); a trend in the martingale `residual` of `cox_augment_by` against a covariate ⇒ non-linear effect. AFT: `cox_snell_residual` should look Exp(1).
- **AIC / BIC:** lower is better; compare *nested or same-data* models. BIC penalizes complexity harder than AIC. GLM result structs already carry an `aic` field; AFT and ALM results carry both `aic` and `bic`.
- **Residuals:** `standardized`/`studentized` residuals with |value| > 2–3 flag potential outliers; high `leverage` flags influential x-positions. Structure in residuals ⇒ mis-specified model.
- **`tidy` inference columns** are NULL when the model has no inference (fit without `compute_inference`, or PLS/LARS; quantile only with its `se` option). The intercept row (first) carries its own inference. Per group use `<model>_tidy_by` (batch skill).
- **Leave-one-out (OLS / WLS / Ridge):** `ols_glance_by` / `wls_glance_by` / `ridge_glance_by` report `press`, `press_rmse` and `loo_r_squared` (predicted R², Q²); a big gap to `r_squared` means the fit leans on individual rows, whose large `|loo_residual|` in `*_augment_by` point to them. Exact, from the leverage, no refits. A row with leverage 1 (single-observation dummy level) has NULL `loo_*` and makes the group's `press` metrics NULL. GLM `approx_loo_*` is an approximation (good unless leverage or the Pearson residual is large); there is no GLM PRESS.
- **Influence (`*_augment_by`):** `|stud_residual| > 3` flags outliers, `leverage > 2·n_params/n` high-leverage points, `cooks_d > 4/n` (or > 0.5–1) influential rows, `|dffits| > 2·sqrt(n_params/n)`. Rows outside training (`split` other than `'train'` / `'training'` in any case, or NULL; NULL y) get `fitted` but NULL diagnostics. GLMs: `fitted` on the response scale, `residual` is the deviance residual (`resid_type = 'deviance'`); penalized GLM fits (`glm_lambda`, `prior`) are rejected.
- **Partial residuals:** curvature of `partial_residual` against `x` ⇒ the term needs a transform; the slope of `y_residual` on `x_residual` through the origin equals the coefficient.
- **Autocorrelation:** ACF lags outside `[conf_low, conf_high]`, a small Ljung-Box p-value or Durbin-Watson well below 2 ⇒ residuals are autocorrelated (standard errors too small) — add lags / time terms or use a time-series model. `ORDER BY` goes **after** the constant arguments: `acf_agg(resid, 12 ORDER BY t)`.
- **Heteroskedasticity:** a small Breusch-Pagan / White p-value ⇒ use robust SEs (`'hc_type': 'hc3'` on OLS / WLS) or WLS. Pass the residuals of a model with an intercept and the same regressors.
- **AID demand classes:** use before forecasting/choosing a model — intermittent series (many zeros; threshold `intermittent_threshold`, default 0.3) need count/intermittent methods rather than plain OLS.

## Return fields

- `residuals_diagnostics_agg(y, y_hat[, x] ORDER BY …)` → STRUCT with `raw DOUBLE[]`, `standardized DOUBLE[]`, `studentized DOUBLE[]`, `leverage DOUBLE[]`, in `ORDER BY` order. Without `x`, `standardized`, `studentized` and `leverage` are NULL. For per-row diagnostics as a table with Cook's distance, prefer `ols_augment_by`.
- `vif_agg([x1, x2, …])` → per-predictor VIF array (`vif(X)` is the scalar form on column-major arrays).

## Worked examples

```sql
-- Residual diagnostics from in-sample OLS predictions
WITH preds AS (
  SELECT
    unnest([120.0,155.0,190.0,225.0,265.0,300.0])::DOUBLE AS actual,
    unnest(predict(
      [[50.0,65.0,80.0,95.0,110.0,125.0]]::DOUBLE[][],
      (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
               [[50.0,65.0,80.0,95.0,110.0,125.0]])).coefficients,
      (ols_fit([120.0,155.0,190.0,225.0,265.0,300.0],
               [[50.0,65.0,80.0,95.0,110.0,125.0]])).intercept
    )) AS yhat
)
SELECT (residuals_diagnostics_agg(actual, yhat)).raw AS raw_residuals FROM preds;
```

```sql
-- Same residual check with an aggregate fit: fitted values from predict(model, x)
CREATE OR REPLACE TABLE houses AS SELECT * FROM (VALUES
  (50.0,120.0),(65.0,155.0),(80.0,190.0),(95.0,225.0),(110.0,265.0),(125.0,300.0)
) t(sqm, price);
WITH m AS (SELECT ols_fit_agg(price, [sqm]) AS fit FROM houses)
SELECT (residuals_diagnostics_agg(price, predict(fit, [sqm]))).raw AS raw_residuals
FROM houses, m;

-- Coefficient table with inference and a one-row model summary
SELECT unnest(tidy(ols_fit_agg(price, [sqm], {'compute_inference': true}), ['sqm']), recursive := true)
FROM houses;
SELECT unnest(glance(ols_fit_agg(price, [sqm]))) FROM houses;
```

```sql
-- Influential rows per group (Cook's distance), from a per-row augment table
CREATE OR REPLACE TABLE mt_demo AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS grp, i::DOUBLE AS x1,
       ((i * 3) % 7)::DOUBLE AS x2, 1.0 + 0.5 * i - 0.3 * ((i * 3) % 7) + sin(i) AS y,
       round(exp(0.2 + 0.05 * i))::DOUBLE AS y_count
FROM range(1, 41) t(i);
SELECT grp, id, round(fitted, 3) AS fitted, round(stud_residual, 3) AS stud, round(leverage, 3) AS h, round(cooks_d, 3) AS cooks_d
FROM ols_augment_by('mt_demo', grp, y, [x1, x2])
QUALIFY row_number() OVER (PARTITION BY grp ORDER BY cooks_d DESC) <= 3
ORDER BY grp, cooks_d DESC;

-- Poisson: deviance residuals; the held-out rows get a prediction but no diagnostics
SELECT id, y_count, round(fitted, 2) AS fitted, round(residual, 3) AS dev_resid, is_training
FROM poisson_augment_by('mt_demo', grp, y_count, [x1], split := CASE WHEN id > 36 THEN 'test' ELSE 'train' END)
WHERE grp = 'A' AND id > 30 ORDER BY id;

-- Component-plus-residual data for one term
SELECT grp, row_id, x, round(partial_residual, 3) AS partial_residual, round(component, 3) AS component
FROM ols_partial_residuals_by('mt_demo', grp, y, [x1, x2], names := ['trend', 'cycle'])
WHERE term = 'cycle' ORDER BY grp, row_id LIMIT 5;
```

```sql
-- The aggregate behind ols_augment_by, inline: one list per group, rows in ORDER BY order
SELECT grp, list_count(a) AS n_rows, round(a[1].leverage, 4) AS first_leverage, round(a[1].cooks_d, 4) AS first_cooks_d
FROM (SELECT grp, ols_augment_agg(y, [x1, x2] ORDER BY id) AS a FROM mt_demo GROUP BY grp)
ORDER BY grp;
```

```sql
-- Residual autocorrelation and heteroskedasticity of a per-group fit
CREATE OR REPLACE TABLE resid_tbl AS
SELECT grp, id, x1, x2, y - fitted AS resid FROM ols_augment_by('mt_demo', grp, y, [x1, x2]);
SELECT grp,
       (ljung_box_agg(resid, 5 ORDER BY id)).p_value  AS ljung_box_p,
       (durbin_watson_agg(resid ORDER BY id)).statistic AS dw,
       (breusch_pagan_agg(resid, [x1, x2])).p_value     AS bp_p
FROM resid_tbl GROUP BY grp ORDER BY grp;
-- ACF with the white-noise band, one curve per group (curve schema)
SELECT model_id, x AS lag, round(y, 3) AS acf, round(y_high, 3) AS band
FROM acf_by('resid_tbl', grp, resid, id, max_lag := 4);
```

```sql
-- Collinearity: condition indexes / variance proportions, and GVIF with a 2-column factor term
CREATE OR REPLACE TABLE coll_demo AS
SELECT i AS id, i::DOUBLE AS x1, 2.0 * i + (i % 3) * 0.1 AS x2, ((i * 7) % 5)::DOUBLE AS x3,
       (i % 3 = 1)::DOUBLE AS region_b, (i % 3 = 2)::DOUBLE AS region_c,
       (sin(0.9 * i) + 0.6 * cos(1.7 * i) + 0.3 > 0)::INTEGER AS y01,
       round(1 / (1 + exp(-(0.2 + 1.1 * sin(0.9 * i)))), 2) AS prob
FROM range(1, 41) t(i);
SELECT r.severity, round(r.scaled_condition_number, 1) AS kappa_scaled,
       list_transform(r.variance_proportions[len(r.variance_proportions)], lambda v: round(v, 3)) AS worst_component
FROM (SELECT condition_diagnostic_agg([x1, x2, x3]) AS r FROM coll_demo);
SELECT e.term, e.df, round(e.gvif, 3) AS gvif, round(e.gvif_adj, 3) AS gvif_adj
FROM (SELECT unnest(gvif_agg([x1, x3, region_b, region_c],
                             {'terms': ['x1', 'x3', 'region', 'region']})) AS e FROM coll_demo);

-- Separation: y = 1 exactly when x1 + x2 > 5 (only the exact check sees the combination)
SELECT (detect_separation_agg(y, [x1, x2])).separated AS separated,
       (separation_check_agg(y, [x1, x2])).has_separation AS single_predictor_flag
FROM (VALUES (1.0, 1.0, 0.0), (2.0, 1.0, 0.0), (3.0, 1.0, 0.0), (4.0, 1.0, 0.0),
             (1.0, 4.0, 0.0), (2.0, 4.0, 1.0), (3.0, 4.0, 1.0), (4.0, 4.0, 1.0),
             (2.0, 2.0, 0.0), (3.0, 3.0, 1.0), (1.0, 3.0, 0.0), (4.0, 2.0, 1.0)) t(x1, x2, y);
```

```sql
-- Accuracy of external probabilities, and lrm stats of an in-database logistic fit
SELECT round(s.c, 4) AS c, round(s.dxy, 4) AS dxy,
       round(v.slope, 3) AS cal_slope, round(v.intercept, 3) AS cal_intercept, round(v.emax, 3) AS emax
FROM (SELECT somers2_agg(y01, prob) AS s, val_prob_agg(y01, prob) AS v FROM coll_demo);
WITH m AS (SELECT logistic_fit_agg(y01, [x1, x3]) AS fit FROM coll_demo),
     scored AS (SELECT y01, predict(m.fit, [x1, x3], {'type': 'link'}) AS lp FROM coll_demo, m)
SELECT round(r.model_lr, 3) AS model_lr, round(r.c, 4) AS c, round(r.r2, 4) AS r2, round(r.brier, 4) AS brier
FROM (SELECT lrm_stats_agg(y01, lp, 2) AS r FROM scored);
-- Harrell's C for censored times: negate a risk score (higher risk = shorter survival)
SELECT round((rcorr_cens_agg(time, event, -risk)).c, 4) AS harrell_c
FROM (SELECT i, exp(1 + 0.5 * cos(0.8 * i)) AS time, i % 4 <> 0 AS event, -cos(0.8 * i) + 0.2 * sin(3 * i) AS risk
      FROM range(40) t(i));
```

```sql
-- Data reduction before modelling: screen, cluster, find redundant features, choose a transform
CREATE OR REPLACE TABLE cars AS
SELECT i AS id, 10 + i * 0.6 + sin(i) * 2 AS mpg, 100 + i * 9 + cos(i * 1.7) * 15 AS disp,
       60 + i * 6 + sin(i * 0.9) * 12 AS hp, 1.5 + i * 0.07 + cos(i * 2.3) * 0.2 AS wt,
       16 + sin(i * 0.5) * 2 AS qsec
FROM range(1, 41) t(i);
SELECT round((spearman2_agg(mpg, hp, {'p': 2})).rho2, 4) AS rho2_hp,
       round((hoeffding_d_agg(qsec, wt)).d, 4) AS hoeffding_d
FROM cars;
SELECT v.variables, v.clusters, list_transform(v.height, lambda h: round(h, 3)) AS height
FROM (SELECT varclus_agg([disp, hp, wt, qsec],
                         {'similarity': 'spearman', 'cut_height': 0.3,
                          'feature_names': ['disp', 'hp', 'wt', 'qsec']}) AS v FROM cars);
SELECT r.removed, r.kept
FROM (SELECT redun_agg([disp, hp, wt, qsec], {'r2': 0.8, 'feature_names': ['disp', 'hp', 'wt', 'qsec']}) AS r FROM cars);
SELECT round(b.lambda_hat, 2) AS lambda, round(b.ci_lower, 2) AS lo, round(b.ci_upper, 2) AS hi,
       round(box_cox_inverse(box_cox_transform(20.0, 0.5), 0.5), 6) AS roundtrip
FROM (SELECT box_cox_agg(mpg, [wt, hp]) AS b FROM cars);
-- Correlation PCA of a cluster (features on different scales) and one row's scores
SELECT list_transform(m.proportion_of_variance, lambda p: round(p, 4)) AS prop_var,
       list_transform(pca_transform(m, [300.0, 200.0, 3.0]), lambda z: round(z, 3)) AS scores
FROM (SELECT pca_agg([disp, hp, wt], {'scale': true}) AS m FROM cars);
```

```sql
-- Cox PH: proportional-hazards test and martingale residuals per group
CREATE OR REPLACE TABLE pts AS
SELECT i AS id, CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS site, 40 + (i * 7) % 30 AS age,
       (i % 3 = 0)::DOUBLE AS treated, round(exp(4 - 0.02 * ((i * 7) % 30) + 0.8 * sin(i)), 1) AS days,
       (i % 5 <> 0)::DOUBLE AS died
FROM range(1, 61) t(i);
SELECT z.term, round(z.statistic, 3) AS chisq, round(z.p_value, 4) AS p
FROM (SELECT unnest(cox_zph_agg(days, [age, treated], died)) AS z FROM pts);
SELECT site, round(age / 10) * 10 AS age_band, round(avg(residual), 3) AS mean_martingale
FROM cox_augment_by('pts', site, days, [age, treated], died)
GROUP BY ALL ORDER BY site, age_band;
```

```sql
-- Multicollinearity check (x3 is nearly x1 + x2)
SELECT vif_agg([x1, x2, x3]) AS vifs
FROM (SELECT i::DOUBLE AS x1, (i % 7)::DOUBLE AS x2, (i + (i % 7) + (i % 2) * 0.1)::DOUBLE AS x3
      FROM range(1, 51) t(i));

-- Model selection from RSS
SELECT aic(12.5, 100, 3) AS aic, bic(12.5, 100, 3) AS bic;

-- AID demand-pattern classification per SKU
SELECT sku, (aid_agg(demand)).demand_type AS demand_type
FROM (SELECT 'sku' || (i % 2) AS sku,
             CASE WHEN i % 2 = 0 THEN 10.0 + (i % 4) WHEN i % 5 = 0 THEN 3.0 ELSE 0.0 END AS demand
      FROM range(1, 61) t(i))
GROUP BY sku ORDER BY sku;

-- AID per-observation flags: unnest ONCE in a subquery, then read the struct fields
SELECT sku, t, flag.stockout, flag.high_outlier
FROM (SELECT sku,
             unnest(list(t ORDER BY t))                 AS t,
             unnest(aid_anomaly_agg(demand ORDER BY t)) AS flag
      FROM (SELECT 'sku' || (i % 2) AS sku, i AS t,
                   CASE WHEN i % 2 = 0 THEN 10.0 + (i % 4) WHEN i = 7 THEN 0.0 ELSE 5.0 END AS demand
            FROM range(1, 41) t(i))
      GROUP BY sku)
WHERE flag.stockout OR flag.high_outlier
ORDER BY sku, t;
```

Read `aid_anomaly_agg` results with the subquery pattern above (or use `aid_anomaly_by`),
not with one `(unnest(flags)).field` per field in the same `SELECT`: that form has
crashed or returned corrupted results in some ADBC-based clients (e.g. Bruin).
