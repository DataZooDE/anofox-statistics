# Anofox Statistics — DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.5%20LTS%20%7C%20v1.5.x-brightgreen.svg)](https://duckdb.org)
[![Build & WASM tests](https://github.com/DataZooDE/anofox-statistics/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main)](https://github.com/DataZooDE/anofox-statistics/actions/workflows/MainDistributionPipeline.yml)
[![Claude Code Plugin](https://img.shields.io/badge/Claude_Code-AI_skills_plugin-8A2BE2)](#-claude-code-skills-ai-pair-programming)

> [!TIP]
> **🤖 Built for AI pair-programming.** anofox-statistics ships an installable **[Claude Code plugin](#-claude-code-skills-ai-pair-programming)** — four skills that give your AI assistant verified, hallucination-free knowledge of the full SQL API. Describe the analysis in plain English; Claude writes correct `ols_fit_agg`, GLM, and batch queries.
>
> Install with `/plugin marketplace add DataZooDE/anofox-statistics` → `/plugin install anofox-statistics@anofox-statistics`.

> [!IMPORTANT]
> This extension is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Claude Code Skills](#-claude-code-skills-ai-pair-programming)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Support](#-support)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Key Features

Every function is unprefixed (`ols_fit_agg`, `theil_sen_fit`, …) and takes model options as a MAP/STRUCT literal (`{'fit_intercept': true, 'compute_inference': true}`). Regression aggregates follow `fn(y, [x1, x2, …][, options])`; WLS adds a positional weight (`wls_fit_agg(y, x, weight[, options])`). Most model families come in several forms:

| Form | Example | Use |
|------|---------|-----|
| Scalar fit | `ols_fit(y_array, x_columns[, options])` | Fit on literal arrays (`x` is column-major) |
| Aggregate fit | `ols_fit_agg(y, [x1, x2][, options])` | One model per table or per `GROUP BY` group |
| Window fit-predict | `ols_fit_predict(y, [x1, x2][, options]) OVER (ORDER BY t ROWS BETWEEN ... AND CURRENT ROW)` | Expanding / rolling in-sample fits; prediction for the last row of each frame |
| Fit-predict aggregate | `ols_fit_predict_agg(y, [x1, x2][, options])` | Fit on rows with non-NULL `y`, predict every row of the group |
| Table macro | `ols_fit_predict_by('table', group, y, [x1, x2][, options])` | Per-group fit + predict in one call |

### Regression Methods

| Method | Functions | Description | Docs |
|--------|-----------|-------------|------|
| OLS | `ols_fit`, `ols_fit_agg` | Ordinary Least Squares | [ols](docs/api/regression/ols.md) |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2 regularization | [ridge](docs/api/regression/ridge.md) |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | Combined L1+L2 regularization | [elasticnet](docs/api/regression/elasticnet.md) |
| LARS / Lasso | `lars_fit_agg` | Least Angle Regression | [lars](docs/api/regression/lars.md) |
| WLS | `wls_fit`, `wls_fit_agg` | Weighted Least Squares | [wls](docs/api/regression/wls.md) |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive Least Squares (online) | [rls](docs/api/regression/rls.md) |
| Huber | `huber_fit`, `huber_fit_agg` | Robust M-estimator (reports robust `scale` and `n_outliers`) | [huber](docs/api/regression/huber.md) |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | Robust consensus regression (reports `n_inliers` and `n_trials`) | [ransac](docs/api/regression/ransac.md) |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Robust nonparametric regression via spatial median over subsamples | [theil_sen](docs/api/regression/theil_sen.md) |
| BLS / NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded / non-negative least squares | [bls](docs/api/regression/bls.md) |
| PLS | `pls_fit_agg`, `pls_fit_predict_agg` | Partial Least Squares <!-- TODO(lead): verify after *_fit_agg lands --> | [pls](docs/api/regression/pls.md) |
| Isotonic | `isotonic_fit_agg`, `isotonic_fit_predict_agg` | Monotonic regression <!-- TODO(lead): verify after *_fit_agg lands --> | [isotonic](docs/api/regression/isotonic.md) |
| Quantile | `quantile_fit_agg`, `quantile_fit_predict_agg` | Quantile / median regression <!-- TODO(lead): verify after *_fit_agg lands --> | [quantile](docs/api/regression/quantile.md) |

### Generalized Linear, Survival and Hierarchical Models

| Method | Functions | Description | Docs |
|--------|-----------|-------------|------|
| Poisson | `poisson_fit_agg` | GLM for count data | [poisson](docs/api/glm/poisson.md) |
| Binomial | `binomial_fit_agg` | GLM for success-rate data (logit / probit / cloglog links) | [binomial](docs/api/glm/binomial.md) |
| Logistic | `logistic_fit_agg` | Binary classification (logit link); reports accuracy and the threshold used | [logistic](docs/api/glm/logistic.md) |
| Negative Binomial | `negbinom_fit_agg` | GLM for overdispersed counts (dispersion estimated jointly) | [negbinom](docs/api/glm/negbinom.md) |
| Gamma | `gamma_fit_agg` | GLM for strictly positive continuous outcomes | [gamma](docs/api/glm/gamma.md) |
| Tweedie | `tweedie_fit_agg` | GLM for zero-inflated positive-skew outcomes | [tweedie](docs/api/glm/tweedie.md) |
| GLM priors | `prior` option | Normal / Laplace priors on GLM coefficients | [priors](docs/api/glm/priors.md) |
| ALM | `alm_fit_agg` | Augmented Linear Model with 24 error distributions | [alm](docs/api/glm/alm.md) |
| AFT survival | `aft_fit_agg`, `aft_cdf`, `aft_quantile` | Accelerated failure time models with right censoring; CDF and quantile helpers | [aft](docs/api/survival/aft.md) |
| Mixed effects | `glmm_fit_agg`, `glmm_fit_by` | Random intercepts / slopes over grouping factors | [glmm](docs/api/glm/glmm.md) |
| EB shrinkage | `eb_shrink_agg`, `eb_shrink_by` | Empirical-Bayes partial pooling of per-group estimates | [eb_shrink](docs/api/glm/eb_shrink.md) |

### Prediction

| Function | Description | Docs |
|----------|-------------|------|
| `predict(x_new, coefficients, intercept)` | Score new points from fitted coefficients (column-major `x_new`) | [API reference](docs/API_REFERENCE.md) |
| `ols_fit_predict`, `ridge_fit_predict`, `elasticnet_fit_predict`, `wls_fit_predict`, `rls_fit_predict`, `huber_fit_predict`, `ransac_fit_predict`, `theil_sen_fit_predict` | Window aggregates (`OVER (...)`): fit on the frame and return `yhat`, `yhat_lower`, `yhat_upper` for the last row of the frame (use frames ending at `CURRENT ROW`) | [fit_predict_window](docs/api/regression/fit_predict_window.md) |
| `ols_fit_predict_agg`, `ridge_fit_predict_agg`, `elasticnet_fit_predict_agg`, `wls_fit_predict_agg`, `rls_fit_predict_agg`, `huber_fit_predict_agg`, `ransac_fit_predict_agg`, `theil_sen_fit_predict_agg`, `bls_fit_predict_agg`, `alm_fit_predict_agg`, `poisson_fit_predict_agg`, `pls_fit_predict_agg`, `isotonic_fit_predict_agg`, `quantile_fit_predict_agg` | Fit on rows with non-NULL `y`, return a LIST of per-row predictions (`y`, `yhat`, `yhat_lower`, `yhat_upper`, `is_training`) | [fit_predict_agg](docs/api/regression/fit_predict_agg.md) |

### Fit-Predict Table Macros (`*_fit_predict_by`)

Per-group model fitting and prediction with a single call. `source` is a table name string; the result is every source row plus `yhat`, `yhat_lower`, `yhat_upper`, `is_training`. See [table_macros](docs/api/macros/table_macros.md).

| Macro | Description |
|-------|-------------|
| `ols_fit_predict_by` | OLS per-group fit + predict |
| `ridge_fit_predict_by` | Ridge per-group fit + predict |
| `elasticnet_fit_predict_by` | Elastic Net per-group fit + predict |
| `wls_fit_predict_by` | WLS per-group fit + predict (extra `weight_col` argument) |
| `rls_fit_predict_by` | RLS per-group fit + predict |
| `huber_fit_predict_by` | Huber robust per-group fit + predict |
| `ransac_fit_predict_by` | RANSAC robust per-group fit + predict |
| `theil_sen_fit_predict_by` | Theil-Sen robust per-group fit + predict |
| `bls_fit_predict_by` | Bounded LS per-group fit + predict |
| `alm_fit_predict_by` | ALM per-group fit + predict |
| `poisson_fit_predict_by` | Poisson GLM per-group fit + predict |
| `pls_fit_predict_by` | PLS per-group fit + predict |
| `isotonic_fit_predict_by` | Isotonic per-group fit + predict (single `x_col`) |
| `quantile_fit_predict_by` | Quantile per-group fit + predict |
| `glmm_fit_by` | One mixed model across groups; returns per-group random effects ([glmm](docs/api/glm/glmm.md)) |
| `eb_shrink_by` | Empirical-Bayes shrinkage of existing per-group estimates ([eb_shrink](docs/api/glm/eb_shrink.md)) |
| `aid_by`, `aid_anomaly_by` | Per-group demand classification / per-row anomaly flags ([aid](docs/api/aid/aid.md)) |

### Statistical Hypothesis Tests

| Category | Functions | Docs |
|----------|-----------|------|
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg`, `dagostino_k2_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Parametric | `t_test_agg`, `one_way_anova_agg`, `yuen_agg`, `brown_forsythe_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`, `permutation_t_test_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Equivalence | `tost_t_test_agg`, `tost_paired_agg`, `tost_correlation_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Distribution | `energy_distance_agg`, `mmd_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Forecast | `diebold_mariano_agg`, `clark_west_agg` | [hypothesis](docs/api/statistics/hypothesis.md) |
| Correlation | `pearson_agg`, `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg` | [correlation](docs/api/statistics/correlation.md) |
| Categorical | `chisq_test_agg`, `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg` | [categorical](docs/api/statistics/categorical.md) |
| Effect size | `cramers_v_agg`, `phi_coefficient_agg`, `contingency_coef_agg`, `cohen_kappa_agg` | [categorical](docs/api/statistics/categorical.md) |
| Proportion | `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg` | [categorical](docs/api/statistics/categorical.md) |

### Diagnostics & Utilities

| Function | Description | Docs |
|----------|-------------|------|
| `vif`, `vif_agg` | Variance Inflation Factor | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `aic`, `bic` | Model selection criteria from RSS, n and k | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `jarque_bera`, `jarque_bera_agg` | Jarque-Bera normality test (scalar on an array, or aggregate) | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `residuals_diagnostics`, `residuals_diagnostics_agg` | Raw, standardized, studentized residuals and leverage | [diagnostics](docs/api/diagnostics/diagnostics.md) |
| `aid_agg`, `aid_anomaly_agg` | Demand classification (regular / intermittent) and per-row anomaly flags | [aid](docs/api/aid/aid.md) |

### Behaviour you can rely on

- **Options are validated**: option keys a function does not support raise an error at bind time instead of being silently ignored, and out-of-range values are rejected early.
- **Early input validation**: dimension mismatches, insufficient rows, all-non-finite input and similar problems produce a descriptive error before any computation.
- **Consistent result fields**: `r_squared`, `residual_std_error`, `n_observations`, `n_features` across linear models; GLM and AFT results report `z_values` (Wald z) instead of `t_values`.
- **Documented NULL handling**: see [docs/NULL_SEMANTICS.md](docs/NULL_SEMANTICS.md). The statistical methods are described in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

The regression algorithms are validated against R's `lm()`, `glmnet`, and other standard statistical packages in the [anofox-regression](https://github.com/DataZooDE/anofox-regression) Rust crate.

### ⚡ Performance

Model fitting runs inside DuckDB's parallel aggregation, so fitting one model per group scales across cores without leaving the database. On a typical developer machine:

| Workload | Scale | Cost |
|----------|-------|------|
| `ols_fit_agg` per group (3 features) | 10K groups / 1M rows | ~3 µs per fit |
| `ols_fit_agg` with `compute_inference: true` (1 feature) | 500 groups / 50K rows | ~5 µs per fit |

Most of the query time is spent in DuckDB's own `GROUP BY` machinery rather than in the model fits. Reproduce the numbers with `bash scripts/bench.sh` (see [bench/README.md](bench/README.md)); profiling notes are in [bench/PROFILING.md](bench/PROFILING.md).

---

## 📦 Installation

Supported DuckDB versions: **v1.4.5 LTS** and **v1.5.x**.

### From the DataZoo extension repository (all supported versions)

The binaries hosted at `get.erpl.io` are not signed by the DuckDB Foundation, so start DuckDB with `-unsigned` (or set `allow_unsigned_extensions` in your client):

```bash
duckdb -unsigned
```

```sql skip
INSTALL anofox_statistics FROM 'https://get.erpl.io';
LOAD anofox_statistics;
```

Binaries are available for Linux (amd64/arm64), macOS (amd64/arm64) and Windows (amd64). No build toolchain is required.

### Community extension (DuckDB v1.5.x)

```sql skip
INSTALL anofox_statistics FROM community;
LOAD anofox_statistics;
```

### Telemetry

The extension sends **pseudonymous** usage telemetry (extension load events and per-session function call counts; never SQL text, table or column names, or data). It is enabled by default and can be turned off with either:

```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

```sql skip
SET anofox_telemetry_enabled = false;
```

See [TELEMETRY.md](TELEMETRY.md) for the complete list of collected properties.

---

## 🚀 Quick Start

All functions use unprefixed names and MAP-style options (the `anofox_stats_` prefix was removed in v0.10.0; see [docs/MIGRATION.md](docs/MIGRATION.md)). [docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md) has the naming and options reference.

### Step 1 — Create a small dataset and fit an OLS model

```sql
-- Dataset: house size (sqm) and sale price (kEUR)
CREATE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

-- Fit: OLS regression (price_keur ~ sqm) via the aggregate form
SELECT
    round(fit.r_squared, 4) AS r_squared,
    fit.coefficients[1]     AS slope,
    fit.intercept           AS intercept
FROM (SELECT ols_fit_agg(price_keur, [sqm]) AS fit FROM houses);
-- r_squared ≈ 0.9995, slope ≈ 2.41, intercept ≈ -1.67
```

### Step 2 — Predict on new data

The scalar `ols_fit` takes `y` and `X` in **column-major** format (each inner array is one feature column across all observations). Use the scalar `predict(X_new, coefficients, intercept)` to score new points:

```sql
-- Predict prices for three new house sizes (70, 100, 140 sqm)
-- ols_fit column-major X: [[sqm_col]] = one feature column with all 6 training values
SELECT
    unnest([70.0, 100.0, 140.0]) AS new_sqm,
    unnest(predict(
        [[70.0, 100.0, 140.0]]::DOUBLE[][],
        (ols_fit([120.0, 155.0, 190.0, 225.0, 265.0, 300.0],
                 [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]])).coefficients,
        (ols_fit([120.0, 155.0, 190.0, 225.0, 265.0, 300.0],
                 [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]])).intercept
    )) AS predicted_keur;
-- 70 sqm → ~167 kEUR, 100 sqm → ~239 kEUR, 140 sqm → ~336 kEUR
```

### Step 3 — Inspect residuals

```sql
-- Residual diagnostics: raw and standardized residuals from in-sample predictions
WITH preds AS (
    SELECT
        unnest([120.0, 155.0, 190.0, 225.0, 265.0, 300.0])::DOUBLE AS actual,
        unnest(predict(
            [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]]::DOUBLE[][],
            (ols_fit([120.0, 155.0, 190.0, 225.0, 265.0, 300.0],
                     [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]])).coefficients,
            (ols_fit([120.0, 155.0, 190.0, 225.0, 265.0, 300.0],
                     [[50.0, 65.0, 80.0, 95.0, 110.0, 125.0]])).intercept
        )) AS yhat
)
SELECT
    (residuals_diagnostics_agg(actual, yhat)).raw AS raw_residuals
FROM preds;
```

### Per-group regression with `GROUP BY`

All `*_fit_agg` functions support `GROUP BY` for per-segment models:

```sql
CREATE TABLE sales_data AS
SELECT
    CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END        AS category,
    i::DOUBLE                                        AS units_sold,
    ((CASE WHEN i % 2 = 0 THEN 3.0 ELSE 5.0 END) * i + (i % 3))::DOUBLE AS revenue
FROM range(1, 41) t(i);

-- Fit a separate OLS model per product category
SELECT
    category,
    round(fit.r_squared, 4)       AS r_squared,
    round(fit.coefficients[1], 3) AS slope,
    round(fit.p_values[1], 6)     AS slope_p_value
FROM (
    SELECT category, ols_fit_agg(revenue, [units_sold], {'compute_inference': true}) AS fit
    FROM sales_data
    GROUP BY category
)
ORDER BY category;
```

### Fit and predict per group in one call

```sql
-- One model per category; every source row comes back with yhat and an interval
SELECT category, units_sold, revenue, round(yhat, 2) AS yhat
FROM ols_fit_predict_by('sales_data', category, revenue, [units_sold])
ORDER BY category, units_sold
LIMIT 4;
```

### Rolling regression with a window function

`*_fit_predict(...) OVER (...)` fits on the window frame and returns the prediction for the **last row of the frame**, so use frames that end at `CURRENT ROW` over a unique ordering:

```sql
-- Expanding-window in-sample fit: each row's prediction uses all rows up to and including it
SELECT
    units_sold,
    revenue,
    round((ols_fit_predict(revenue, [units_sold]) OVER (
        PARTITION BY category ORDER BY units_sold
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )).yhat, 2) AS yhat_expanding
FROM sales_data
WHERE category = 'A'
ORDER BY units_sold
LIMIT 5;
```

Frames that end before the current row (`... AND 1 PRECEDING`) do not give a one-step-ahead forecast, and `OVER (PARTITION BY g)` without `ORDER BY` gives every row the same prediction. For one prediction per row of a whole group use `ols_fit_predict_agg` or `ols_fit_predict_by`; see [fit_predict_window](docs/api/regression/fit_predict_window.md).

---

## 🤖 Claude Code Skills (AI pair-programming)

**Statistics is easier when your AI assistant actually knows the API.** anofox-statistics ships an installable [Claude Code](https://claude.com/claude-code) plugin — four skills that give Claude deep, *source-verified* knowledge of every function, MAP option, and return-struct field. Instead of guessing at signatures, Claude writes correct SQL from a plain-English request like *"fit a robust regression per store and flag outliers"* or *"run a per-segment Poisson GLM and give me the z-values."*

Install in any Claude Code session:

```
/plugin marketplace add DataZooDE/anofox-statistics
/plugin install anofox-statistics@anofox-statistics
```

| Skill | Covers |
|-------|--------|
| `anofox-statistics-regression` | OLS, robust (Huber/RANSAC/Theil-Sen), Ridge/Elastic Net, WLS/RLS, BLS/NNLS, PLS, isotonic, quantile, GLMs, ALM, AFT, GLMM, EB — options + return-struct fields |
| `anofox-statistics-tests` | Normality, parametric & nonparametric tests, correlation, categorical/contingency, effect sizes, TOST equivalence, distribution comparison, forecast-evaluation tests |
| `anofox-statistics-diagnostics` | VIF, AIC/BIC, residual diagnostics, AID demand-pattern classification, model-selection guidance |
| `anofox-statistics-batch` | AI-ready batch fitting — train thousands of models in one SQL query via `*_fit_predict_by`, `*_fit_agg` + `GROUP BY`, and rolling `*_fit_predict_agg` |

The skills live in [`plugins/anofox-statistics/`](plugins/anofox-statistics/). For in-repo development, load them directly with `claude --plugin-dir ./plugins/anofox-statistics`.

---

## 📚 API Reference

- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — function signatures, option keys, and return-struct fields for all functions.
- **[docs/api/](docs/api/)** — per-family reference pages (linked from the tables above).
- **[docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md)** — naming convention, MAP-option keys, return-struct field names and error taxonomy.
- **[docs/MIGRATION.md](docs/MIGRATION.md)** — upgrading from v0.9.x to v0.10.0 (prefix removal, renames).
- **[docs/NULL_SEMANTICS.md](docs/NULL_SEMANTICS.md)** — how NULL and NaN inputs are handled.
- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — standard errors, p-values, IRLS convergence and prediction intervals.
- **[CHANGELOG.md](CHANGELOG.md)** — release history.

User-facing guides are in the [`guides/`](guides/) directory:

- **[guides/01_quick_start.md](guides/01_quick_start.md)** — getting started with worked examples
- **[guides/02_technical_guide.md](guides/02_technical_guide.md)** — architecture and implementation details
- **[guides/03_business_guide.md](guides/03_business_guide.md)** — real-world business use cases
- **[guides/04_advanced_use_cases.md](guides/04_advanced_use_cases.md)** — complex analytical workflows

---

## 🛠️ Development

### Building from source

**Prerequisites:** Rust stable toolchain, a C++17 compiler, and CMake:

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-statistics.git
cd anofox-statistics
make release
```

This produces `build/release/duckdb` (the CLI) and `build/release/extension/anofox_statistics/anofox_statistics.duckdb_extension`.

**Rust unit tests** (the core regression library):

```bash
cargo test
```

**DuckDB SQL test suite**:

```bash
make test
```

**Doc-SQL validation**: every fenced `sql` block in this README, `guides/*.md`, `docs/*.md` and `docs/api/**/*.md` is executed against the built extension in CI (blocks marked `sql skip`, such as network installs, are excluded):

```bash
python3 scripts/validate_docs_sql.py                     # all documentation files
python3 scripts/validate_docs_sql.py --file README.md    # a single file
```

**Benchmark harness**:

```bash
bash scripts/bench.sh          # default workloads, ~1 s total
bash scripts/bench.sh --full   # adds a 1M-group workload (~8 GB RAM, ~160 s)
```

Results are written to `bench/results/` as diffable markdown files. See [bench/README.md](bench/README.md) for details.

### Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the build, test and documentation workflow.

**Areas for contribution:** additional statistical tests, documentation and examples, bug reports and fixes, performance optimizations.

---

## 💬 Support

- **API docs**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) and [docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md)
- **Guides**: [guides/](guides/)
- **Issues**: [GitHub Issues](https://github.com/DataZooDE/anofox-statistics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DataZooDE/anofox-statistics/discussions)
- **Email**: contact@datazoo.de

If a fit misbehaves or a result looks wrong, please open an issue — regression against real data has failure modes we cannot reproduce from synthetic tests, so a report with your data shape is the fastest path to a fix. Errors from the fit and predict functions include that link.

If it saved you time, a star on the repo helps other people find it.

The first time you load the extension in an interactive terminal each day, a small banner says the same thing. It never prints when output is piped, in notebooks, or in CI. Silence it with `SET datazoo_banner = false;` or `DATAZOO_NO_BANNER=1`.

---

## 📖 Citation

If you use this extension in research, please cite:

```bibtex
@software{anofox_statistics,
  title   = {Anofox Statistics: Statistical Analysis Extension for DuckDB},
  author  = {{DataZoo GmbH}},
  year    = {2026},
  url     = {https://github.com/DataZooDE/anofox-statistics},
  version = {0.10.0}
}
```

---

## ⚖️ License

This project is licensed under the **Business Source License 1.1** (BSL 1.1). The [LICENSE](LICENSE) file is authoritative; in summary:

- **Licensor:** DataZoo GmbH
- **Additional Use Grant:** you may use the Licensed Work for production purposes, but you may not offer the Licensed Work to third parties on a hosted or embedded basis.
- **Change Date:** five years from the date of first publication of each version.
- **Change License:** Mozilla Public License 2.0 (MPL 2.0). On the Change Date (or the fifth anniversary of the first public distribution of a version, whichever comes first) that version becomes available under the MPL 2.0.

Non-production use (copying, modifying, creating derivative works, redistributing) is always permitted. For uses not covered by the Additional Use Grant — for example offering the extension as part of a hosted or embedded service — please contact contact@datazoo.de for a commercial license.
