# Anofox Statistics — DuckDB Extension

A statistical analysis extension for DuckDB, providing regression analysis, diagnostics, and inference capabilities directly within your database.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![DuckDB Version](https://img.shields.io/badge/DuckDB-v1.4.5%20LTS%20%7C%20v1.5.4-brightgreen.svg)](https://duckdb.org)
[![WASM](https://github.com/DataZooDE/anofox-statistics/actions/workflows/WasmTest.yml/badge.svg?branch=main)](https://github.com/DataZooDE/anofox-statistics/actions/workflows/WasmTest.yml)

> [!IMPORTANT]
> This extension is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/DataZooDE/anofox-statistics/issues) to report bugs or request features.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Support](#-support)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Key Features

### Regression Methods

| Method | Function | Description |
|--------|----------|-------------|
| OLS | `ols_fit`, `ols_fit_agg` | Ordinary Least Squares |
| Huber | `huber_fit`, `huber_fit_agg` | Robust M-estimator (outlier-resistant; reports MAD scale + outlier mask) |
| RANSAC | `ransac_fit`, `ransac_fit_agg` | Robust consensus regression (outlier-resistant; reports inlier mask + trial count) |
| Theil-Sen | `theil_sen_fit`, `theil_sen_fit_agg` | Robust nonparametric regression via spatial-median over OLS subsamples |
| Ridge | `ridge_fit`, `ridge_fit_agg` | L2 regularization |
| Elastic Net | `elasticnet_fit`, `elasticnet_fit_agg` | Combined L1+L2 regularization |
| WLS | `wls_fit`, `wls_fit_agg` | Weighted Least Squares |
| RLS | `rls_fit`, `rls_fit_agg` | Recursive Least Squares (online) |
| Poisson | `poisson_fit_agg` | GLM for count data |
| Binomial | `binomial_fit_agg` | GLM for success-rate data (logit / probit / cloglog links) |
| Negative Binomial | `negbinom_fit_agg` | GLM for overdispersed counts (dispersion α estimated jointly) |
| Tweedie | `tweedie_fit_agg` | GLM for positive-skew continuous outcomes |
| Gamma | `gamma_fit_agg` | GLM for strictly-positive continuous outcomes (claims, durations) |
| Logistic | `logistic_fit_agg` | Binary classification (binomial GLM with logit link); reports accuracy + threshold echo, optional L2 penalty |
| ALM | `alm_fit_agg` | 24 error distributions |
| BLS/NNLS | `bls_fit_agg`, `nnls_fit_agg` | Bounded/Non-negative LS |
| PLS | `pls_fit`, `pls_fit_agg` | Partial Least Squares |
| Isotonic | `isotonic_fit`, `isotonic_fit_agg` | Monotonic regression |
| Quantile | `quantile_fit`, `quantile_fit_agg` | Quantile/median regression |
| AFT survival | `aft_fit_agg` | Duration models with right censoring |
| Mixed effects | `glmm_fit_agg` | Random intercept over a grouping factor |
| EB shrinkage | `eb_shrink_agg` | Partial pooling of per-group estimates |

### Statistical Hypothesis Tests

| Category | Function | Description |
|----------|----------|-------------|
| Normality | `shapiro_wilk_agg`, `jarque_bera_agg`, `dagostino_k2_agg` | Normality tests |
| Parametric | `t_test_agg`, `one_way_anova_agg`, `yuen_agg`, `brown_forsythe_agg` | Parametric tests |
| Nonparametric | `mann_whitney_u_agg`, `kruskal_wallis_agg`, `wilcoxon_signed_rank_agg`, `brunner_munzel_agg`, `permutation_t_test_agg` | Nonparametric tests |
| Correlation | `pearson_agg`, `spearman_agg`, `kendall_agg`, `distance_cor_agg`, `icc_agg` | Correlation tests |
| Categorical | `chisq_test_agg`, `chisq_gof_agg`, `g_test_agg`, `fisher_exact_agg`, `mcnemar_agg` | Contingency table tests |
| Effect Size | `cramers_v_agg`, `phi_coefficient_agg`, `contingency_coef_agg`, `cohen_kappa_agg` | Association measures |
| Proportion | `prop_test_one_agg`, `prop_test_two_agg`, `binom_test_agg` | Proportion tests |
| Equivalence | `tost_t_test_agg`, `tost_paired_agg`, `tost_correlation_agg` | TOST equivalence tests |
| Distribution | `energy_distance_agg`, `mmd_agg` | Distribution comparison |
| Forecast | `diebold_mariano_agg`, `clark_west_agg` | Forecast evaluation |

### Diagnostics & Utilities

| Function | Description |
|----------|-------------|
| `vif`, `vif_agg` | Variance Inflation Factor |
| `aic`, `bic` | Model selection criteria |
| `residuals_diagnostics_agg` | Residual analysis (raw, standardized, studentized, leverage arrays) |
| `aid_agg`, `aid_anomaly_agg` | Demand pattern classification |

### Fit-Predict Table Macros (`*_fit_predict_by`)

Table macros for easy per-group model fitting and prediction with a single function call:

| Macro | Description |
|-------|-------------|
| `ols_fit_predict_by` | OLS per-group fit + predict |
| `huber_fit_predict_by` | Huber robust per-group fit + predict |
| `ransac_fit_predict_by` | RANSAC robust per-group fit + predict |
| `theil_sen_fit_predict_by` | Theil-Sen robust per-group fit + predict |
| `ridge_fit_predict_by` | Ridge per-group fit + predict |
| `elasticnet_fit_predict_by` | ElasticNet per-group fit + predict |
| `wls_fit_predict_by` | WLS per-group fit + predict |
| `rls_fit_predict_by` | RLS per-group fit + predict |
| `bls_fit_predict_by` | Bounded LS per-group fit + predict |
| `alm_fit_predict_by` | ALM per-group fit + predict |
| `poisson_fit_predict_by` | Poisson GLM per-group fit + predict |
| `pls_fit_predict_by` | PLS per-group fit + predict |
| `isotonic_fit_predict_by` | Isotonic per-group fit + predict |
| `quantile_fit_predict_by` | Quantile per-group fit + predict |

### ⚡ Performance

The extension is built with performance as a first-class concern. The Phase-4 benchmark harness (`scripts/bench.sh`) covers three representative workloads:

| Workload | Scale | Per-group cost |
|----------|-------|----------------|
| W1 — aggregate dispatch | 10K groups / 1M rows | ~3.2 µs per OLS fit (3 features) |
| W2 — fit + predict | 10K groups / 1M rows | fit → predict → marshal pipeline |
| W3 — FFI micro (inference) | 500 groups / 50K rows | ~4.8 µs per fit with `compute_inference: true` |

Profiling showed that the dominant cost is DuckDB's own `HASH_GROUP_BY` dispatch (~66% of query time) — the extension's per-call overhead is a minority. Two optimizations landed in Phase 4:

- **`DataArray::to_vec` bulk-copy fast path** (no-NULL path): eliminated per-element branching for dense columns, yielding a consistent ~3–4% query-time reduction at 5M rows / 50K groups (controlled A/B).
- **`FfiVec<T>` RAII wrapper + `alloc_inference_arrays!` macro**: replaced 6 hand-written `libc::malloc` inference blocks, removing manual free/OOM boilerplate without changing allocation count (inherent to the FFI ABI).

Benchmark results and before/after numbers live in `bench/PROFILING.md` and `bench/baseline/`. Run `bash scripts/bench.sh` to reproduce.

### 🎨 User-Friendly API

Phase 5 (ERGO) made the API consistent and ergonomic:

- **Unprefixed function names**: all functions are now `ols_fit_agg(...)`, `theil_sen_fit(...)`, etc. The old `anofox_stats_` prefix is gone — see `docs/API_CONVENTIONS.md` §5 for migration.
- **MAP-style options** with documented keys (e.g. `{'fit_intercept': true, 'compute_inference': true}`); unknown keys raise `InvalidInputException` at bind time instead of being silently ignored.
- **Early input validation**: dimension mismatches, insufficient rows, all-non-finite input, and constant columns are caught with a descriptive message before any computation.
- **Consistent return-struct field names** across families: `r_squared` (not `.r2`), `residual_std_error`, `n_observations`, `z_values` for GLM/AFT (not `t_values`).

The regression algorithms are validated against R's `lm()`, `glmnet`, and other standard statistical packages in the [anofox-regression](https://github.com/DataZooDE/anofox-regression) Rust crate.

---

## 🚀 Quick Start

All functions use the **v0.3.0 API** — unprefixed names and MAP-style options. See `docs/API_CONVENTIONS.md` for the full naming and options reference.

### Step 1 — Create a small dataset and fit an OLS model

```sql
-- Dataset: house size (sqm) and sale price (kEUR)
CREATE TABLE houses AS SELECT * FROM (VALUES
    (50.0, 120.0), (65.0, 155.0), (80.0, 190.0),
    (95.0, 225.0), (110.0, 265.0), (125.0, 300.0)
) t(sqm, price_keur);

-- Fit: OLS regression (price_keur ~ sqm) via the aggregate form
SELECT
    round((ols_fit_agg(price_keur, [sqm])).r_squared, 4) AS r_squared,
    (ols_fit_agg(price_keur, [sqm])).coefficients[1]      AS slope,
    (ols_fit_agg(price_keur, [sqm])).intercept            AS intercept
FROM houses;
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

```sql skip
-- Fit a separate OLS model per product category
SELECT
    category,
    (ols_fit_agg(revenue, [units_sold], {'compute_inference': true})).r_squared AS r_squared,
    (ols_fit_agg(revenue, [units_sold], {'compute_inference': true})).coefficients[1] AS slope
FROM sales_data
GROUP BY category;
```

---

## 📦 Installation

### From erpl.io (recommended)

Install directly from our public distribution bucket. DuckDB must be started
with the `-unsigned` flag, since the extension binary is not signed by the
DuckDB Foundation:

```sql skip
INSTALL 'anofox_statistics' FROM 'http://get.erpl.io';
LOAD 'anofox_statistics';
```

This pulls the right binary for your platform (Linux amd64/arm64, macOS
amd64/arm64, Windows amd64). No build toolchain, no vcpkg, no submodules
required.

### Community Extension

```sql skip
INSTALL anofox_statistics FROM community;
LOAD anofox_statistics;
```

### Telemetry

This extension collects anonymous usage telemetry to help improve the product. Telemetry is **enabled by default** and includes:

- Extension load events (extension name, version, platform)
- Function execution events (which functions are used)
- No personal data or query contents are collected

**Disable telemetry:**

```bash
export DATAZOO_DISABLE_TELEMETRY=1
```

```sql skip
SET anofox_telemetry_enabled = false;
```

For more information, see the [posthog-telemetry](https://github.com/DataZooDE/posthog-telemetry) repository.

---

## 📚 API Reference

The authoritative function reference and naming conventions live in the `docs/` directory — not duplicated here to avoid drift:

- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — complete function signatures, option keys, and return-struct fields for all 100+ functions.
- **[docs/API_CONVENTIONS.md](docs/API_CONVENTIONS.md)** — naming convention, MAP-option keys, return-struct field names, error taxonomy, and the v0.3.0 breaking-changes migration guide.

User-facing guides are in the [`guides/`](guides/) directory:

- **[guides/01_quick_start.md](guides/01_quick_start.md)** — getting started with worked examples
- **[guides/02_technical_guide.md](guides/02_technical_guide.md)** — architecture and implementation details
- **[guides/03_business_guide.md](guides/03_business_guide.md)** — real-world business use cases
- **[guides/04_advanced_use_cases.md](guides/04_advanced_use_cases.md)** — complex analytical workflows

---

## 🛠️ Development

### Building from source

**Prerequisites:** Rust stable toolchain, a C++17 compiler, and CMake (all provided by `extension-ci-tools`):

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

**DuckDB SQL test suite** (2 000+ assertions across all function families):

```bash
build/release/test/unittest --test-dir=test/sql
```

**Doc-SQL validation** (all SQL examples in README + guides + API docs must pass):

```bash
python3 scripts/validate_docs_sql.py          # full 7-file sweep
python3 scripts/validate_docs_sql.py --file README.md   # single-file fast path
```

**Benchmark harness** (repeatable timing across three workloads):

```bash
bash scripts/bench.sh          # default: ~1 s total
bash scripts/bench.sh --full   # adds 1M-group workload (~8 GB RAM, ~160 s)
```

Results are written to `bench/results/` as diffable markdown files. See `bench/README.md` for details.

### Contributing

Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

**Areas for contribution:** additional statistical tests, visualization helpers for diagnostics, documentation and examples, bug reports and fixes, performance optimizations.

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
  title = {Anofox Statistics: Statistical Analysis Extension for DuckDB},
  author = {DataZoo DE},
  year = {2025},
  url = {https://github.com/DataZooDE/anofox-statistics},
  version = {1.0.0}
}
```

---

## ⚖️ License

This project is licensed under the **Business Source License 1.1** (BSL 1.1).

### Key Terms

- **Usage Grant**: Free to use, modify, and distribute for non-production purposes
- **Production Use**: Permitted after 4 years from release date, or under a commercial license
- **Change Date**: [Release Date + 4 years]
- **Change License**: Apache License 2.0

See [LICENSE](LICENSE) for full terms.

### Why BSL?

The BSL allows:
- Free use for development, testing, and research
- Open source collaboration and contributions
- Academic and educational use
- Small-scale production use

While ensuring:
- Sustainable development funding
- Protection for the project's long-term viability
- Future conversion to fully open source (Apache 2.0)

For commercial production use before the Change Date, please contact: contact@datazoo.de
