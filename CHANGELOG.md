# Changelog

All notable changes to the Anofox Statistics DuckDB extension are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is below 1.0, minor releases may contain breaking changes.
Breaking changes are called out explicitly.

## [Unreleased]

### Added

- TODO(lead): new functions, e.g. `pls_fit_agg`, `quantile_fit_agg`, `isotonic_fit_agg`,
  model-aware `predict(model, x)`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/MIGRATION.md`, `docs/NULL_SEMANTICS.md`,
  `docs/METHODOLOGY.md`. New reference pages for Huber, RANSAC, Theil-Sen, LARS,
  the binomial, logistic, gamma and Tweedie GLMs, the window and fit-predict
  aggregates.
- The doc-SQL validation now covers every page under `docs/` and `docs/api/`.
- A Claude Code plugin (`plugins/anofox-statistics`) with skills for regression,
  tests, diagnostics and batch use.
- Option aliases (old keys keep working): `link` sets the link of the fitted
  family (`poisson_fit_agg`, `poisson_fit_predict_agg`, `binomial_fit_agg`);
  `lambda`/`alpha` alias `glm_lambda` on the GLMs (poisson, binomial, logistic,
  negbinom, gamma, tweedie); `tau` aliases `quantile` on ALM and `quantile`
  aliases `tau` on quantile regression; `lambda` aliases `forgetting_factor` on
  RLS; `lambda` aliases `alpha` on `ridge_fit_predict_agg` /
  `elasticnet_fit_predict_agg`; `seed`/`random_state` on every seeded function.
- TOST: `tost_t_test_agg`, `tost_paired_agg` and `tost_correlation_agg` all accept
  both `alpha` and `confidence_level` (alpha = 1 - confidence_level; supplying
  inconsistent values is an error) and `lower`/`upper` as aliases of
  `bound_lower`/`bound_upper`.
- Seeds for `permutation_t_test_agg`, `energy_distance_agg`, `mmd_agg` and
  `distance_cor_agg` (`seed` / `random_state`).
- `mann_whitney_u_agg` options `exact` and `mu`; `fisher_exact_agg` option
  `confidence_level`; `mcnemar_agg` option `exact`; `lars_fit_agg` options
  `method` (`'lar'` | `'lasso'`), `n_nonzero_coefs` and `standardize`.
- Hypothesis-test results now all carry `statistic`, `p_value`, `n` (total
  sample size), `method` and `alternative` (`'two_sided'` | `'less'` |
  `'greater'`, NULL where not applicable). Missing fields were appended at the
  end of each struct, e.g. `cohen_kappa_agg` gains `statistic` (= z), `n`,
  `method`; `icc_agg` gains `statistic` (= F), `p_value` (one-way F test of
  ICC = 0), `n`; `tost_*_agg` gain `statistic`; `jarque_bera`/`jarque_bera_agg`
  gain `method`.

### Changed

- TODO(lead): option keys a function does not support now raise an error
  (per-function key validation).
- **Breaking:** every regression-family function (aggregates, table/scalar
  fits, window functions, fit-predict aggregates) declares the option keys it
  reads; any other key raises an error naming the function and listing its
  supported keys. Keys that were silently ignored before, e.g.
  `ols_fit_agg(y, x, {'alpha': 1})` or `compute_inference` on fit-predict
  functions, now error. The same holds for the hypothesis-test aggregates;
  `mmd_agg` rejects `bandwidth`/`sigma` (never used) and `tost_t_test_agg`
  rejects `alternative`, `paired` and `mu` (never used).
- **Breaking:** options must be a constant expression; a per-row expression
  used to be skipped silently and now errors.
- **Breaking:** `confidence_level` must lie strictly inside (0, 1) everywhere;
  integer options reject negative, fractional and overflowing values instead of
  truncating them.
- `lars_fit_agg`: `alpha` > 0 without a `method` now selects the LassoLars path
  (plain LAR ignores alpha, so it used to have no effect); `{'method': 'lar',
  'alpha': > 0}` is an error.
- `fisher_exact_agg` reports R's conditional MLE odds ratio and exact
  conditional CI (`fisher.test` semantics) instead of the sample odds ratio and
  Woolf interval, and returns a result for any non-empty table (n < 4 used to
  return NULL).
- RANSAC and Theil-Sen (aggregates, fit-predict aggregates and window
  functions) fit on a canonical row order, so a given `random_state` yields the
  same model whatever the thread count or input order. Results can differ from
  earlier versions for the same seed.
- TODO(lead): leverage-aware prediction intervals
  (`s·sqrt(1 + x₀ᵀ(XᵀX)⁻¹x₀)` instead of `s·sqrt(1 + 1/n)`).
- License metadata is consistent with `LICENSE`: BSL 1.1 that converts to MPL 2.0
  five years after each version is first published. The Cargo workspace license
  is now `BUSL-1.1`.

### Fixed

- TODO(lead): bug fixes from the review remediation.
- Hypothesis-test aggregates with hand-written option parsing (yuen, permutation
  t-test, TOST paired/correlation, Diebold-Mariano, Clark-West, binomial and
  proportion tests, ICC, McNemar, Cohen's kappa, distance correlation) ignored
  STRUCT literals `{'k': v}` and silently fell back to defaults on invalid enum
  values (e.g. `alternative: 'sideways'`); they now accept MAP and STRUCT, and
  reject unknown keys and invalid values.
- `MAP {...}` options to the shared test-option parsers (`t_test_agg`,
  `mann_whitney_u_agg`, ...) failed with "Invalid MAP structure".
- `binomial_fit_agg(..., {'link': 'probit'})` failed with a Poisson-link error;
  `quantile_fit_predict_agg` ignored `quantile`; GLMs ignored `lambda`.
- `t_test_agg` silently ignored `paired: true` (it has no pairing information);
  it now raises an error pointing to `tost_paired_agg` /
  `wilcoxon_signed_rank_agg`.
- `mann_whitney_u_agg` with `exact: true`: the two-sided p-value was 1.0
  whenever U lay above its mean; it now matches `wilcox.test(exact = TRUE)`.
- `fisher_exact_agg` silently dropped values other than 0/1; it now errors.
- Seeded tests and RANSAC/Theil-Sen gave different results depending on the
  number of threads.
- Documentation: removed the stale `anofox_stats_` prefixes, the wrong calling
  conventions, the wrong option keys and field names, and the broken links.

## [0.10.0] - 2026-09-02

### Changed (breaking)

- All SQL functions are registered under their unprefixed names only. The
  `anofox_stats_` prefix and the deprecated aliases are gone. See
  [docs/MIGRATION.md](docs/MIGRATION.md).
- `theilsen_*` was renamed to `theil_sen_*`. The `*_predict_agg` aliases were
  removed; use `*_fit_predict_agg`.
- Unknown option keys in an options MAP now raise an error. Before, they were
  silently ignored.

### Added

- A benchmark harness (`scripts/bench.sh`, `bench/`).
- Doc-SQL validation (`scripts/validate_docs_sql.py`) and a CI gate that runs it.
- `docs/API_CONVENTIONS.md`, which describes naming, option keys and return fields.
- WASM support: a load fix, a DuckDB-Wasm test harness and a CI gate.
- GLM results report the curvature (observed information), with a mapping for its rows.

### Fixed

- `ols_fit_agg` used as a window function returns NULL for degenerate frames,
  instead of a saturated fit with NaN statistics.
- Errors name the SQL function that raised them, and numerical failures use a
  distinct error class.

### Performance

- Refactored the FFI marshalling layer, which reduces per-group allocation
  overhead (see `bench/PROFILING.md`).

## [0.9.0] - 2026-08-08

### Added

- MCMC convergence diagnostics: split R-hat and effective sample size.
- AFT models report the full covariance matrix and the observed information.
- Fit errors include a feedback banner and links for reporting issues.

## [0.8.1] - 2026-08-05

### Fixed

- macOS and WASM builds (FFI index type mismatch).

## [0.8.0] - 2026-08-04

### Added

- Hierarchical GLMs (`glmm_fit_agg`, `glmm_fit_by`) with random slopes and
  crossed or nested factors.
- Censored likelihoods: AFT survival models (`aft_fit_agg`, `aft_cdf`, `aft_quantile`).
- Explicit coefficient priors with Laplace intervals.
- Empirical-Bayes shrinkage (`eb_shrink_agg`, `eb_shrink_by`).
- GLM offsets, NULL list elements, and a `converged` field in GLM results.

## [0.7.3] - 2026-07-25

### Changed

- Built against DuckDB v1.5.5, keeping v1.4 LTS support.

## [0.7.2.1] - 2026-07-18

### Changed

- Updated the telemetry dependency.

## [0.7.1] - 2026-07-13

### Changed

- Adopted telemetry schema 2.

### Fixed

- The Rust FFI archive is linked into the WASM build.
- A crash at telemetry teardown.

## [0.7.0] - 2026-06-25

### Added

- LARS / LassoLars regression (`lars_fit_agg`).
- Binary logistic regression (`logistic_fit_agg`).
- GLM aggregates: binomial, negative binomial, gamma and Tweedie.

### Fixed

- An uninitialized-memory read on NULL features in the fit-predict aggregates.

## [0.6.1] - 2026-06-18

### Fixed

- CI: the extension-ci-tools version used for deploys is pinned.

## [0.6.0] - 2026-06-11

### Added

- Robust regression: Huber, RANSAC and Theil-Sen, each with scalar, aggregate,
  window, fit-predict and `*_by` macro forms.
- Publication to the erpl.io distribution bucket.

### Changed

- Upgraded DuckDB to v1.5.3.

## [0.5.3] - 2026-03-10

### Added

- `solver`, `hc_type`, `lambda_scaling` and `glm_lambda` options.

### Changed

- SVD is the default decomposition.
- Upgraded to DuckDB 1.5.0.

## [0.5.2] - 2026-02-13

### Added

- `split` column support in the `*_fit_predict_agg` functions and the `*_fit_predict_by` macros.
- Passthrough columns in the `*_fit_predict_by` macros.

## [0.5.1] - 2026-01-30

### Added

- `aid_by` and `aid_anomaly_by` table macros.

### Changed

- Updated DuckDB from v1.4.3 to v1.4.4.

## [0.5.0] - 2026-01-16

### Added

- PLS, isotonic and quantile regression, as fit-predict aggregates and `*_by` macros.

### Fixed

- A segfault in `aid_anomaly_agg` on very large inputs.

## [0.4.0] - 2026-01-08

### Added

- `*_fit_predict_by` table macros for grouped regression.
- Fit-predict aggregates for BLS, ALM and Poisson.

### Fixed

- Zero-variance feature handling in all regression methods.
- Undefined behaviour in aggregate functions.

## [0.3.0] - 2025-12-19

### Added

- Statistical hypothesis tests: t-test, ANOVA, Mann-Whitney, correlation tests and more.
- Automatic Identification of Demand (AID).

## [0.2.0] - 2025-11-30

### Changed (breaking)

- Function prefix unified from `anofox_statistics_*` to `anofox_stats_*`.

## [0.1.0] - 2025-11-05

### Added

- Initial release: OLS, Ridge, WLS, RLS and Elastic Net regression as scalar
  and aggregate functions, plus diagnostics.

[Unreleased]: https://github.com/DataZooDE/anofox-statistics/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.2.1...v0.7.3
[0.7.2.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.1...v0.7.2.1
[0.7.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.3...v0.6.0
[0.5.3]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/DataZooDE/anofox-statistics/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DataZooDE/anofox-statistics/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DataZooDE/anofox-statistics/releases/tag/v0.1.0
