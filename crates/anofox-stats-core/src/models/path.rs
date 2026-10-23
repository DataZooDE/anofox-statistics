//! Regularisation paths, cross-validation curves (#154) and CV strategies /
//! nested CV (#204).
//!
//! Marshalling only: the ridge / elastic-net path, glmnet's default lambda
//! grid, the fold generators, the explicit splits, the `cv.glmnet` aggregates
//! and nested CV come from `anofox_regression::solvers::{path, cv}`, the LARS
//! path from `FittedLars::path`. This module drops rows with a non-finite value, drops
//! the columns the fits drop (reported as NaN coefficients) and lays the path
//! out per lambda.

use crate::errors::{StatsError, StatsResult};
use crate::types::LarsOptions;
use anofox_regression::solvers::cv::{
    cv_path_splits, group_codes, group_kfold_ids, group_time_series_splits, logo_ids,
    nested_cv_path, quantile_strata, splits_from_fold_ids, stratified_kfold_ids, CvMetric, CvSplit,
    GroupBalance, InnerFolds, LambdaRule, NestedCvOptions, TimeSeriesSplitOptions, TrainWindow,
};
use anofox_regression::solvers::path::RegPath as UpPath;
use anofox_regression::solvers::path::{cv_path, fit_path, kfold_ids, lambda_grid, PathOptions};
use anofox_regression::solvers::{LarsMethod, LarsRegressor, Regressor};
use faer::{Col, Mat};

/// Options of [`fit_reg_path`] / [`cv_reg_path`].
#[derive(Debug, Clone)]
pub struct RegPathOptions {
    /// Elastic-net mixing in `[0, 1]`: 0 is ridge, 1 the lasso (glmnet's `alpha`).
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    /// glmnet's `standardize`.
    pub standardize: bool,
    /// Penalties (glmnet scale); empty for glmnet's default grid.
    pub lambdas: Vec<f64>,
    /// Size of the default grid.
    pub n_lambda: usize,
    /// Smallest lambda of the default grid relative to the largest; NaN for
    /// glmnet's default (1e-4 when n > p, else 0.01).
    pub lambda_min_ratio: f64,
}

impl Default for RegPathOptions {
    fn default() -> Self {
        Self {
            l1_ratio: 0.0,
            fit_intercept: true,
            standardize: false,
            lambdas: Vec::new(),
            n_lambda: 100,
            lambda_min_ratio: f64::NAN,
        }
    }
}

/// A coefficient path, one entry per lambda.
#[derive(Debug, Clone)]
pub struct RegPathResult {
    pub lambdas: Vec<f64>,
    /// NaN without an intercept.
    pub intercepts: Vec<f64>,
    /// Row-major `lambdas.len() x n_features`; NaN for dropped columns.
    pub coefficients: Vec<f64>,
    /// Number of non-zero coefficients at each lambda.
    pub df: Vec<usize>,
    pub n_features: usize,
    pub n_observations: usize,
}

/// A k-fold cross-validation curve over a lambda grid.
#[derive(Debug, Clone)]
pub struct CvPathResult {
    pub lambdas: Vec<f64>,
    pub cv_mean: Vec<f64>,
    pub cv_se: Vec<f64>,
    pub df: Vec<usize>,
    pub index_min: usize,
    pub index_1se: usize,
    pub n_folds: usize,
    pub n_observations: usize,
}

/// Rows with a finite `y`, features (and fold id), and the columns kept.
struct PathData {
    x: Mat<f64>,
    y: Col<f64>,
    rows: Vec<usize>,
    kept: Vec<usize>,
    n_features: usize,
}

fn prepare(y: &[f64], x: &[Vec<f64>], fit_intercept: bool) -> StatsResult<PathData> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    crate::validation::validate_x_columns(y.len(), x)?;
    let rows: Vec<usize> = (0..y.len())
        .filter(|&i| y[i].is_finite() && x.iter().all(|c| c[i].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err(StatsError::NoValidData);
    }
    let dropped = crate::validation::droppable_columns(x, &rows, fit_intercept);
    let kept: Vec<usize> = (0..x.len()).filter(|&j| !dropped[j]).collect();
    if kept.is_empty() || rows.len() < 2 {
        return Err(StatsError::InsufficientData {
            rows: rows.len(),
            cols: x.len(),
        });
    }
    Ok(PathData {
        x: Mat::from_fn(rows.len(), kept.len(), |i, j| x[kept[j]][rows[i]]),
        y: Col::from_fn(rows.len(), |i| y[rows[i]]),
        rows,
        kept,
        n_features: x.len(),
    })
}

fn upstream_options(opts: &RegPathOptions) -> StatsResult<PathOptions> {
    if !(0.0..=1.0).contains(&opts.l1_ratio) {
        return Err(StatsError::InvalidL1Ratio(opts.l1_ratio));
    }
    Ok(PathOptions::elastic_net(opts.l1_ratio)
        .with_intercept(opts.fit_intercept)
        .with_standardize(opts.standardize))
}

/// The user's lambdas, or glmnet's default grid on the data.
fn lambdas_for(data: &PathData, opts: &RegPathOptions, up: &PathOptions) -> StatsResult<Vec<f64>> {
    if !opts.lambdas.is_empty() {
        return Ok(opts.lambdas.clone());
    }
    let ratio = if opts.lambda_min_ratio.is_nan() {
        None
    } else {
        Some(opts.lambda_min_ratio)
    };
    lambda_grid(&data.x, &data.y, up, opts.n_lambda, ratio).map_err(StatsError::from)
}

fn expand_path(path: &UpPath, data: &PathData, fit_intercept: bool) -> RegPathResult {
    let l = path.len();
    let p = data.n_features;
    let mut coefficients = vec![f64::NAN; l * p];
    for c in 0..l {
        for (r, &j) in data.kept.iter().enumerate() {
            coefficients[c * p + j] = path.coefficients[(r, c)];
        }
    }
    RegPathResult {
        lambdas: path.lambdas.clone(),
        intercepts: if fit_intercept {
            path.intercepts.clone()
        } else {
            vec![f64::NAN; l]
        },
        coefficients,
        df: path.df.clone(),
        n_features: p,
        n_observations: data.rows.len(),
    }
}

/// Ridge / elastic-net coefficient path (`glmnet(family = "gaussian")`).
pub fn fit_reg_path(
    y: &[f64],
    x: &[Vec<f64>],
    opts: &RegPathOptions,
) -> StatsResult<RegPathResult> {
    let data = prepare(y, x, opts.fit_intercept)?;
    let up = upstream_options(opts)?;
    let lambdas = lambdas_for(&data, opts, &up)?;
    let path = fit_path(&data.x, &data.y, &lambdas, &up).map_err(StatsError::from)?;
    Ok(expand_path(&path, &data, opts.fit_intercept))
}

/// k-fold cross-validation of the ridge / elastic-net path (`cv.glmnet`,
/// `type.measure = "mse"`).
///
/// `fold_ids` gives each row's fold (rows dropped for missing values drop
/// their id); without it, `n_folds` seeded folds are drawn with upstream's
/// `kfold_ids(n, n_folds, seed)` over the usable rows in input order.
pub fn cv_reg_path(
    y: &[f64],
    x: &[Vec<f64>],
    fold_ids: Option<&[i64]>,
    n_folds: usize,
    seed: u64,
    opts: &RegPathOptions,
) -> StatsResult<CvPathResult> {
    if let Some(f) = fold_ids {
        if f.len() != y.len() {
            return Err(StatsError::DimensionMismatch {
                y_len: y.len(),
                x_rows: f.len(),
            });
        }
    }
    let data = prepare(y, x, opts.fit_intercept)?;
    let up = upstream_options(opts)?;
    let lambdas = lambdas_for(&data, opts, &up)?;
    let folds: Vec<usize> = match fold_ids {
        Some(f) => {
            // Any labels; map them to 0..k in order of first appearance.
            let mut labels: Vec<i64> = Vec::new();
            let ids: Vec<usize> = data
                .rows
                .iter()
                .map(|&i| match labels.iter().position(|&l| l == f[i]) {
                    Some(k) => k,
                    None => {
                        labels.push(f[i]);
                        labels.len() - 1
                    }
                })
                .collect();
            if labels.len() < 3 {
                return Err(StatsError::InvalidValue {
                    field: "fold_id",
                    message: format!(
                        "cross-validation needs at least 3 folds, got {}",
                        labels.len()
                    ),
                });
            }
            ids
        }
        None => {
            if n_folds < 3 {
                return Err(StatsError::InvalidValue {
                    field: "folds",
                    message: format!("cross-validation needs at least 3 folds, got {n_folds}"),
                });
            }
            if n_folds > data.rows.len() {
                return Err(StatsError::InvalidValue {
                    field: "folds",
                    message: format!(
                        "{n_folds} folds for {} usable rows; use at most one fold per row",
                        data.rows.len()
                    ),
                });
            }
            kfold_ids(data.rows.len(), n_folds, seed)
        }
    };
    let cv = cv_path(&data.x, &data.y, &lambdas, &folds, &up).map_err(StatsError::from)?;
    Ok(CvPathResult {
        lambdas: cv.lambdas,
        cv_mean: cv.cv_mean,
        cv_se: cv.cv_se,
        df: cv.df,
        index_min: cv.index_min,
        index_1se: cv.index_1se,
        n_folds: cv.n_folds,
        n_observations: data.rows.len(),
    })
}

/// How the cross-validation splits are built (`cv_strategy`, #204).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvStrategy {
    /// Seeded random k-fold (`kfold_ids`), no key.
    Kfold,
    /// The key is the fold label of each row (any labels, at least 3).
    FoldIds,
    /// Group k-fold balanced by rows (`group_kfold_ids`, sklearn `GroupKFold`):
    /// the key is the group.
    Group,
    /// Leave-one-group-out (`logo_ids`): the key is the group.
    Logo,
    /// Stratified k-fold (`stratified_kfold_ids`): the key is the stratum;
    /// without a key, `quantile_strata(y, n_bins)`.
    Stratified,
    /// Rolling-origin splits over the distinct key values in increasing order
    /// (`group_time_series_splits`): the key is the time.
    TimeSeries,
}

/// Options of [`cv_reg_path_strategy`] / [`nested_cv_reg_path`].
#[derive(Debug, Clone)]
pub struct CvSpec {
    pub strategy: CvStrategy,
    /// k of the k-fold strategies, `n_splits` of `time_series` (the outer
    /// folds of nested CV).
    pub n_folds: usize,
    pub seed: u64,
    /// Repetitions of `kfold` / `stratified` with seeds `seed, seed + 1, ...`;
    /// the splits of all repetitions form one curve.
    pub repeats: usize,
    /// `time_series`: time points per test block (0: sklearn's default).
    pub test_size: usize,
    /// `time_series`: time points left out between training and test.
    pub gap: usize,
    /// `time_series`: sliding window of at most this many time points
    /// (0: expanding window).
    pub max_train_size: usize,
    /// `stratified` without a key: number of quantile bins of `y`.
    pub n_bins: usize,
    /// Mean absolute error instead of the mean squared error.
    pub mae: bool,
    /// Nested CV: inner folds (k, or the time-series splits).
    pub inner_folds: usize,
    /// Nested CV: choose `lambda_1se` instead of `lambda_min`.
    pub one_se: bool,
}

impl Default for CvSpec {
    fn default() -> Self {
        Self {
            strategy: CvStrategy::Kfold,
            n_folds: 10,
            seed: 42,
            repeats: 1,
            test_size: 0,
            gap: 0,
            max_train_size: 0,
            n_bins: 5,
            mae: false,
            inner_folds: 5,
            one_se: false,
        }
    }
}

impl CvSpec {
    fn metric(&self) -> CvMetric {
        if self.mae {
            CvMetric::Mae
        } else {
            CvMetric::Mse
        }
    }

    fn time_options(&self, n_splits: usize, test_size: usize) -> TimeSeriesSplitOptions {
        TimeSeriesSplitOptions {
            n_splits,
            test_size: (test_size > 0).then_some(test_size),
            gap: self.gap,
            window: if self.max_train_size > 0 {
                TrainWindow::Sliding(self.max_train_size)
            } else {
                TrainWindow::Expanding
            },
            min_train_size: None,
        }
    }
}

/// A total order on the f64 keys (integers, hashes or times as doubles).
fn ord_key(v: f64) -> i64 {
    let b = v.to_bits() as i64;
    b ^ (((b >> 63) as u64) >> 1) as i64
}

fn invalid(field: &'static str, message: String) -> StatsError {
    StatsError::InvalidValue { field, message }
}

/// The keys of the usable rows, as ordered codes.
fn key_codes(keys: Option<&[f64]>, data: &PathData, spec: &CvSpec) -> StatsResult<Vec<usize>> {
    let keys = keys.ok_or_else(|| {
        invalid(
            "cv_key",
            format!(
                "cv_strategy '{}' needs a cv_key argument",
                strategy_name(spec.strategy)
            ),
        )
    })?;
    let ord: Vec<i64> = data.rows.iter().map(|&i| ord_key(keys[i])).collect();
    Ok(group_codes(&ord).0)
}

fn strategy_name(s: CvStrategy) -> &'static str {
    match s {
        CvStrategy::Kfold => "kfold",
        CvStrategy::FoldIds => "fold_id",
        CvStrategy::Group => "group",
        CvStrategy::Logo => "logo",
        CvStrategy::Stratified => "stratified",
        CvStrategy::TimeSeries => "time_series",
    }
}

/// The outer splits of a strategy over the usable rows.
fn strategy_splits(
    keys: Option<&[f64]>,
    data: &PathData,
    spec: &CvSpec,
) -> StatsResult<Vec<CvSplit>> {
    let n = data.rows.len();
    let k = spec.n_folds;
    let repeats = spec.repeats.max(1);
    if repeats > 1 && !matches!(spec.strategy, CvStrategy::Kfold | CvStrategy::Stratified) {
        return Err(invalid(
            "repeats",
            "repeats applies to cv_strategy 'kfold' and 'stratified' only".to_string(),
        ));
    }
    if keys.is_some() && spec.strategy == CvStrategy::Kfold {
        return Err(invalid(
            "cv_key",
            "cv_strategy 'kfold' draws random folds and takes no cv_key; use 'fold_id' for given folds"
                .to_string(),
        ));
    }
    let up = |e| StatsError::from(e);
    let mut splits = Vec::new();
    match spec.strategy {
        CvStrategy::Kfold => {
            if k > n {
                return Err(invalid(
                    "folds",
                    format!("{k} folds for {n} usable rows; use at most one fold per row"),
                ));
            }
            for r in 0..repeats {
                splits.extend(splits_from_fold_ids(&kfold_ids(
                    n,
                    k,
                    spec.seed.wrapping_add(r as u64),
                )));
            }
        }
        CvStrategy::FoldIds => {
            let codes = key_codes(keys, data, spec)?;
            splits = splits_from_fold_ids(&codes);
            if splits.len() < 3 {
                return Err(invalid(
                    "fold_id",
                    format!(
                        "cross-validation needs at least 3 folds, got {}",
                        splits.len()
                    ),
                ));
            }
        }
        CvStrategy::Group => {
            let codes = key_codes(keys, data, spec)?;
            splits = splits_from_fold_ids(
                &group_kfold_ids(&codes, k, GroupBalance::Rows, spec.seed).map_err(up)?,
            );
        }
        CvStrategy::Logo => {
            let codes = key_codes(keys, data, spec)?;
            splits = splits_from_fold_ids(&logo_ids(&codes));
        }
        CvStrategy::Stratified => {
            let strata = match keys {
                Some(_) => key_codes(keys, data, spec)?,
                None => quantile_strata(&data.y, spec.n_bins),
            };
            for r in 0..repeats {
                let ids = stratified_kfold_ids(&strata, k, Some(spec.seed.wrapping_add(r as u64)))
                    .map_err(up)?;
                splits.extend(splits_from_fold_ids(&ids));
            }
        }
        CvStrategy::TimeSeries => {
            let codes = key_codes(keys, data, spec)?;
            splits = group_time_series_splits(&codes, &spec.time_options(k, spec.test_size))
                .map_err(up)?;
        }
    }
    Ok(splits)
}

/// The inner folds of nested CV for a strategy: the same key, `inner_folds`
/// folds. `fold_id` and `kfold` use seeded random inner folds, `logo` uses
/// group k-fold on the same groups.
fn inner_folds(keys: Option<&[f64]>, data: &PathData, spec: &CvSpec) -> StatsResult<InnerFolds> {
    let k = spec.inner_folds;
    Ok(match spec.strategy {
        CvStrategy::Kfold | CvStrategy::FoldIds => InnerFolds::Kfold { k, seed: spec.seed },
        CvStrategy::Group | CvStrategy::Logo => InnerFolds::Group {
            groups: key_codes(keys, data, spec)?,
            k,
            balance: GroupBalance::Rows,
            seed: spec.seed,
        },
        CvStrategy::Stratified => InnerFolds::Stratified {
            strata: match keys {
                Some(_) => key_codes(keys, data, spec)?,
                None => quantile_strata(&data.y, spec.n_bins),
            },
            k,
            seed: Some(spec.seed),
        },
        CvStrategy::TimeSeries => InnerFolds::TimeSeries {
            opts: spec.time_options(k, 0),
            time_keys: Some(key_codes(keys, data, spec)?),
        },
    })
}

fn check_keys(keys: Option<&[f64]>, n: usize) -> StatsResult<()> {
    if let Some(k) = keys {
        if k.len() != n {
            return Err(StatsError::DimensionMismatch {
                y_len: n,
                x_rows: k.len(),
            });
        }
    }
    Ok(())
}

/// Cross-validation of the ridge / elastic-net path with a CV strategy
/// (`cv_path_splits` on the strategy's splits).
///
/// `keys` holds one value per row (the fold label, group, stratum or time,
/// depending on the strategy; integers, hashes or times as doubles). Rows
/// dropped for missing values drop their key. Rows are used in the order
/// given (the caller orders them canonically).
pub fn cv_reg_path_strategy(
    y: &[f64],
    x: &[Vec<f64>],
    keys: Option<&[f64]>,
    spec: &CvSpec,
    opts: &RegPathOptions,
) -> StatsResult<CvPathResult> {
    check_keys(keys, y.len())?;
    let data = prepare(y, x, opts.fit_intercept)?;
    let up = upstream_options(opts)?;
    let splits = strategy_splits(keys, &data, spec)?;
    let lambdas = lambdas_for(&data, opts, &up)?;
    let cv = cv_path_splits(&data.x, &data.y, &lambdas, &splits, &up, spec.metric())
        .map_err(StatsError::from)?;
    Ok(CvPathResult {
        lambdas: cv.lambdas,
        cv_mean: cv.cv_mean,
        cv_se: cv.cv_se,
        df: cv.df,
        index_min: cv.index_min,
        index_1se: cv.index_1se,
        n_folds: cv.n_folds,
        n_observations: data.rows.len(),
    })
}

/// Nested cross-validation, one entry per outer split.
#[derive(Debug, Clone)]
pub struct NestedCvResult {
    /// Selected lambda of each outer split.
    pub lambda: Vec<f64>,
    /// Test error (MSE or MAE) of the refit at that lambda on the outer test rows.
    pub test_error: Vec<f64>,
    /// Inner CV error and its standard error at the selected lambda.
    pub inner_cv_mean: Vec<f64>,
    pub inner_cv_se: Vec<f64>,
    pub n_train: Vec<usize>,
    pub n_test: Vec<usize>,
    pub n_inner_folds: Vec<usize>,
    /// Outer test error averaged with the test sizes as weights, and its
    /// standard error.
    pub mean: f64,
    pub se: f64,
    pub n_observations: usize,
}

/// Nested cross-validation of the ridge / elastic-net path
/// (`anofox_regression::solvers::cv::nested_cv_path`): the outer splits come
/// from the strategy, the inner folds from the outer training rows only
/// (same key, `inner_folds` folds), and lambda is chosen by the 'min' or
/// '1se' rule. Without `lambdas` each outer split uses glmnet's default grid
/// of its own training rows.
pub fn nested_cv_reg_path(
    y: &[f64],
    x: &[Vec<f64>],
    keys: Option<&[f64]>,
    spec: &CvSpec,
    opts: &RegPathOptions,
) -> StatsResult<NestedCvResult> {
    check_keys(keys, y.len())?;
    if spec.repeats > 1 {
        return Err(invalid(
            "repeats",
            "repeats is not supported by nested cross-validation".to_string(),
        ));
    }
    let data = prepare(y, x, opts.fit_intercept)?;
    let up = upstream_options(opts)?;
    let outer = strategy_splits(keys, &data, spec)?;
    let nested = NestedCvOptions {
        inner: inner_folds(keys, &data, spec)?,
        rule: if spec.one_se {
            LambdaRule::OneSe
        } else {
            LambdaRule::Min
        },
        metric: spec.metric(),
        n_lambda: opts.n_lambda,
        lambda_min_ratio: (!opts.lambda_min_ratio.is_nan()).then_some(opts.lambda_min_ratio),
    };
    let lambdas = (!opts.lambdas.is_empty()).then_some(opts.lambdas.as_slice());
    let r = nested_cv_path(&data.x, &data.y, lambdas, &outer, &up, &nested)
        .map_err(StatsError::from)?;
    Ok(NestedCvResult {
        lambda: r.folds.iter().map(|f| f.lambda).collect(),
        test_error: r.folds.iter().map(|f| f.test_metric).collect(),
        inner_cv_mean: r.folds.iter().map(|f| f.inner.cv_mean[f.index]).collect(),
        inner_cv_se: r.folds.iter().map(|f| f.inner.cv_se[f.index]).collect(),
        n_train: r.folds.iter().map(|f| f.n_train).collect(),
        n_test: r.folds.iter().map(|f| f.n_test).collect(),
        n_inner_folds: r.folds.iter().map(|f| f.inner.n_folds).collect(),
        mean: r.mean,
        se: r.se,
        n_observations: data.rows.len(),
    })
}

/// The LARS / Lasso-LARS coefficient path (`FittedLars::path`), one entry per
/// knot; `lambdas` are the knots' regularisation levels (`alphas`).
pub fn lars_path(y: &[f64], x: &[Vec<f64>], options: &LarsOptions) -> StatsResult<RegPathResult> {
    let data = prepare(y, x, options.fit_intercept)?;
    let mut builder = LarsRegressor::builder()
        .fit_intercept(options.fit_intercept)
        .method(if options.method_lasso {
            LarsMethod::Lasso
        } else {
            LarsMethod::Lar
        })
        .alpha(options.alpha)
        .standardize(options.standardize);
    if options.n_nonzero_coefs > 0 {
        builder = builder.n_nonzero_coefs(options.n_nonzero_coefs as usize);
    }
    let fitted = builder
        .build()
        .fit(&data.x, &data.y)
        .map_err(StatsError::from)?;
    Ok(expand_path(&fitted.path(), &data, options.fit_intercept))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 / 3.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 / 2.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + x1[i] - 0.5 * x2[i] + ((i * 3) % 7) as f64 / 4.0)
            .collect();
        (y, vec![x1, x2])
    }

    #[test]
    fn default_grid_and_layout() {
        let (y, x) = data();
        let opts = RegPathOptions {
            l1_ratio: 1.0,
            n_lambda: 10,
            ..Default::default()
        };
        let p = fit_reg_path(&y, &x, &opts).unwrap();
        assert_eq!(p.lambdas.len(), 10);
        assert_eq!(p.coefficients.len(), 20);
        assert_eq!(p.df[0], 0);
        assert!(p.lambdas[0] > p.lambdas[9]);
    }

    #[test]
    fn nan_rows_and_constant_columns() {
        let (mut y, mut x) = data();
        y[3] = f64::NAN;
        x.push(vec![2.0; y.len()]);
        let opts = RegPathOptions {
            lambdas: vec![0.5, 0.1],
            ..Default::default()
        };
        let p = fit_reg_path(&y, &x, &opts).unwrap();
        assert_eq!(p.n_observations, 39);
        assert!(p.coefficients[2].is_nan() && p.coefficients[5].is_nan());
        assert!(p.coefficients[0].is_finite());
    }

    #[test]
    fn cv_seeded_and_fold_ids_agree() {
        let (y, x) = data();
        let opts = RegPathOptions {
            l1_ratio: 0.5,
            n_lambda: 8,
            ..Default::default()
        };
        let a = cv_reg_path(&y, &x, None, 5, 3, &opts).unwrap();
        let ids: Vec<i64> = kfold_ids(y.len(), 5, 3)
            .iter()
            .map(|&k| k as i64 + 10)
            .collect();
        let b = cv_reg_path(&y, &x, Some(&ids), 0, 0, &opts).unwrap();
        // Same folds; only the summation order of the folds differs.
        for (u, v) in a.cv_mean.iter().zip(&b.cv_mean) {
            assert!((u - v).abs() < 1e-12 * (1.0 + v.abs()));
        }
        assert_eq!(a.n_folds, 5);
        assert!(cv_reg_path(&y, &x, None, 2, 3, &opts).is_err());
    }

    #[test]
    fn strategies_map_to_upstream_folds() {
        let (y, x) = data();
        let opts = RegPathOptions {
            n_lambda: 6,
            ..Default::default()
        };
        // Group keys (hashes as doubles in the extension): 8 groups of 5 rows.
        let groups: Vec<f64> = (0..40).map(|i| ((i % 8) * 1_000_003) as f64).collect();
        let spec = CvSpec {
            strategy: CvStrategy::Group,
            n_folds: 4,
            ..Default::default()
        };
        let a = cv_reg_path_strategy(&y, &x, Some(&groups), &spec, &opts).unwrap();
        // The same folds passed as fold ids.
        let codes: Vec<usize> = (0..40).map(|i| i % 8).collect();
        let ids: Vec<f64> = group_kfold_ids(&codes, 4, GroupBalance::Rows, 0)
            .unwrap()
            .iter()
            .map(|&f| f as f64)
            .collect();
        let fold_spec = CvSpec {
            strategy: CvStrategy::FoldIds,
            ..Default::default()
        };
        let b = cv_reg_path_strategy(&y, &x, Some(&ids), &fold_spec, &opts).unwrap();
        assert_eq!(a.cv_mean, b.cv_mean);
        assert_eq!(a.n_folds, 4);

        let logo = CvSpec {
            strategy: CvStrategy::Logo,
            ..Default::default()
        };
        assert_eq!(
            cv_reg_path_strategy(&y, &x, Some(&groups), &logo, &opts)
                .unwrap()
                .n_folds,
            8
        );

        // Time series over 20 time points of 2 rows each.
        let time: Vec<f64> = (0..40).map(|i| (i / 2) as f64).collect();
        let ts = CvSpec {
            strategy: CvStrategy::TimeSeries,
            n_folds: 4,
            test_size: 3,
            gap: 1,
            ..Default::default()
        };
        assert_eq!(
            cv_reg_path_strategy(&y, &x, Some(&time), &ts, &opts)
                .unwrap()
                .n_folds,
            4
        );

        // Repeated stratified k-fold on quantile strata of y.
        let strat = CvSpec {
            strategy: CvStrategy::Stratified,
            n_folds: 4,
            repeats: 3,
            ..Default::default()
        };
        assert_eq!(
            cv_reg_path_strategy(&y, &x, None, &strat, &opts)
                .unwrap()
                .n_folds,
            12
        );

        // A strategy that needs a key, and repeats where they do not apply.
        assert!(cv_reg_path_strategy(&y, &x, None, &spec, &opts).is_err());
        let bad = CvSpec {
            repeats: 2,
            ..spec.clone()
        };
        assert!(cv_reg_path_strategy(&y, &x, Some(&groups), &bad, &opts).is_err());
    }

    #[test]
    fn nested_cv_shapes() {
        let (y, x) = data();
        let opts = RegPathOptions {
            n_lambda: 8,
            ..Default::default()
        };
        let groups: Vec<f64> = (0..40).map(|i| (i % 8) as f64).collect();
        let spec = CvSpec {
            strategy: CvStrategy::Group,
            n_folds: 4,
            inner_folds: 3,
            one_se: true,
            ..Default::default()
        };
        let r = nested_cv_reg_path(&y, &x, Some(&groups), &spec, &opts).unwrap();
        assert_eq!(r.lambda.len(), 4);
        assert_eq!(r.n_test.iter().sum::<usize>(), 40);
        assert!(r.n_inner_folds.iter().all(|&k| k == 3));
        assert!(r.mean.is_finite() && r.se.is_finite());
        let plain = CvSpec {
            n_folds: 5,
            ..Default::default()
        };
        let r = nested_cv_reg_path(&y, &x, None, &plain, &opts).unwrap();
        assert_eq!(r.lambda.len(), 5);
    }

    #[test]
    fn lars_path_ends_at_ols() {
        let (y, x) = data();
        let p = lars_path(
            &y,
            &x,
            &LarsOptions {
                method_lasso: true,
                standardize: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.df[0], 0);
        assert!(p.lambdas.len() >= 3);
    }
}
