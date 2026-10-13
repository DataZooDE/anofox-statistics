//! Regularisation paths and k-fold cross-validation curves (#154).
//!
//! Marshalling only: the ridge / elastic-net path, glmnet's default lambda
//! grid, the seeded folds and the `cv.glmnet` aggregates come from
//! `anofox_regression::solvers::path`, the LARS path from
//! `FittedLars::path`. This module drops rows with a non-finite value, drops
//! the columns the fits drop (reported as NaN coefficients) and lays the path
//! out per lambda.

use crate::errors::{StatsError, StatsResult};
use crate::types::LarsOptions;
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
