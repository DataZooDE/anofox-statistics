//! Weighted Least Squares (WLS) regression using native WlsRegressor

use crate::errors::{StatsError, StatsResult};
use crate::models::cluster::{
    inference_from_vcov, lm_cluster_inference, ClusterFit, ClusterSpec, LmClusterInput,
};
use crate::types::{
    FitResult, FitResultCore, FitResultInference, HcType, ModelSummary, SolverType, WlsOptions,
};
use anofox_regression::core::{HasModelInfo, ModelInfo};
use anofox_regression::prelude::*;
use faer::{Col, Mat};

/// Convert our SolverType to anofox_regression's SolverType
fn convert_solver(solver: SolverType) -> anofox_regression::SolverType {
    match solver {
        SolverType::Qr => anofox_regression::SolverType::Qr,
        SolverType::Svd => anofox_regression::SolverType::Svd,
        SolverType::Cholesky => anofox_regression::SolverType::Cholesky,
    }
}

/// Fit a Weighted Least Squares regression model
///
/// # Arguments
/// * `y` - Response variable (n observations)
/// * `x` - Feature matrix (p features, each with n observations)
/// * `weights` - Observation weights (n observations, must be positive)
/// * `options` - Fitting options
///
/// # Returns
/// * `FitResult` containing coefficients, R-squared, and optionally inference statistics
pub fn fit_wls(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
) -> StatsResult<FitResult> {
    fit_wls_impl(y, x, weights, options, None).map(|c| c.fit)
}

/// `fit_wls`, with the cluster-robust inference of `cluster` when given (#309).
pub(crate) fn fit_wls_impl(
    y: &[f64],
    x: &[Vec<f64>],
    weights: &[f64],
    options: &WlsOptions,
    cluster: Option<&ClusterSpec>,
) -> StatsResult<ClusterFit> {
    // Validate inputs
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }
    if x.is_empty() {
        return Err(StatsError::EmptyInput { field: "x" });
    }
    if weights.is_empty() {
        return Err(StatsError::EmptyInput { field: "weights" });
    }

    let n_obs = y.len();
    let n_features = x.len();

    // Check weights length matches y
    if weights.len() != n_obs {
        return Err(StatsError::DimensionMismatch {
            y_len: n_obs,
            x_rows: weights.len(),
        });
    }

    // Check all feature vectors have same length as y
    crate::validation::validate_x_columns(n_obs, x)?;

    // Filter out rows with NaN/Inf values or non-positive weights
    let valid_indices: Vec<usize> = (0..n_obs)
        .filter(|&i| {
            !y[i].is_nan()
                && !y[i].is_infinite()
                && weights[i] > 0.0
                && !weights[i].is_nan()
                && !weights[i].is_infinite()
                && x.iter()
                    .all(|col| !col[i].is_nan() && !col[i].is_infinite())
        })
        .collect();

    if valid_indices.is_empty() {
        return Err(StatsError::NoValidData);
    }

    let n_valid = valid_indices.len();

    // Non-estimable columns (constant with an intercept, all-zero without) are
    // left out of the design and reported as NaN; aliasing, the no-intercept
    // (uncentered) R²/F and intercept-only fits are upstream's.
    let dropped = crate::validation::droppable_columns(x, &valid_indices, options.fit_intercept);
    let kept: Vec<usize> = (0..n_features).filter(|&j| !dropped[j]).collect();
    if (kept.is_empty() && !options.fit_intercept)
        || n_valid < kept.len() + usize::from(options.fit_intercept)
    {
        return Err(StatsError::InsufficientData {
            rows: n_valid,
            cols: n_features,
        });
    }
    if kept.is_empty() && n_valid == 1 {
        // A single observation: the intercept is that value, its residual
        // variance is undefined (upstream needs two rows).
        return Ok(ClusterFit::plain(FitResult {
            core: FitResultCore {
                coefficients: vec![f64::NAN; n_features],
                intercept: Some(y[valid_indices[0]]),
                r_squared: 0.0,
                adj_r_squared: 0.0,
                residual_std_error: f64::NAN,
                n_observations: 1,
                n_features,
            },
            inference: None,
            diagnostics: None,
            summary: ModelSummary::new(ModelInfo::gaussian("wls")),
        }));
    }

    let y_col = Col::from_fn(n_valid, |i| y[valid_indices[i]]);
    let x_mat = Mat::from_fn(n_valid, kept.len(), |i, j| x[kept[j]][valid_indices[i]]);
    let w_col = Col::from_fn(n_valid, |i| weights[valid_indices[i]]);

    let mut builder = WlsRegressor::builder()
        .with_intercept(options.fit_intercept)
        .weights(w_col)
        .compute_inference(options.compute_inference && cluster.is_none())
        .confidence_level(options.confidence_level)
        .solve_method(convert_solver(options.solver));
    if let Some(tol) = options.alias_tolerance {
        builder = builder.alias_tolerance(tol);
    }
    let fitted = builder
        .build()
        .fit(&x_mat, &y_col)
        .map_err(StatsError::from)?;
    let result = fitted.result();

    // Scatter a reduced (kept-columns) vector back to full width, NaN elsewhere.
    let expand = |reduced: Option<&Col<f64>>| -> Vec<f64> {
        let mut full = vec![f64::NAN; n_features];
        if let Some(col) = reduced {
            for (r, &j) in kept.iter().enumerate() {
                full[j] = col[r];
            }
        }
        full
    };

    let core = FitResultCore {
        coefficients: expand(Some(&result.coefficients)),
        intercept: result.intercept,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        residual_std_error: result.rmse,
        n_observations: n_valid,
        n_features,
    };

    let classical = || FitResultInference {
        std_errors: expand(result.std_errors.as_ref()),
        t_values: expand(result.t_statistics.as_ref()),
        p_values: expand(result.p_values.as_ref()),
        ci_lower: expand(result.conf_interval_lower.as_ref()),
        ci_upper: expand(result.conf_interval_upper.as_ref()),
        confidence_level: options.confidence_level,
        f_statistic: Some(result.f_statistic),
        f_pvalue: Some(result.f_pvalue),
    };
    let mut summary = ModelSummary::from_result(
        fitted.model_info(),
        result,
        options.compute_inference && options.hc_type.is_none() && cluster.is_none(),
    );
    let mut clustered = None;
    let inference = if !options.compute_inference {
        None
    } else if let Some(spec) = cluster {
        let input = LmClusterInput {
            spec,
            valid_indices: &valid_indices,
            kept: &kept,
            n_features,
            result,
            hc_type: options.hc_type,
            confidence_level: options.confidence_level,
        };
        let (inf, vcov, n_clusters) = lm_cluster_inference(&input, &mut summary, |codes, opts| {
            fitted.cluster_vcov(&x_mat, codes, opts)
        });
        clustered = Some((vcov, n_clusters));
        Some(inf)
    } else if let Some(hc_type) = options.hc_type {
        // Heteroskedasticity-consistent covariance of the *weighted* fit
        // (sandwich::vcovHC(lm(..., weights = w), type = hc_type), upstream
        // `FittedWls::hc_vcov`); the standard errors and tests come from it
        // and it is the model's vcov (so predict(se_fit) uses it). The F
        // test stays the classical one, as with `ols_fit_agg(hc_type)`.
        let no_clusters = ClusterSpec {
            codes: Vec::new(),
            cadjust: false,
        };
        let input = LmClusterInput {
            spec: &no_clusters,
            valid_indices: &valid_indices,
            kept: &kept,
            n_features,
            result,
            hc_type: Some(hc_type),
            confidence_level: options.confidence_level,
        };
        let upstream_hc = match hc_type {
            HcType::HC0 => anofox_regression::HcType::HC0,
            HcType::HC1 => anofox_regression::HcType::HC1,
            HcType::HC2 => anofox_regression::HcType::HC2,
            HcType::HC3 => anofox_regression::HcType::HC3,
        };
        match fitted.hc_vcov(&x_mat, upstream_hc) {
            Ok(v) => {
                let mut hc_summary = summary;
                let (inf, full) = inference_from_vcov(&input, &mut hc_summary, &v);
                summary = hc_summary;
                clustered = Some((Some(full), Vec::new()));
                Some(FitResultInference {
                    f_statistic: Some(result.f_statistic),
                    f_pvalue: Some(result.f_pvalue),
                    ..inf
                })
            }
            Err(_) => {
                // Fall back to classical inference if HC fails (e.g. a row
                // with leverage 1 for HC2 / HC3); the intercept inference and
                // the covariance then are the classical ones too.
                summary = ModelSummary::from_result(fitted.model_info(), result, true);
                Some(classical())
            }
        }
    } else {
        Some(classical())
    };

    // Exact leave-one-out statistics from the leverage (#220); a fit they
    // cannot be computed for keeps them NaN.
    if options.compute_loo {
        if let Ok(p) = anofox_regression::diagnostics::press(&fitted, &x_mat, &y_col, None) {
            summary.set_press(&p);
        }
    }

    let fit = FitResult {
        core,
        inference,
        diagnostics: None,
        summary,
    };
    Ok(match clustered {
        Some((vcov, n_clusters)) => ClusterFit {
            fit,
            vcov,
            n_clusters,
        },
        None => ClusterFit::plain(fit),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wls_uniform_weights() {
        // With uniform weights, WLS should give same results as OLS
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0; 10]; // Uniform weights

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should be close to y = 2*x
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_wls_heteroscedastic() {
        // Data with increasing variance - higher weights for more reliable observations
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        // y = 2*x + noise, where noise variance increases with x
        let y = vec![2.1, 4.0, 5.9, 8.3, 9.5, 12.5, 13.0, 17.0, 17.0, 22.0];
        // Higher weights for observations with lower variance (early observations)
        let weights = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let options = WlsOptions {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Coefficient should still be close to 2
        assert!(result.core.coefficients[0] > 1.5 && result.core.coefficients[0] < 2.5);
    }

    #[test]
    fn test_wls_zero_weights() {
        // Zero weights should be filtered out
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        // Last two observations have zero weight - should be ignored
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // Should still get good fit on remaining 8 observations
        assert!(result.core.r_squared > 0.99);
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_negative_weights_filtered() {
        // Negative weights should be filtered out (not valid)
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let weights = vec![1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options).unwrap();

        // 8 valid observations (2 with negative weights filtered)
        assert_eq!(result.core.n_observations, 8);
    }

    #[test]
    fn test_wls_dimension_mismatch() {
        let x = vec![vec![1.0, 2.0, 3.0]];
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0]; // Wrong length

        let options = WlsOptions::default();
        let result = fit_wls(&y, &x, &weights, &options);

        assert!(matches!(result, Err(StatsError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_wls_with_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let options = WlsOptions {
            compute_inference: true,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!(result.inference.is_some());
        let inf = result.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        assert!(inf.p_values[0] < 0.05);
    }

    #[test]
    fn test_wls_hc1_inference() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];

        let options = WlsOptions {
            compute_inference: true,
            hc_type: Some(HcType::HC1),
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!(result.inference.is_some());
        let inf = result.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
    }

    #[test]
    fn test_wls_svd_solver() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1];
        let weights = vec![1.0; 10];

        let options = WlsOptions {
            solver: SolverType::Svd,
            ..Default::default()
        };

        let result = fit_wls(&y, &x, &weights, &options).unwrap();
        assert!((result.core.coefficients[0] - 2.0).abs() < 0.1);
        assert!(result.core.r_squared > 0.99);
    }

    #[test]
    fn test_wls_no_intercept_ones_column_matches_intercept_model() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];
        let w = vec![1.0; 8];
        let opts_int = WlsOptions {
            compute_inference: true,
            ..Default::default()
        };
        let with_int = fit_wls(&y, std::slice::from_ref(&x1), &w, &opts_int).unwrap();
        let opts = WlsOptions {
            fit_intercept: false,
            compute_inference: true,
            ..Default::default()
        };
        // Constant column of 2s: its coefficient is intercept / 2.
        let no_int = fit_wls(&y, &[vec![2.0; 8], x1], &w, &opts).unwrap();
        let b0 = no_int.core.coefficients[0];
        assert!((b0 * 2.0 - with_int.core.intercept.unwrap()).abs() < 1e-10);
        assert!((no_int.core.coefficients[1] - with_int.core.coefficients[0]).abs() < 1e-10);
        let inf = no_int.inference.unwrap();
        assert!(inf.std_errors[0].is_finite() && inf.std_errors[0] > 0.0);
        assert!(inf.ci_lower[0] < b0 && b0 < inf.ci_upper[0]);
    }

    #[test]
    fn test_wls_hc_uses_the_weights() {
        // R: sqrt(diag(sandwich::vcovHC(lm(y ~ x1, weights = w), type = t)))
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.3, 3.9, 6.8, 7.2, 10.9, 11.1, 15.2, 15.0, 19.9, 18.7];
        let w = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0];
        let cases = [
            (HcType::HC0, 0.429959251233901, 0.0885272074455763),
            (HcType::HC1, 0.480709056656957, 0.0989764268532671),
            (HcType::HC2, 0.490611034256085, 0.101923272316287),
            (HcType::HC3, 0.565324842199749, 0.118861951834527),
        ];
        for (hc, se0, se1) in cases {
            let opts = WlsOptions {
                compute_inference: true,
                hc_type: Some(hc),
                ..Default::default()
            };
            let r = fit_wls(&y, std::slice::from_ref(&x1), &w, &opts).unwrap();
            let inf = r.inference.unwrap();
            assert!((inf.std_errors[0] - se1).abs() < 1e-9 * se1, "{hc:?}");
            let i_se = r.summary.intercept_std_error;
            assert!((i_se - se0).abs() < 1e-9 * se0, "{hc:?}: {i_se}");
        }
    }
}
