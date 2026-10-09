//! Regression model implementations

mod aft;
mod aft_dist;
mod aid;
mod alm;
mod bls;
mod eb_shrink;
mod elasticnet;
mod glm;
pub mod glm_engine;
mod glmm;
mod huber;
mod interval;
mod isotonic;
mod lars;
mod lm_dynamic;
mod lowess;
mod ols;
mod pls;
mod predict;
mod quantile;
mod ransac;
mod ridge;
mod rls;
mod theil_sen;
mod wls;

pub use aft::{fit_aft, AftFitResult, AftInference, AftOptions, AftResult};
pub use aft_dist::AftDistribution;
pub use aid::{compute_aid, compute_aid_anomalies};
pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use bls::{fit_bls, fit_nnls};
pub use eb_shrink::{eb_shrink, EbShrinkOptions, EbShrinkResult, ShrunkenGroup, TauMethod};
pub use elasticnet::fit_elasticnet;
pub use glm::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult,
    LogisticResult,
};
pub use glmm::{
    fit_glmm, fit_glmm_crossed, FactorVariance, GlmmFamily, GlmmOptions, GlmmResult, RandomEffect,
};
pub use huber::{fit_huber, HuberResult};
pub use interval::{
    interval_matrix, predict_with_centroid_interval, predict_with_interval_matrix, t_critical,
    IntervalType,
};
pub use isotonic::fit_isotonic;
pub use lars::fit_lars;
pub use lm_dynamic::fit_lm_dynamic;
pub use lowess::fit_lowess;
pub use ols::fit_ols;
pub use pls::fit_pls;
pub use predict::predict;
pub use quantile::fit_quantile;
pub use ransac::{fit_ransac, RansacResult};
pub use ridge::fit_ridge;
pub use rls::{fit_rls, RlsOptions};
pub use theil_sen::{fit_theilsen, TheilSenResult};
pub use wls::fit_wls;

/// Project upstream coefficient inference onto `FitResultInference`; fields
/// upstream leaves unset become one NaN (NULL in SQL) per feature.
pub(crate) fn inference_from_result(
    result: &anofox_regression::core::RegressionResult,
    n_features: usize,
    confidence_level: f64,
) -> crate::types::FitResultInference {
    let col = |c: Option<&faer::Col<f64>>| -> Vec<f64> {
        c.map_or_else(
            || vec![f64::NAN; n_features],
            |c| c.iter().copied().collect(),
        )
    };
    crate::types::FitResultInference {
        std_errors: col(result.std_errors.as_ref()),
        t_values: col(result.t_statistics.as_ref()),
        p_values: col(result.p_values.as_ref()),
        ci_lower: col(result.conf_interval_lower.as_ref()),
        ci_upper: col(result.conf_interval_upper.as_ref()),
        confidence_level,
        f_statistic: Some(result.f_statistic),
        f_pvalue: Some(result.f_pvalue),
    }
}
