//! Regression model implementations

mod aft;
mod aid;
mod alm;
mod augment;
mod bayesian;
mod bls;
mod cluster;
pub(crate) mod cox;
mod eb_shrink;
mod elasticnet;
mod glm;
mod glm_weighted;
mod glmm;
mod gls;
mod huber;
mod interval;
mod isotonic;
mod lars;
mod likelihood;
mod lm_dynamic;
mod lowess;
mod ls_state;
mod mediation;
mod ols;
mod ols_anova;
mod ols_anova_table;
mod passive_aggressive;
mod path;
mod pentrace;
mod pls;
mod predict;
mod predict_se;
mod pspline;
mod quantile;
mod ransac;
mod ridge;
mod rls;
mod theil_sen;
mod wald_anova;
mod wls;

pub use aft::{
    aft_augment, aft_survival_curve, fit_aft, AftAugment, AftDistribution, AftFitResult,
    AftInference, AftOptions, AftResult,
};
pub use aid::{compute_aid, compute_aid_anomalies};
pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use anofox_regression::solvers::{OrdinalLink, PentraceCriterion};
pub use augment::{
    augment_binomial, augment_gamma, augment_logistic, augment_negbinomial, augment_ols,
    augment_poisson, augment_ridge, augment_tweedie, augment_wls, partial_residuals_binomial,
    partial_residuals_gamma, partial_residuals_logistic, partial_residuals_negbinomial,
    partial_residuals_ols, partial_residuals_poisson, partial_residuals_tweedie,
    partial_residuals_wls, AugmentResult, PartialResidualsResult,
};
pub use bayesian::{
    fit_bayesian, fit_predict_bayesian, BayesianFitResult, BayesianOptions, BayesianPredictions,
};
pub use bls::{fit_bls, fit_nnls};
pub use cluster::{fit_ols_cluster, fit_wls_cluster, ClusterFit, ClusterSpec};
pub use cox::{
    cox_survival_curve, fit_cox, CoxAugment, CoxModel, CoxRows, CoxSpec, CoxSummary, CoxTies,
    ZphTransform,
};
pub use eb_shrink::{eb_shrink, EbShrinkOptions, EbShrinkResult, ShrunkenGroup, TauMethod};
pub use elasticnet::fit_elasticnet;
pub use glm::{
    fit_binomial, fit_gamma, fit_logistic, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult,
    LogisticResult,
};
pub use glm_weighted::{
    fit_gamma_weighted, fit_poisson_weighted, fit_tweedie_weighted, WeightedGlmFit,
};
pub use glmm::{
    fit_glmm, fit_glmm_crossed, FactorVariance, GlmmDdf, GlmmFamily, GlmmObservations, GlmmOptions,
    GlmmResult, RandomEffect,
};
pub use gls::{
    fit_gls, GlsAugment, GlsCorrelationKind, GlsModel, GlsResidualType, GlsRows, GlsSpec,
    GlsSummary, GlsVarianceKind, GlsVariogramBin,
};
pub use huber::{fit_huber, HuberResult};
pub use interval::{
    interval_matrix, predict_with_centroid_interval, predict_with_interval_matrix,
    predict_with_interval_matrix_batch, t_critical, IntervalType,
};
pub use isotonic::fit_isotonic;
pub use lars::fit_lars;
pub use likelihood::{
    glm_profile_confint, glm_score_test, lr_test, model_lr_test, GlmFamilySpec, GlmScoreTestResult,
    LrTestResult, ModelLikelihood,
};
pub use lm_dynamic::fit_lm_dynamic;
pub use lowess::fit_lowess;
pub use ls_state::{
    fit_ols_from_state, fit_ridge_from_state, fit_wls_from_state, ls_state_from_rows,
    ls_state_info, ls_state_merge, LsStateInfo, StateFit,
};
pub use mediation::{
    mediate, BootCi, MediationEffect, MediationMethod, MediationOptions, MediationResult,
};
pub use ols::fit_ols;
pub use ols_anova::{ols_anova, NestedFTest, NestedFTestOptions};
pub use ols_anova_table::{ols_anova_table, AnovaTable, AnovaTableOptions, AnovaType};
pub use passive_aggressive::{
    fit_passive_aggressive, PassiveAggressiveOptions, PassiveAggressiveResult,
};
pub use path::{
    cv_reg_path, cv_reg_path_strategy, fit_reg_path, lars_path, nested_cv_reg_path, CvPathResult,
    CvSpec, CvStrategy, NestedCvResult, RegPathOptions, RegPathResult,
};
pub use pentrace::{
    penalized_ols, pentrace, PenalizedOlsResult, Penalty, PenaltyGrid, PenaltyTerms, PentraceModel,
    PentraceResult, PentraceRowResult,
};
pub use pls::fit_pls;
pub use predict::predict;
pub use predict_se::{
    coefficient_vcov, glm_vcov, predict_se_glm, predict_se_linear, PredictInterval, PredictSe,
};
pub use pspline::{
    fit_predict_pspline, pspline_predict, PSplineFit, PSplineOptions, PSplinePredictions,
    StoredPSpline,
};
pub use quantile::{fit_quantile, quantile_process, QuantileProcessRow, QuantileSeType};
pub use ransac::{fit_ransac, RansacResult};
pub use ridge::{fit_ridge, ridge_selected_lambda, DEFAULT_COND_THRESHOLD, DEFAULT_LAMBDA_SMALL};
pub use rls::{fit_rls, RlsOptions};
pub use theil_sen::{fit_theilsen, TheilSenResult};
pub use wald_anova::{
    fastbw, wald_anova, FastbwResult, FastbwRule, FastbwStep, ModelCovariance, TermSpec,
    WaldAnovaResult, WaldAnovaRow,
};
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
