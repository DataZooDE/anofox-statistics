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
mod huber;
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
pub use huber::{fit_huber, HuberResult};
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
pub use rls::{fit_rls, RlsOptions, RlsState};
pub use theil_sen::{fit_theilsen, TheilSenResult};
pub use wls::fit_wls;
