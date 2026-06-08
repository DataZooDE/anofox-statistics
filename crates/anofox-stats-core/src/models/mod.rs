//! Regression model implementations

mod aid;
mod alm;
mod bls;
mod elasticnet;
mod glm;
mod huber;
mod isotonic;
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

pub use aid::{compute_aid, compute_aid_anomalies};
pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use bls::{fit_bls, fit_nnls};
pub use elasticnet::fit_elasticnet;
pub use glm::{fit_binomial, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult};
pub use huber::{fit_huber, HuberResult};
pub use isotonic::fit_isotonic;
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
