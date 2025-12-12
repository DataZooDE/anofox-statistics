//! Regression model implementations

mod alm;
mod bls;
mod elasticnet;
mod glm;
mod ols;
mod predict;
mod ridge;
mod rls;
mod wls;

pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use bls::{fit_bls, fit_nnls};
pub use elasticnet::fit_elasticnet;
pub use glm::{fit_binomial, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult};
pub use ols::fit_ols;
pub use predict::predict;
pub use ridge::fit_ridge;
pub use rls::{fit_rls, RlsOptions, RlsState};
pub use wls::fit_wls;
