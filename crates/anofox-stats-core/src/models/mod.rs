//! Regression model implementations

mod aid;
mod alm;
mod bls;
mod elasticnet;
mod glm;
mod isotonic;
mod ols;
mod pls;
mod predict;
mod quantile;
mod ridge;
mod rls;
mod wls;

pub use aid::{compute_aid, compute_aid_anomalies};
pub use alm::{fit_alm, AlmInferenceResult, AlmResult};
pub use bls::{fit_bls, fit_nnls};
pub use elasticnet::fit_elasticnet;
pub use glm::{fit_binomial, fit_negbinomial, fit_poisson, fit_tweedie, GlmResult};
pub use isotonic::fit_isotonic;
pub use ols::fit_ols;
pub use pls::fit_pls;
pub use predict::predict;
pub use quantile::fit_quantile;
pub use ridge::fit_ridge;
pub use rls::{fit_rls, RlsOptions, RlsState};
pub use wls::fit_wls;
