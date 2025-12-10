//! Diagnostic functions for regression models

mod information_criteria;
mod jarque_bera;
mod residuals;
mod vif;

pub use information_criteria::{compute_aic, compute_aic_bic, compute_bic};
pub use jarque_bera::{jarque_bera, JarqueBeraResult};
pub use residuals::{compute_residuals, ResidualType, ResidualsResult};
pub use vif::compute_vif;
