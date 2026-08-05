//! Diagnostic functions for regression models, and for Markov-chain output.

mod condition;
mod information_criteria;
mod jarque_bera;
mod mcmc;
mod residuals;
mod separation;
mod vif;

pub use condition::{compute_condition_diagnostic, compute_condition_number};
pub use information_criteria::{compute_aic, compute_aic_bic, compute_bic};
pub use jarque_bera::{jarque_bera, JarqueBeraResult};
pub use mcmc::{ess_bulk, ess_tail, rhat};
pub use residuals::{compute_residuals, ResidualType, ResidualsResult};
pub use separation::{check_binary_separation_diagnostic, check_count_sparsity_diagnostic};
pub use vif::compute_vif;
