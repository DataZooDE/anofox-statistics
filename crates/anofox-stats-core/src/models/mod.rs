//! Regression model implementations

mod elasticnet;
mod ols;
mod ridge;

pub use elasticnet::fit_elasticnet;
pub use ols::fit_ols;
pub use ridge::fit_ridge;
