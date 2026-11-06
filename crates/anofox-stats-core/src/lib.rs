//! anofox-stats-core: Core statistics library for DuckDB extension
//!
//! This crate provides statistical regression models wrapping regress-rs,
//! designed for use via FFI in a DuckDB extension.

pub mod diagnostics;
pub mod errors;
pub mod imputation;
pub mod models;
pub mod multinomial;
pub mod multivariate;
pub mod tests;
pub mod types;
mod validation;

pub use errors::{StatsError, StatsResult};
pub use types::*;
