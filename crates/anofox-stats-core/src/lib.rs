//! anofox-stats-core: Core statistics library for DuckDB extension
//!
//! This crate provides statistical regression models wrapping regress-rs,
//! designed for use via FFI in a DuckDB extension.

pub mod diagnostics;
pub mod errors;
pub mod models;
pub mod tests;
pub mod types;

pub use errors::{StatsError, StatsResult};
pub use types::*;
