pub mod dagostino;
pub mod normality;

pub use dagostino::{dagostino_k_squared, DAgostinoResult};
pub use normality::{shapiro_wilk, ShapiroWilkResult};
