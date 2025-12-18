pub mod energy;
pub mod mmd;

pub use energy::{energy_distance_test, energy_distance_test_1d, EnergyDistanceResult};
pub use mmd::{mmd_test, mmd_test_1d, Kernel, MMDResult};
