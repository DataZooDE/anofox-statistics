//! Causal mediation analysis (issue #223): thin delegation to
//! `anofox_regression::solvers::mediation` (the linear no-interaction case of R's
//! `mediation::mediate`). This module only maps errors.

pub use anofox_regression::solvers::mediation::{
    BootCi, MediationEffect, MediationMethod, MediationOptions, MediationResult,
};

use crate::errors::StatsResult;

/// Mediation of the effect of `x` on `y` through `m`, adjusting for
/// `covariates` (one slice per covariate).
pub fn mediate(
    y: &[f64],
    m: &[f64],
    x: &[f64],
    covariates: &[Vec<f64>],
    options: &MediationOptions,
) -> StatsResult<MediationResult> {
    Ok(anofox_regression::solvers::mediation::mediate(
        y, m, x, covariates, options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::StatsError;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..20).map(|i| (i % 2) as f64).collect();
        let m: Vec<f64> = (0..20)
            .map(|i| 0.5 * x[i] + ((i * 7 % 5) as f64) * 0.1)
            .collect();
        let y: Vec<f64> = (0..20)
            .map(|i| 0.3 * x[i] + 0.8 * m[i] + ((i * 3 % 7) as f64) * 0.05)
            .collect();
        (y, m, x)
    }

    #[test]
    fn delegates_mediation() {
        let (y, m, x) = data();
        let o = MediationOptions {
            n_sims: 50,
            ..Default::default()
        };
        let r = mediate(&y, &m, &x, &[], &o).unwrap();
        assert_eq!(r.n_obs, 20);
        assert!((r.acme.estimate + r.ade.estimate - r.total.estimate).abs() < 1e-12);
    }

    #[test]
    fn maps_errors() {
        let (y, m, x) = data();
        let o = MediationOptions::default();
        assert!(matches!(
            mediate(&y[..3], &m[..3], &x[..3], &[], &o),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        assert!(matches!(
            mediate(&y, &m[1..], &x, &[], &o),
            Err(StatsError::DimensionMismatchMsg(_))
        ));
        assert!(matches!(
            mediate(&y, &m, &[1.0; 20], &[], &o),
            Err(StatsError::SingularMatrix)
        ));
        let bad = MediationOptions {
            conf_level: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            mediate(&y, &m, &x, &[], &bad),
            Err(StatsError::InvalidValue { .. })
        ));
    }
}
