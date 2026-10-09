//! AID (Automatic Identification of Demand): thin delegation to
//! `anofox_regression::solvers::aid::heuristic`, which holds the moment-heuristic
//! classification and anomaly flagging. This module only maps errors.

use anofox_regression::solvers::aid::heuristic;

use crate::errors::StatsResult;
use crate::types::{AidAnomalyFlags, AidOptions, AidResult};

/// Classify a demand series (regular/intermittent, best-fit distribution,
/// anomaly counts). See `anofox_regression::solvers::aid::heuristic::compute_aid`.
pub fn compute_aid(y: &[f64], options: &AidOptions) -> StatsResult<AidResult> {
    Ok(heuristic::compute_aid(y, options)?)
}

/// Per-observation anomaly flags, in input order.
/// See `anofox_regression::solvers::aid::heuristic::compute_aid_anomalies`.
pub fn compute_aid_anomalies(y: &[f64], options: &AidOptions) -> StatsResult<Vec<AidAnomalyFlags>> {
    Ok(heuristic::compute_aid_anomalies(y, options)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::StatsError;

    #[test]
    fn delegates_classification() {
        let y = [0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 0.0, 6.0];
        let r = compute_aid(&y, &AidOptions::default()).unwrap();
        assert_eq!(r.demand_type, "intermittent");
        assert_eq!(r.n_observations, 10);
        assert_eq!(
            compute_aid_anomalies(&y, &AidOptions::default())
                .unwrap()
                .len(),
            10
        );
    }

    #[test]
    fn maps_errors() {
        let opts = AidOptions::default();
        assert!(matches!(
            compute_aid(&[], &opts),
            Err(StatsError::EmptyInput { field: "y" })
        ));
        assert!(matches!(
            compute_aid_anomalies(&[], &opts),
            Err(StatsError::EmptyInput { .. })
        ));
        assert!(matches!(
            compute_aid(&[f64::NAN], &opts),
            Err(StatsError::NoValidData)
        ));
    }
}
