//! Non-parametric survival analysis (#156)
//!
//! - Kaplan-Meier estimator with Greenwood standard errors
//! - k-group log-rank test
//!
//! Both delegate to `anofox_statistics::survival` (R `survival::survfit`,
//! `survival::survdiff`).

use super::{convert_error, UnifiedTestResult};
use crate::{StatsError, StatsResult};

pub use anofox_tests::{KaplanMeierResult, KmConfType};

/// Kaplan-Meier curve of right-censored `time` (`event = true` observed).
pub fn kaplan_meier(
    time: &[f64],
    event: &[bool],
    conf_type: KmConfType,
    conf_level: f64,
) -> StatsResult<KaplanMeierResult> {
    if time.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "Kaplan-Meier requires at least 1 observation".into(),
        ));
    }
    anofox_tests::kaplan_meier(time, event, conf_type, conf_level).map_err(convert_error)
}

/// Log-rank test of equal survival across the groups coded by `group`.
/// Fewer than two groups is insufficient data.
pub fn log_rank(time: &[f64], event: &[bool], group: &[u64]) -> StatsResult<UnifiedTestResult> {
    let mut levels: Vec<u64> = group.to_vec();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "log-rank test needs at least 2 groups".into(),
        ));
    }
    let result = anofox_tests::log_rank(time, event, group).map_err(convert_error)?;
    Ok(UnifiedTestResult::from(&result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn km_steps_at_event_times() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0];
        let event = [true, true, false, true, false];
        let km = kaplan_meier(&time, &event, KmConfType::Log, 0.95).unwrap();
        assert_eq!(km.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(km.survival[0], 0.8);
        assert_relative_eq!(km.survival[1], 0.6);
        assert_relative_eq!(km.survival[2], 0.3);
        assert_eq!(km.n_censor, vec![0, 1, 0, 1]);
    }

    #[test]
    fn log_rank_needs_two_groups() {
        let time = [1.0, 2.0, 3.0];
        let event = [true, true, false];
        assert!(matches!(
            log_rank(&time, &event, &[0, 0, 0]),
            Err(StatsError::InsufficientDataMsg(_))
        ));
        let t = log_rank(&time, &event, &[0, 1, 0]).unwrap();
        assert_eq!(t.df1, Some(1.0));
        assert_eq!(t.n, Some(3));
    }
}
