//! Pairwise post-hoc tests (#159): thin wrappers over the upstream
//! `pairwise_t_test`, `tukey_hsd` and `dunn_test`.
//!
//! Groups are passed as integer codes; upstream sorts them, so the caller
//! assigns codes in the label order it wants (R's factor-level order).

use super::{convert_error, Alternative};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    dunn_test as lib_dunn_test, pairwise_t_test as lib_pairwise_t_test, tukey_hsd as lib_tukey_hsd,
    PAdjustMethod,
};

pub use anofox_tests::{PairwiseComparison, PairwiseResult};

/// Parse an R `p.adjust` method name (holm, hochberg, hommel, bonferroni, BH,
/// fdr, BY, none), case-insensitively, with the upstream parser.
pub fn parse_p_adjust(name: &str) -> StatsResult<PAdjustMethod> {
    name.parse::<PAdjustMethod>()
        .map_err(|e| StatsError::InvalidInput(e.to_string()))
}

/// Fewer than two groups is degenerate data (SQL NULL), not an invalid argument.
fn check_groups(values: &[f64], groups: &[u64]) -> StatsResult<()> {
    if values.len() != groups.len() {
        return Err(StatsError::DimensionMismatchMsg(format!(
            "values and groups must have the same length ({} vs {})",
            values.len(),
            groups.len()
        )));
    }
    let first = groups.first();
    if first.is_none() || groups.iter().all(|g| Some(g) == first) {
        return Err(StatsError::InsufficientDataMsg(
            "post-hoc comparisons need at least 2 groups".into(),
        ));
    }
    Ok(())
}

/// Pairwise t-tests with p-value adjustment (R `pairwise.t.test`).
pub fn pairwise_t_test(
    values: &[f64],
    groups: &[u64],
    pooled_sd: bool,
    p_adjust: PAdjustMethod,
    alternative: Alternative,
) -> StatsResult<PairwiseResult<u64>> {
    check_groups(values, groups)?;
    lib_pairwise_t_test(values, groups, pooled_sd, p_adjust, alternative).map_err(convert_error)
}

/// Tukey's honest significant difference (R `TukeyHSD(aov(...))`).
pub fn tukey_hsd(
    values: &[f64],
    groups: &[u64],
    conf_level: f64,
) -> StatsResult<PairwiseResult<u64>> {
    check_groups(values, groups)?;
    lib_tukey_hsd(values, groups, conf_level).map_err(convert_error)
}

/// Dunn's rank test with p-value adjustment (`FSA::dunnTest`).
pub fn dunn_test(
    values: &[f64],
    groups: &[u64],
    p_adjust: PAdjustMethod,
) -> StatsResult<PairwiseResult<u64>> {
    check_groups(values, groups)?;
    lib_dunn_test(values, groups, p_adjust).map_err(convert_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_t_test_three_groups() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let groups = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        let res = pairwise_t_test(
            &values,
            &groups,
            true,
            parse_p_adjust("holm").unwrap(),
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(res.comparisons.len(), 3);
        assert_eq!(
            (res.comparisons[0].group1, res.comparisons[0].group2),
            (0, 1)
        );
    }

    #[test]
    fn single_group_is_insufficient_data() {
        let err = tukey_hsd(&[1.0, 2.0, 3.0], &[4, 4, 4], 0.95).unwrap_err();
        assert!(matches!(err, StatsError::InsufficientDataMsg(_)));
    }

    #[test]
    fn unknown_p_adjust_is_invalid_input() {
        assert!(matches!(
            parse_p_adjust("nope"),
            Err(StatsError::InvalidInput(_))
        ));
    }
}
