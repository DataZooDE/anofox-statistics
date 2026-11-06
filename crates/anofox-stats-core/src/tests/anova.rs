//! Factorial and repeated-measures ANOVA (#297)
//!
//! - Two-way ANOVA with interaction, Type III sums of squares
//!   (`anofox_statistics::two_way_anova`, R's `car::Anova(type = 3)` with sum
//!   contrasts)
//! - One-way repeated-measures ANOVA with Mauchly's sphericity test and the
//!   Greenhouse-Geisser correction (`anofox_statistics::repeated_measures_anova`)
//!
//! The statistics are computed upstream; these wrappers only validate shapes
//! and copy the results.

use super::{convert_error, UnifiedTestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    repeated_measures_anova as lib_repeated_measures_anova, two_way_anova as lib_two_way_anova,
    AnovaTableRow, CorrectedResult,
};

/// One F test of a two-way ANOVA table (factor A, factor B or A x B).
#[derive(Debug, Clone)]
pub struct AnovaTerm {
    pub sum_sq: f64,
    pub df: f64,
    pub mean_sq: f64,
    pub f_statistic: f64,
    pub p_value: f64,
    /// Unified result (contract `test` row): F, df1, df2, p, partial eta squared.
    pub unified: UnifiedTestResult,
}

/// Two-way ANOVA table (Type III sums of squares).
#[derive(Debug, Clone)]
pub struct TwoWayAnovaResult {
    /// Factor A, factor B and the A x B interaction, in that order.
    pub terms: [AnovaTerm; 3],
    pub residual_sum_sq: f64,
    pub residual_df: f64,
    pub residual_mean_sq: f64,
    pub total_sum_sq: f64,
    pub total_df: f64,
    pub levels_a: usize,
    pub levels_b: usize,
    pub n: usize,
    pub grand_mean: f64,
}

fn term(row: &AnovaTableRow, unified: UnifiedTestResult) -> AnovaTerm {
    AnovaTerm {
        sum_sq: row.ss,
        df: row.df,
        mean_sq: row.ms,
        f_statistic: row.f_statistic.unwrap_or(f64::NAN),
        p_value: row.p_value.unwrap_or(f64::NAN),
        unified,
    }
}

/// Two-way ANOVA of `values` by the 0-based level codes `factor_a` and
/// `factor_b` (every code below the maximum must occur; every cell must be
/// observed).
pub fn two_way_anova(
    values: &[f64],
    factor_a: &[usize],
    factor_b: &[usize],
) -> StatsResult<TwoWayAnovaResult> {
    if values.len() != factor_a.len() || values.len() != factor_b.len() {
        return Err(StatsError::InvalidInput(format!(
            "two-way ANOVA: mismatched lengths (values {}, factor_a {}, factor_b {})",
            values.len(),
            factor_a.len(),
            factor_b.len()
        )));
    }
    let r = lib_two_way_anova(values, factor_a, factor_b).map_err(convert_error)?;
    let mut tests = r.to_test_results().into_iter();
    let mut next = || tests.next().unwrap_or_default();
    let terms = [
        term(&r.factor_a, next()),
        term(&r.factor_b, next()),
        term(&r.interaction, next()),
    ];
    Ok(TwoWayAnovaResult {
        terms,
        residual_sum_sq: r.residual.ss,
        residual_df: r.residual.df,
        residual_mean_sq: r.residual.ms,
        total_sum_sq: r.total.ss,
        total_df: r.total.df,
        levels_a: r.levels_a,
        levels_b: r.levels_b,
        n: r.n,
        grand_mean: r.grand_mean,
    })
}

/// Epsilon-corrected within-subjects F test.
#[derive(Debug, Clone, Copy)]
pub struct SphericityCorrection {
    pub epsilon: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
}

impl From<&CorrectedResult> for SphericityCorrection {
    fn from(c: &CorrectedResult) -> Self {
        Self {
            epsilon: c.epsilon,
            df1: c.df_num_corrected,
            df2: c.df_den_corrected,
            p_value: c.p_value,
        }
    }
}

/// Mauchly's test of sphericity.
#[derive(Debug, Clone, Copy)]
pub struct MauchlyTest {
    pub w: f64,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
}

/// One-way repeated-measures ANOVA.
#[derive(Debug, Clone)]
pub struct RmAnovaResult {
    pub ss_condition: f64,
    pub df_condition: f64,
    pub ms_condition: f64,
    pub f_statistic: f64,
    pub p_value: f64,
    pub ss_subjects: f64,
    pub df_subjects: f64,
    pub ss_error: f64,
    pub df_error: f64,
    pub ms_error: f64,
    pub ss_total: f64,
    pub n_subjects: usize,
    pub n_conditions: usize,
    /// None when fewer than 3 conditions or sphericity was not requested.
    pub mauchly: Option<MauchlyTest>,
    pub greenhouse_geisser: Option<SphericityCorrection>,
    /// Huynh-Feldt correction (Lecoutre-corrected epsilon, capped at 1, as
    /// car / afex / ez report it). None like `greenhouse_geisser`.
    pub huynh_feldt: Option<SphericityCorrection>,
    /// Unified result of the uncorrected F test (n = number of subjects).
    pub unified: UnifiedTestResult,
}

/// Repeated-measures ANOVA of a complete subjects x conditions matrix (one
/// slice per subject, every slice with one value per condition).
pub fn repeated_measures_anova(data: &[&[f64]], sphericity: bool) -> StatsResult<RmAnovaResult> {
    let r = lib_repeated_measures_anova(data, sphericity).map_err(convert_error)?;
    let unified = r.to_test_results().into_iter().next().unwrap_or_default();
    Ok(RmAnovaResult {
        ss_condition: r.within_subjects.ss,
        df_condition: r.within_subjects.df,
        ms_condition: r.within_subjects.ms,
        f_statistic: r.within_subjects.f_statistic.unwrap_or(f64::NAN),
        p_value: r.within_subjects.p_value.unwrap_or(f64::NAN),
        ss_subjects: r.subjects.ss,
        df_subjects: r.subjects.df,
        ss_error: r.error.ss,
        df_error: r.error.df,
        ms_error: r.error.ms,
        ss_total: r.total.ss,
        n_subjects: r.subject_means.len(),
        n_conditions: r.condition_means.len(),
        mauchly: r.sphericity.as_ref().map(|s| MauchlyTest {
            w: s.w,
            chi_square: s.chi_square,
            df: s.df,
            p_value: s.p_value,
        }),
        greenhouse_geisser: r
            .greenhouse_geisser
            .as_ref()
            .map(SphericityCorrection::from),
        huynh_feldt: r.huynh_feldt.as_ref().map(SphericityCorrection::from),
        unified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values: validation/generators/make_reference_anova.R
    // (car::Anova(type = 3) with contr.sum; aov(y ~ cond + Error(subject/cond))).
    fn assert_close(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn two_way_unbalanced_matches_car_type3() {
        let y = [
            12.1, 13.4, 11.8, 14.2, 15.0, 13.1, 14.7, 18.3, 17.2, 11.0, 12.6, 10.9, 11.7, 12.2,
            16.4, 17.8, 15.9, 16.1, 15.2, 17.0, 16.6,
        ];
        let a = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        let b = [
            0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2,
        ];
        let r = two_way_anova(&y, &a, &b).unwrap();
        assert_close(r.terms[0].sum_sq, 0.01578720238095, 1e-9);
        assert_close(r.terms[1].sum_sq, 84.15898810963, 1e-9);
        assert_close(r.terms[2].sum_sq, 14.33937505038, 1e-9);
        assert_close(r.terms[2].p_value, 0.001310409919425, 1e-9);
        assert_close(r.residual_sum_sq, 10.07716666667, 1e-9);
        assert_eq!(r.residual_df, 15.0);
        assert_eq!(r.n, 21);
    }

    #[test]
    fn two_way_rejects_mismatched_lengths() {
        assert!(matches!(
            two_way_anova(&[1.0, 2.0], &[0], &[0, 1]),
            Err(StatsError::InvalidInput(_))
        ));
    }

    #[test]
    fn rm_anova_matches_base_r() {
        let rows: [[f64; 4]; 8] = [
            [45.0, 50.0, 55.0, 70.0],
            [42.0, 42.0, 45.0, 60.0],
            [36.0, 41.0, 43.0, 62.0],
            [39.0, 35.0, 40.0, 51.0],
            [51.0, 55.0, 59.0, 77.0],
            [44.0, 49.0, 56.0, 68.0],
            [40.0, 41.0, 48.0, 55.0],
            [48.0, 52.0, 56.0, 74.0],
        ];
        let data: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let r = repeated_measures_anova(&data, true).unwrap();
        assert_close(r.f_statistic, 105.7994886000, 1e-9);
        assert_close(r.ss_error, 146.65625, 1e-9);
        assert_eq!(r.n_subjects, 8);
        assert_eq!(r.n_conditions, 4);
        let m = r.mauchly.unwrap();
        assert_close(m.w, 0.05077975866551, 1e-9);
        assert_close(r.greenhouse_geisser.unwrap().epsilon, 0.5163544344870, 1e-9);
        let no_sph = repeated_measures_anova(&data, false).unwrap();
        assert!(no_sph.mauchly.is_none() && no_sph.greenhouse_geisser.is_none());
    }
}
