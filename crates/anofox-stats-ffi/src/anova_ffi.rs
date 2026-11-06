//! C FFI for the two-way and repeated-measures ANOVA (#297). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::anova::{
    repeated_measures_anova, two_way_anova, AnovaTerm, RmAnovaResult, SphericityCorrection,
};

/// One F test of the two-way ANOVA table.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AnovaTermFFI {
    pub sum_sq: f64,
    pub df: f64,
    pub mean_sq: f64,
    pub f_statistic: f64,
    pub p_value: f64,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

impl From<&AnovaTerm> for AnovaTermFFI {
    fn from(t: &AnovaTerm) -> Self {
        Self {
            sum_sq: t.sum_sq,
            df: t.df,
            mean_sq: t.mean_sq,
            f_statistic: t.f_statistic,
            p_value: t.p_value,
            unified: TestSummaryFFI::from(&t.unified),
        }
    }
}

/// Result of `anofox_two_way_anova`: factor A, factor B and the interaction,
/// then the residual and total rows.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TwoWayAnovaResultFFI {
    pub terms: [AnovaTermFFI; 3],
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

/// Result of `anofox_repeated_measures_anova`. The sphericity fields are NaN
/// when fewer than three conditions were observed or `sphericity` was false.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RmAnovaResultFFI {
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
    pub mauchly_w: f64,
    pub mauchly_chi_square: f64,
    pub mauchly_df: f64,
    pub mauchly_p_value: f64,
    pub gg_epsilon: f64,
    pub gg_df1: f64,
    pub gg_df2: f64,
    pub gg_p_value: f64,
    pub hf_epsilon: f64,
    pub hf_df1: f64,
    pub hf_df2: f64,
    pub hf_p_value: f64,
    /// Unified test fields of the uncorrected F test (#159); keep last.
    pub unified: TestSummaryFFI,
}

impl From<&RmAnovaResult> for RmAnovaResultFFI {
    fn from(r: &RmAnovaResult) -> Self {
        let nan = f64::NAN;
        let missing = SphericityCorrection {
            epsilon: nan,
            df1: nan,
            df2: nan,
            p_value: nan,
        };
        let gg = r.greenhouse_geisser.unwrap_or(missing);
        let hf = r.huynh_feldt.unwrap_or(missing);
        Self {
            ss_condition: r.ss_condition,
            df_condition: r.df_condition,
            ms_condition: r.ms_condition,
            f_statistic: r.f_statistic,
            p_value: r.p_value,
            ss_subjects: r.ss_subjects,
            df_subjects: r.df_subjects,
            ss_error: r.ss_error,
            df_error: r.df_error,
            ms_error: r.ms_error,
            ss_total: r.ss_total,
            n_subjects: r.n_subjects,
            n_conditions: r.n_conditions,
            mauchly_w: r.mauchly.map_or(nan, |m| m.w),
            mauchly_chi_square: r.mauchly.map_or(nan, |m| m.chi_square),
            mauchly_df: r.mauchly.map_or(nan, |m| m.df),
            mauchly_p_value: r.mauchly.map_or(nan, |m| m.p_value),
            gg_epsilon: gg.epsilon,
            gg_df1: gg.df1,
            gg_df2: gg.df2,
            gg_p_value: gg.p_value,
            hf_epsilon: hf.epsilon,
            hf_df1: hf.df1,
            hf_df2: hf.df2,
            hf_p_value: hf.p_value,
            unified: TestSummaryFFI::from(&r.unified),
        }
    }
}

/// 0-based level codes from the doubles the C++ side passes.
fn level_codes(codes: &[f64]) -> Option<Vec<usize>> {
    codes
        .iter()
        .map(|&c| (c.is_finite() && c >= 0.0 && c.fract() == 0.0).then_some(c as usize))
        .collect()
}

/// Two-way ANOVA (Type III sums of squares) of `values` by the dense 0-based
/// level codes `factor_a` and `factor_b` (equal lengths).
///
/// # Safety
/// - `values`, `factor_a` and `factor_b` must be valid DataArrays
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_two_way_anova(
    values: DataArray,
    factor_a: DataArray,
    factor_b: DataArray,
    out_result: *mut TwoWayAnovaResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (Some(a), Some(b)) = (
            level_codes(&factor_a.to_vec()),
            level_codes(&factor_b.to_vec()),
        ) else {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "invalid factor level code");
            }
            return false;
        };
        match two_way_anova(&values.to_vec(), &a, &b) {
            Ok(r) => {
                *out_result = TwoWayAnovaResultFFI {
                    terms: [
                        AnovaTermFFI::from(&r.terms[0]),
                        AnovaTermFFI::from(&r.terms[1]),
                        AnovaTermFFI::from(&r.terms[2]),
                    ],
                    residual_sum_sq: r.residual_sum_sq,
                    residual_df: r.residual_df,
                    residual_mean_sq: r.residual_mean_sq,
                    total_sum_sq: r.total_sum_sq,
                    total_df: r.total_df,
                    levels_a: r.levels_a,
                    levels_b: r.levels_b,
                    n: r.n,
                    grand_mean: r.grand_mean,
                };
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// One-way repeated-measures ANOVA of the complete row-major
/// `n_subjects` x `n_conditions` matrix `data`; `sphericity` adds Mauchly's
/// test and the Greenhouse-Geisser correction (three or more conditions).
///
/// # Safety
/// - `data` must point to `n_subjects * n_conditions` doubles
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_repeated_measures_anova(
    data: *const f64,
    n_subjects: usize,
    n_conditions: usize,
    sphericity: bool,
    out_result: *mut RmAnovaResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if data.is_null() || n_subjects == 0 || n_conditions == 0 {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "Invalid data dimensions");
            }
            return false;
        }
        let flat = slice::from_raw_parts(data, n_subjects * n_conditions);
        let rows: Vec<&[f64]> = flat.chunks(n_conditions).collect();
        match repeated_measures_anova(&rows, sphericity) {
            Ok(r) => {
                *out_result = RmAnovaResultFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::MaybeUninit;
    use std::ptr;

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: ptr::null(),
            len: v.len(),
        }
    }

    #[test]
    fn two_way_round_trip_and_errors() {
        unsafe {
            let y = [4.2, 5.1, 3.8, 5.0, 5.9, 6.3, 7.9, 8.4, 7.1, 9.6, 10.9, 9.2];
            let a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
            let b = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            let mut out = MaybeUninit::<TwoWayAnovaResultFFI>::uninit();
            let mut e = AnofoxError::success();
            assert!(anofox_two_way_anova(
                arr(&y),
                arr(&a),
                arr(&b),
                out.as_mut_ptr(),
                &mut e
            ));
            let r = out.assume_init();
            assert_eq!((r.levels_a, r.levels_b, r.n), (2, 2, 12));
            assert_eq!(r.residual_df, 8.0);
            assert_eq!(r.terms[2].unified.df1, 1.0);

            // A non-integer level code is an invalid input.
            let bad = [0.5; 12];
            let mut e = AnofoxError::success();
            assert!(!anofox_two_way_anova(
                arr(&y),
                arr(&bad),
                arr(&b),
                out.as_mut_ptr(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
            // NULL out-pointer.
            let mut e = AnofoxError::success();
            assert!(!anofox_two_way_anova(
                arr(&y),
                arr(&a),
                arr(&b),
                ptr::null_mut(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
        }
    }

    #[test]
    fn rm_anova_round_trip_and_errors() {
        unsafe {
            let data = [
                45.0, 50.0, 55.0, 42.0, 42.0, 45.0, 36.0, 41.0, 43.0, 39.0, 35.0, 40.0,
            ];
            let mut out = MaybeUninit::<RmAnovaResultFFI>::uninit();
            let mut e = AnofoxError::success();
            assert!(anofox_repeated_measures_anova(
                data.as_ptr(),
                4,
                3,
                true,
                out.as_mut_ptr(),
                &mut e
            ));
            let r = out.assume_init();
            assert_eq!((r.n_subjects, r.n_conditions), (4, 3));
            assert!(r.mauchly_w.is_finite() && r.gg_epsilon.is_finite());
            assert!(anofox_repeated_measures_anova(
                data.as_ptr(),
                4,
                3,
                false,
                out.as_mut_ptr(),
                &mut e
            ));
            assert!(out.assume_init().gg_epsilon.is_nan());
            let mut e = AnofoxError::success();
            assert!(!anofox_repeated_measures_anova(
                ptr::null(),
                4,
                3,
                true,
                out.as_mut_ptr(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
        }
    }
}
