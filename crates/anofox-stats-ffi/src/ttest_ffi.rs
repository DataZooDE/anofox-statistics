//! C FFI for the one-sample and paired t-tests (#299). Both reuse
//! `AnofoxTTestOptions` (`var_equal` is ignored) and `AnofoxTestResult`; the
//! declarations are in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::parametric::{t_test_one_sample, t_test_paired};
use anofox_stats_core::tests::TestResult;
use anofox_stats_core::StatsResult;

unsafe fn write_t_result(
    result: StatsResult<TestResult>,
    out_result: *mut TestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    match result {
        Ok(r) => {
            (*out_result) = TestResultFFI {
                statistic: r.statistic,
                p_value: r.p_value,
                df: r.df,
                effect_size: r.effect_size,
                ci_lower: r.ci_lower,
                ci_upper: r.ci_upper,
                confidence_level: r.confidence_level,
                n: r.n,
                n1: r.n1,
                n2: r.n2,
                alternative: r.alternative.into(),
                method: alloc_string(&r.method),
                unified: TestSummaryFFI::from(&r.unified),
            };
            true
        }
        Err(e) => {
            if !out_error.is_null() {
                (*out_error).set(error_to_code(&e), &e.to_string());
            }
            false
        }
    }
}

fn t_options(options: &TTestOptionsFFI) -> TTestOptions {
    TTestOptions {
        alternative: options.alternative.into(),
        kind: TTestKind::Paired,
        confidence_level: Some(options.confidence_level),
        mu: options.mu,
    }
}

/// One-sample t-test of H0: mean(x) = `options.mu`.
///
/// # Safety
/// - `x` must be a valid DataArray
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_t_test_one_sample(
    x: DataArray,
    options: TTestOptionsFFI,
    out_result: *mut TestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        if out_result.is_null() {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
            }
            return false;
        }
        let xs = x.to_vec();
        let opts = t_options(&options);
        write_t_result(t_test_one_sample(&xs, &opts), out_result, out_error)
    })
}

/// Paired t-test of H0: mean(x - y) = `options.mu`.
///
/// # Safety
/// - `x` and `y` must be valid DataArrays of equal length
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_t_test_paired(
    x: DataArray,
    y: DataArray,
    options: TTestOptionsFFI,
    out_result: *mut TestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        if !out_error.is_null() {
            *out_error = AnofoxError::success();
        }
        if !check_confidence_level(options.confidence_level, out_error) {
            return false;
        }
        if out_result.is_null() {
            if !out_error.is_null() {
                (*out_error).set(ErrorCode::InvalidInput, "out_result is NULL");
            }
            return false;
        }
        let xs = x.to_vec();
        let ys = y.to_vec();
        let opts = t_options(&options);
        write_t_result(t_test_paired(&xs, &ys, &opts), out_result, out_error)
    })
}

#[cfg(test)]
mod ttest_ffi_tests {
    use super::*;

    fn arr(v: &[f64]) -> DataArray {
        DataArray {
            data: v.as_ptr(),
            validity: std::ptr::null(),
            len: v.len(),
        }
    }

    #[test]
    fn one_sample_and_paired_round_trip() {
        let x = [5.1, 4.9, 6.2, 5.8, 6.05, 5.5, 5.3, 6.1];
        let y = [5.0, 5.2, 5.9, 5.5, 6.0, 5.6, 5.0, 5.7];
        unsafe {
            let mut r: TestResultFFI = std::mem::zeroed();
            let mut e = AnofoxError::success();
            let opts = TTestOptionsFFI {
                mu: 5.0,
                ..TTestOptionsFFI::default()
            };
            assert!(anofox_t_test_one_sample(arr(&x), opts, &mut r, &mut e));
            assert!((r.statistic - 3.562099364).abs() < 1e-8);
            anofox_free_test_result(&mut r);

            assert!(anofox_t_test_paired(
                arr(&x),
                arr(&y),
                TTestOptionsFFI::default(),
                &mut r,
                &mut e
            ));
            assert_eq!(r.n, 8);
            anofox_free_test_result(&mut r);

            assert!(!anofox_t_test_paired(
                arr(&x),
                arr(&y[..3]),
                TTestOptionsFFI::default(),
                &mut r,
                &mut e
            ));
            assert!(!anofox_t_test_one_sample(
                arr(&[1.0]),
                TTestOptionsFFI::default(),
                &mut r,
                &mut e
            ));
        }
    }
}
