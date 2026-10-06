//! FFI boundary tests: NULL out-pointers, empty / mismatched / NaN-only input,
//! allocate -> free round trips. The crate is a `staticlib`, so these live
//! in-crate and call the `extern "C"` functions directly.

use super::*;
use std::ptr;

fn arr(v: &[f64]) -> DataArray {
    DataArray {
        data: if v.is_empty() {
            ptr::null()
        } else {
            v.as_ptr()
        },
        validity: ptr::null(),
        len: v.len(),
    }
}

fn err() -> AnofoxError {
    AnofoxError::success()
}

const X1: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
const Y: [f64; 8] = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];
const NAN8: [f64; 8] = [f64::NAN; 8];

// ---------------------------------------------------------------------------
// Panic guard
// ---------------------------------------------------------------------------

#[test]
fn ffi_guard_converts_panic_to_internal_error() {
    let mut e = err();
    let r = ffi_guard(&mut e, false, || -> bool { panic!("boom") });
    assert!(!r);
    assert_eq!(e.code, ErrorCode::InternalError);
    // NULL error pointer is fine too.
    let v = ffi_guard(ptr::null_mut(), f64::NAN, || -> f64 { panic!("boom") });
    assert!(v.is_nan());
}

// ---------------------------------------------------------------------------
// OLS
// ---------------------------------------------------------------------------

#[test]
fn ols_round_trip_and_free() {
    unsafe {
        let x = [arr(&X1)];
        let mut core = FitResultCore::default();
        let mut inf = FitResultInference::default();
        let mut e = err();
        let opts = || OlsOptionsFFI {
            compute_inference: true,
            ..Default::default()
        };
        assert!(anofox_ols_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            &mut core,
            &mut inf,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::Success);
        assert_eq!(core.coefficients_len, 1);
        assert!((*core.coefficients - 1.9976190476190476).abs() < 1e-10);
        assert_eq!(inf.len, 1);
        anofox_free_result_core(&mut core);
        anofox_free_result_inference(&mut inf);
        assert!(core.coefficients.is_null());
        // Double free / NULL free are no-ops.
        anofox_free_result_core(&mut core);
        anofox_free_result_core(ptr::null_mut());
        anofox_free_result_inference(ptr::null_mut());
    }
}

#[test]
fn ols_invalid_inputs() {
    unsafe {
        let x = [arr(&X1)];
        let opts = OlsOptionsFFI::default;
        let mut core = FitResultCore::default();
        let mut e = err();

        // NULL out_core / NULL x / zero x_count.
        assert!(!anofox_ols_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_ols_fit(
            arr(&Y),
            ptr::null(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_ols_fit(
            arr(&Y),
            x.as_ptr(),
            0,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        // NULL out_error must not crash.
        assert!(!anofox_ols_fit(
            arr(&Y),
            ptr::null(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            ptr::null_mut()
        ));

        // Zero-length input.
        let x0 = [arr(&[])];
        assert!(!anofox_ols_fit(
            arr(&[]),
            x0.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);

        // Mismatched lengths (second column short).
        let xm = [arr(&X1), arr(&X1[..5])];
        assert!(!anofox_ols_fit(
            arr(&Y),
            xm.as_ptr(),
            2,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);

        // NaN-only.
        let xn = [arr(&NAN8)];
        assert!(!anofox_ols_fit(
            arr(&NAN8),
            xn.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);

        // Invalid confidence level.
        for cl in [0.0, 1.0, f64::NAN, -1.0, 2.0] {
            let bad = OlsOptionsFFI {
                confidence_level: cl,
                ..Default::default()
            };
            assert!(!anofox_ols_fit(
                arr(&Y),
                x.as_ptr(),
                1,
                bad,
                &mut core,
                ptr::null_mut(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
        }

        // DataArray with NULL data but non-zero length is treated as all-NULL.
        let null_data = DataArray {
            data: ptr::null(),
            validity: ptr::null(),
            len: 8,
        };
        let xnd = [null_data];
        assert!(!anofox_ols_fit(
            arr(&Y),
            xnd.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
    }
}

// ---------------------------------------------------------------------------
// Ridge / WLS / RLS
// ---------------------------------------------------------------------------

#[test]
fn ridge_round_trip_and_invalid() {
    unsafe {
        let x = [arr(&X1)];
        let mut core = FitResultCore::default();
        let mut e = err();
        let opts = || RidgeOptionsFFI {
            alpha: 1.0,
            ..Default::default()
        };
        assert!(anofox_ridge_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(core.coefficients_len, 1);
        anofox_free_result_core(&mut core);

        assert!(!anofox_ridge_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        let xm = [arr(&X1), arr(&X1[..3])];
        assert!(!anofox_ridge_fit(
            arr(&Y),
            xm.as_ptr(),
            2,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        let xn = [arr(&NAN8)];
        assert!(!anofox_ridge_fit(
            arr(&NAN8),
            xn.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
        let x0 = [arr(&[])];
        assert!(!anofox_ridge_fit(
            arr(&[]),
            x0.as_ptr(),
            1,
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
    }
}

#[test]
fn wls_round_trip_and_invalid() {
    unsafe {
        let x = [arr(&X1)];
        let w = [1.0; 8];
        let mut core = FitResultCore::default();
        let mut inf = FitResultInference::default();
        let mut e = err();
        let opts = || WlsOptionsFFI {
            compute_inference: true,
            ..Default::default()
        };
        assert!(anofox_wls_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            arr(&w),
            opts(),
            &mut core,
            &mut inf,
            &mut e
        ));
        anofox_free_result_core(&mut core);
        anofox_free_result_inference(&mut inf);

        // Weights length mismatch.
        assert!(!anofox_wls_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            arr(&w[..4]),
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        // Column length mismatch.
        let xm = [arr(&X1), arr(&X1[..2])];
        assert!(!anofox_wls_fit(
            arr(&Y),
            xm.as_ptr(),
            2,
            arr(&w),
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        // NaN-only.
        let xn = [arr(&NAN8)];
        assert!(!anofox_wls_fit(
            arr(&NAN8),
            xn.as_ptr(),
            1,
            arr(&w),
            opts(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
        // NULL out_core.
        assert!(!anofox_wls_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            arr(&w),
            opts(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

#[test]
fn rls_round_trip_and_mismatched_columns() {
    unsafe {
        let x = [arr(&X1)];
        let mut core = FitResultCore::default();
        let mut e = err();
        let opts = RlsOptionsFFI::default;
        assert!(anofox_rls_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            &mut core,
            &mut e
        ));
        assert_eq!(core.coefficients_len, 1);
        anofox_free_result_core(&mut core);

        // Previously panicked (only x[0] was length-checked).
        let xm = [arr(&X1), arr(&X1[..3])];
        assert!(!anofox_rls_fit(
            arr(&Y),
            xm.as_ptr(),
            2,
            opts(),
            &mut core,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        let xn = [arr(&NAN8)];
        assert!(!anofox_rls_fit(
            arr(&NAN8),
            xn.as_ptr(),
            1,
            opts(),
            &mut core,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
        let x0 = [arr(&[])];
        assert!(!anofox_rls_fit(
            arr(&[]),
            x0.as_ptr(),
            1,
            opts(),
            &mut core,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_rls_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            opts(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

// ---------------------------------------------------------------------------
// Predict / VIF
// ---------------------------------------------------------------------------

#[test]
fn predict_round_trip_and_invalid() {
    unsafe {
        let x = [arr(&X1)];
        let coef = [2.0];
        let mut out: *mut f64 = ptr::null_mut();
        let mut n = 0usize;
        let mut e = err();
        assert!(anofox_predict(
            x.as_ptr(),
            1,
            coef.as_ptr(),
            1,
            1.0,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(n, 8);
        assert!((*out.add(7) - 17.0).abs() < 1e-12);
        anofox_free_predictions(out);
        anofox_free_predictions(ptr::null_mut());

        assert!(!anofox_predict(
            x.as_ptr(),
            1,
            coef.as_ptr(),
            1,
            1.0,
            ptr::null_mut(),
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_predict(
            x.as_ptr(),
            1,
            ptr::null(),
            1,
            1.0,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        // Coefficient count != column count.
        let xm = [arr(&X1), arr(&X1)];
        assert!(!anofox_predict(
            xm.as_ptr(),
            2,
            coef.as_ptr(),
            1,
            1.0,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        // Column length mismatch.
        let coef2 = [1.0, 1.0];
        let xl = [arr(&X1), arr(&X1[..3])];
        assert!(!anofox_predict(
            xl.as_ptr(),
            2,
            coef2.as_ptr(),
            2,
            1.0,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
    }
}

#[test]
fn vif_round_trip_and_invalid() {
    unsafe {
        let x2: Vec<f64> = X1.iter().map(|v| (v * 1.7).sin()).collect();
        let x = [arr(&X1), arr(&x2)];
        let mut out: *mut f64 = ptr::null_mut();
        let mut n = 0usize;
        let mut e = err();
        assert!(anofox_compute_vif(x.as_ptr(), 2, &mut out, &mut n, &mut e));
        assert_eq!(n, 2);
        anofox_free_vif(out);
        anofox_free_vif(ptr::null_mut());

        assert!(!anofox_compute_vif(
            x.as_ptr(),
            2,
            ptr::null_mut(),
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_compute_vif(
            ptr::null(),
            2,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        let xm = [arr(&X1), arr(&X1[..3])];
        assert!(!anofox_compute_vif(
            xm.as_ptr(),
            2,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        let xn = [arr(&NAN8), arr(&NAN8)];
        assert!(!anofox_compute_vif(
            xn.as_ptr(),
            2,
            &mut out,
            &mut n,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
    }
}

// ---------------------------------------------------------------------------
// Poisson
// ---------------------------------------------------------------------------

#[test]
fn poisson_round_trip_and_invalid() {
    unsafe {
        let y = [1.0, 0.0, 2.0, 3.0, 2.0, 5.0, 4.0, 7.0];
        let x = [arr(&X1)];
        let mut res = GlmFitResultCore::default();
        let mut e = err();
        let opts = PoissonOptionsFFI::default;
        assert!(anofox_poisson_fit(
            arr(&y),
            x.as_ptr(),
            1,
            opts(),
            &mut res,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(res.coefficients_len, 1);
        anofox_free_glm_result(&mut res);
        anofox_free_glm_result(ptr::null_mut());

        assert!(!anofox_poisson_fit(
            arr(&y),
            x.as_ptr(),
            1,
            opts(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        let xm = [arr(&X1), arr(&X1[..3])];
        assert!(!anofox_poisson_fit(
            arr(&y),
            xm.as_ptr(),
            2,
            opts(),
            &mut res,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        let xn = [arr(&NAN8)];
        assert!(!anofox_poisson_fit(
            arr(&NAN8),
            xn.as_ptr(),
            1,
            opts(),
            &mut res,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
        let x0 = [arr(&[])];
        assert!(!anofox_poisson_fit(
            arr(&[]),
            x0.as_ptr(),
            1,
            opts(),
            &mut res,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

// ---------------------------------------------------------------------------
// t-test / Mann-Whitney
// ---------------------------------------------------------------------------

#[test]
fn t_test_round_trip_and_invalid() {
    unsafe {
        let g2: Vec<f64> = X1.iter().map(|v| v + 3.0).collect();
        let mut r = TestResultFFI::default();
        let mut e = err();
        let opts = TTestOptionsFFI::default;
        assert!(anofox_t_test(arr(&X1), arr(&g2), opts(), &mut r, &mut e));
        assert!(r.p_value < 0.05);
        assert!(!r.method.is_null());
        anofox_free_test_result(&mut r);
        assert!(r.method.is_null());
        anofox_free_test_result(&mut r);
        anofox_free_test_result(ptr::null_mut());

        assert!(!anofox_t_test(
            arr(&X1),
            arr(&g2),
            opts(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_t_test(arr(&[]), arr(&[]), opts(), &mut r, &mut e));
        assert_ne!(e.code, ErrorCode::Success);
        assert!(!anofox_t_test(
            arr(&NAN8),
            arr(&NAN8),
            opts(),
            &mut r,
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
        let bad = TTestOptionsFFI {
            confidence_level: 1.5,
            ..Default::default()
        };
        assert!(!anofox_t_test(arr(&X1), arr(&g2), bad, &mut r, &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

#[test]
fn mann_whitney_effect_size_and_invalid() {
    unsafe {
        let g2: Vec<f64> = X1.iter().map(|v| v + 10.0).collect();
        let mut r = TestResultFFI::default();
        let mut e = err();
        let opts = MannWhitneyOptionsFFI::default;
        assert!(anofox_mann_whitney_u(
            arr(&X1),
            arr(&g2),
            opts(),
            &mut r,
            &mut e
        ));
        assert!(
            (r.effect_size - 1.0).abs() < 1e-12,
            "rank-biserial r = {}",
            r.effect_size
        );
        anofox_free_test_result(&mut r);

        assert!(!anofox_mann_whitney_u(
            arr(&X1),
            arr(&g2),
            opts(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_mann_whitney_u(
            arr(&[]),
            arr(&g2),
            opts(),
            &mut r,
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
        assert!(!anofox_mann_whitney_u(
            arr(&NAN8),
            arr(&NAN8),
            opts(),
            &mut r,
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
        // confidence_level <= 0 is the "no CI" sentinel; >= 1 is invalid.
        let no_ci = MannWhitneyOptionsFFI {
            confidence_level: 0.0,
            ..Default::default()
        };
        assert!(anofox_mann_whitney_u(
            arr(&X1),
            arr(&g2),
            no_ci,
            &mut r,
            &mut e
        ));
        anofox_free_test_result(&mut r);
        let bad = MannWhitneyOptionsFFI {
            confidence_level: 1.0,
            ..Default::default()
        };
        assert!(!anofox_mann_whitney_u(
            arr(&X1),
            arr(&g2),
            bad,
            &mut r,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

// ---------------------------------------------------------------------------
// Fisher / seeded distance correlation / lm_dynamic
// ---------------------------------------------------------------------------

#[test]
fn fisher_variants() {
    unsafe {
        let mut r = TestResultFFI::default();
        let mut e = err();
        let opts = FisherExactOptionsFFI::default;
        assert!(anofox_fisher_exact_conditional(
            3,
            1,
            1,
            3,
            opts(),
            &mut r,
            &mut e
        ));
        assert!((r.p_value - 0.4857142857142856).abs() < 1e-12);
        anofox_free_test_result(&mut r);
        // Tiny table works.
        assert!(anofox_fisher_exact_conditional(
            1,
            0,
            0,
            1,
            opts(),
            &mut r,
            &mut e
        ));
        anofox_free_test_result(&mut r);
        assert!(!anofox_fisher_exact_conditional(
            0,
            0,
            0,
            0,
            opts(),
            &mut r,
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
        let bad = FisherExactOptionsFFI {
            confidence_level: 0.0,
            ..Default::default()
        };
        assert!(!anofox_fisher_exact(3, 1, 1, 3, bad, &mut r, &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_fisher_exact(
            3,
            1,
            1,
            3,
            opts(),
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_fisher_exact(
            usize::MAX,
            1,
            0,
            0,
            opts(),
            &mut r,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

#[test]
fn distance_cor_test_seed_is_reproducible() {
    unsafe {
        let y: Vec<f64> = X1.iter().map(|v| (v * 0.9).cos() + v * 0.1).collect();
        let run = |seed: u64| {
            let mut r = TestResultFFI::default();
            let mut e = err();
            assert!(anofox_distance_cor_test_seeded(
                arr(&X1),
                arr(&y),
                199,
                seed,
                true,
                &mut r,
                &mut e
            ));
            let p = r.p_value;
            anofox_free_test_result(&mut r);
            p
        };
        assert_eq!(run(42), run(42));
        let mut r = TestResultFFI::default();
        let mut e = err();
        assert!(!anofox_distance_cor_test_seeded(
            arr(&X1),
            arr(&y),
            10,
            1,
            true,
            ptr::null_mut(),
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!anofox_distance_cor_test_seeded(
            arr(&X1),
            arr(&y[..3]),
            10,
            1,
            true,
            &mut r,
            &mut e
        ));
        assert_ne!(e.code, ErrorCode::Success);
    }
}

#[test]
fn lm_dynamic_null_pointers_do_not_crash() {
    unsafe {
        let opts = LmDynamicOptionsFFI::default();
        let mut res = LmDynamicFitResultFFI::default();
        let mut e = err();
        // NULL error / result: silently returns.
        anofox_fit_lm_dynamic(
            Y.as_ptr(),
            8,
            X1.as_ptr(),
            1,
            &opts,
            &mut res,
            ptr::null_mut(),
        );
        anofox_fit_lm_dynamic(
            Y.as_ptr(),
            8,
            X1.as_ptr(),
            1,
            &opts,
            ptr::null_mut(),
            &mut e,
        );
        // NULL inputs / overflow.
        anofox_fit_lm_dynamic(ptr::null(), 8, X1.as_ptr(), 1, &opts, &mut res, &mut e);
        assert_eq!(e.code, ErrorCode::InvalidInput);
        anofox_fit_lm_dynamic(Y.as_ptr(), 8, X1.as_ptr(), 1, ptr::null(), &mut res, &mut e);
        assert_eq!(e.code, ErrorCode::InvalidInput);
        anofox_fit_lm_dynamic(
            Y.as_ptr(),
            usize::MAX,
            X1.as_ptr(),
            2,
            &opts,
            &mut res,
            &mut e,
        );
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}

// ---------------------------------------------------------------------------
// Leverage-aware intervals
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::excessive_precision)]
fn interval_matrix_round_trip_matches_r() {
    unsafe {
        // Fit via the FFI, then build M and predict; R reference:
        // predict(lm(y ~ x), data.frame(x = 12), interval = "prediction")
        let x = [arr(&X1)];
        let mut core = FitResultCore::default();
        let mut e = err();
        assert!(anofox_ols_fit(
            arr(&Y),
            x.as_ptr(),
            1,
            OlsOptionsFFI::default(),
            &mut core,
            ptr::null_mut(),
            &mut e
        ));

        let mut m: *mut f64 = ptr::null_mut();
        let mut dim = 0usize;
        assert!(anofox_interval_matrix(
            x.as_ptr(),
            1,
            core.coefficients,
            core.coefficients_len,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(dim, 2);

        let mut pr = PredictionResult::default();
        let x_new = [12.0];
        assert!(anofox_predict_with_interval_matrix(
            core.coefficients,
            1,
            core.intercept,
            x_new.as_ptr(),
            1,
            m,
            dim,
            core.n_observations,
            2,
            core.residual_std_error,
            0.95,
            0,
            &mut pr,
            &mut e
        ));
        assert!((pr.yhat - 24.00714285714286).abs() < 1e-8);
        assert!((pr.yhat_lower - 23.315088455013225).abs() < 1e-8);
        assert!((pr.yhat_upper - 24.699197259272495).abs() < 1e-8);

        // Confidence interval.
        assert!(anofox_predict_with_interval_matrix(
            core.coefficients,
            1,
            core.intercept,
            x_new.as_ptr(),
            1,
            m,
            dim,
            8,
            2,
            core.residual_std_error,
            0.95,
            1,
            &mut pr,
            &mut e
        ));
        assert!((pr.yhat_lower - 23.473675784475667).abs() < 1e-8);
        assert!((pr.yhat_upper - 24.540609929810053).abs() < 1e-8);

        // Invalid interval type / confidence / dim / NULLs.
        let call =
            |kind: i32, cl: f64, d: usize, out: *mut PredictionResult, e: &mut AnofoxError| {
                anofox_predict_with_interval_matrix(
                    core.coefficients,
                    1,
                    core.intercept,
                    x_new.as_ptr(),
                    1,
                    m,
                    d,
                    8,
                    2,
                    core.residual_std_error,
                    cl,
                    kind,
                    out,
                    e,
                )
            };
        assert!(!call(2, 0.95, 2, &mut pr, &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!call(0, 1.0, 2, &mut pr, &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!call(0, 0.95, 1, &mut pr, &mut e));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        assert!(!call(0, 0.95, 0, &mut pr, &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        assert!(!call(0, 0.95, 2, ptr::null_mut(), &mut e));
        assert_eq!(e.code, ErrorCode::InvalidInput);

        anofox_free_interval_matrix(m);
        anofox_free_interval_matrix(ptr::null_mut());
        anofox_free_result_core(&mut core);
    }
}

#[test]
fn interval_matrix_invalid_inputs() {
    unsafe {
        let coef = [2.0, 1.0];
        let mut m: *mut f64 = ptr::null_mut();
        let mut dim = 0usize;
        let mut e = err();
        let x = [arr(&X1), arr(&X1)];

        // NULL outputs.
        assert!(!anofox_interval_matrix(
            x.as_ptr(),
            2,
            coef.as_ptr(),
            2,
            true,
            ptr::null(),
            0.0,
            ptr::null_mut(),
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        // Zero-length x.
        assert!(!anofox_interval_matrix(
            x.as_ptr(),
            0,
            coef.as_ptr(),
            0,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::InvalidInput);
        // coefficients_len mismatch.
        assert!(!anofox_interval_matrix(
            x.as_ptr(),
            2,
            coef.as_ptr(),
            1,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        // Column length mismatch.
        let xm = [arr(&X1), arr(&X1[..4])];
        assert!(!anofox_interval_matrix(
            xm.as_ptr(),
            2,
            coef.as_ptr(),
            2,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
        // Singular (duplicated column) -> graceful failure; with ridge OK.
        assert!(!anofox_interval_matrix(
            x.as_ptr(),
            2,
            coef.as_ptr(),
            2,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::SingularMatrix);
        assert!(m.is_null());
        assert!(anofox_interval_matrix(
            x.as_ptr(),
            2,
            coef.as_ptr(),
            2,
            true,
            ptr::null(),
            0.5,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(dim, 3);
        anofox_free_interval_matrix(m);
        // NaN-only rows.
        let xn = [arr(&NAN8)];
        assert!(!anofox_interval_matrix(
            xn.as_ptr(),
            1,
            coef.as_ptr(),
            1,
            true,
            ptr::null(),
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::NoValidData);
        // Weights length mismatch.
        let w = [1.0; 3];
        let w_arr = arr(&w);
        let x1 = [arr(&X1)];
        assert!(!anofox_interval_matrix(
            x1.as_ptr(),
            1,
            coef.as_ptr(),
            1,
            true,
            &w_arr,
            0.0,
            &mut m,
            &mut dim,
            &mut e
        ));
        assert_eq!(e.code, ErrorCode::DimensionMismatch);
    }
}
