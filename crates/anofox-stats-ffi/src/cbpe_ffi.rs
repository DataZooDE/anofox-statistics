//! C FFI for confidence-based performance estimation (#365): upstream
//! anofox-statistics `monitoring::cbpe`. The structs are mirrored in
//! `src/include/anofox_cbpe_ffi.h`.
//!
//! Calibration codes: 0 isotonic, 1 platt, 2 none. Predictions are NULL (all
//! rows use `score >= threshold`) or one code per row: 0 negative, 1
//! positive, any other value missing (falls back to the threshold). Metrics
//! are in the order roc_auc, f1, precision, recall, specificity, accuracy,
//! true_positive, false_positive, false_negative, true_negative.

use super::*;
use anofox_stats_core::tests::cbpe::{
    cbpe_estimate, cbpe_model, cbpe_realized, fit_cbpe, sampling_to_array, Calibrator,
    CbpeCalibration, CbpeModel, MetricEstimate, SAMPLING_FIELDS,
};

pub const CBPE_METRICS: usize = 10;

fn calibration_of(code: i32) -> Option<CbpeCalibration> {
    match code {
        0 => Some(CbpeCalibration::Isotonic),
        1 => Some(CbpeCalibration::Platt),
        2 => Some(CbpeCalibration::None),
        _ => None,
    }
}

unsafe fn read<T: Copy>(p: *const T, len: usize) -> Vec<T> {
    if p.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(p, len).to_vec()
    }
}

unsafe fn predictions_of(p: *const i8, n: usize) -> Option<Vec<Option<bool>>> {
    if p.is_null() {
        return None;
    }
    Some(
        read(p, n)
            .into_iter()
            .map(|c| match c {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
            .collect(),
    )
}

unsafe fn malloc_values(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

/// A fitted CBPE model. `knots_x` / `knots_y` (`n_knots`, isotonic only) are
/// allocated by `anofox_cbpe_fit` and released with
/// `anofox_free_cbpe_model`; `platt_a` / `platt_b` are NaN unless Platt.
#[repr(C)]
#[derive(Debug)]
pub struct CbpeModelFFI {
    pub knots_x: *mut f64,
    pub knots_y: *mut f64,
    pub n_knots: usize,
    pub n_reference: usize,
    pub calibration: i32,
    pub threshold: f64,
    pub platt_a: f64,
    pub platt_b: f64,
    pub sampling: [f64; SAMPLING_FIELDS],
}

/// The parts of a stored model, as laid out in `CbpeModelFFI`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CbpeModelPartsFFI {
    pub knots_x: *const f64,
    pub knots_y: *const f64,
    pub n_knots: usize,
    pub calibration: i32,
    pub threshold: f64,
    pub platt_a: f64,
    pub platt_b: f64,
    pub sampling: [f64; SAMPLING_FIELDS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CbpeMetricFFI {
    pub value: f64,
    pub sampling_error: f64,
    pub lower_confidence: f64,
    pub upper_confidence: f64,
}

impl From<&MetricEstimate> for CbpeMetricFFI {
    fn from(m: &MetricEstimate) -> Self {
        Self {
            value: m.value,
            sampling_error: m.sampling_error,
            lower_confidence: m.lower_confidence,
            upper_confidence: m.upper_confidence,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CbpeEstimateFFI {
    pub metrics: [CbpeMetricFFI; CBPE_METRICS],
    pub n: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CbpeRealizedFFI {
    pub metrics: [f64; CBPE_METRICS],
    pub n: usize,
}

unsafe fn invalid(out_error: *mut AnofoxError, msg: String) -> bool {
    set_error(out_error, &StatsError::InvalidInput(msg));
    false
}

unsafe fn model_of(parts: &CbpeModelPartsFFI) -> Result<CbpeModel, StatsError> {
    let Some(cal) = calibration_of(parts.calibration) else {
        return Err(StatsError::InvalidInput(format!(
            "unknown calibration code {}",
            parts.calibration
        )));
    };
    cbpe_model(
        cal,
        &read(parts.knots_x, parts.n_knots),
        &read(parts.knots_y, parts.n_knots),
        parts.platt_a,
        parts.platt_b,
        parts.threshold,
        &parts.sampling,
    )
}

/// Fits CBPE on `n` labelled reference rows (`monitoring::fit_reference` /
/// `fit_reference_with_predictions`).
///
/// # Safety
/// `scores` and `labels` must point to `n` values, `predictions` to `n`
/// codes or be NULL; `out_result` and `out_error` valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_cbpe_fit(
    scores: *const f64,
    labels: *const bool,
    predictions: *const i8,
    n: usize,
    calibration: i32,
    threshold: f64,
    out_result: *mut CbpeModelFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let Some(cal) = calibration_of(calibration) else {
            return invalid(out_error, format!("unknown calibration code {calibration}"));
        };
        let s = read(scores, n);
        let y = read(labels, n);
        let p = predictions_of(predictions, n);
        match fit_cbpe(&s, &y, p.as_deref(), cal, threshold) {
            Ok(m) => {
                let (kx, ky, a, b) = match &m.calibrator {
                    Calibrator::Isotonic(c) => (c.x.as_slice(), c.y.as_slice(), f64::NAN, f64::NAN),
                    Calibrator::Platt(c) => (&[][..], &[][..], c.a, c.b),
                    Calibrator::Identity => (&[][..], &[][..], f64::NAN, f64::NAN),
                };
                let out = CbpeModelFFI {
                    knots_x: malloc_values(kx),
                    knots_y: malloc_values(ky),
                    n_knots: kx.len(),
                    n_reference: n,
                    calibration,
                    threshold: m.threshold,
                    platt_a: a,
                    platt_b: b,
                    sampling: sampling_to_array(&m.sampling),
                };
                let failed = out.n_knots > 0 && (out.knots_x.is_null() || out.knots_y.is_null());
                *out_result = out;
                if failed {
                    anofox_free_cbpe_model(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate CBPE model",
                        );
                    }
                    return false;
                }
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Free the knots of a model filled by `anofox_cbpe_fit`.
///
/// # Safety
/// `model` must be NULL or a model filled by `anofox_cbpe_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_cbpe_model(model: *mut CbpeModelFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if model.is_null() {
            return;
        }
        let m = &mut *model;
        for p in [&mut m.knots_x, &mut m.knots_y] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
        m.n_knots = 0;
    })
}

/// Estimated performance of `n` unlabelled rows with a stored model
/// (`CbpeModel::from_parts`, then `estimate` / `estimate_with_predictions`).
///
/// # Safety
/// `scores` must point to `n` values, `predictions` to `n` codes or be NULL;
/// the model's knots to `n_knots` values; `out_result` and `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_cbpe_estimate(
    model: CbpeModelPartsFFI,
    scores: *const f64,
    predictions: *const i8,
    n: usize,
    out_result: *mut CbpeEstimateFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let s = read(scores, n);
        let p = predictions_of(predictions, n);
        match model_of(&model).and_then(|m| cbpe_estimate(&m, &s, p.as_deref())) {
            Ok(e) => {
                let m = [
                    &e.roc_auc,
                    &e.f1,
                    &e.precision,
                    &e.recall,
                    &e.specificity,
                    &e.accuracy,
                    &e.true_positive,
                    &e.false_positive,
                    &e.false_negative,
                    &e.true_negative,
                ];
                *out_result = CbpeEstimateFFI {
                    metrics: m.map(CbpeMetricFFI::from),
                    n: e.n,
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

/// Realized performance of `n` labelled rows with a stored model's threshold
/// (`CbpeModel::realized` / `realized_performance`); NaN where NannyML's
/// rules leave a metric undefined.
///
/// # Safety
/// `scores` and `labels` must point to `n` values, `predictions` to `n`
/// codes or be NULL; the model's knots to `n_knots` values; `out_result` and
/// `out_error` valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_cbpe_realized(
    model: CbpeModelPartsFFI,
    scores: *const f64,
    labels: *const bool,
    predictions: *const i8,
    n: usize,
    out_result: *mut CbpeRealizedFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let s = read(scores, n);
        let y = read(labels, n);
        let p = predictions_of(predictions, n);
        match model_of(&model).and_then(|m| cbpe_realized(&m, &s, &y, p.as_deref())) {
            Ok(r) => {
                *out_result = CbpeRealizedFFI {
                    metrics: [
                        r.roc_auc,
                        r.f1,
                        r.precision,
                        r.recall,
                        r.specificity,
                        r.accuracy,
                        r.true_positive,
                        r.false_positive,
                        r.false_negative,
                        r.true_negative,
                    ],
                    n: r.n,
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
