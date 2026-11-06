//! C FFI for multivariate drift by PCA reconstruction error (#366) and the
//! reference-band alert thresholds (#367). The structs are mirrored in
//! `src/include/reconstruction_ffi.h`.
//!
//! A sample is `n_rows` rows of `n_features` doubles each, row-major; NaN is
//! a missing value (imputed by the reference means when the model says so).

use super::*;
use anofox_stats_core::tests::reconstruction::{
    alert_band, reconstruction_band, reconstruction_chunk, reconstruction_fit,
    reconstruction_row_errors, ComponentSelection, ReconstructionParts, ThresholdBand,
    ThresholdMethod, ValueLimits,
};

/// A fitted reconstruction model. The arrays are allocated by
/// `anofox_reconstruction_fit` and released with
/// `anofox_free_reconstruction_model`. As input (`ReconstructionModelFFI`
/// filled by the caller) the arrays are only read.
#[repr(C)]
#[derive(Debug)]
pub struct ReconstructionModelFFI {
    /// Reference column means, `n_features` values.
    pub center: *mut f64,
    /// Population column standard deviations, `n_features` values; NULL for
    /// an unscaled model.
    pub scale: *mut f64,
    /// Standard deviations of the kept components, `n_components` values.
    pub sdev: *mut f64,
    /// Share of the total variance per kept component, `n_components` values.
    pub proportion_of_variance: *mut f64,
    /// Loadings, component-major: `loadings[j * n_features + i]` is variable
    /// `i` of component `j`.
    pub loadings: *mut f64,
    pub n_features: usize,
    pub n_components: usize,
    pub n_observations: usize,
    pub impute_nan: bool,
}

/// Reconstruction error of one chunk (upstream `ReconstructionChunk`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReconstructionChunkFFI {
    pub n: usize,
    pub value: f64,
    pub std: f64,
    pub sampling_error: f64,
    pub lower_confidence: f64,
    pub upper_confidence: f64,
}

/// How a band is derived; a side whose value is NaN is disabled (`None`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ThresholdMethodFFI {
    /// 0 standard deviation, 1 constant, 2 relative.
    pub method: i32,
    pub lower: f64,
    pub upper: f64,
}

/// Value limits of a metric; NaN is no limit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ValueLimitsFFI {
    pub lower: f64,
    pub upper: f64,
    pub override_with_none: bool,
}

/// An alert band; NaN is no threshold on that side.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ThresholdBandFFI {
    pub lower: f64,
    pub upper: f64,
}

impl From<ThresholdBand> for ThresholdBandFFI {
    fn from(b: ThresholdBand) -> Self {
        Self {
            lower: b.lower.unwrap_or(f64::NAN),
            upper: b.upper.unwrap_or(f64::NAN),
        }
    }
}

fn nan_none(v: f64) -> Option<f64> {
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}

fn threshold_method(m: &ThresholdMethodFFI) -> Result<ThresholdMethod, &'static str> {
    let (lower, upper) = (nan_none(m.lower), nan_none(m.upper));
    Ok(match m.method {
        0 => ThresholdMethod::StandardDeviation { lower, upper },
        1 => ThresholdMethod::Constant { lower, upper },
        2 => ThresholdMethod::Relative { lower, upper },
        _ => return Err("invalid threshold method"),
    })
}

unsafe fn invalid_input(out_error: *mut AnofoxError, msg: &str) {
    if !out_error.is_null() {
        (*out_error).set(ErrorCode::InvalidInput, msg);
    }
}

/// The rows of a row-major sample.
///
/// # Safety
/// `data` must point to `n_rows * dim` doubles (or be NULL with `n_rows == 0`).
unsafe fn rows(data: *const f64, n_rows: usize, dim: usize) -> Vec<Vec<f64>> {
    if data.is_null() || n_rows == 0 {
        return Vec::new();
    }
    if dim == 0 {
        return vec![Vec::new(); n_rows];
    }
    slice::from_raw_parts(data, n_rows * dim)
        .chunks_exact(dim)
        .map(<[f64]>::to_vec)
        .collect()
}

unsafe fn view(p: *const f64, len: usize) -> Vec<f64> {
    if p.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(p, len).to_vec()
    }
}

unsafe fn malloc_copy(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

/// The model parts of a caller-filled `ReconstructionModelFFI`.
unsafe fn parts(m: &ReconstructionModelFFI) -> ReconstructionParts {
    let (d, k) = (m.n_features, m.n_components);
    let flat = view(m.loadings, k * d);
    let rotation = if flat.len() == k * d {
        (0..d)
            .map(|i| (0..k).map(|j| flat[j * d + i]).collect())
            .collect()
    } else {
        Vec::new()
    };
    ReconstructionParts {
        center: view(m.center, d),
        scale: (!m.scale.is_null()).then(|| view(m.scale, d)),
        sdev: view(m.sdev, k),
        proportion_of_variance: view(m.proportion_of_variance, k),
        rotation,
        n_components: k,
        impute_nan: m.impute_nan,
    }
}

/// Fits NannyML's reconstruction model on `n_rows` reference rows: the
/// components explaining more than `variance_share` of the variance, or
/// `n_components` components when it is non-zero; `scale` standardises the
/// columns (population sd), `impute_nan` replaces NaN by the reference means.
///
/// # Safety
/// - `data` must point to `n_rows * n_features` doubles
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_reconstruction_fit(
    data: *const f64,
    n_rows: usize,
    n_features: usize,
    variance_share: f64,
    n_components: usize,
    scale: bool,
    impute_nan: bool,
    out_result: *mut ReconstructionModelFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let selection = if n_components > 0 {
            ComponentSelection::Fixed(n_components)
        } else {
            ComponentSelection::VarianceShare(variance_share)
        };
        let x = rows(data, n_rows, n_features);
        match reconstruction_fit(&x, selection, scale, impute_nan) {
            Ok(fit) => {
                let k = fit.n_components;
                let d = fit.center.len();
                let loadings: Vec<f64> = (0..k)
                    .flat_map(|j| fit.rotation.iter().map(move |row| row[j]))
                    .collect();
                let out = ReconstructionModelFFI {
                    center: malloc_copy(&fit.center),
                    scale: fit
                        .scale
                        .as_deref()
                        .map_or(std::ptr::null_mut(), |s| malloc_copy(s)),
                    sdev: malloc_copy(&fit.sdev),
                    proportion_of_variance: malloc_copy(&fit.proportion_of_variance),
                    loadings: malloc_copy(&loadings),
                    n_features: d,
                    n_components: k,
                    n_observations: x.len(),
                    impute_nan: fit.impute_nan,
                };
                let failed = (d > 0 && out.center.is_null())
                    || (fit.scale.is_some() && d > 0 && out.scale.is_null())
                    || (k > 0
                        && (out.sdev.is_null()
                            || out.proportion_of_variance.is_null()
                            || out.loadings.is_null()));
                *out_result = out;
                if failed {
                    anofox_free_reconstruction_model(out_result);
                    if !out_error.is_null() {
                        (*out_error).set(
                            ErrorCode::AllocationFailure,
                            "Failed to allocate the reconstruction model",
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

/// Free the arrays of a model filled by `anofox_reconstruction_fit`.
///
/// # Safety
/// - `model` must be NULL or a model filled by `anofox_reconstruction_fit`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_reconstruction_model(model: *mut ReconstructionModelFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if model.is_null() {
            return;
        }
        let m = &mut *model;
        for p in [
            &mut m.center,
            &mut m.scale,
            &mut m.sdev,
            &mut m.proportion_of_variance,
            &mut m.loadings,
        ] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
    })
}

/// Reconstruction error of each of `n_rows` rows (`model.n_features` values
/// each), written to `out_errors` (`n_rows` doubles).
///
/// # Safety
/// - `model` must point to a model whose arrays hold the documented counts
/// - `data` must point to `n_rows * model.n_features` doubles
/// - `out_errors` must point to `n_rows` writable doubles
/// - `out_error` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn anofox_reconstruction_row_errors(
    model: *const ReconstructionModelFFI,
    data: *const f64,
    n_rows: usize,
    out_errors: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if model.is_null() {
            invalid_input(out_error, "model is NULL");
            return false;
        }
        if !check_out(out_errors, out_error) {
            return false;
        }
        let m = parts(&*model);
        let x = rows(data, n_rows, (*model).n_features);
        match reconstruction_row_errors(&m, &x) {
            Ok(e) => {
                std::ptr::copy_nonoverlapping(e.as_ptr(), out_errors, e.len().min(n_rows));
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Mean reconstruction error of a chunk of `n_rows` rows with its sampling
/// error and NannyML's confidence band.
///
/// # Safety
/// - as `anofox_reconstruction_row_errors`; `out_result` must be valid
#[no_mangle]
pub unsafe extern "C" fn anofox_reconstruction_chunk(
    model: *const ReconstructionModelFFI,
    data: *const f64,
    n_rows: usize,
    out_result: *mut ReconstructionChunkFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if model.is_null() {
            invalid_input(out_error, "model is NULL");
            return false;
        }
        if !check_out(out_result, out_error) {
            return false;
        }
        let m = parts(&*model);
        let x = rows(data, n_rows, (*model).n_features);
        match reconstruction_chunk(&m, &x) {
            Ok(c) => {
                *out_result = ReconstructionChunkFFI {
                    n: c.n,
                    value: c.value,
                    std: c.std,
                    sampling_error: c.sampling_error,
                    lower_confidence: c.lower_confidence,
                    upper_confidence: c.upper_confidence,
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

/// Alert band of a metric from its `n` reference values (NaN ignored):
/// upstream `threshold_band(values, method, limits)`.
///
/// # Safety
/// - `values` must point to `n` doubles (or be NULL with `n == 0`)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_threshold_band(
    values: *const f64,
    n: usize,
    method: ThresholdMethodFFI,
    limits: ValueLimitsFFI,
    out_result: *mut ThresholdBandFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let method = match threshold_method(&method) {
            Ok(m) => m,
            Err(msg) => {
                invalid_input(out_error, msg);
                return false;
            }
        };
        let limits = ValueLimits {
            lower: nan_none(limits.lower),
            upper: nan_none(limits.upper),
            override_with_none: limits.override_with_none,
        };
        match alert_band(&view(values, n), &method, &limits) {
            Ok(b) => {
                *out_result = b.into();
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Alert band of the reconstruction error from its `n` reference chunk
/// values: upstream `reconstruction_thresholds(values, method)` (a lower
/// threshold at or below 0 dropped).
///
/// # Safety
/// - as `anofox_threshold_band`
#[no_mangle]
pub unsafe extern "C" fn anofox_reconstruction_thresholds(
    values: *const f64,
    n: usize,
    method: ThresholdMethodFFI,
    out_result: *mut ThresholdBandFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let method = match threshold_method(&method) {
            Ok(m) => m,
            Err(msg) => {
                invalid_input(out_error, msg);
                return false;
            }
        };
        match reconstruction_band(&view(values, n), &method) {
            Ok(b) => {
                *out_result = b.into();
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

    #[test]
    fn fit_chunk_and_band_round_trip() {
        let x: Vec<f64> = (0..60)
            .flat_map(|i| {
                let t = i as f64;
                [t.sin(), (0.7 * t).cos() + 0.1 * t, (1.3 * t).sin()]
            })
            .collect();
        unsafe {
            let mut m: ReconstructionModelFFI = std::mem::zeroed();
            let mut err = AnofoxError::success();
            assert!(anofox_reconstruction_fit(
                x.as_ptr(),
                20,
                3,
                0.65,
                0,
                true,
                true,
                &mut m,
                &mut err
            ));
            assert!(m.n_components >= 1 && !m.scale.is_null());
            let mut c = ReconstructionChunkFFI {
                n: 0,
                value: 0.0,
                std: 0.0,
                sampling_error: 0.0,
                lower_confidence: 0.0,
                upper_confidence: 0.0,
            };
            assert!(anofox_reconstruction_chunk(
                &m,
                x.as_ptr(),
                20,
                &mut c,
                &mut err
            ));
            let mut e = vec![0.0; 20];
            assert!(anofox_reconstruction_row_errors(
                &m,
                x.as_ptr(),
                20,
                e.as_mut_ptr(),
                &mut err
            ));
            assert!((e.iter().sum::<f64>() / 20.0 - c.value).abs() < 1e-12);
            anofox_free_reconstruction_model(&mut m);
            assert!(m.center.is_null());

            let v = [0.91, 0.93, f64::NAN, 0.92, 0.95, 0.90];
            let mut b = ThresholdBandFFI {
                lower: 0.0,
                upper: 0.0,
            };
            let std3 = ThresholdMethodFFI {
                method: 0,
                lower: 3.0,
                upper: 3.0,
            };
            let limits = ValueLimitsFFI {
                lower: 0.9,
                upper: 0.96,
                override_with_none: true,
            };
            assert!(anofox_threshold_band(
                v.as_ptr(),
                v.len(),
                std3,
                limits,
                &mut b,
                &mut err
            ));
            assert!(b.lower.is_nan() && b.upper.is_nan());
            let bad = ThresholdMethodFFI {
                method: 7,
                lower: 1.0,
                upper: 1.0,
            };
            assert!(!anofox_threshold_band(
                v.as_ptr(),
                v.len(),
                bad,
                limits,
                &mut b,
                &mut err
            ));
        }
    }
}
