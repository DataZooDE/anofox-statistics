//! C FFI for resampling model validation and calibration (#318):
//! `rms::validate`, `rms::calibrate`, `rms::calibrate.cph` and `rms::bootcov`
//! through `anofox_stats_core::model_validation`. The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`. Every array of a result is allocated
//! here and released with the matching `anofox_free_*` function.

use super::*;
use anofox_stats_core::model_validation::{
    self as mv, CalibrateCphSpec, ResampleScheme, ResampleSpec, ValidationModel,
};

/// Bytes reserved for each index name in `ValidationResultFFI::names`.
pub const VALIDATION_NAME_LEN: usize = 32;

/// Model, resampling scheme, number of resamples and seed.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResampleOptionsFFI {
    /// 0 = linear (ols), 1 = binary logistic (lrm).
    pub model: i32,
    /// 0 = bootstrap, 1 = cross-validation, 2 = .632 bootstrap.
    pub method: i32,
    /// Bootstrap repetitions, or cross-validation groups.
    pub b: u32,
    pub seed: u64,
}

/// `rms::validate` table: one entry per index.
#[repr(C)]
#[derive(Debug)]
pub struct ValidationResultFFI {
    /// `n_indexes` NUL-terminated names of `VALIDATION_NAME_LEN` bytes each.
    pub names: *mut c_char,
    pub index_orig: *mut f64,
    pub training: *mut f64,
    pub test: *mut f64,
    pub optimism: *mut f64,
    pub corrected: *mut f64,
    /// Resamples with a finite index, per index.
    pub n: *mut u64,
    pub n_indexes: usize,
    /// Resamples whose fit failed.
    pub failed: usize,
}

/// `rms::calibrate` curve: one entry per grid point.
#[repr(C)]
#[derive(Debug)]
pub struct CalibrationResultFFI {
    pub predy: *mut f64,
    pub calibrated_orig: *mut f64,
    pub calibrated_corrected: *mut f64,
    pub optimism: *mut f64,
    pub n: *mut u64,
    pub len: usize,
    pub mean_abs_error: f64,
    pub mean_squared_error: f64,
    pub quantile90_abs_error: f64,
}

/// Options of `anofox_calibrate_cph`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CalibrateCphOptionsFFI {
    pub u: f64,
    pub m: usize,
    /// Explicit group boundaries (NULL: quantile groups of about `m`).
    pub cuts: *const f64,
    pub cuts_len: usize,
    /// 0 = Efron, 1 = Breslow.
    pub ties: i32,
    pub max_iterations: u32,
    pub tolerance: f64,
}

/// `rms::calibrate.cph` table: one entry per group of predicted survival;
/// `cuts` holds `n_groups + 1` boundaries.
#[repr(C)]
#[derive(Debug)]
pub struct SurvivalCalibrationResultFFI {
    pub cuts: *mut f64,
    pub index_orig: *mut f64,
    pub training: *mut f64,
    pub test: *mut f64,
    pub optimism: *mut f64,
    pub corrected: *mut f64,
    pub n: *mut u64,
    pub mean_predicted: *mut f64,
    pub km: *mut f64,
    pub km_corrected: *mut f64,
    pub std_err: *mut f64,
    pub n_groups: usize,
}

/// `rms::bootcov` result: `n_coefficients` coefficients (intercept first).
#[repr(C)]
#[derive(Debug)]
pub struct BootcovResultFFI {
    pub estimate: *mut f64,
    pub boot_mean: *mut f64,
    pub std_error: *mut f64,
    pub conf_low: *mut f64,
    pub conf_high: *mut f64,
    /// Row-major `n_coefficients * n_coefficients` bootstrap covariance.
    pub vcov: *mut f64,
    pub n_coefficients: usize,
    /// Successful replicates.
    pub n_boot: usize,
    pub failed: usize,
}

unsafe fn malloc_f64(values: &[f64]) -> *mut f64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut f64;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn malloc_u64(values: &[usize]) -> *mut u64 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(values.len() * std::mem::size_of::<u64>()) as *mut u64;
    if !ptr.is_null() {
        for (k, &v) in values.iter().enumerate() {
            *ptr.add(k) = v as u64;
        }
    }
    ptr
}

unsafe fn free_ptr<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn alloc_failure(out_error: *mut AnofoxError) -> bool {
    if !out_error.is_null() {
        (*out_error).set(
            ErrorCode::AllocationFailure,
            "Failed to allocate validation result",
        );
    }
    false
}

fn model_of(code: i32) -> Result<ValidationModel, StatsError> {
    match code {
        0 => Ok(ValidationModel::Ols),
        1 => Ok(ValidationModel::Logistic),
        _ => Err(StatsError::InvalidInput(format!(
            "unknown model code {code}"
        ))),
    }
}

fn spec_of(o: &ResampleOptionsFFI) -> Result<ResampleSpec, StatsError> {
    let scheme = match o.method {
        0 => ResampleScheme::Bootstrap,
        1 => ResampleScheme::Crossvalidation,
        2 => ResampleScheme::Point632,
        m => return Err(StatsError::InvalidInput(format!("unknown method code {m}"))),
    };
    Ok(ResampleSpec {
        scheme,
        b: o.b as usize,
        seed: o.seed,
    })
}

/// `n_rows` values of `y` and `n_features` column-major columns of `x`.
unsafe fn columns(
    y: *const f64,
    x: *const f64,
    n_rows: usize,
    n_features: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    if y.is_null() || n_rows == 0 {
        return (Vec::new(), vec![Vec::new(); n_features]);
    }
    let yv = slice::from_raw_parts(y, n_rows).to_vec();
    let xv = if x.is_null() || n_features == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(x, n_rows * n_features)
            .chunks(n_rows)
            .map(<[f64]>::to_vec)
            .collect()
    };
    (yv, xv)
}

/// `rms::validate` of a linear or binary logistic model of `y` on the
/// column-major `x` (`n_rows * n_features`, no intercept column).
///
/// # Safety
/// - `y` must point to `n_rows` doubles, `x` to `n_rows * n_features`
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_validate(
    y: *const f64,
    x: *const f64,
    n_rows: usize,
    n_features: usize,
    options: ResampleOptionsFFI,
    out_result: *mut ValidationResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (yv, xv) = columns(y, x, n_rows, n_features);
        let table = model_of(options.model)
            .and_then(|m| spec_of(&options).map(|s| (m, s)))
            .and_then(|(m, s)| mv::validate(m, &yv, &xv, &s));
        let t = match table {
            Ok(t) => t,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let k = t.names.len();
        let names = libc::calloc(k.max(1), VALIDATION_NAME_LEN) as *mut c_char;
        if !names.is_null() {
            for (i, name) in t.names.iter().enumerate() {
                let bytes = name.as_bytes();
                let len = bytes.len().min(VALIDATION_NAME_LEN - 1);
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr() as *const c_char,
                    names.add(i * VALIDATION_NAME_LEN),
                    len,
                );
            }
        }
        let out = ValidationResultFFI {
            names,
            index_orig: malloc_f64(&t.index_orig),
            training: malloc_f64(&t.training),
            test: malloc_f64(&t.test),
            optimism: malloc_f64(&t.optimism),
            corrected: malloc_f64(&t.corrected),
            n: malloc_u64(&t.n),
            n_indexes: k,
            failed: t.failed,
        };
        let failed = out.names.is_null()
            || (k > 0
                && (out.index_orig.is_null()
                    || out.training.is_null()
                    || out.test.is_null()
                    || out.optimism.is_null()
                    || out.corrected.is_null()
                    || out.n.is_null()));
        *out_result = out;
        if failed {
            anofox_free_validation_result(out_result);
            return alloc_failure(out_error);
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_validate`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_validate`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_validation_result(result: *mut ValidationResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        free_ptr(&mut r.names);
        for p in [
            &mut r.index_orig,
            &mut r.training,
            &mut r.test,
            &mut r.optimism,
            &mut r.corrected,
        ] {
            free_ptr(p);
        }
        free_ptr(&mut r.n);
        r.n_indexes = 0;
    })
}

/// `rms::calibrate` of a linear or binary logistic model: the
/// overfitting-corrected calibration curve at `predy` (`predy_len` values;
/// NULL: upstream's default grid of 50 points).
///
/// # Safety
/// - `y` must point to `n_rows` doubles, `x` to `n_rows * n_features`,
///   `predy` (if not NULL) to `predy_len`
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_calibrate(
    y: *const f64,
    x: *const f64,
    n_rows: usize,
    n_features: usize,
    options: ResampleOptionsFFI,
    predy: *const f64,
    predy_len: usize,
    out_result: *mut CalibrationResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (yv, xv) = columns(y, x, n_rows, n_features);
        let grid = if predy.is_null() {
            None
        } else {
            Some(slice::from_raw_parts(predy, predy_len).to_vec())
        };
        let cal = model_of(options.model)
            .and_then(|m| spec_of(&options).map(|s| (m, s)))
            .and_then(|(m, s)| mv::calibrate(m, &yv, &xv, &s, grid));
        let c = match cal {
            Ok(c) => c,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let len = c.predy.len();
        let out = CalibrationResultFFI {
            predy: malloc_f64(&c.predy),
            calibrated_orig: malloc_f64(&c.calibrated_orig),
            calibrated_corrected: malloc_f64(&c.calibrated_corrected),
            optimism: malloc_f64(&c.optimism),
            n: malloc_u64(&c.n),
            len,
            mean_abs_error: c.mean_abs_error,
            mean_squared_error: c.mean_squared_error,
            quantile90_abs_error: c.quantile90_abs_error,
        };
        let failed = len > 0
            && (out.predy.is_null()
                || out.calibrated_orig.is_null()
                || out.calibrated_corrected.is_null()
                || out.optimism.is_null()
                || out.n.is_null());
        *out_result = out;
        if failed {
            anofox_free_calibration_result(out_result);
            return alloc_failure(out_error);
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_calibrate`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_calibrate`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_calibration_result(result: *mut CalibrationResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.predy,
            &mut r.calibrated_orig,
            &mut r.calibrated_corrected,
            &mut r.optimism,
        ] {
            free_ptr(p);
        }
        free_ptr(&mut r.n);
        r.len = 0;
    })
}

/// `rms::calibrate.cph(cmethod = "KM")` of a Cox model of `time` (with
/// `event` 1 / 0) on the column-major `x`. `options.model` is ignored.
///
/// # Safety
/// - `time` and `event` must point to `n_rows` doubles, `x` to
///   `n_rows * n_features`, `cph.cuts` (if not NULL) to `cph.cuts_len`
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_calibrate_cph(
    time: *const f64,
    x: *const f64,
    event: *const f64,
    n_rows: usize,
    n_features: usize,
    options: ResampleOptionsFFI,
    cph: CalibrateCphOptionsFFI,
    out_result: *mut SurvivalCalibrationResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (tv, xv) = columns(time, x, n_rows, n_features);
        let ev = if event.is_null() || n_rows == 0 {
            Vec::new()
        } else {
            slice::from_raw_parts(event, n_rows).to_vec()
        };
        let spec = CalibrateCphSpec {
            u: cph.u,
            m: cph.m,
            cuts: if cph.cuts.is_null() {
                None
            } else {
                Some(slice::from_raw_parts(cph.cuts, cph.cuts_len).to_vec())
            },
            breslow: cph.ties == 1,
            max_iter: cph.max_iterations as usize,
            eps: cph.tolerance,
        };
        let cal = spec_of(&options).and_then(|s| mv::calibrate_cph(&tv, &xv, &ev, &s, &spec));
        let c = match cal {
            Ok(c) => c,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let g = c.km.len();
        let out = SurvivalCalibrationResultFFI {
            cuts: malloc_f64(&c.cuts),
            index_orig: malloc_f64(&c.index_orig),
            training: malloc_f64(&c.training),
            test: malloc_f64(&c.test),
            optimism: malloc_f64(&c.optimism),
            corrected: malloc_f64(&c.corrected),
            n: malloc_u64(&c.n),
            mean_predicted: malloc_f64(&c.mean_predicted),
            km: malloc_f64(&c.km),
            km_corrected: malloc_f64(&c.km_corrected),
            std_err: malloc_f64(&c.std_err),
            n_groups: g,
        };
        let failed = g > 0
            && (out.cuts.is_null()
                || out.index_orig.is_null()
                || out.training.is_null()
                || out.test.is_null()
                || out.optimism.is_null()
                || out.corrected.is_null()
                || out.n.is_null()
                || out.mean_predicted.is_null()
                || out.km.is_null()
                || out.km_corrected.is_null()
                || out.std_err.is_null());
        *out_result = out;
        if failed {
            anofox_free_survival_calibration_result(out_result);
            return alloc_failure(out_error);
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_calibrate_cph`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_calibrate_cph`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_survival_calibration_result(
    result: *mut SurvivalCalibrationResultFFI,
) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.cuts,
            &mut r.index_orig,
            &mut r.training,
            &mut r.test,
            &mut r.optimism,
            &mut r.corrected,
            &mut r.mean_predicted,
            &mut r.km,
            &mut r.km_corrected,
            &mut r.std_err,
        ] {
            free_ptr(p);
        }
        free_ptr(&mut r.n);
        r.n_groups = 0;
    })
}

/// `rms::bootcov` of a linear or binary logistic model (`options.method` is
/// ignored: always the ordinary bootstrap, or with `cluster` — `n_rows`
/// dense cluster codes, NULL for none — the cluster bootstrap).
///
/// # Safety
/// - `y` must point to `n_rows` doubles, `x` to `n_rows * n_features`,
///   `cluster` (if not NULL) to `n_rows` values
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_bootcov(
    y: *const f64,
    x: *const f64,
    cluster: *const u64,
    n_rows: usize,
    n_features: usize,
    options: ResampleOptionsFFI,
    conf_level: f64,
    out_result: *mut BootcovResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let (yv, xv) = columns(y, x, n_rows, n_features);
        let codes: Option<Vec<usize>> = if cluster.is_null() || n_rows == 0 {
            None
        } else {
            Some(
                slice::from_raw_parts(cluster, n_rows)
                    .iter()
                    .map(|&c| c as usize)
                    .collect(),
            )
        };
        let res = model_of(options.model).and_then(|m| {
            mv::bootcov(
                m,
                &yv,
                &xv,
                codes.as_deref(),
                options.b as usize,
                options.seed,
                conf_level,
            )
        });
        let r = match res {
            Ok(r) => r,
            Err(e) => {
                set_error(out_error, &e);
                return false;
            }
        };
        let k = r.estimate.len();
        let vcov: Vec<f64> = (0..k * k).map(|i| r.boot.vcov[(i / k, i % k)]).collect();
        let se: Vec<f64> = (0..k).map(|j| r.boot.vcov[(j, j)].sqrt()).collect();
        let out = BootcovResultFFI {
            estimate: malloc_f64(&r.estimate),
            boot_mean: malloc_f64(&r.boot.mean),
            std_error: malloc_f64(&se),
            conf_low: malloc_f64(&r.conf_low),
            conf_high: malloc_f64(&r.conf_high),
            vcov: malloc_f64(&vcov),
            n_coefficients: k,
            n_boot: r.boot.replicates.nrows(),
            failed: r.boot.failed,
        };
        let failed = k > 0
            && (out.estimate.is_null()
                || out.boot_mean.is_null()
                || out.std_error.is_null()
                || out.conf_low.is_null()
                || out.conf_high.is_null()
                || out.vcov.is_null());
        *out_result = out;
        if failed {
            anofox_free_bootcov_result(out_result);
            return alloc_failure(out_error);
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_bootcov`.
///
/// # Safety
/// - `result` must be NULL or a result filled by `anofox_bootcov`
#[no_mangle]
pub unsafe extern "C" fn anofox_free_bootcov_result(result: *mut BootcovResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.estimate,
            &mut r.boot_mean,
            &mut r.std_error,
            &mut r.conf_low,
            &mut r.conf_high,
            &mut r.vcov,
        ] {
            free_ptr(p);
        }
        r.n_coefficients = 0;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 23) as f64 / 7.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.8 * x[i] + ((i * 7) % 13) as f64 / 6.0)
            .collect();
        (y, x)
    }

    #[test]
    fn validate_round_trip() {
        let (y, x) = data(60);
        let o = ResampleOptionsFFI {
            model: 0,
            method: 0,
            b: 20,
            seed: 1,
        };
        let mut r = std::mem::MaybeUninit::<ValidationResultFFI>::zeroed();
        let mut e = AnofoxError::success();
        unsafe {
            assert!(anofox_validate(
                y.as_ptr(),
                x.as_ptr(),
                60,
                1,
                o,
                r.as_mut_ptr(),
                &mut e
            ));
            let r = r.assume_init_mut();
            assert_eq!(r.n_indexes, 5);
            let first = std::ffi::CStr::from_ptr(r.names);
            assert_eq!(first.to_str().unwrap(), "R-square");
            anofox_free_validation_result(r);
            assert!(r.names.is_null());
        }
    }

    #[test]
    fn bootcov_and_errors() {
        let (y, x) = data(60);
        let o = ResampleOptionsFFI {
            model: 0,
            method: 0,
            b: 30,
            seed: 1,
        };
        let mut r = std::mem::MaybeUninit::<BootcovResultFFI>::zeroed();
        let mut e = AnofoxError::success();
        unsafe {
            assert!(anofox_bootcov(
                y.as_ptr(),
                x.as_ptr(),
                std::ptr::null(),
                60,
                1,
                o,
                0.95,
                r.as_mut_ptr(),
                &mut e
            ));
            let r = r.assume_init_mut();
            assert_eq!(r.n_coefficients, 2);
            assert_eq!(r.n_boot + r.failed, 30);
            anofox_free_bootcov_result(r);
            let bad = ResampleOptionsFFI { model: 7, ..o };
            let mut v = std::mem::MaybeUninit::<ValidationResultFFI>::zeroed();
            assert!(!anofox_validate(
                y.as_ptr(),
                x.as_ptr(),
                60,
                1,
                bad,
                v.as_mut_ptr(),
                &mut e
            ));
            assert_eq!(e.code, ErrorCode::InvalidInput);
        }
    }
}
