//! C FFI for generalized least squares (#315, `nlme::gls`). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::{
    fit_gls, GlsCorrelationKind, GlsModel, GlsResidualType, GlsRows, GlsSpec, GlsVarianceKind,
};

/// Options of a GLS fit.
///
/// * `correlation`: 0 independent, 1 compound symmetry, 2 AR(1), 3 continuous
///   AR(1), 4 exponential, 5 Gaussian, 6 linear, 7 spherical, 8 unstructured.
/// * `variance`: 0 constant, 1 ident, 2 power, 3 exp.
/// * `variogram_bins`: 0 = no variogram; `variogram_residuals`: 0 pearson,
///   1 normalized, 2 response (only read by `anofox_gls_fit`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GlsOptionsFFI {
    pub confidence_level: f64,
    pub max_iterations: u32,
    pub correlation: i32,
    pub variance: i32,
    pub variogram_bins: u32,
    pub variogram_residuals: i32,
    pub nugget: bool,
    pub reml: bool,
    pub fit_intercept: bool,
}

/// The rows of a GLS fit. `x` is column-major (`n_rows * n_features`);
/// `variance_covariate` is NULL without a variance function. `subject` holds
/// dense subject codes (negative: no subject). Rows with a non-finite value
/// in a column the fit uses are left out.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GlsDataFFI {
    pub y: *const f64,
    pub x: *const f64,
    pub subject: *const i32,
    pub time: *const f64,
    pub variance_covariate: *const f64,
    pub n_rows: usize,
    pub n_features: usize,
}

/// A fitted GLS model. Arrays are `libc::malloc`ed and released with
/// `anofox_free_gls_fit_result`. The coefficient arrays have `n_coefficients`
/// entries, the intercept first when `has_intercept`; `vcov` is
/// `n_coefficients²`, row-major. The interval arrays are NULL (and the sigma
/// limits NaN) when `has_intervals` is false.
#[repr(C)]
#[derive(Debug)]
pub struct GlsFitResultFFI {
    pub coefficients: *mut f64,
    pub std_errors: *mut f64,
    pub t_values: *mut f64,
    pub p_values: *mut f64,
    pub conf_low: *mut f64,
    pub conf_high: *mut f64,
    pub vcov: *mut f64,
    pub n_coefficients: usize,
    pub correlation_params: *mut f64,
    pub correlation_conf_low: *mut f64,
    pub correlation_conf_high: *mut f64,
    pub n_correlation_params: usize,
    pub variance_params: *mut f64,
    pub variance_conf_low: *mut f64,
    pub variance_conf_high: *mut f64,
    pub n_variance_params: usize,
    /// varIdent strata codes, the reference first then one per variance
    /// parameter (`n_variance_params + 1` entries); NULL for the other
    /// variance functions.
    pub variance_strata: *mut f64,
    pub variogram_dist: *mut f64,
    pub variogram_variog: *mut f64,
    pub variogram_n_pairs: *mut f64,
    pub n_variogram: usize,
    pub sigma: f64,
    pub sigma_conf_low: f64,
    pub sigma_conf_high: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub df_residual: f64,
    pub confidence_level: f64,
    pub n_params: usize,
    pub n_observations: usize,
    pub n_groups: usize,
    pub iterations: usize,
    pub converged: bool,
    pub has_intercept: bool,
    pub has_intervals: bool,
    pub summary: ModelSummaryFFI,
}

/// Per-input-row results (NaN where a value does not exist). Released with
/// `anofox_free_gls_augment_result`.
#[repr(C)]
#[derive(Debug)]
pub struct GlsAugmentResultFFI {
    pub fitted: *mut f64,
    pub se_fit: *mut f64,
    pub residual: *mut f64,
    pub pearson_residual: *mut f64,
    pub normalized_residual: *mut f64,
    pub is_training: *mut bool,
    pub n_rows: usize,
}

unsafe fn gls_malloc<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn gls_free<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn gls_view<'a, T>(p: *const T, n: usize) -> Option<&'a [T]> {
    if p.is_null() {
        None
    } else if n == 0 {
        Some(&[])
    } else {
        Some(slice::from_raw_parts(p, n))
    }
}

fn spec_from_ffi(o: &GlsOptionsFFI) -> Result<GlsSpec, StatsError> {
    let correlation = match o.correlation {
        0 => GlsCorrelationKind::Independent,
        1 => GlsCorrelationKind::CompoundSymmetry,
        2 => GlsCorrelationKind::Ar1,
        3 => GlsCorrelationKind::ContinuousAr1,
        4 => GlsCorrelationKind::Exponential,
        5 => GlsCorrelationKind::Gaussian,
        6 => GlsCorrelationKind::Linear,
        7 => GlsCorrelationKind::Spherical,
        8 => GlsCorrelationKind::Unstructured,
        c => {
            return Err(StatsError::InvalidInput(format!(
                "unknown correlation code {c}"
            )))
        }
    };
    let variance = match o.variance {
        0 => GlsVarianceKind::Constant,
        1 => GlsVarianceKind::Ident,
        2 => GlsVarianceKind::Power,
        3 => GlsVarianceKind::Exponential,
        v => {
            return Err(StatsError::InvalidInput(format!(
                "unknown variance code {v}"
            )))
        }
    };
    Ok(GlsSpec {
        correlation,
        nugget: o.nugget,
        variance,
        reml: o.reml,
        fit_intercept: o.fit_intercept,
        confidence_level: o.confidence_level,
        max_iterations: o.max_iterations as usize,
    })
}

struct GlsInput {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    subject: Vec<i32>,
    time: Vec<f64>,
    variance_covariate: Option<Vec<f64>>,
}

/// Read the rows, fit, and hand the model and the input to `body`.
unsafe fn with_gls_model<R>(
    data: *const GlsDataFFI,
    options: &GlsOptionsFFI,
    out_error: *mut AnofoxError,
    body: impl FnOnce(&GlsModel, &GlsInput) -> Result<R, StatsError>,
) -> Option<R> {
    if data.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "data is NULL");
        }
        return None;
    }
    let d = &*data;
    let n = d.n_rows;
    let (Some(y), Some(subject), Some(time)) = (
        gls_view(d.y, n),
        gls_view(d.subject, n),
        gls_view(d.time, n),
    ) else {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "y, subject or time is NULL");
        }
        return None;
    };
    let x: Vec<Vec<f64>> = match gls_view(d.x, n * d.n_features) {
        Some(all) if n > 0 && d.n_features > 0 => all.chunks(n).map(<[f64]>::to_vec).collect(),
        _ => vec![Vec::new(); d.n_features],
    };
    let input = GlsInput {
        y: y.to_vec(),
        x,
        subject: subject.to_vec(),
        time: time.to_vec(),
        variance_covariate: gls_view(d.variance_covariate, n).map(<[f64]>::to_vec),
    };
    let rows = GlsRows {
        y: &input.y,
        x: &input.x,
        subject: &input.subject,
        time: &input.time,
        variance_covariate: input.variance_covariate.as_deref(),
    };
    let result = spec_from_ffi(options)
        .and_then(|spec| fit_gls(&rows, &spec))
        .and_then(|m| body(&m, &input));
    match result {
        Ok(r) => Some(r),
        Err(e) => {
            set_error(out_error, &e);
            None
        }
    }
}

/// Fit a generalized least squares model (`nlme::gls`).
///
/// # Safety
/// - `data` must point to a valid `GlsDataFFI` whose arrays hold `n_rows`
///   values (`x`: `n_rows * n_features`) or are NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_gls_fit(
    data: *const GlsDataFFI,
    options: GlsOptionsFFI,
    out_result: *mut GlsFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let fitted = with_gls_model(data, &options, out_error, |m, input| {
            let variogram = if options.variogram_bins > 0 {
                let kind = match options.variogram_residuals {
                    0 => GlsResidualType::Pearson,
                    1 => GlsResidualType::Normalized,
                    2 => GlsResidualType::Response,
                    r => {
                        return Err(StatsError::InvalidInput(format!(
                            "unknown variogram residual code {r}"
                        )))
                    }
                };
                // A fit without within-subject pairs has no variogram: NULL,
                // not an error.
                m.variogram(&input.time, kind, options.variogram_bins as usize)
                    .ok()
            } else {
                None
            };
            Ok((m.summary(), m.strata.clone(), variogram))
        });
        let Some((s, strata, variogram)) = fitted else {
            return false;
        };
        let p = s.coefficients.len();
        let mut summary = ModelSummaryFFI {
            model_type: name_buffer("gls"),
            family: name_buffer("gaussian"),
            link: name_buffer("identity"),
            log_likelihood: s.log_likelihood,
            aic: s.aic,
            bic: s.bic,
            ..ModelSummaryFFI::default()
        };
        if options.fit_intercept && p > 0 {
            summary.intercept_std_error = s.std_errors[0];
            summary.intercept_statistic = s.t_values[0];
            summary.intercept_p_value = s.p_values[0];
            summary.intercept_conf_low = s.conf_low[0];
            summary.intercept_conf_high = s.conf_high[0];
        }
        let (cl, ch) = s.correlation_conf.clone().unwrap_or_default();
        let (vl, vh) = s.variance_conf.clone().unwrap_or_default();
        let has_intervals = s.sigma_conf.is_some();
        let (sl, sh) = s.sigma_conf.unwrap_or((f64::NAN, f64::NAN));
        let vg = variogram.unwrap_or_default();
        let vd: Vec<f64> = vg.iter().map(|b| b.dist).collect();
        let vv: Vec<f64> = vg.iter().map(|b| b.variog).collect();
        let vn: Vec<f64> = vg.iter().map(|b| b.n_pairs as f64).collect();
        let nc = s.correlation_params.len();
        let nv = s.variance_params.len();
        let out = GlsFitResultFFI {
            coefficients: gls_malloc(&s.coefficients),
            std_errors: gls_malloc(&s.std_errors),
            t_values: gls_malloc(&s.t_values),
            p_values: gls_malloc(&s.p_values),
            conf_low: gls_malloc(&s.conf_low),
            conf_high: gls_malloc(&s.conf_high),
            vcov: gls_malloc(&s.vcov),
            n_coefficients: p,
            correlation_params: gls_malloc(&s.correlation_params),
            correlation_conf_low: gls_malloc(&cl),
            correlation_conf_high: gls_malloc(&ch),
            n_correlation_params: nc,
            variance_params: gls_malloc(&s.variance_params),
            variance_conf_low: gls_malloc(&vl),
            variance_conf_high: gls_malloc(&vh),
            n_variance_params: nv,
            variance_strata: gls_malloc(&strata),
            variogram_dist: gls_malloc(&vd),
            variogram_variog: gls_malloc(&vv),
            variogram_n_pairs: gls_malloc(&vn),
            n_variogram: vg.len(),
            sigma: s.sigma,
            sigma_conf_low: sl,
            sigma_conf_high: sh,
            log_likelihood: s.log_likelihood,
            aic: s.aic,
            bic: s.bic,
            df_residual: s.df_residual,
            confidence_level: options.confidence_level,
            n_params: s.n_params,
            n_observations: s.n_observations,
            n_groups: s.n_groups,
            iterations: s.iterations,
            converged: s.converged,
            has_intercept: options.fit_intercept,
            has_intervals,
            summary,
        };
        let coef_failed = p > 0
            && [
                out.coefficients,
                out.std_errors,
                out.t_values,
                out.p_values,
                out.conf_low,
                out.conf_high,
                out.vcov,
            ]
            .iter()
            .any(|q| q.is_null());
        let cor_failed = nc > 0
            && (out.correlation_params.is_null()
                || (has_intervals
                    && (out.correlation_conf_low.is_null()
                        || out.correlation_conf_high.is_null())));
        let var_failed = nv > 0
            && (out.variance_params.is_null()
                || (has_intervals
                    && (out.variance_conf_low.is_null() || out.variance_conf_high.is_null())));
        let strata_failed = !strata.is_empty() && out.variance_strata.is_null();
        let vg_failed = !vg.is_empty()
            && (out.variogram_dist.is_null()
                || out.variogram_variog.is_null()
                || out.variogram_n_pairs.is_null());
        *out_result = out;
        if coef_failed || cor_failed || var_failed || strata_failed || vg_failed {
            anofox_free_gls_fit_result(out_result);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate GLS result",
                );
            }
            return false;
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_gls_fit`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_gls_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_gls_fit_result(result: *mut GlsFitResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.coefficients,
            &mut r.std_errors,
            &mut r.t_values,
            &mut r.p_values,
            &mut r.conf_low,
            &mut r.conf_high,
            &mut r.vcov,
            &mut r.correlation_params,
            &mut r.correlation_conf_low,
            &mut r.correlation_conf_high,
            &mut r.variance_params,
            &mut r.variance_conf_low,
            &mut r.variance_conf_high,
            &mut r.variance_strata,
            &mut r.variogram_dist,
            &mut r.variogram_variog,
            &mut r.variogram_n_pairs,
        ] {
            gls_free(p);
        }
    })
}

/// Per-row results of a GLS fit: `x'β̂` and its standard error for every row
/// with finite x, and the response / Pearson / normalized residuals of the
/// training rows.
///
/// # Safety
/// As `anofox_gls_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_gls_augment(
    data: *const GlsDataFFI,
    options: GlsOptionsFFI,
    out_result: *mut GlsAugmentResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let Some(a) = with_gls_model(data, &options, out_error, |m, input| m.augment(&input.x))
        else {
            return false;
        };
        let n = a.is_training.len();
        let out = GlsAugmentResultFFI {
            fitted: gls_malloc(&a.fitted),
            se_fit: gls_malloc(&a.se_fit),
            residual: gls_malloc(&a.residual),
            pearson_residual: gls_malloc(&a.pearson),
            normalized_residual: gls_malloc(&a.normalized),
            is_training: gls_malloc(&a.is_training),
            n_rows: n,
        };
        let failed = n > 0
            && (out.fitted.is_null()
                || out.se_fit.is_null()
                || out.residual.is_null()
                || out.pearson_residual.is_null()
                || out.normalized_residual.is_null()
                || out.is_training.is_null());
        *out_result = out;
        if failed {
            anofox_free_gls_augment_result(out_result);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate GLS augment result",
                );
            }
            return false;
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_gls_augment`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_gls_augment`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_gls_augment_result(result: *mut GlsAugmentResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        gls_free(&mut r.fitted);
        gls_free(&mut r.se_fit);
        gls_free(&mut r.residual);
        gls_free(&mut r.pearson_residual);
        gls_free(&mut r.normalized_residual);
        gls_free(&mut r.is_training);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> GlsOptionsFFI {
        GlsOptionsFFI {
            confidence_level: 0.95,
            max_iterations: 500,
            correlation: 2,
            variance: 0,
            variogram_bins: 20,
            variogram_residuals: 0,
            nugget: false,
            reml: true,
            fit_intercept: true,
        }
    }

    #[test]
    fn fit_and_augment_round_trip() {
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| (i % 6) as f64).collect();
        let subject: Vec<i32> = (0..n).map(|i| (i / 6) as i32).collect();
        let time = x.clone();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x[i] + 0.2 * subject[i] as f64 + ((i * 13) % 7) as f64 / 10.0)
            .collect();
        let data = GlsDataFFI {
            y: y.as_ptr(),
            x: x.as_ptr(),
            subject: subject.as_ptr(),
            time: time.as_ptr(),
            variance_covariate: std::ptr::null(),
            n_rows: n,
            n_features: 1,
        };
        unsafe {
            let mut res: GlsFitResultFFI = std::mem::zeroed();
            let mut err = AnofoxError::success();
            assert!(anofox_gls_fit(&data, options(), &mut res, &mut err));
            assert_eq!(res.n_coefficients, 2);
            assert_eq!(res.n_correlation_params, 1);
            assert_eq!(res.n_groups, 10);
            assert!(res.n_variogram > 0);
            anofox_free_gls_fit_result(&mut res);

            let mut aug: GlsAugmentResultFFI = std::mem::zeroed();
            assert!(anofox_gls_augment(&data, options(), &mut aug, &mut err));
            assert_eq!(aug.n_rows, n);
            anofox_free_gls_augment_result(&mut aug);

            let mut bad = options();
            bad.correlation = 99;
            assert!(!anofox_gls_fit(&data, bad, &mut res, &mut err));
        }
    }
}
