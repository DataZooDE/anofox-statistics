//! C FFI for Cox proportional hazards regression (#312). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::{
    cox_survival_curve, fit_cox, CoxModel, CoxRows, CoxSpec, CoxTies, ZphTransform,
};

/// Options of a Cox fit. `ties`: 0 Efron, 1 Breslow. `zph_transform` (only
/// read by `anofox_cox_zph`): 0 km, 1 rank, 2 identity, 3 log.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CoxOptionsFFI {
    pub tolerance: f64,
    pub ridge: f64,
    pub confidence_level: f64,
    pub max_iterations: u32,
    pub ties: i32,
    pub zph_transform: i32,
    pub timefix: bool,
    pub robust: bool,
}

/// The rows of a Cox fit. `x` is column-major (`n_rows * n_features`); the
/// optional columns are NULL when absent. Rows with a non-finite value in any
/// column are left out of the fit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CoxDataFFI {
    pub time: *const f64,
    pub event: *const f64,
    pub x: *const f64,
    pub weights: *const f64,
    pub entry: *const f64,
    pub offset: *const f64,
    pub strata: *const f64,
    pub cluster: *const f64,
    pub n_rows: usize,
    pub n_features: usize,
}

/// A fitted Cox model. Arrays are `libc::malloc`ed and released with
/// `anofox_free_cox_fit_result`.
#[repr(C)]
#[derive(Debug)]
pub struct CoxFitResultFFI {
    pub coefficients: *mut f64,
    pub std_errors: *mut f64,
    pub naive_std_errors: *mut f64,
    pub z_values: *mut f64,
    pub p_values: *mut f64,
    pub conf_low: *mut f64,
    pub conf_high: *mut f64,
    pub hazard_ratios: *mut f64,
    pub hr_conf_low: *mut f64,
    pub hr_conf_high: *mut f64,
    /// `n_features * n_features`, row-major.
    pub vcov: *mut f64,
    pub n_features: usize,
    /// Baseline cumulative hazard at covariates 0, per stratum (stratum NaN
    /// when unstratified), `n_baseline` points.
    pub baseline_stratum: *mut f64,
    pub baseline_time: *mut f64,
    pub baseline_cumhaz: *mut f64,
    pub n_baseline: usize,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub lr_statistic: f64,
    pub lr_p_value: f64,
    pub wald_statistic: f64,
    pub wald_p_value: f64,
    pub score_statistic: f64,
    pub score_p_value: f64,
    pub concordance: f64,
    pub r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub confidence_level: f64,
    pub df: usize,
    pub n_observations: usize,
    pub n_events: usize,
    pub n_strata: usize,
    pub iterations: usize,
    pub converged: bool,
    pub robust: bool,
    pub summary: ModelSummaryFFI,
}

/// Proportional-hazards tests: one per covariate, the global test last
/// (`len = n_features + 1`). Released with `anofox_free_cox_zph_result`.
#[repr(C)]
#[derive(Debug)]
pub struct CoxZphResultFFI {
    pub chisq: *mut f64,
    pub df: *mut f64,
    pub p_value: *mut f64,
    pub len: usize,
}

/// Per-input-row results (NaN for rows left out of the fit); the `*_mat`
/// arrays are `n_rows * n_features`, row-major. Released with
/// `anofox_free_cox_augment_result`.
#[repr(C)]
#[derive(Debug)]
pub struct CoxAugmentResultFFI {
    pub linear_predictor: *mut f64,
    pub risk: *mut f64,
    pub martingale: *mut f64,
    pub deviance: *mut f64,
    pub stratum: *mut f64,
    pub score_mat: *mut f64,
    pub dfbeta_mat: *mut f64,
    pub dfbetas_mat: *mut f64,
    pub schoenfeld_mat: *mut f64,
    pub scaled_schoenfeld_mat: *mut f64,
    pub is_training: *mut bool,
    pub n_rows: usize,
    pub n_features: usize,
}

unsafe fn cox_malloc<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

unsafe fn cox_free<T>(p: &mut *mut T) {
    if !p.is_null() {
        libc::free(*p as *mut libc::c_void);
        *p = std::ptr::null_mut();
    }
}

unsafe fn view<'a>(p: *const f64, n: usize) -> Option<&'a [f64]> {
    if p.is_null() {
        None
    } else if n == 0 {
        Some(&[])
    } else {
        Some(slice::from_raw_parts(p, n))
    }
}

fn spec_from_ffi(o: &CoxOptionsFFI) -> Result<CoxSpec, StatsError> {
    let ties = match o.ties {
        0 => CoxTies::Efron,
        1 => CoxTies::Breslow,
        t => return Err(StatsError::InvalidInput(format!("unknown ties code {t}"))),
    };
    if !(o.tolerance > 0.0 && o.tolerance.is_finite()) {
        return Err(StatsError::InvalidValue {
            field: "tolerance",
            message: format!("must be positive, got {}", o.tolerance),
        });
    }
    if o.max_iterations == 0 {
        return Err(StatsError::InvalidValue {
            field: "max_iterations",
            message: "must be at least 1".into(),
        });
    }
    Ok(CoxSpec {
        ties,
        max_iter: o.max_iterations as usize,
        eps: o.tolerance,
        ridge: o.ridge,
        conf_level: o.confidence_level,
        timefix: o.timefix,
        robust: o.robust,
    })
}

/// Read the rows, fit, and hand the model to `body`.
unsafe fn with_cox_model<R>(
    data: *const CoxDataFFI,
    options: &CoxOptionsFFI,
    out_error: *mut AnofoxError,
    body: impl FnOnce(CoxModel) -> Result<R, StatsError>,
) -> Option<R> {
    if data.is_null() {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "data is NULL");
        }
        return None;
    }
    let d = &*data;
    let n = d.n_rows;
    let (Some(time), Some(event)) = (view(d.time, n), view(d.event, n)) else {
        if !out_error.is_null() {
            (*out_error).set(ErrorCode::InvalidInput, "time or event is NULL");
        }
        return None;
    };
    let x: Vec<Vec<f64>> = match view(d.x, n * d.n_features) {
        Some(all) if n > 0 => all.chunks(n).map(<[f64]>::to_vec).collect(),
        _ => vec![Vec::new(); d.n_features],
    };
    let rows = CoxRows {
        time,
        event,
        x: &x,
        weights: view(d.weights, n),
        entry: view(d.entry, n),
        offset: view(d.offset, n),
        strata: view(d.strata, n),
        cluster: view(d.cluster, n),
    };
    let result = spec_from_ffi(options)
        .and_then(|spec| fit_cox(&rows, &spec))
        .and_then(body);
    match result {
        Ok(r) => Some(r),
        Err(e) => {
            set_error(out_error, &e);
            None
        }
    }
}

/// Fit a Cox proportional hazards model (`survival::coxph`).
///
/// # Safety
/// - `data` must point to a valid `CoxDataFFI` whose arrays hold `n_rows`
///   values (`x`: `n_rows * n_features`) or are NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_cox_fit(
    data: *const CoxDataFFI,
    options: CoxOptionsFFI,
    out_result: *mut CoxFitResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let fitted = with_cox_model(data, &options, out_error, |m| {
            let base = m.baseline()?;
            Ok((m.summary(), base))
        });
        let Some((s, base)) = fitted else {
            return false;
        };
        let p = s.coefficients.len();
        let bs: Vec<f64> = base.iter().map(|b| b.0).collect();
        let bt: Vec<f64> = base.iter().map(|b| b.1).collect();
        let bh: Vec<f64> = base.iter().map(|b| b.2).collect();
        let mut summary = ModelSummaryFFI {
            model_type: name_buffer("cox"),
            family: name_buffer("coxph"),
            link: name_buffer("log"),
            log_likelihood: s.log_likelihood,
            aic: s.aic,
            bic: s.bic,
            ..ModelSummaryFFI::default()
        };
        summary.intercept_std_error = f64::NAN;
        let out = CoxFitResultFFI {
            coefficients: cox_malloc(&s.coefficients),
            std_errors: cox_malloc(&s.std_errors),
            naive_std_errors: cox_malloc(&s.naive_std_errors),
            z_values: cox_malloc(&s.z_values),
            p_values: cox_malloc(&s.p_values),
            conf_low: cox_malloc(&s.conf_low),
            conf_high: cox_malloc(&s.conf_high),
            hazard_ratios: cox_malloc(&s.hazard_ratios),
            hr_conf_low: cox_malloc(&s.hr_conf_low),
            hr_conf_high: cox_malloc(&s.hr_conf_high),
            vcov: cox_malloc(&s.vcov),
            n_features: p,
            baseline_stratum: cox_malloc(&bs),
            baseline_time: cox_malloc(&bt),
            baseline_cumhaz: cox_malloc(&bh),
            n_baseline: base.len(),
            log_likelihood: s.log_likelihood,
            null_log_likelihood: s.null_log_likelihood,
            lr_statistic: s.lr_statistic,
            lr_p_value: s.lr_p_value,
            wald_statistic: s.wald_statistic,
            wald_p_value: s.wald_p_value,
            score_statistic: s.score_statistic,
            score_p_value: s.score_p_value,
            concordance: s.concordance,
            r_squared: s.r_squared,
            aic: s.aic,
            bic: s.bic,
            confidence_level: options.confidence_level,
            df: s.df,
            n_observations: s.n_observations,
            n_events: s.n_events,
            n_strata: s.n_strata,
            iterations: s.iterations,
            converged: s.converged,
            robust: s.robust,
            summary,
        };
        let failed = (p > 0
            && [
                out.coefficients,
                out.std_errors,
                out.naive_std_errors,
                out.z_values,
                out.p_values,
                out.conf_low,
                out.conf_high,
                out.hazard_ratios,
                out.hr_conf_low,
                out.hr_conf_high,
                out.vcov,
            ]
            .iter()
            .any(|p| p.is_null()))
            || (!base.is_empty()
                && (out.baseline_stratum.is_null()
                    || out.baseline_time.is_null()
                    || out.baseline_cumhaz.is_null()));
        *out_result = out;
        if failed {
            anofox_free_cox_fit_result(out_result);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate Cox result",
                );
            }
            return false;
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_cox_fit`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_cox_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_cox_fit_result(result: *mut CoxFitResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.coefficients,
            &mut r.std_errors,
            &mut r.naive_std_errors,
            &mut r.z_values,
            &mut r.p_values,
            &mut r.conf_low,
            &mut r.conf_high,
            &mut r.hazard_ratios,
            &mut r.hr_conf_low,
            &mut r.hr_conf_high,
            &mut r.vcov,
            &mut r.baseline_stratum,
            &mut r.baseline_time,
            &mut r.baseline_cumhaz,
        ] {
            cox_free(p);
        }
    })
}

/// Grambsch–Therneau proportional-hazards test of a Cox fit
/// (`survival::cox.zph`), one row per covariate and the global test last.
///
/// # Safety
/// As `anofox_cox_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_cox_zph(
    data: *const CoxDataFFI,
    options: CoxOptionsFFI,
    out_result: *mut CoxZphResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let transform = match options.zph_transform {
            0 => ZphTransform::Km,
            1 => ZphTransform::Rank,
            2 => ZphTransform::Identity,
            3 => ZphTransform::Log,
            t => {
                set_error(
                    out_error,
                    &StatsError::InvalidInput(format!("unknown zph transform code {t}")),
                );
                return false;
            }
        };
        let Some(rows) = with_cox_model(data, &options, out_error, |m| m.zph(transform)) else {
            return false;
        };
        let chisq: Vec<f64> = rows.iter().map(|r| r.0).collect();
        let df: Vec<f64> = rows.iter().map(|r| r.1 as f64).collect();
        let pv: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let out = CoxZphResultFFI {
            chisq: cox_malloc(&chisq),
            df: cox_malloc(&df),
            p_value: cox_malloc(&pv),
            len: rows.len(),
        };
        let failed =
            !rows.is_empty() && (out.chisq.is_null() || out.df.is_null() || out.p_value.is_null());
        *out_result = out;
        if failed {
            anofox_free_cox_zph_result(out_result);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate zph result",
                );
            }
            return false;
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_cox_zph`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_cox_zph`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_cox_zph_result(result: *mut CoxZphResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        cox_free(&mut r.chisq);
        cox_free(&mut r.df);
        cox_free(&mut r.p_value);
    })
}

/// Per-row results of a Cox fit (linear predictor, martingale / deviance /
/// score / dfbeta / Schoenfeld residuals), one entry per input row.
///
/// # Safety
/// As `anofox_cox_fit`.
#[no_mangle]
pub unsafe extern "C" fn anofox_cox_augment(
    data: *const CoxDataFFI,
    options: CoxOptionsFFI,
    out_result: *mut CoxAugmentResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let Some(a) = with_cox_model(data, &options, out_error, |m| Ok(m.augment())) else {
            return false;
        };
        let n = a.is_training.len();
        let out = CoxAugmentResultFFI {
            linear_predictor: cox_malloc(&a.linear_predictor),
            risk: cox_malloc(&a.risk),
            martingale: cox_malloc(&a.martingale),
            deviance: cox_malloc(&a.deviance),
            stratum: cox_malloc(&a.stratum),
            score_mat: cox_malloc(&a.score),
            dfbeta_mat: cox_malloc(&a.dfbeta),
            dfbetas_mat: cox_malloc(&a.dfbetas),
            schoenfeld_mat: cox_malloc(&a.schoenfeld),
            scaled_schoenfeld_mat: cox_malloc(&a.scaled_schoenfeld),
            is_training: cox_malloc(&a.is_training),
            n_rows: n,
            n_features: a.n_features,
        };
        let failed = n > 0
            && ([
                out.linear_predictor,
                out.risk,
                out.martingale,
                out.deviance,
                out.stratum,
            ]
            .iter()
            .any(|p| p.is_null())
                || out.is_training.is_null()
                || (a.n_features > 0
                    && [
                        out.score_mat,
                        out.dfbeta_mat,
                        out.dfbetas_mat,
                        out.schoenfeld_mat,
                        out.scaled_schoenfeld_mat,
                    ]
                    .iter()
                    .any(|p| p.is_null())));
        *out_result = out;
        if failed {
            anofox_free_cox_augment_result(out_result);
            if !out_error.is_null() {
                (*out_error).set(
                    ErrorCode::AllocationFailure,
                    "Failed to allocate Cox augment result",
                );
            }
            return false;
        }
        true
    })
}

/// Free the arrays of a result filled by `anofox_cox_augment`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_cox_augment`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_cox_augment_result(result: *mut CoxAugmentResultFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        for p in [
            &mut r.linear_predictor,
            &mut r.risk,
            &mut r.martingale,
            &mut r.deviance,
            &mut r.stratum,
            &mut r.score_mat,
            &mut r.dfbeta_mat,
            &mut r.dfbetas_mat,
            &mut r.schoenfeld_mat,
            &mut r.scaled_schoenfeld_mat,
        ] {
            cox_free(p);
        }
        cox_free(&mut r.is_training);
    })
}

/// Predicted survival `S(t | x)` of a fitted Cox model at `n_times` times from
/// its baseline cumulative hazard at covariates 0 (`n_baseline` points,
/// increasing times), written to `out_survival`. Returns false only for NULL
/// pointers.
///
/// # Safety
/// `coefficients` and `x` must hold `n_coefficients` values, `base_time` and
/// `base_cumhaz` `n_baseline`, `times` and `out_survival` `n_times`.
#[no_mangle]
pub unsafe extern "C" fn anofox_cox_survival(
    coefficients: *const f64,
    n_coefficients: usize,
    base_time: *const f64,
    base_cumhaz: *const f64,
    n_baseline: usize,
    x: *const f64,
    times: *const f64,
    n_times: usize,
    out_survival: *mut f64,
) -> bool {
    ffi_guard(std::ptr::null_mut(), false, || {
        let (Some(c), Some(bt), Some(bh), Some(xr), Some(t)) = (
            view(coefficients, n_coefficients).or(if n_coefficients == 0 {
                Some(&[])
            } else {
                None
            }),
            view(base_time, n_baseline).or(if n_baseline == 0 { Some(&[]) } else { None }),
            view(base_cumhaz, n_baseline).or(if n_baseline == 0 { Some(&[]) } else { None }),
            view(x, n_coefficients).or(if n_coefficients == 0 { Some(&[]) } else { None }),
            view(times, n_times).or(if n_times == 0 { Some(&[]) } else { None }),
        ) else {
            return false;
        };
        if n_times > 0 && out_survival.is_null() {
            return false;
        }
        let s = cox_survival_curve(c, bt, bh, xr, t);
        for (i, v) in s.iter().enumerate() {
            *out_survival.add(i) = *v;
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> CoxOptionsFFI {
        CoxOptionsFFI {
            tolerance: 1e-9,
            ridge: 0.0,
            confidence_level: 0.95,
            max_iterations: 20,
            ties: 0,
            zph_transform: 0,
            timefix: true,
            robust: false,
        }
    }

    #[test]
    fn options_layout_is_doubles_then_ints_then_bools() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(CoxOptionsFFI, max_iterations), 24);
        assert_eq!(offset_of!(CoxOptionsFFI, timefix), 36);
        assert_eq!(offset_of!(CoxOptionsFFI, robust), 37);
    }

    #[test]
    fn fit_zph_augment_and_survival_round_trip() {
        let time = [6.0, 7.0, 10.0, 15.0, 19.0, 25.0, 3.0, 9.0, 11.0, 20.0];
        let event = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let x = [0.5, 1.2, -0.3, 0.8, 2.0, -1.0, 1.5, 0.1, -0.7, 0.4];
        let data = CoxDataFFI {
            time: time.as_ptr(),
            event: event.as_ptr(),
            x: x.as_ptr(),
            weights: std::ptr::null(),
            entry: std::ptr::null(),
            offset: std::ptr::null(),
            strata: std::ptr::null(),
            cluster: std::ptr::null(),
            n_rows: 10,
            n_features: 1,
        };
        unsafe {
            let mut err = AnofoxError::success();
            let mut fit: CoxFitResultFFI = std::mem::zeroed();
            assert!(anofox_cox_fit(&data, options(), &mut fit, &mut err));
            assert_eq!(fit.n_features, 1);
            assert!(fit.n_baseline > 0);
            let mut s = [0.0; 2];
            assert!(anofox_cox_survival(
                fit.coefficients,
                1,
                fit.baseline_time,
                fit.baseline_cumhaz,
                fit.n_baseline,
                [0.0].as_ptr(),
                [0.0, 30.0].as_ptr(),
                2,
                s.as_mut_ptr()
            ));
            assert_eq!(s[0], 1.0);
            assert!(s[1] < 1.0);
            anofox_free_cox_fit_result(&mut fit);

            let mut z: CoxZphResultFFI = std::mem::zeroed();
            assert!(anofox_cox_zph(&data, options(), &mut z, &mut err));
            assert_eq!(z.len, 2);
            anofox_free_cox_zph_result(&mut z);

            let mut a: CoxAugmentResultFFI = std::mem::zeroed();
            assert!(anofox_cox_augment(&data, options(), &mut a, &mut err));
            assert_eq!(a.n_rows, 10);
            anofox_free_cox_augment_result(&mut a);

            let mut bad = options();
            bad.ties = 7;
            assert!(!anofox_cox_fit(&data, bad, &mut fit, &mut err));
            assert!(matches!(err.code, ErrorCode::InvalidInput));
        }
    }
}
