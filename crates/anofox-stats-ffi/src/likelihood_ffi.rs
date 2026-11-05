//! C FFI for the likelihood-ratio and score tests and the profile-likelihood
//! confidence intervals of GLM fits (#310). The structs are mirrored in
//! `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::models::{
    glm_profile_confint, glm_score_test, lr_test, model_lr_test, GlmFamilySpec, ModelLikelihood,
};
use anofox_stats_core::types::{BinomialLink, PoissonLink};

/// The likelihood summary of a fitted model STRUCT. Unknown numbers are NaN,
/// unknown names empty, an unknown `n_observations` is -1.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelLikelihoodFFI {
    pub model_type: [c_char; MODEL_NAME_LEN],
    pub family: [c_char; MODEL_NAME_LEN],
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub dispersion: f64,
    /// Estimated mean parameters, intercept included.
    pub n_parameters: u64,
    pub n_observations: i64,
}

/// Result of `anofox_lr_test` and `anofox_glm_score_test`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ChiSqTestResultFFI {
    pub statistic: f64,
    pub df: f64,
    pub p_value: f64,
    /// Dispersion the statistic was divided by.
    pub dispersion: f64,
    /// Negative Binomial shape used by the score test (NaN otherwise).
    pub theta: f64,
    /// Rows used (-1 when unknown).
    pub n_observations: i64,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

/// GLM family of the score test / profile intervals. `kind`: 0 gaussian,
/// 1 poisson, 2 binomial, 3 negative binomial, 4 gamma, 5 tweedie (the
/// `family` codes of the GLMM options). `link`: the Poisson / Binomial link
/// code. `param`: Negative Binomial theta (NaN: estimate) or Tweedie power.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LikelihoodFamilyFFI {
    pub kind: i32,
    pub link: i32,
    pub param: f64,
}

fn name_of(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl From<&ModelLikelihoodFFI> for ModelLikelihood {
    fn from(m: &ModelLikelihoodFFI) -> Self {
        ModelLikelihood {
            model_type: name_of(&m.model_type),
            family: name_of(&m.family),
            log_likelihood: m.log_likelihood,
            null_log_likelihood: m.null_log_likelihood,
            deviance: m.deviance,
            null_deviance: m.null_deviance,
            dispersion: m.dispersion,
            n_parameters: m.n_parameters as usize,
            n_observations: usize::try_from(m.n_observations).ok(),
        }
    }
}

fn family_spec(f: &LikelihoodFamilyFFI) -> Result<GlmFamilySpec, &'static str> {
    Ok(match f.kind {
        0 => GlmFamilySpec::Gaussian,
        1 => GlmFamilySpec::Poisson(match f.link {
            0 => PoissonLink::Log,
            1 => PoissonLink::Identity,
            2 => PoissonLink::Sqrt,
            _ => return Err("invalid Poisson link"),
        }),
        2 => GlmFamilySpec::Binomial(match f.link {
            0 => BinomialLink::Logit,
            1 => BinomialLink::Probit,
            2 => BinomialLink::Cloglog,
            _ => return Err("invalid binomial link"),
        }),
        3 => GlmFamilySpec::NegativeBinomial(if f.param.is_nan() {
            None
        } else {
            Some(f.param)
        }),
        4 => GlmFamilySpec::Gamma,
        5 => {
            if !(f.param.is_finite() && (1.0..=2.0).contains(&f.param)) {
                return Err("Tweedie power must be in [1, 2]");
            }
            GlmFamilySpec::Tweedie(f.param)
        }
        _ => return Err("invalid GLM family"),
    })
}

unsafe fn columns(ptr: *const DataArray, count: usize) -> Vec<Vec<f64>> {
    if ptr.is_null() || count == 0 {
        return Vec::new();
    }
    slice::from_raw_parts(ptr, count)
        .iter()
        .map(|a| a.to_vec())
        .collect()
}

fn offset_of(offset_column: u64) -> Option<usize> {
    (offset_column > 0).then_some(offset_column as usize)
}

fn unified(method: &str, statistic: f64, df: f64, p_value: f64, n: i64) -> TestSummaryFFI {
    TestSummaryFFI {
        statistic,
        df1: df,
        p_value,
        n,
        method: fixed_buffer(method),
        ..TestSummaryFFI::default()
    }
}

/// Likelihood-ratio test of `full` against `reduced` (R's
/// `anova(reduced, full, test = "LRT")`), or of `full` against its null model
/// when `reduced` is NULL. Incompatible models are INVALID_INPUT.
///
/// # Safety
/// `reduced` may be NULL; `out_result` must be valid.
#[no_mangle]
pub unsafe extern "C" fn anofox_lr_test(
    full: ModelLikelihoodFFI,
    reduced: *const ModelLikelihoodFFI,
    out_result: *mut ChiSqTestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let f = ModelLikelihood::from(&full);
        let res = if reduced.is_null() {
            model_lr_test(&f)
        } else {
            lr_test(&f, &ModelLikelihood::from(&*reduced))
        };
        match res {
            Ok(r) => {
                let t = r.test;
                let disp = r.dispersion;
                *out_result = ChiSqTestResultFFI {
                    statistic: t.statistic,
                    df: t.df as f64,
                    p_value: t.p_value,
                    dispersion: disp,
                    theta: f64::NAN,
                    n_observations: full.n_observations,
                    unified: unified(
                        "likelihood ratio",
                        t.statistic,
                        t.df as f64,
                        t.p_value,
                        full.n_observations,
                    ),
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

/// Rao score test for adding the `x_add` columns to the GLM of `y` on
/// `x_reduced` (R's `anova(reduced, full, test = "Rao")`), on the rows where
/// every value is finite. `offset_column` is a 1-based column of `x_reduced`
/// (0: none). `out_z` receives the `x_add_count` per-column z statistics.
///
/// # Safety
/// `x_reduced` (may be NULL when the count is 0) and `x_add` must point to
/// that many valid arrays as long as `y`; `out_z` to `x_add_count` doubles;
/// `out_result` must be valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_glm_score_test(
    family: LikelihoodFamilyFFI,
    y: DataArray,
    x_reduced: *const DataArray,
    x_reduced_count: usize,
    x_add: *const DataArray,
    x_add_count: usize,
    fit_intercept: bool,
    offset_column: u64,
    out_z: *mut f64,
    out_result: *mut ChiSqTestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        if x_add.is_null() || x_add_count == 0 || out_z.is_null() {
            (*out_error).set(
                ErrorCode::InvalidInput,
                "x_add must have at least one column",
            );
            return false;
        }
        let spec = match family_spec(&family) {
            Ok(s) => s,
            Err(msg) => {
                (*out_error).set(ErrorCode::InvalidInput, msg);
                return false;
            }
        };
        match glm_score_test(
            spec,
            &y.to_vec(),
            &columns(x_reduced, x_reduced_count),
            &columns(x_add, x_add_count),
            fit_intercept,
            offset_of(offset_column),
        ) {
            Ok(r) => {
                let z = slice::from_raw_parts_mut(out_z, x_add_count);
                for (dst, src) in z.iter_mut().zip(r.z.iter()) {
                    *dst = *src;
                }
                let n = i64::try_from(r.n_observations).unwrap_or(-1);
                *out_result = ChiSqTestResultFFI {
                    statistic: r.test.statistic,
                    df: r.test.df as f64,
                    p_value: r.test.p_value,
                    dispersion: r.dispersion,
                    theta: r.theta,
                    n_observations: n,
                    unified: unified(
                        "Rao score",
                        r.test.statistic,
                        r.test.df as f64,
                        r.test.p_value,
                        n,
                    ),
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

/// Profile-likelihood confidence intervals (R's `confint(glm)`) of the GLM of
/// `y` on `x` whose fitted feature coefficients are `coefficients` (NaN: a
/// column the fit dropped, which gets a NaN interval). `offset_column` is a
/// 1-based column of `x` (0: none); `coefficients_len` is the number of the
/// other columns. Writes one bound per coefficient to `out_lower` /
/// `out_upper` and the intercept's to `out_intercept` (2 doubles, NaN without
/// an intercept). An unreached bound is NaN.
///
/// # Safety
/// `x` must point to `x_count` valid arrays as long as `y`; `coefficients`,
/// `out_lower` and `out_upper` to `coefficients_len` doubles; `out_intercept`
/// to 2 doubles.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_glm_profile_confint(
    family: LikelihoodFamilyFFI,
    y: DataArray,
    x: *const DataArray,
    x_count: usize,
    coefficients: *const f64,
    coefficients_len: usize,
    fit_intercept: bool,
    offset_column: u64,
    level: f64,
    out_lower: *mut f64,
    out_upper: *mut f64,
    out_intercept: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if out_intercept.is_null()
            || (coefficients_len > 0
                && (coefficients.is_null() || out_lower.is_null() || out_upper.is_null()))
        {
            (*out_error).set(ErrorCode::InvalidInput, "output buffer is NULL");
            return false;
        }
        let spec = match family_spec(&family) {
            Ok(s) => s,
            Err(msg) => {
                (*out_error).set(ErrorCode::InvalidInput, msg);
                return false;
            }
        };
        let coef = if coefficients_len == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(coefficients, coefficients_len)
        };
        match glm_profile_confint(
            spec,
            &y.to_vec(),
            &columns(x, x_count),
            coef,
            fit_intercept,
            offset_of(offset_column),
            level,
        ) {
            Ok((intercept, slopes)) => {
                let (lo, hi) = intercept.unwrap_or((f64::NAN, f64::NAN));
                *out_intercept = lo;
                *out_intercept.add(1) = hi;
                for (j, (l, u)) in slopes.into_iter().enumerate() {
                    *out_lower.add(j) = l;
                    *out_upper.add(j) = u;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model(family: &str, dev: f64, k: u64) -> ModelLikelihoodFFI {
        ModelLikelihoodFFI {
            model_type: fixed_buffer("glm"),
            family: fixed_buffer(family),
            log_likelihood: f64::NAN,
            null_log_likelihood: f64::NAN,
            deviance: dev,
            null_deviance: 40.0,
            dispersion: 1.0,
            n_parameters: k,
            n_observations: 30,
        }
    }

    #[test]
    fn lr_test_round_trips_through_ffi() {
        let mut out = std::mem::MaybeUninit::<ChiSqTestResultFFI>::uninit();
        let mut err = AnofoxError::success();
        let reduced = model("binomial", 25.0, 2);
        let ok = unsafe {
            anofox_lr_test(
                model("binomial", 20.0, 4),
                &reduced,
                out.as_mut_ptr(),
                &mut err,
            )
        };
        assert!(ok);
        let out = unsafe { out.assume_init() };
        assert!((out.statistic - 5.0).abs() < 1e-12);
        assert_eq!(out.df, 2.0);

        let ok = unsafe {
            anofox_lr_test(
                model("poisson", 20.0, 4),
                &reduced,
                &mut std::mem::zeroed(),
                &mut err,
            )
        };
        assert!(!ok);
        assert_eq!(err.code, ErrorCode::InvalidInput);
    }
}
