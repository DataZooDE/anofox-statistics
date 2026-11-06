//! C FFI for effects, contrasts, effect grids and nomograms (#319).
//! Declarations mirrored in `src/include/anofox_stats_ffi.h`.
//!
//! Every function takes an [`EffectsInputFFI`]: the term list, the training
//! data of the predictors (column-major) and the fitted model. Variable-length
//! results are written into caller buffers: `*out_len` is set to the length of
//! the result, which is written only when it fits `capacity`, so the caller
//! can grow its buffer and call again. The nomogram is returned in arrays
//! allocated here and released with `anofox_free_nomogram`.

use super::*;
use anofox_stats_core::effects::{Effects, FittedModel, NomogramFun, Setting, TermKind};

/// One term. Must match `AnofoxEffectsTerm`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EffectsTermFFI {
    /// 0 linear, 1 rcs, 2 lsp, 3 categorical, 4 interaction, 5 restricted
    /// interaction (`%ia%`)
    pub kind: i32,
    /// rcs: `norm` of `rcspline.eval` (0, 1, 2)
    pub norm: i32,
    /// rcs / lsp: the knots; categorical: the levels (`n_values == 0`: from
    /// the data)
    pub values: *const f64,
    pub n_values: usize,
    /// interactions: the two predictors (indices among the non-interaction
    /// terms)
    pub a: usize,
    pub b: usize,
}

/// Terms, predictor data and fitted model. Must match `AnofoxEffectsInput`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EffectsInputFFI {
    pub terms: *const EffectsTermFFI,
    pub n_terms: usize,
    /// Column-major: predictor `k` of row `i` is `data[k * n_rows + i]`.
    pub data: *const f64,
    pub n_rows: usize,
    /// Coefficients, intercept first when `intercept`.
    pub beta: *const f64,
    pub n_beta: usize,
    /// Row-major `n_beta x n_beta` covariance (may be NULL for the nomogram
    /// and the design).
    pub vcov: *const f64,
    pub intercept: bool,
    /// Residual degrees of freedom of a t reference; NaN: standard normal.
    pub df: f64,
    pub level: f64,
}

/// A predictor setting. Must match `AnofoxEffectsSetting`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EffectsSettingFFI {
    pub predictors: *const usize,
    pub values: *const f64,
    pub n: usize,
}

/// A contrast estimate. Must match `AnofoxContrastEstimate`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContrastEstimateFFI {
    pub estimate: f64,
    pub se: f64,
    pub lower: f64,
    pub upper: f64,
    pub statistic: f64,
    pub p_value: f64,
}

/// A `summary.rms` row. Must match `AnofoxEffectRow`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EffectRowFFI {
    pub predictor: usize,
    pub low: f64,
    pub high: f64,
    pub effect: ContrastEstimateFFI,
}

/// A nomogram. Must match `AnofoxNomogramResult`. The arrays are allocated by
/// `anofox_effects_nomogram` and released with `anofox_free_nomogram`.
#[repr(C)]
#[derive(Debug)]
pub struct NomogramFFI {
    /// `n_axes + 1` offsets into the axis arrays.
    pub axis_offsets: *mut usize,
    pub n_axes: usize,
    pub axis_values: *mut f64,
    pub axis_xbeta: *mut f64,
    pub axis_points: *mut f64,
    pub scale: f64,
    pub intercept: f64,
    pub total_points: *mut f64,
    pub n_total_points: usize,
    pub lp_at: *mut f64,
    pub lp_points: *mut f64,
    pub n_lp: usize,
    pub fun_at: *mut f64,
    pub fun_points: *mut f64,
    pub n_fun: usize,
}

fn invalid(msg: &str) -> anofox_stats_core::StatsError {
    anofox_stats_core::StatsError::InvalidInput(msg.to_string())
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

unsafe fn read_terms(input: &EffectsInputFFI) -> anofox_stats_core::StatsResult<Vec<TermKind>> {
    slice(input.terms, input.n_terms)
        .iter()
        .map(|t| {
            let values = slice(t.values, t.n_values).to_vec();
            Ok(match t.kind {
                0 => TermKind::Linear,
                1 => TermKind::Rcs {
                    knots: values,
                    norm: t.norm,
                },
                2 => TermKind::Lsp { knots: values },
                3 => TermKind::Categorical {
                    levels: (!values.is_empty()).then_some(values),
                },
                4 | 5 => TermKind::Interaction {
                    a: t.a,
                    b: t.b,
                    restricted: t.kind == 5,
                },
                _ => return Err(invalid("invalid term kind")),
            })
        })
        .collect()
}

unsafe fn read_columns(input: &EffectsInputFFI, n_predictors: usize) -> Vec<Vec<f64>> {
    let data = slice(input.data, input.n_rows * n_predictors);
    (0..n_predictors)
        .map(|k| data[k * input.n_rows..(k + 1) * input.n_rows].to_vec())
        .collect()
}

unsafe fn build(
    input: *const EffectsInputFFI,
    levels_from_terms: bool,
) -> anofox_stats_core::StatsResult<(Effects, Vec<Vec<f64>>, FittedModel)> {
    if input.is_null() {
        return Err(invalid("input is NULL"));
    }
    let input = &*input;
    let terms = read_terms(input)?;
    let n_predictors = terms
        .iter()
        .filter(|t| !matches!(t, TermKind::Interaction { .. }))
        .count();
    let columns = read_columns(input, n_predictors);
    let effects = Effects::new(&terms, &columns, levels_from_terms)?;
    let beta = slice(input.beta, input.n_beta).to_vec();
    let vcov = slice(input.vcov, input.n_beta * input.n_beta).to_vec();
    let model = FittedModel {
        beta,
        vcov,
        intercept: input.intercept,
        df: input.df,
        level: input.level,
    };
    Ok((effects, columns, model))
}

unsafe fn read_setting(s: *const EffectsSettingFFI) -> Setting {
    if s.is_null() {
        return Vec::new();
    }
    let s = &*s;
    slice(s.predictors, s.n)
        .iter()
        .copied()
        .zip(slice(s.values, s.n).iter().copied())
        .collect()
}

/// Write `values` to `out` when they fit; always report the length.
unsafe fn write_values<T: Copy>(values: &[T], out: *mut T, capacity: usize, out_len: *mut usize) {
    *out_len = values.len();
    if values.len() <= capacity && !values.is_empty() && !out.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    }
}

unsafe fn finish<T>(
    result: anofox_stats_core::StatsResult<T>,
    out_error: *mut AnofoxError,
    write: impl FnOnce(T),
) -> bool {
    match result {
        Ok(v) => {
            write(v);
            true
        }
        Err(e) => {
            set_error(out_error, &e);
            false
        }
    }
}

fn estimate(e: &anofox_stats_core::effects::ContrastEstimate) -> ContrastEstimateFFI {
    ContrastEstimateFFI {
        estimate: e.estimate,
        se: e.se,
        lower: e.lower,
        upper: e.upper,
        statistic: e.statistic,
        p_value: e.p_value,
    }
}

/// Design rows (row-major `n_rows x n_columns`, no intercept) of the
/// predictor data of `input`; categorical terms must give their levels. The
/// model fields are ignored.
///
/// # Safety
/// `input` must be valid; `out_values` must hold `capacity` doubles; the
/// `out_*` pointers must be valid; `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_effects_design(
    input: *const EffectsInputFFI,
    out_values: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_n_columns: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) || !check_out(out_n_columns, out_error) {
            return false;
        }
        let r = build(input, true).and_then(|(e, cols, _)| Ok((e.design(&cols)?, e.n_columns())));
        finish(r, out_error, |(values, n_columns)| {
            *out_n_columns = n_columns;
            write_values(&values, out_values, capacity, out_len)
        })
    })
}

/// `summary.rms`: one row per numeric predictor (low / high effect values)
/// and per non-reference level of a categorical predictor.
///
/// # Safety
/// As `anofox_effects_design`, with `out_rows` holding `capacity` rows.
#[no_mangle]
pub unsafe extern "C" fn anofox_effects_summary(
    input: *const EffectsInputFFI,
    out_rows: *mut EffectRowFFI,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = build(input, false).and_then(|(e, _, m)| e.summary(&m));
        finish(r, out_error, |rows| {
            let rows: Vec<EffectRowFFI> = rows
                .iter()
                .map(|r| EffectRowFFI {
                    predictor: r.predictor,
                    low: r.low,
                    high: r.high,
                    effect: estimate(&r.effect),
                })
                .collect();
            write_values(&rows, out_rows, capacity, out_len)
        })
    })
}

/// The adjust-to value (median, or mode of a categorical predictor) of every
/// predictor (`datadist`), `out_values` holding `capacity` doubles.
///
/// # Safety
/// As `anofox_effects_design`.
#[no_mangle]
pub unsafe extern "C" fn anofox_effects_adjust_to(
    input: *const EffectsInputFFI,
    out_values: *mut f64,
    capacity: usize,
    out_len: *mut usize,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) {
            return false;
        }
        let r = build(input, false).map(|(e, _, _)| {
            (0..e.n_predictors())
                .map(|k| e.adjust_to(k))
                .collect::<Vec<f64>>()
        });
        finish(r, out_error, |v| {
            write_values(&v, out_values, capacity, out_len)
        })
    })
}

/// `contrast.rms(fit, a, b)`; with non-NULL `a2` and `b2` the double
/// difference `(a - b) - (a2 - b2)`.
///
/// # Safety
/// `input`, `a`, `b` and `out` must be valid; `a2` / `b2` may be NULL;
/// `out_error` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn anofox_effects_contrast(
    input: *const EffectsInputFFI,
    a: *const EffectsSettingFFI,
    b: *const EffectsSettingFFI,
    a2: *const EffectsSettingFFI,
    b2: *const EffectsSettingFFI,
    out: *mut ContrastEstimateFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out, out_error) {
            return false;
        }
        let (sa, sb) = (read_setting(a), read_setting(b));
        let second = if a2.is_null() || b2.is_null() {
            None
        } else {
            Some((read_setting(a2), read_setting(b2)))
        };
        let r = build(input, false).and_then(|(e, _, m)| {
            e.contrast(
                &m,
                &sa,
                &sb,
                second.as_ref().map(|(x, y)| (x.as_slice(), y.as_slice())),
            )
        });
        finish(r, out_error, |c| *out = estimate(&c))
    })
}

/// `Predict(fit, x_k)` over `values` (`n_values == 0`: the default grid)
/// with pointwise or (`simultaneous`) simultaneous bands; the grid values go
/// to `out_values`, the estimates to `out_estimates` (both `capacity`), the
/// simultaneous critical value to `out_critical` (NaN when pointwise).
///
/// # Safety
/// As `anofox_effects_design`; `values` must hold `n_values` doubles.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_effects_predict_grid(
    input: *const EffectsInputFFI,
    predictor: usize,
    values: *const f64,
    n_values: usize,
    simultaneous: bool,
    out_values: *mut f64,
    out_estimates: *mut ContrastEstimateFFI,
    capacity: usize,
    out_len: *mut usize,
    out_critical: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_len, out_error) || !check_out(out_critical, out_error) {
            return false;
        }
        let grid = (n_values > 0).then(|| slice(values, n_values).to_vec());
        let r = build(input, false)
            .and_then(|(e, _, m)| e.predict_grid(&m, predictor, grid, simultaneous));
        finish(r, out_error, |(v, est, crit)| {
            let est: Vec<ContrastEstimateFFI> = est.iter().map(estimate).collect();
            *out_critical = crit;
            write_values(&est, out_estimates, capacity, out_len);
            write_values(&v, out_values, capacity, out_len);
        })
    })
}

unsafe fn malloc_copy<T: Copy>(values: &[T]) -> *mut T {
    if values.is_empty() {
        return std::ptr::null_mut();
    }
    let ptr = libc::malloc(std::mem::size_of_val(values)) as *mut T;
    if !ptr.is_null() {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
}

/// `rms::nomogram` of a main-effects model: `lp` are the fit's linear
/// predictors (`n_lp`); `fun` 0 none, 1 plogis, 2 exp adds a function axis.
///
/// # Safety
/// `input` and `out_result` must be valid; `lp` must hold `n_lp` doubles;
/// `out_error` may be NULL. Release the result with `anofox_free_nomogram`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_effects_nomogram(
    input: *const EffectsInputFFI,
    lp: *const f64,
    n_lp: usize,
    maxscale: f64,
    nint: usize,
    fun: i32,
    out_result: *mut NomogramFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let fun = match fun {
            0 => Ok(NomogramFun::None),
            1 => Ok(NomogramFun::Plogis),
            2 => Ok(NomogramFun::Exp),
            _ => Err(invalid("invalid nomogram function")),
        };
        let lp = slice(lp, n_lp).to_vec();
        let r = fun.and_then(|f| {
            build(input, false).and_then(|(e, _, m)| e.nomogram(&m, &lp, maxscale, nint, f))
        });
        finish(r, out_error, |nom| {
            let mut offsets = vec![0usize];
            let (mut values, mut xbeta, mut points) = (Vec::new(), Vec::new(), Vec::new());
            for a in &nom.axes {
                values.extend_from_slice(&a.values);
                xbeta.extend_from_slice(&a.xbeta);
                points.extend_from_slice(&a.points);
                offsets.push(values.len());
            }
            *out_result = NomogramFFI {
                axis_offsets: malloc_copy(&offsets),
                n_axes: nom.axes.len(),
                axis_values: malloc_copy(&values),
                axis_xbeta: malloc_copy(&xbeta),
                axis_points: malloc_copy(&points),
                scale: nom.scale,
                intercept: nom.intercept,
                total_points: malloc_copy(&nom.total_points),
                n_total_points: nom.total_points.len(),
                lp_at: malloc_copy(&nom.lp_at),
                lp_points: malloc_copy(&nom.lp_points),
                n_lp: nom.lp_at.len(),
                fun_at: malloc_copy(&nom.fun_at),
                fun_points: malloc_copy(&nom.fun_points),
                n_fun: nom.fun_at.len(),
            };
        })
    })
}

/// Free the arrays of a result filled by `anofox_effects_nomogram`.
///
/// # Safety
/// `result` must be NULL or a result filled by `anofox_effects_nomogram`.
#[no_mangle]
pub unsafe extern "C" fn anofox_free_nomogram(result: *mut NomogramFFI) {
    ffi_guard(std::ptr::null_mut(), (), || {
        if result.is_null() {
            return;
        }
        let r = &mut *result;
        if !r.axis_offsets.is_null() {
            libc::free(r.axis_offsets as *mut libc::c_void);
            r.axis_offsets = std::ptr::null_mut();
        }
        for p in [
            &mut r.axis_values,
            &mut r.axis_xbeta,
            &mut r.axis_points,
            &mut r.total_points,
            &mut r.lp_at,
            &mut r.lp_points,
            &mut r.fun_at,
            &mut r.fun_points,
        ] {
            if !p.is_null() {
                libc::free(*p as *mut libc::c_void);
                *p = std::ptr::null_mut();
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(kind: i32) -> EffectsTermFFI {
        EffectsTermFFI {
            kind,
            norm: 2,
            values: std::ptr::null(),
            n_values: 0,
            a: 0,
            b: 0,
        }
    }

    #[test]
    fn summary_contrast_and_grid() {
        let terms = [term(0), term(3)];
        let n = 30;
        let mut data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        data.extend((0..n).map(|i| (i % 2) as f64));
        let beta = [1.0, 0.5, -1.0];
        let mut vcov = [0.0; 9];
        for i in 0..3 {
            vcov[i * 3 + i] = 0.04;
        }
        let input = EffectsInputFFI {
            terms: terms.as_ptr(),
            n_terms: 2,
            data: data.as_ptr(),
            n_rows: n,
            beta: beta.as_ptr(),
            n_beta: 3,
            vcov: vcov.as_ptr(),
            intercept: true,
            df: f64::NAN,
            level: 0.95,
        };
        let mut err = AnofoxError::success();
        let mut rows = [EffectRowFFI::default(); 4];
        let mut len = 0usize;
        unsafe {
            assert!(anofox_effects_summary(
                &input,
                rows.as_mut_ptr(),
                0,
                &mut len,
                &mut err
            ));
            assert_eq!(len, 2);
            assert!(anofox_effects_summary(
                &input,
                rows.as_mut_ptr(),
                4,
                &mut len,
                &mut err
            ));
            assert_eq!(rows[1].predictor, 1);

            let (p, v) = ([0usize], [10.0]);
            let (p2, v2) = ([0usize], [4.0]);
            let a = EffectsSettingFFI {
                predictors: p.as_ptr(),
                values: v.as_ptr(),
                n: 1,
            };
            let b = EffectsSettingFFI {
                predictors: p2.as_ptr(),
                values: v2.as_ptr(),
                n: 1,
            };
            let mut c = ContrastEstimateFFI::default();
            assert!(anofox_effects_contrast(
                &input,
                &a,
                &b,
                std::ptr::null(),
                std::ptr::null(),
                &mut c,
                &mut err
            ));
            assert!((c.estimate - 3.0).abs() < 1e-12);

            let mut vals = [0.0; 200];
            let mut est = [ContrastEstimateFFI::default(); 200];
            let mut crit = 0.0;
            assert!(anofox_effects_predict_grid(
                &input,
                0,
                std::ptr::null(),
                0,
                false,
                vals.as_mut_ptr(),
                est.as_mut_ptr(),
                200,
                &mut len,
                &mut crit,
                &mut err
            ));
            assert_eq!(len, 200);
            assert!(crit.is_nan());

            let lp: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * i as f64).collect();
            let mut nom = std::mem::zeroed::<NomogramFFI>();
            assert!(anofox_effects_nomogram(
                &input,
                lp.as_ptr(),
                lp.len(),
                100.0,
                10,
                1,
                &mut nom,
                &mut err
            ));
            assert_eq!(nom.n_axes, 2);
            assert!(nom.n_fun > 0);
            anofox_free_nomogram(&mut nom);

            // wrong number of coefficients
            let bad = EffectsInputFFI { n_beta: 2, ..input };
            assert!(!anofox_effects_summary(
                &bad,
                rows.as_mut_ptr(),
                4,
                &mut len,
                &mut err
            ));
            assert_eq!(err.code, ErrorCode::InvalidInput);
        }
    }
}
