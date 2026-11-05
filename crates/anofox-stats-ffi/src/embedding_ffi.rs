//! C FFI for embedding drift and the multivariate two-sample tests (#302). The
//! structs are mirrored in `src/include/anofox_stats_ffi.h`.
//!
//! A sample is `n_rows` embeddings of `dim` doubles each, row-major.

use super::*;
use anofox_stats_core::tests::embedding::{
    embedding_drift, multivariate_two_sample, DriftTest, EmbeddingDistance, EmbeddingDrift,
    EmbeddingDriftOptions, EmbeddingMethod, Kernel, MultivariateTest, MultivariateTestResult,
};
use libc::c_char;
use std::ffi::CStr;

/// Options of `anofox_embedding_drift`; each method reads only its own.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingDriftOptionsFFI {
    /// evidently's drift test name of each component (`ratio`); NULL for
    /// `"wasserstein"`.
    pub component_test: *const c_char,
    /// Fixed threshold; NaN for evidently's default.
    pub threshold: f64,
    /// Bootstrap quantile; NaN for evidently's default.
    pub quantile: f64,
    /// Per-component threshold (`ratio`); NaN for the test's default.
    pub component_threshold: f64,
    pub n_bootstrap: usize,
    pub seed: u64,
    /// 0 distance, 1 ratio, 2 mmd.
    pub method: i32,
    /// 0 euclidean, 1 cosine, 2 cityblock, 3 chebyshev.
    pub metric: i32,
    pub bootstrap: bool,
    pub has_seed: bool,
}

/// Result of `anofox_embedding_drift`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingDriftResultFFI {
    pub score: f64,
    pub threshold: f64,
    pub n1: usize,
    pub n2: usize,
    pub dim: usize,
    pub drifted: bool,
}

impl From<&EmbeddingDrift> for EmbeddingDriftResultFFI {
    fn from(d: &EmbeddingDrift) -> Self {
        Self {
            score: d.score,
            threshold: d.threshold,
            n1: d.n1,
            n2: d.n2,
            dim: d.dim,
            drifted: d.drifted,
        }
    }
}

/// Options of `anofox_multivariate_two_sample`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MultivariateTestOptionsFFI {
    /// Bandwidth of the gaussian / laplacian kernel.
    pub bandwidth: f64,
    /// Polynomial kernel `(scale x.y + offset)^degree`.
    pub scale: f64,
    pub offset: f64,
    pub n_permutations: usize,
    pub seed: u64,
    pub degree: u32,
    /// MMD kernel: 0 gaussian, 1 laplacian, 2 linear, 3 polynomial.
    pub kernel: i32,
    pub has_seed: bool,
}

/// Result of `anofox_multivariate_two_sample`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MultivariateTestResultFFI {
    pub statistic: f64,
    pub p_value: f64,
    pub n1: usize,
    pub n2: usize,
    pub dim: usize,
    pub n_permutations: usize,
    /// Unified test fields (#159); keep last.
    pub unified: TestSummaryFFI,
}

impl From<&MultivariateTestResult> for MultivariateTestResultFFI {
    fn from(r: &MultivariateTestResult) -> Self {
        Self {
            statistic: r.statistic,
            p_value: r.p_value,
            n1: r.n1,
            n2: r.n2,
            dim: r.dim,
            n_permutations: r.n_permutations,
            unified: TestSummaryFFI::from(&r.unified),
        }
    }
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

unsafe fn embedding_options(o: &EmbeddingDriftOptionsFFI) -> Result<EmbeddingDriftOptions, String> {
    let nan_none = |v: f64| if v.is_nan() { None } else { Some(v) };
    let component_test = if o.component_test.is_null() {
        DriftTest::Wasserstein
    } else {
        let name = CStr::from_ptr(o.component_test).to_string_lossy();
        DriftTest::from_name(&name).ok_or_else(|| {
            let names: Vec<&str> = DriftTest::ALL.iter().map(|t| t.name()).collect();
            format!(
                "unknown component_test '{}' (expected one of {})",
                name,
                names.join(", ")
            )
        })?
    };
    Ok(EmbeddingDriftOptions {
        method: match o.method {
            0 => EmbeddingMethod::Distance,
            1 => EmbeddingMethod::Ratio,
            2 => EmbeddingMethod::Mmd,
            _ => return Err("invalid embedding drift method".into()),
        },
        metric: match o.metric {
            0 => EmbeddingDistance::Euclidean,
            1 => EmbeddingDistance::Cosine,
            2 => EmbeddingDistance::Cityblock,
            3 => EmbeddingDistance::Chebyshev,
            _ => return Err("invalid embedding distance metric".into()),
        },
        threshold: nan_none(o.threshold),
        bootstrap: o.bootstrap,
        n_bootstrap: o.n_bootstrap,
        quantile: nan_none(o.quantile),
        component_test,
        component_threshold: nan_none(o.component_threshold),
        seed: if o.has_seed { Some(o.seed) } else { None },
    })
}

/// evidently's embedding drift of `current` against `reference` (#302).
/// The options are parsed before the samples are looked at, so a call with
/// empty samples validates them (INVALID_INPUT for a bad name, otherwise an
/// insufficient-data error).
///
/// # Safety
/// - `reference` / `current` must point to `n_reference * dim` /
///   `n_current * dim` doubles (or be NULL with zero rows)
/// - `options.component_test` must be a NUL-terminated string or NULL
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_embedding_drift(
    reference: *const f64,
    n_reference: usize,
    current: *const f64,
    n_current: usize,
    dim: usize,
    options: EmbeddingDriftOptionsFFI,
    out_result: *mut EmbeddingDriftResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let opts = match embedding_options(&options) {
            Ok(o) => o,
            Err(msg) => {
                invalid_input(out_error, &msg);
                return false;
            }
        };
        let (r, c) = (
            rows(reference, n_reference, dim),
            rows(current, n_current, dim),
        );
        match embedding_drift(&r, &c, &opts) {
            Ok(d) => {
                *out_result = EmbeddingDriftResultFFI::from(&d);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}

/// Multivariate two-sample permutation test of `x` against `y` (#302).
/// `kind`: 0 energy distance, 1 MMD (with `options.kernel`).
///
/// # Safety
/// - `x` / `y` must point to `n_x * dim` / `n_y * dim` doubles (or be NULL
///   with zero rows)
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
pub unsafe extern "C" fn anofox_multivariate_two_sample(
    kind: i32,
    x: *const f64,
    n_x: usize,
    y: *const f64,
    n_y: usize,
    dim: usize,
    options: MultivariateTestOptionsFFI,
    out_result: *mut MultivariateTestResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let test = match kind {
            0 => MultivariateTest::Energy,
            1 => MultivariateTest::Mmd(match options.kernel {
                0 => Kernel::Gaussian {
                    bandwidth: options.bandwidth,
                },
                1 => Kernel::Laplacian {
                    bandwidth: options.bandwidth,
                },
                2 => Kernel::Linear,
                3 => Kernel::Polynomial {
                    degree: options.degree,
                    scale: options.scale,
                    offset: options.offset,
                },
                _ => {
                    invalid_input(out_error, "invalid MMD kernel");
                    return false;
                }
            }),
            _ => {
                invalid_input(out_error, "invalid multivariate test kind");
                return false;
            }
        };
        let seed = if options.has_seed {
            Some(options.seed)
        } else {
            None
        };
        let (a, b) = (rows(x, n_x, dim), rows(y, n_y, dim));
        match multivariate_two_sample(test, &a, &b, options.n_permutations, seed) {
            Ok(r) => {
                *out_result = MultivariateTestResultFFI::from(&r);
                true
            }
            Err(e) => {
                set_error(out_error, &e);
                false
            }
        }
    })
}
