//! C FFI for the percentile bootstrap CI of the mean (#294). The structs are
//! mirrored in `src/include/anofox_stats_ffi.h`.

use super::*;
use anofox_stats_core::tests::resampling::{
    bootstrap_mean, BootstrapMethod, BootstrapOptions, BootstrapResult,
};

/// Result of `anofox_bootstrap_mean_ci`. Must match AnofoxBootstrapMeanResult.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BootstrapMeanResultFFI {
    /// Mean of the data
    pub estimate: f64,
    /// Bootstrap standard error (SD of the replicates)
    pub std_error: f64,
    /// Percentile interval
    pub conf_low: f64,
    pub conf_high: f64,
    pub conf_level: f64,
    /// Observations used (NaN dropped)
    pub n: usize,
    pub n_bootstrap: usize,
    /// Block length of the circular block bootstrap, 0 for IID
    pub block_length: usize,
}

/// Percentile bootstrap confidence interval of the mean of `data` (NaN entries
/// are dropped; for the block bootstrap the order of `data` is the series
/// order). `method`: 0 IID, 1 circular block bootstrap with `block_length >= 2`.
///
/// # Safety
/// - `data` must be a valid DataArray
/// - `out_result` and `out_error` must be valid pointers
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn anofox_bootstrap_mean_ci(
    data: DataArray,
    method: i32,
    n_bootstrap: usize,
    confidence_level: f64,
    block_length: usize,
    seed: u64,
    has_seed: bool,
    out_result: *mut BootstrapMeanResultFFI,
    out_error: *mut AnofoxError,
) -> bool {
    ffi_guard(out_error, false, || {
        reset_error(out_error);
        if !check_out(out_result, out_error) {
            return false;
        }
        let method = match method {
            0 => BootstrapMethod::Iid,
            1 => BootstrapMethod::CircularBlock { block_length },
            _ => {
                if !out_error.is_null() {
                    (*out_error).set(ErrorCode::InvalidInput, "invalid bootstrap method");
                }
                return false;
            }
        };
        let values: Vec<f64> = data.to_vec().into_iter().filter(|v| !v.is_nan()).collect();
        let result = method.upstream_block_length().and_then(|block_length| {
            let opts = BootstrapOptions {
                n_bootstrap,
                confidence_level,
                block_length,
                seed: if has_seed { Some(seed) } else { None },
            };
            bootstrap_mean(&values, &opts).map(|r| (r, block_length))
        });
        match result {
            Ok((r, block_length)) => {
                let BootstrapResult {
                    statistic,
                    se,
                    ci_lower,
                    ci_upper,
                    n_bootstrap,
                } = r;
                *out_result = BootstrapMeanResultFFI {
                    estimate: statistic,
                    std_error: se,
                    conf_low: ci_lower,
                    conf_high: ci_upper,
                    conf_level: confidence_level,
                    n: values.len(),
                    n_bootstrap,
                    block_length,
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

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn run(
        data: &[f64],
        method: i32,
        block_length: usize,
    ) -> (bool, BootstrapMeanResultFFI) {
        let arr = DataArray {
            data: data.as_ptr(),
            validity: std::ptr::null(),
            len: data.len(),
        };
        let mut out = std::mem::zeroed::<BootstrapMeanResultFFI>();
        let mut err = AnofoxError::success();
        let ok = anofox_bootstrap_mean_ci(
            arr,
            method,
            500,
            0.9,
            block_length,
            7,
            true,
            &mut out,
            &mut err,
        );
        (ok, out)
    }

    #[test]
    fn bootstrap_mean_ci_ffi() {
        let x: Vec<f64> = (1..=20).map(|v| v as f64).collect();
        unsafe {
            let (ok, r) = run(&x, 0, 0);
            assert!(ok);
            assert!((r.estimate - 10.5).abs() < 1e-12);
            assert!(r.conf_low < 10.5 && r.conf_high > 10.5);
            assert_eq!((r.n, r.n_bootstrap, r.block_length), (20, 500, 0));
            assert_eq!(r.conf_level, 0.9);

            let (ok, r) = run(&x, 1, 4);
            assert!(ok);
            assert_eq!(r.block_length, 4);
            assert!(r.std_error > 0.0);

            assert!(!run(&x, 1, 1).0);
            assert!(!run(&x, 7, 0).0);
            assert!(!run(&[1.0, f64::NAN], 0, 0).0);
        }
    }
}
