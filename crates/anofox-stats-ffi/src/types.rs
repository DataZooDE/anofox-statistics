//! C-compatible types for FFI boundary

use libc::c_char;

/// Error codes for FFI boundary
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Success = 0,
    InvalidInput = 1,
    SingularMatrix = 2,
    ConvergenceFailure = 3,
    InvalidAlpha = 4,
    InvalidL1Ratio = 5,
    InsufficientData = 6,
    AllocationFailure = 7,
    SerializationError = 8,
    DimensionMismatch = 9,
    NoValidData = 10,
    InternalError = 99,
}

/// Error information for FFI
#[repr(C)]
pub struct AnofoxError {
    pub code: ErrorCode,
    pub message: [c_char; 256],
}

impl AnofoxError {
    pub fn success() -> Self {
        Self {
            code: ErrorCode::Success,
            message: [0; 256],
        }
    }

    pub fn set(&mut self, code: ErrorCode, msg: &str) {
        self.code = code;
        let bytes = msg.as_bytes();
        let len = bytes.len().min(255);
        for (i, &b) in bytes[..len].iter().enumerate() {
            self.message[i] = b as c_char;
        }
        self.message[len] = 0;
    }
}

/// Array of f64 values with validity mask for NULL handling
#[repr(C)]
pub struct DataArray {
    /// Pointer to data values
    pub data: *const f64,
    /// Validity bitmask: bit i is 1 if data[i] is valid, 0 if NULL
    /// Can be NULL if all values are valid
    pub validity: *const u8,
    /// Number of elements
    pub len: usize,
}

impl DataArray {
    /// Check if index i is valid (not NULL)
    ///
    /// # Safety
    /// Caller must ensure index is within bounds
    pub unsafe fn is_valid(&self, i: usize) -> bool {
        if self.validity.is_null() {
            return true;
        }
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        ((*self.validity.add(byte_idx)) >> bit_idx) & 1 == 1
    }

    /// Convert to Vec<f64>, replacing NULL with NaN
    ///
    /// # Safety
    /// Caller must ensure pointers are valid and len is correct
    pub unsafe fn to_vec(&self) -> Vec<f64> {
        if self.len == 0 {
            return Vec::new();
        }
        // Fast path: no validity mask means every value is valid (the common case
        // for dense/non-nullable columns). Bulk-copy the slice instead of the
        // per-element validity branch + push. This returns the same owned
        // Vec<f64> — no borrow crosses the FFI boundary, so there is no aliasing
        // or lifetime risk; only the NULL→NaN branch (unreachable here) is skipped.
        if self.validity.is_null() {
            return std::slice::from_raw_parts(self.data, self.len).to_vec();
        }
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            if self.is_valid(i) {
                result.push(*self.data.add(i));
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }
}

/// An owning heap buffer of `T`, allocated with `libc::malloc`.
///
/// RAII wrapper for FFI result arrays: it frees its buffer with `libc::free` on
/// `Drop`, or relinquishes ownership via [`FfiVec::into_raw`] to a caller that
/// will free it with C `free()`.
///
/// # Safety / ABI invariant
/// The pointer is allocated with `libc::malloc` and MUST be released with C
/// `free()` — which is exactly what the `anofox_free_*` FFI functions call. This
/// is why the buffer is NOT backed by `Box`, `Vec`, or Rust's global allocator:
/// on musl targets (WASM, some CI) the Rust global allocator and libc's malloc
/// can differ, so freeing a `Box`/`Vec` pointer with C `free()` is undefined
/// behavior. Changing the allocator here is a published-ABI break — it would
/// require changing every C++ `free` site. See PERF-04 / phase-04 CONTEXT.
pub struct FfiVec<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> FfiVec<T> {
    /// Allocate an uninitialized buffer of `len` elements via `libc::malloc`.
    ///
    /// `len == 0` yields a null pointer and no allocation (mirrors the previous
    /// hand-written behavior, where a zero-length request produced a null ptr).
    /// Returns `None` on allocation failure (OOM) for `len > 0`.
    pub fn alloc(len: usize) -> Option<Self> {
        if len == 0 {
            return Some(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        // Safety: size computed from a checked element count; null is handled below.
        let ptr = unsafe { libc::malloc(len * std::mem::size_of::<T>()) as *mut T };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, len })
        }
    }

    /// Copy `src` element-for-element into the buffer.
    ///
    /// # Safety
    /// `src.len()` must equal the allocated `len`. No-op for a zero-length buffer.
    pub unsafe fn copy_from_slice(&self, src: &[T])
    where
        T: Copy,
    {
        if self.len == 0 {
            return;
        }
        // WR-01: assert in release too. A length mismatch here would read past the
        // source slice or the destination allocation (UB) — at an FFI boundary a
        // hard panic is strictly safer than silent memory corruption.
        assert_eq!(
            src.len(),
            self.len,
            "FfiVec::copy_from_slice: length mismatch ({} vs {})",
            src.len(),
            self.len
        );
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, self.len);
    }

    /// Consume the wrapper, returning the raw pointer and suppressing `Drop`.
    ///
    /// The returned pointer is transferred to the caller, which MUST release it
    /// with C `free()` (i.e. via the `anofox_free_*` functions). For a
    /// zero-length buffer this returns a null pointer.
    pub fn into_raw(self) -> *mut T {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl<T> Drop for FfiVec<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Safety: `ptr` was allocated by `libc::malloc` in `alloc` and is only
            // freed once (Drop is suppressed by `into_raw`).
            unsafe { libc::free(self.ptr as *mut libc::c_void) };
        }
    }
}

#[cfg(test)]
mod ffi_vec_tests {
    use super::FfiVec;

    /// Proves the `into_raw()` pointer is freeable by `libc::free` — i.e. the
    /// buffer really is libc-malloc-backed. This test would be undefined behavior
    /// (and flagged by ASan/valgrind) if `FfiVec` ever switched to `Box`/`Vec`.
    #[test]
    fn ffi_vec_ptr_is_freeable_by_libc() {
        let v = FfiVec::<f64>::alloc(4).expect("alloc(4) failed");
        let src = [1.0f64, 2.0, 3.0, 4.0];
        unsafe { v.copy_from_slice(&src) };
        let raw = v.into_raw();
        assert!(!raw.is_null());
        unsafe {
            let back = std::slice::from_raw_parts(raw, 4);
            assert_eq!(back, &src, "values must round-trip through the raw buffer");
            libc::free(raw as *mut libc::c_void);
        }
    }

    /// A zero-length allocation is a null pointer with no allocation.
    #[test]
    fn ffi_vec_alloc_zero_is_null() {
        let v = FfiVec::<f64>::alloc(0).expect("alloc(0) failed");
        assert!(v.into_raw().is_null());
    }

    /// Dropping a non-into_raw'd FfiVec frees via libc::free without leaking or
    /// double-freeing (observable under ASan/valgrind; here we just exercise it).
    #[test]
    fn ffi_vec_drop_frees() {
        let v = FfiVec::<f64>::alloc(8).expect("alloc(8) failed");
        unsafe { v.copy_from_slice(&[0.0f64; 8]) };
        drop(v);
    }
}

/// Core fit result (always returned)
#[repr(C)]
pub struct FitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Residual standard error
    pub residual_std_error: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
}

impl Default for FitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            r_squared: f64::NAN,
            adj_r_squared: f64::NAN,
            residual_std_error: f64::NAN,
            n_observations: 0,
            n_features: 0,
        }
    }
}

/// Inference results (optional)
#[repr(C)]
pub struct FitResultInference {
    /// Standard errors of coefficients
    pub std_errors: *mut f64,
    /// t-values
    pub t_values: *mut f64,
    /// p-values
    pub p_values: *mut f64,
    /// Lower confidence interval bounds
    pub ci_lower: *mut f64,
    /// Upper confidence interval bounds
    pub ci_upper: *mut f64,
    /// Number of elements in each array
    pub len: usize,
    /// Confidence level used
    pub confidence_level: f64,
    /// F-statistic (NaN if not computed)
    pub f_statistic: f64,
    /// F p-value (NaN if not computed)
    pub f_pvalue: f64,
}

impl Default for FitResultInference {
    fn default() -> Self {
        Self {
            std_errors: std::ptr::null_mut(),
            t_values: std::ptr::null_mut(),
            p_values: std::ptr::null_mut(),
            ci_lower: std::ptr::null_mut(),
            ci_upper: std::ptr::null_mut(),
            len: 0,
            confidence_level: 0.95,
            f_statistic: f64::NAN,
            f_pvalue: f64::NAN,
        }
    }
}

/// Decomposition method for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverTypeFFI {
    /// QR decomposition with column pivoting (default)
    Qr = 0,
    /// SVD decomposition (most robust)
    #[default]
    Svd = 1,
    /// Cholesky decomposition (fastest)
    Cholesky = 2,
}

/// Lambda scaling convention for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LambdaScalingFFI {
    /// Use lambda as-is (default)
    #[default]
    Raw = 0,
    /// Scale to match R's glmnet convention
    Glmnet = 1,
}

/// Heteroscedasticity-consistent SE type for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HcTypeFFI {
    /// No HC inference (use classical)
    #[default]
    None = 0,
    /// HC0: White's original
    HC0 = 1,
    /// HC1: With df correction (default)
    HC1 = 2,
    /// HC2: Leverage-based
    HC2 = 3,
    /// HC3: Jackknife-like
    HC3 = 4,
}

/// OLS options for FFI
#[repr(C)]
pub struct OlsOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// Decomposition method (0=QR, 1=SVD, 2=Cholesky)
    pub solver: SolverTypeFFI,
    /// Heteroscedasticity-consistent SE type
    pub hc_type: HcTypeFFI,
}

impl Default for OlsOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            solver: SolverTypeFFI::Qr,
            hc_type: HcTypeFFI::None,
        }
    }
}

/// Huber M-estimator robust regression options for FFI
#[repr(C)]
pub struct HuberOptionsFFI {
    /// Huber threshold parameter (must be > 1.0). Default 1.35.
    pub epsilon: f64,
    /// L2 regularization (must be >= 0). Default 0.0001.
    pub alpha: f64,
    /// Whether to fit intercept.
    pub fit_intercept: bool,
    /// Whether to compute inference statistics.
    pub compute_inference: bool,
    /// Confidence level for CIs.
    pub confidence_level: f64,
    /// Maximum IRLS iterations.
    pub max_iterations: u32,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for HuberOptionsFFI {
    fn default() -> Self {
        Self {
            epsilon: 1.35,
            alpha: 0.0001,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            max_iterations: 100,
            tolerance: 1e-5,
        }
    }
}

/// Binary Logistic regression options for FFI (logit link; classifier API).
#[repr(C)]
pub struct LogisticOptionsFFI {
    pub fit_intercept: bool,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// L2 (ridge) penalty strength. 0.0 = unpenalised.
    pub lambda: f64,
    /// Classification threshold on the predicted probability.
    pub threshold: f64,
    pub max_iterations: u32,
    pub tolerance: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for LogisticOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            threshold: 0.5,
            max_iterations: 100,
            tolerance: 1e-8,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// Logistic-specific diagnostics returned alongside GlmFitResultCore.
/// All fields are scalars — no allocations to free.
#[repr(C)]
pub struct LogisticFitExtras {
    /// Classification accuracy on the training data with the configured threshold.
    pub accuracy: f64,
    /// Classification threshold actually used (echoed from options).
    pub threshold: f64,
}

impl Default for LogisticFitExtras {
    fn default() -> Self {
        Self {
            accuracy: f64::NAN,
            threshold: 0.5,
        }
    }
}

/// Huber-specific diagnostics returned alongside FitResultCore / FitResultInference.
/// Memory rules: `outliers` is allocated by `anofox_huber_fit` and must be freed
/// via `anofox_free_huber_extras` (boolean array exposed as u8: 0 = inlier, 1 = outlier).
#[repr(C)]
pub struct HuberFitExtras {
    /// MAD-based scale estimate (sigma).
    pub scale: f64,
    /// Echoed epsilon used for the fit.
    pub epsilon: f64,
    /// Per-observation outlier mask (1 = |r_i| > epsilon * scale).
    pub outliers: *mut u8,
    /// Number of valid (non-NaN) observations the fit was computed on
    /// — equals `outliers` array length.
    pub outliers_len: usize,
    /// Number of observations flagged as outliers.
    pub n_outliers: usize,
}

impl Default for HuberFitExtras {
    fn default() -> Self {
        Self {
            scale: f64::NAN,
            epsilon: f64::NAN,
            outliers: std::ptr::null_mut(),
            outliers_len: 0,
            n_outliers: 0,
        }
    }
}

/// RANSAC robust regression options for FFI.
///
/// `min_samples_set` / `min_samples_value` together encode an
/// `Option<usize>`: when `min_samples_set` is false the upstream solver
/// picks the default (`n_features + 1` with intercept). Same encoding for
/// `residual_threshold_*` and `stop_n_inliers_*`.
#[repr(C)]
pub struct RansacOptionsFFI {
    /// Whether to fit intercept.
    pub fit_intercept: bool,
    /// Whether to compute inference statistics on the inlier-only final fit.
    pub compute_inference: bool,
    /// Confidence level for any inference intervals.
    pub confidence_level: f64,
    /// Maximum number of RANSAC trials.
    pub max_trials: u32,
    /// Fischler-Bolles stop probability (must be in [0, 1]).
    pub stop_probability: f64,
    /// Random seed for the trial subsampler.
    pub random_state: u64,

    /// `Option<usize>` for `min_samples`.
    pub min_samples_set: bool,
    pub min_samples_value: usize,

    /// `Option<f64>` for `residual_threshold` (must be finite and > 0 when set).
    pub residual_threshold_set: bool,
    pub residual_threshold_value: f64,

    /// `Option<usize>` for `stop_n_inliers`.
    pub stop_n_inliers_set: bool,
    pub stop_n_inliers_value: usize,
}

impl Default for RansacOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            max_trials: 100,
            stop_probability: 0.99,
            random_state: 0,
            min_samples_set: false,
            min_samples_value: 0,
            residual_threshold_set: false,
            residual_threshold_value: 0.0,
            stop_n_inliers_set: false,
            stop_n_inliers_value: 0,
        }
    }
}

/// RANSAC-specific diagnostics returned alongside FitResultCore /
/// FitResultInference.
///
/// Memory rules: `inliers` is allocated by `anofox_ransac_fit` (1 byte per
/// observation, 0 = outlier, 1 = inlier) and must be freed via
/// `anofox_free_ransac_extras`.
#[repr(C)]
pub struct RansacFitExtras {
    /// Residual threshold actually used (either user-supplied or MAD(y)).
    pub residual_threshold: f64,
    /// Per-observation inlier mask (1 = inlier).
    pub inliers: *mut u8,
    /// Length of the inliers array.
    pub inliers_len: usize,
    /// Number of observations classified as inliers in the final consensus.
    pub n_inliers: usize,
    /// Actual number of RANSAC trials run before early termination.
    pub n_trials: usize,
}

impl Default for RansacFitExtras {
    fn default() -> Self {
        Self {
            residual_threshold: f64::NAN,
            inliers: std::ptr::null_mut(),
            inliers_len: 0,
            n_inliers: 0,
            n_trials: 0,
        }
    }
}

/// Theil-Sen robust regression options for FFI.
///
/// `n_subsamples_*` together encode Option<usize>: when `n_subsamples_set` is
/// false the upstream solver picks `n_features + 1` (sklearn default).
#[repr(C)]
pub struct TheilSenOptionsFFI {
    pub fit_intercept: bool,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Cap on the number of subsamples examined (sklearn default 10_000).
    pub max_subpopulation: u32,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub random_state: u64,

    pub n_subsamples_set: bool,
    pub n_subsamples_value: usize,
}

impl Default for TheilSenOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            max_subpopulation: 10_000,
            max_iterations: 300,
            tolerance: 1e-3,
            random_state: 0,
            n_subsamples_set: false,
            n_subsamples_value: 0,
        }
    }
}

/// Ridge regression options for FFI
#[repr(C)]
pub struct RidgeOptionsFFI {
    /// L2 regularization parameter (alpha/lambda)
    pub alpha: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// Decomposition method (0=QR, 1=SVD, 2=Cholesky)
    pub solver: SolverTypeFFI,
    /// Lambda scaling convention
    pub lambda_scaling: LambdaScalingFFI,
}

impl Default for RidgeOptionsFFI {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            solver: SolverTypeFFI::Qr,
            lambda_scaling: LambdaScalingFFI::Raw,
        }
    }
}

/// Elastic Net regression options for FFI
#[repr(C)]
pub struct ElasticNetOptionsFFI {
    /// Regularization strength (must be >= 0)
    pub alpha: f64,
    /// L1 ratio: 0 = Ridge, 1 = Lasso (must be in [0, 1])
    pub l1_ratio: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Maximum iterations for coordinate descent
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Lambda scaling convention
    pub lambda_scaling: LambdaScalingFFI,
}

/// Options for Least Angle Regression (LARS / LassoLars) for FFI
#[repr(C)]
pub struct LarsOptionsFFI {
    /// false = plain LARS, true = LassoLars (exact Lasso path)
    pub method_lasso: bool,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// LassoLars early-stop alpha (0.0 = full path)
    pub alpha: f64,
    /// Cap on non-zero coefficients (<= 0 = unlimited)
    pub n_nonzero_coefs: i64,
    /// Standardize features before fitting
    pub standardize: bool,
}

impl Default for ElasticNetOptionsFFI {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
            lambda_scaling: LambdaScalingFFI::Raw,
        }
    }
}

/// WLS (Weighted Least Squares) options for FFI
#[repr(C)]
pub struct WlsOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// Decomposition method (0=QR, 1=SVD, 2=Cholesky)
    pub solver: SolverTypeFFI,
    /// Heteroscedasticity-consistent SE type
    pub hc_type: HcTypeFFI,
}

impl Default for WlsOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
            solver: SolverTypeFFI::Qr,
            hc_type: HcTypeFFI::None,
        }
    }
}

// =============================================================================
// GLM (Generalized Linear Models) FFI Types
// =============================================================================

/// GLM family codes for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmFamilyFFI {
    Poisson = 0,
    Binomial = 1,
    NegBinomial = 2,
    Tweedie = 3,
}

/// Poisson link function codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonLinkFFI {
    Log = 0,
    Identity = 1,
    Sqrt = 2,
}

/// Binomial link function codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinomialLinkFFI {
    Logit = 0,
    Probit = 1,
    Cloglog = 2,
}

/// Prior family code, mirroring `anofox_stats_core::types::PriorKind`.
// repr(C), not repr(u8): a C enum is int-sized, so a 1-byte Rust enum shifts
// every field that follows it. Harmless while such a field sits last in a
// struct, fatal when it sits first (AftOptionsFFI::dist).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorKindFFI {
    /// No prior on this coefficient.
    Flat = 0,
    /// Gaussian prior: `scale` is the prior standard deviation.
    Normal = 1,
    /// Laplace prior: `scale` is the Laplace scale `b`.
    Laplace = 2,
}

/// One coefficient's prior. Flat POD; the C++ side resolves feature names to
/// positions before crossing this boundary, so names never appear here.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PriorSpecFFI {
    pub kind: PriorKindFFI,
    pub loc: f64,
    pub scale: f64,
}

impl Default for PriorSpecFFI {
    fn default() -> Self {
        Self {
            kind: PriorKindFFI::Flat,
            loc: 0.0,
            scale: f64::INFINITY,
        }
    }
}

/// Covariance type code, mirroring `anofox_stats_core::types::VcovType`.
// repr(C), not repr(u8): a C enum is int-sized, so a 1-byte Rust enum shifts
// every field that follows it. Harmless while such a field sits last in a
// struct, fatal when it sits first (AftOptionsFFI::dist).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VcovTypeFFI {
    /// `(X'WX + P)^-1`, the curvature of the log posterior at the mode.
    #[default]
    Laplace = 0,
    /// `(X'WX + P)^-1 X'WX (X'WX + P)^-1`.
    Sandwich = 1,
    /// `(X'WX)^-1`, ignoring the penalty.
    Naive = 2,
}

/// Prior and covariance settings carried by every GLM options struct.
///
/// These three fields are appended verbatim to each `*OptionsFFI` rather than
/// nested, keeping the C ABI structs flat POD as the rest of the header is.
#[derive(Debug, Clone, Copy)]
pub struct GlmPriorFields {
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    pub vcov: VcovTypeFFI,
}

/// Rebuild the core-crate prior vector from a raw FFI array.
///
/// # Safety
/// `priors` must be null or point to `priors_len` initialised `PriorSpecFFI`.
pub unsafe fn priors_from_ffi(
    priors: *const PriorSpecFFI,
    priors_len: usize,
) -> Vec<anofox_stats_core::types::PriorSpec> {
    use anofox_stats_core::types::{PriorKind, PriorSpec};
    if priors.is_null() || priors_len == 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(priors, priors_len)
        .iter()
        .map(|p| PriorSpec {
            kind: match p.kind {
                PriorKindFFI::Flat => PriorKind::Flat,
                PriorKindFFI::Normal => PriorKind::Normal,
                PriorKindFFI::Laplace => PriorKind::Laplace,
            },
            loc: p.loc,
            scale: p.scale,
        })
        .collect()
}

/// Map the FFI covariance code onto the core enum.
pub fn vcov_from_ffi(v: VcovTypeFFI) -> anofox_stats_core::types::VcovType {
    use anofox_stats_core::types::VcovType;
    match v {
        VcovTypeFFI::Laplace => VcovType::Laplace,
        VcovTypeFFI::Sandwich => VcovType::Sandwich,
        VcovTypeFFI::Naive => VcovType::Naive,
    }
}

/// GLM options for Poisson regression
#[repr(C)]
pub struct PoissonOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Link function
    pub link: PoissonLinkFFI,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// L2 regularization parameter (0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for PoissonOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: PoissonLinkFFI::Log,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// GLM options for Binomial regression
#[repr(C)]
pub struct BinomialOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Link function
    pub link: BinomialLinkFFI,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// L2 regularization parameter (0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for BinomialOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: BinomialLinkFFI::Logit,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// GLM options for Negative Binomial regression
#[repr(C)]
pub struct NegBinomialOptionsFFI {
    /// Dispersion (theta). NaN means "estimate from the data".
    pub alpha: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// L2 regularization parameter (0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for NegBinomialOptionsFFI {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// GLM options for Tweedie regression
#[repr(C)]
pub struct TweedieOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma)
    pub power: f64,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
    /// L2 regularization parameter (0 = no regularization)
    pub lambda: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for TweedieOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            power: 1.5,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// Gamma GLM options for FFI (var_power = 2.0 fixed; log link).
#[repr(C)]
pub struct GammaOptionsFFI {
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    pub lambda: f64,
    /// Explicit per-coefficient priors, positionally aligned with the design
    /// (intercept first when one is fitted). Null or zero-length means none.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    /// How to compute the coefficient covariance.
    pub vcov: VcovTypeFFI,
    /// 1-based index into `x` of an offset column (0 = none). The column is added
    /// to the linear predictor with coefficient fixed at 1 and dropped from the
    /// design. Used as-is; take logs upstream if the link requires it.
    pub offset_column: usize,
}

impl Default for GammaOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            lambda: 0.0,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
            offset_column: 0,
        }
    }
}

/// GLM fit result (different from standard regression - uses deviance)
#[repr(C)]
pub struct GlmFitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// Model deviance
    pub deviance: f64,
    /// Null deviance
    pub null_deviance: f64,
    /// Pseudo R-squared (1 - deviance/null_deviance)
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// Dispersion parameter (if applicable)
    pub dispersion: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of iterations to converge
    pub iterations: u32,
    /// Whether the IRLS solver reached the convergence tolerance. Appended last to
    /// preserve the ABI; every construction site must set it explicitly.
    pub converged: bool,
}

impl Default for GlmFitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            deviance: f64::NAN,
            null_deviance: f64::NAN,
            pseudo_r_squared: f64::NAN,
            aic: f64::NAN,
            dispersion: f64::NAN,
            n_observations: 0,
            n_features: 0,
            iterations: 0,
            converged: false,
        }
    }
}

// =============================================================================
// ALM (Augmented Linear Models) FFI Types
// =============================================================================

/// ALM distribution codes for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlmDistributionFFI {
    Normal = 0,
    Laplace = 1,
    StudentT = 2,
    Logistic = 3,
    AsymmetricLaplace = 4,
    GeneralisedNormal = 5,
    S = 6,
    LogNormal = 7,
    LogLaplace = 8,
    LogS = 9,
    LogGeneralisedNormal = 10,
    FoldedNormal = 11,
    RectifiedNormal = 12,
    BoxCoxNormal = 13,
    Gamma = 14,
    InverseGaussian = 15,
    Exponential = 16,
    Beta = 17,
    LogitNormal = 18,
    Poisson = 19,
    NegativeBinomial = 20,
    Binomial = 21,
    Geometric = 22,
    CumulativeLogistic = 23,
    CumulativeNormal = 24,
}

/// ALM loss function codes for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlmLossFFI {
    Likelihood = 0,
    MSE = 1,
    MAE = 2,
    HAM = 3,
    ROLE = 4,
}

/// ALM options for FFI
#[repr(C)]
pub struct AlmOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Distribution family
    pub distribution: AlmDistributionFFI,
    /// Loss function
    pub loss: AlmLossFFI,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Quantile for AsymmetricLaplace (0-1)
    pub quantile: f64,
    /// ROLE trim fraction
    pub role_trim: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
}

impl Default for AlmOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            distribution: AlmDistributionFFI::Normal,
            loss: AlmLossFFI::Likelihood,
            max_iterations: 100,
            tolerance: 1e-8,
            quantile: 0.5,
            role_trim: 0.05,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// ALM fit result
#[repr(C)]
pub struct AlmFitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Scale parameter
    pub scale: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of iterations to converge
    pub iterations: u32,
}

impl Default for AlmFitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            log_likelihood: f64::NAN,
            aic: f64::NAN,
            bic: f64::NAN,
            scale: f64::NAN,
            n_observations: 0,
            n_features: 0,
            iterations: 0,
        }
    }
}

// =============================================================================
// BLS (Bounded Least Squares) FFI Types
// =============================================================================

/// BLS options for FFI
#[repr(C)]
pub struct BlsOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Pointer to lower bounds (NULL = no lower bounds, single value = apply to all)
    pub lower_bounds: *const f64,
    /// Number of lower bounds (0 = no bounds, 1 = single value for all)
    pub lower_bounds_len: usize,
    /// Pointer to upper bounds (NULL = no upper bounds, single value = apply to all)
    pub upper_bounds: *const f64,
    /// Number of upper bounds (0 = no bounds, 1 = single value for all)
    pub upper_bounds_len: usize,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for BlsOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: false,
            lower_bounds: std::ptr::null(),
            lower_bounds_len: 0,
            upper_bounds: std::ptr::null(),
            upper_bounds_len: 0,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// BLS fit result
#[repr(C)]
pub struct BlsFitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// Sum of squared residuals
    pub ssr: f64,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of active constraints
    pub n_active_constraints: usize,
    /// Pointer to at_lower_bound flags
    pub at_lower_bound: *mut bool,
    /// Pointer to at_upper_bound flags
    pub at_upper_bound: *mut bool,
}

impl Default for BlsFitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            ssr: f64::NAN,
            r_squared: f64::NAN,
            n_observations: 0,
            n_features: 0,
            n_active_constraints: 0,
            at_lower_bound: std::ptr::null_mut(),
            at_upper_bound: std::ptr::null_mut(),
        }
    }
}

// =============================================================================
// AID (Automatic Identification of Demand) FFI Types
// =============================================================================

/// Outlier detection method codes for AID
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierMethodFFI {
    ZScore = 0,
    Iqr = 1,
}

/// AID options for FFI
#[repr(C)]
pub struct AidOptionsFFI {
    /// Zero proportion threshold for intermittent classification (default: 0.3)
    pub intermittent_threshold: f64,
    /// Outlier detection method
    pub outlier_method: OutlierMethodFFI,
}

impl Default for AidOptionsFFI {
    fn default() -> Self {
        Self {
            intermittent_threshold: 0.3,
            outlier_method: OutlierMethodFFI::ZScore,
        }
    }
}

/// AID classification result
#[repr(C)]
pub struct AidResultFFI {
    /// Demand type string pointer ("regular" or "intermittent")
    pub demand_type: *mut c_char,
    /// Whether demand is intermittent
    pub is_intermittent: bool,
    /// Best-fit distribution name pointer
    pub distribution: *mut c_char,
    /// Mean of values
    pub mean: f64,
    /// Variance of values
    pub variance: f64,
    /// Proportion of zero values
    pub zero_proportion: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Whether stockouts were detected
    pub has_stockouts: bool,
    /// Whether new product pattern was detected
    pub is_new_product: bool,
    /// Whether obsolete product pattern was detected
    pub is_obsolete_product: bool,
    /// Number of stockout observations
    pub stockout_count: usize,
    /// Number of new product observations
    pub new_product_count: usize,
    /// Number of obsolete product observations
    pub obsolete_product_count: usize,
    /// Number of high outlier observations
    pub high_outlier_count: usize,
    /// Number of low outlier observations
    pub low_outlier_count: usize,
}

impl Default for AidResultFFI {
    fn default() -> Self {
        Self {
            demand_type: std::ptr::null_mut(),
            is_intermittent: false,
            distribution: std::ptr::null_mut(),
            mean: f64::NAN,
            variance: f64::NAN,
            zero_proportion: f64::NAN,
            n_observations: 0,
            has_stockouts: false,
            is_new_product: false,
            is_obsolete_product: false,
            stockout_count: 0,
            new_product_count: 0,
            obsolete_product_count: 0,
            high_outlier_count: 0,
            low_outlier_count: 0,
        }
    }
}

/// Per-observation anomaly flags for AID
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AidAnomalyFlagsFFI {
    /// Unexpected zero in positive demand (stockout)
    pub stockout: bool,
    /// Leading zeros pattern (new product)
    pub new_product: bool,
    /// Trailing zeros pattern (obsolete product)
    pub obsolete_product: bool,
    /// Unusually high value
    pub high_outlier: bool,
    /// Unusually low value
    pub low_outlier: bool,
}

/// AID anomaly result (array of per-observation flags)
#[repr(C)]
pub struct AidAnomalyResultFFI {
    /// Pointer to array of anomaly flags
    pub flags: *mut AidAnomalyFlagsFFI,
    /// Number of observations
    pub len: usize,
}

impl Default for AidAnomalyResultFFI {
    fn default() -> Self {
        Self {
            flags: std::ptr::null_mut(),
            len: 0,
        }
    }
}

// =============================================================================
// Statistical Hypothesis Testing FFI Types
// =============================================================================

/// Alternative hypothesis codes for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlternativeFFI {
    TwoSided = 0,
    Less = 1,
    Greater = 2,
}

impl From<AlternativeFFI> for anofox_stats_core::tests::Alternative {
    fn from(alt: AlternativeFFI) -> Self {
        match alt {
            AlternativeFFI::TwoSided => anofox_stats_core::tests::Alternative::TwoSided,
            AlternativeFFI::Less => anofox_stats_core::tests::Alternative::Less,
            AlternativeFFI::Greater => anofox_stats_core::tests::Alternative::Greater,
        }
    }
}

impl From<anofox_stats_core::tests::Alternative> for AlternativeFFI {
    fn from(alt: anofox_stats_core::tests::Alternative) -> Self {
        match alt {
            anofox_stats_core::tests::Alternative::TwoSided => AlternativeFFI::TwoSided,
            anofox_stats_core::tests::Alternative::Less => AlternativeFFI::Less,
            anofox_stats_core::tests::Alternative::Greater => AlternativeFFI::Greater,
        }
    }
}

/// Generic test result for FFI
#[repr(C)]
pub struct TestResultFFI {
    /// Test statistic (t, U, chi2, F, etc.)
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom (NaN if not applicable)
    pub df: f64,
    /// Effect size (Cohen's d, r, etc.) (NaN if not applicable)
    pub effect_size: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Total sample size
    pub n: usize,
    /// Group 1 sample size (for two-sample tests)
    pub n1: usize,
    /// Group 2 sample size (for two-sample tests)
    pub n2: usize,
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Test method/name (must be freed)
    pub method: *mut c_char,
}

impl Default for TestResultFFI {
    fn default() -> Self {
        Self {
            statistic: f64::NAN,
            p_value: f64::NAN,
            df: f64::NAN,
            effect_size: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            confidence_level: 0.95,
            n: 0,
            n1: 0,
            n2: 0,
            alternative: AlternativeFFI::TwoSided,
            method: std::ptr::null_mut(),
        }
    }
}

/// ANOVA result for FFI
#[repr(C)]
pub struct AnovaResultFFI {
    /// F statistic
    pub f_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Between-groups degrees of freedom
    pub df_between: usize,
    /// Within-groups degrees of freedom
    pub df_within: usize,
    /// Between-groups sum of squares
    pub ss_between: f64,
    /// Within-groups sum of squares
    pub ss_within: f64,
    /// Number of groups
    pub n_groups: usize,
    /// Total sample size
    pub n: usize,
    /// Test method (must be freed)
    pub method: *mut c_char,
}

impl Default for AnovaResultFFI {
    fn default() -> Self {
        Self {
            f_statistic: f64::NAN,
            p_value: f64::NAN,
            df_between: 0,
            df_within: 0,
            ss_between: f64::NAN,
            ss_within: f64::NAN,
            n_groups: 0,
            n: 0,
            method: std::ptr::null_mut(),
        }
    }
}

/// Correlation result for FFI
#[repr(C)]
pub struct CorrelationResultFFI {
    /// Correlation coefficient
    pub r: f64,
    /// Test statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Sample size
    pub n: usize,
    /// Method name (must be freed)
    pub method: *mut c_char,
}

impl Default for CorrelationResultFFI {
    fn default() -> Self {
        Self {
            r: f64::NAN,
            statistic: f64::NAN,
            p_value: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            confidence_level: 0.95,
            n: 0,
            method: std::ptr::null_mut(),
        }
    }
}

/// Chi-square test result for FFI
#[repr(C)]
pub struct ChiSquareResultFFI {
    /// Chi-square statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Method name (must be freed)
    pub method: *mut c_char,
}

impl Default for ChiSquareResultFFI {
    fn default() -> Self {
        Self {
            statistic: f64::NAN,
            p_value: f64::NAN,
            df: 0,
            method: std::ptr::null_mut(),
        }
    }
}

/// TOST (equivalence) result for FFI
#[repr(C)]
pub struct TostResultFFI {
    /// Lower bound test statistic
    pub t_lower: f64,
    /// Upper bound test statistic
    pub t_upper: f64,
    /// p-value for lower test
    pub p_lower: f64,
    /// p-value for upper test
    pub p_upper: f64,
    /// Overall p-value (max of p_lower, p_upper)
    pub p_value: f64,
    /// Degrees of freedom
    pub df: f64,
    /// Point estimate (mean difference, correlation, etc.)
    pub estimate: f64,
    /// Confidence interval lower bound (for 1-2*alpha CI)
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Equivalence bound (lower)
    pub bound_lower: f64,
    /// Equivalence bound (upper)
    pub bound_upper: f64,
    /// Whether equivalence was established
    pub equivalent: bool,
    /// Sample size
    pub n: usize,
    /// Method name (must be freed)
    pub method: *mut c_char,
}

impl Default for TostResultFFI {
    fn default() -> Self {
        Self {
            t_lower: f64::NAN,
            t_upper: f64::NAN,
            p_lower: f64::NAN,
            p_upper: f64::NAN,
            p_value: f64::NAN,
            df: f64::NAN,
            estimate: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            bound_lower: f64::NAN,
            bound_upper: f64::NAN,
            equivalent: false,
            n: 0,
            method: std::ptr::null_mut(),
        }
    }
}

// =============================================================================
// Test Options FFI Types
// =============================================================================

/// T-test options for FFI
#[repr(C)]
pub struct TTestOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Confidence level for CI
    pub confidence_level: f64,
    /// Assumed equal variance (Student's t) vs Welch
    pub var_equal: bool,
    /// Hypothesized mean difference
    pub mu: f64,
}

impl Default for TTestOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            confidence_level: 0.95,
            var_equal: false,
            mu: 0.0,
        }
    }
}

/// Mann-Whitney U test options for FFI
#[repr(C)]
pub struct MannWhitneyOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Use exact distribution
    pub exact: bool,
    /// Apply continuity correction
    pub continuity_correction: bool,
    /// Confidence level for CI
    pub confidence_level: f64,
    /// Hypothesized location shift
    pub mu: f64,
}

impl Default for MannWhitneyOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            exact: false,
            continuity_correction: true,
            confidence_level: 0.95,
            mu: 0.0,
        }
    }
}

/// Wilcoxon signed-rank test options for FFI
#[repr(C)]
pub struct WilcoxonOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Use exact distribution
    pub exact: bool,
    /// Apply continuity correction
    pub continuity_correction: bool,
    /// Confidence level for CI
    pub confidence_level: f64,
    /// Hypothesized median
    pub mu: f64,
}

impl Default for WilcoxonOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            exact: false,
            continuity_correction: true,
            confidence_level: 0.95,
            mu: 0.0,
        }
    }
}

/// Correlation options for FFI
#[repr(C)]
pub struct CorrelationOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Confidence level for CI
    pub confidence_level: f64,
}

impl Default for CorrelationOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            confidence_level: 0.95,
        }
    }
}

/// Kendall tau type for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KendallTypeFFI {
    TauA = 0,
    TauB = 1,
    TauC = 2,
}

/// Kendall correlation options for FFI
#[repr(C)]
pub struct KendallOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Tau type (a, b, or c)
    pub tau_type: KendallTypeFFI,
    /// Confidence level for CI
    pub confidence_level: f64,
}

impl Default for KendallOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            tau_type: KendallTypeFFI::TauB,
            confidence_level: 0.95,
        }
    }
}

/// Chi-square test options for FFI
#[derive(Default)]
#[repr(C)]
pub struct ChiSquareOptionsFFI {
    /// Apply Yates continuity correction (for 2x2 tables)
    pub correction: bool,
}

/// Fisher's exact test options for FFI
#[repr(C)]
pub struct FisherExactOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Confidence level for odds ratio CI
    pub confidence_level: f64,
}

impl Default for FisherExactOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            confidence_level: 0.95,
        }
    }
}

/// Energy distance test options for FFI
#[repr(C)]
pub struct EnergyDistanceOptionsFFI {
    /// Number of permutations
    pub n_permutations: usize,
    /// Seed for reproducibility (0 = random)
    pub seed: u64,
    /// Whether seed is set
    pub has_seed: bool,
}

impl Default for EnergyDistanceOptionsFFI {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: 0,
            has_seed: false,
        }
    }
}

/// MMD test options for FFI
#[repr(C)]
pub struct MmdOptionsFFI {
    /// Number of permutations
    pub n_permutations: usize,
    /// Seed for reproducibility (0 = random)
    pub seed: u64,
    /// Whether seed is set
    pub has_seed: bool,
}

impl Default for MmdOptionsFFI {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: 0,
            has_seed: false,
        }
    }
}

/// TOST options for FFI
#[repr(C)]
pub struct TostOptionsFFI {
    /// Lower equivalence bound
    pub bound_lower: f64,
    /// Upper equivalence bound
    pub bound_upper: f64,
    /// Alpha level
    pub alpha: f64,
    /// Use pooled variance (only for two-sample t-test)
    pub pooled: bool,
}

impl Default for TostOptionsFFI {
    fn default() -> Self {
        Self {
            bound_lower: -0.5,
            bound_upper: 0.5,
            alpha: 0.05,
            pooled: false,
        }
    }
}

/// Brunner-Munzel test options for FFI
#[repr(C)]
pub struct BrunnerMunzelOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Confidence level
    pub confidence_level: f64,
}

impl Default for BrunnerMunzelOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            confidence_level: 0.95,
        }
    }
}

/// Yuen test options for FFI
#[repr(C)]
pub struct YuenOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Trim proportion (0.0-0.5)
    pub trim: f64,
    /// Confidence level for CI
    pub confidence_level: f64,
}

impl Default for YuenOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            trim: 0.2,
            confidence_level: 0.95,
        }
    }
}

/// Forecast loss function for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastLossFFI {
    SquaredError = 0,
    AbsoluteError = 1,
}

/// Diebold-Mariano test options for FFI
#[repr(C)]
pub struct DieboldMarianoOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Loss function
    pub loss: ForecastLossFFI,
    /// Forecast horizon
    pub horizon: usize,
}

impl Default for DieboldMarianoOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            loss: ForecastLossFFI::SquaredError,
            horizon: 1,
        }
    }
}

/// Proportion test options for FFI
#[repr(C)]
pub struct PropTestOptionsFFI {
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Apply continuity correction
    pub correction: bool,
}

impl Default for PropTestOptionsFFI {
    fn default() -> Self {
        Self {
            alternative: AlternativeFFI::TwoSided,
            correction: true,
        }
    }
}

/// ICC (Intraclass Correlation Coefficient) model type for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccModelFFI {
    OnewayRandom = 0,
    TwowayRandom = 1,
    TwowayMixed = 2,
}

/// ICC type for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccTypeFFI {
    Single = 0,
    Average = 1,
}

/// ICC options for FFI
#[repr(C)]
pub struct IccOptionsFFI {
    /// ICC model
    pub model: IccModelFFI,
    /// ICC type (single or average)
    pub icc_type: IccTypeFFI,
    /// Confidence level
    pub confidence_level: f64,
}

impl Default for IccOptionsFFI {
    fn default() -> Self {
        Self {
            model: IccModelFFI::TwowayRandom,
            icc_type: IccTypeFFI::Single,
            confidence_level: 0.95,
        }
    }
}

/// ICC result for FFI
#[repr(C)]
pub struct IccResultFFI {
    /// ICC value
    pub icc: f64,
    /// F-statistic
    pub f_statistic: f64,
    /// Lower CI bound
    pub ci_lower: f64,
    /// Upper CI bound
    pub ci_upper: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// Number of subjects
    pub n_subjects: usize,
    /// Number of raters
    pub n_raters: usize,
    /// Method name (must be freed)
    pub method: *mut c_char,
}

impl Default for IccResultFFI {
    fn default() -> Self {
        Self {
            icc: f64::NAN,
            f_statistic: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            confidence_level: 0.95,
            n_subjects: 0,
            n_raters: 0,
            method: std::ptr::null_mut(),
        }
    }
}

// =============================================================================
// Categorical Tests Result Types
// =============================================================================

/// Proportion test result for FFI
#[repr(C)]
pub struct PropTestResultFFI {
    /// Test statistic (z)
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Estimated proportion
    pub estimate: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// Sample size
    pub n: usize,
    /// Alternative hypothesis
    pub alternative: AlternativeFFI,
    /// Method name (must be freed)
    pub method: *mut c_char,
}

impl Default for PropTestResultFFI {
    fn default() -> Self {
        Self {
            statistic: f64::NAN,
            p_value: f64::NAN,
            estimate: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            n: 0,
            alternative: AlternativeFFI::TwoSided,
            method: std::ptr::null_mut(),
        }
    }
}

/// Cohen's kappa result for FFI
#[repr(C)]
pub struct KappaResultFFI {
    /// Kappa coefficient
    pub kappa: f64,
    /// Standard error
    pub se: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// z-statistic
    pub z: f64,
    /// p-value
    pub p_value: f64,
}

impl Default for KappaResultFFI {
    fn default() -> Self {
        Self {
            kappa: f64::NAN,
            se: f64::NAN,
            ci_lower: f64::NAN,
            ci_upper: f64::NAN,
            z: f64::NAN,
            p_value: f64::NAN,
        }
    }
}

// =============================================================================
// Correlation Result Types
// =============================================================================

/// Distance correlation result for FFI
#[repr(C)]
pub struct DistanceCorResultFFI {
    /// Distance correlation coefficient
    pub dcor: f64,
    /// Distance covariance
    pub dcov: f64,
    /// Distance variance of x
    pub dvar_x: f64,
    /// Distance variance of y
    pub dvar_y: f64,
    /// Sample size
    pub n: usize,
}

impl Default for DistanceCorResultFFI {
    fn default() -> Self {
        Self {
            dcor: f64::NAN,
            dcov: f64::NAN,
            dvar_x: f64::NAN,
            dvar_y: f64::NAN,
            n: 0,
        }
    }
}

// =============================================================================
// PLS (Partial Least Squares) FFI Types
// =============================================================================

/// PLS options for FFI
#[repr(C)]
pub struct PlsOptionsFFI {
    /// Number of components to extract
    pub n_components: usize,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}

impl Default for PlsOptionsFFI {
    fn default() -> Self {
        Self {
            n_components: 1,
            fit_intercept: true,
        }
    }
}

/// PLS fit result
#[repr(C)]
pub struct PlsFitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// R-squared
    pub r_squared: f64,
    /// Number of components used
    pub n_components: usize,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
}

impl Default for PlsFitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            r_squared: f64::NAN,
            n_components: 0,
            n_observations: 0,
            n_features: 0,
        }
    }
}

// =============================================================================
// Isotonic Regression FFI Types
// =============================================================================

/// Isotonic regression options for FFI
#[repr(C)]
pub struct IsotonicOptionsFFI {
    /// Whether the function should be increasing (true) or decreasing (false)
    pub increasing: bool,
}

impl Default for IsotonicOptionsFFI {
    fn default() -> Self {
        Self { increasing: true }
    }
}

/// Isotonic fit result
#[repr(C)]
pub struct IsotonicFitResultCore {
    /// Pointer to fitted values array (same length as input)
    pub fitted_values: *mut f64,
    /// Number of fitted values
    pub fitted_values_len: usize,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Whether increasing constraint was used
    pub increasing: bool,
}

impl Default for IsotonicFitResultCore {
    fn default() -> Self {
        Self {
            fitted_values: std::ptr::null_mut(),
            fitted_values_len: 0,
            r_squared: f64::NAN,
            n_observations: 0,
            increasing: true,
        }
    }
}

// =============================================================================
// Quantile Regression FFI Types
// =============================================================================

/// Quantile regression options for FFI
#[repr(C)]
pub struct QuantileOptionsFFI {
    /// Quantile to estimate (0 < tau < 1, e.g., 0.5 for median)
    pub tau: f64,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for QuantileOptionsFFI {
    fn default() -> Self {
        Self {
            tau: 0.5,
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Quantile fit result
#[repr(C)]
pub struct QuantileFitResultCore {
    /// Pointer to coefficients array
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// Quantile estimated
    pub tau: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
}

impl Default for QuantileFitResultCore {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            tau: f64::NAN,
            n_observations: 0,
            n_features: 0,
        }
    }
}

// ============================================================================
// LmDynamic FFI Types
// ============================================================================

/// Information criterion for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InformationCriterionFFI {
    /// Akaike Information Criterion
    AIC = 0,
    /// Corrected AIC (default)
    #[default]
    AICc = 1,
    /// Bayesian Information Criterion
    BIC = 2,
}

/// LmDynamic options for FFI
#[repr(C)]
pub struct LmDynamicOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Information criterion for model weighting
    pub ic: InformationCriterionFFI,
    /// Distribution family
    pub distribution: AlmDistributionFFI,
    /// LOWESS smoothing span (0.0 = no smoothing, >0 = span value)
    pub lowess_span: f64,
    /// Maximum number of candidate models (0 = default)
    pub max_models: u32,
    /// Confidence level for intervals
    pub confidence_level: f64,
}

impl Default for LmDynamicOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            ic: InformationCriterionFFI::AICc,
            distribution: AlmDistributionFFI::Normal,
            lowess_span: 0.3,
            max_models: 0,
            confidence_level: 0.95,
        }
    }
}

/// LmDynamic result for FFI
#[repr(C)]
pub struct LmDynamicFitResultFFI {
    /// Averaged coefficients (heap-allocated, caller must free)
    pub coefficients: *mut f64,
    /// Number of coefficients
    pub coefficients_len: usize,
    /// Intercept value (NaN if no intercept)
    pub intercept: f64,
    /// R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// RMSE
    pub rmse: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features
    pub n_features: usize,
    /// Flattened dynamic coefficients (n_observations x n_coefs_per_obs, row-major)
    pub dynamic_coefficients: *mut f64,
    /// Number of coefficient columns per observation
    pub n_coefs_per_obs: usize,
}

impl Default for LmDynamicFitResultFFI {
    fn default() -> Self {
        Self {
            coefficients: std::ptr::null_mut(),
            coefficients_len: 0,
            intercept: f64::NAN,
            r_squared: f64::NAN,
            adj_r_squared: f64::NAN,
            rmse: f64::NAN,
            n_observations: 0,
            n_features: 0,
            dynamic_coefficients: std::ptr::null_mut(),
            n_coefs_per_obs: 0,
        }
    }
}

// =============================================================================
// AFT (accelerated failure time) survival regression — issue #107
// =============================================================================

/// AFT error distribution code.
// repr(C), not repr(u8): a C enum is int-sized, so a 1-byte Rust enum shifts
// every field that follows it. Harmless while such a field sits last in a
// struct, fatal when it sits first (AftOptionsFFI::dist).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AftDistributionFFI {
    #[default]
    Weibull = 0,
    LogNormal = 1,
    LogLogistic = 2,
    Exponential = 3,
}

/// Options for an AFT fit. Flat POD, like every other options struct here.
#[repr(C)]
pub struct AftOptionsFFI {
    pub dist: AftDistributionFFI,
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    /// Per-coefficient priors, or null.
    pub priors: *const PriorSpecFFI,
    pub priors_len: usize,
    pub vcov: VcovTypeFFI,
}

impl Default for AftOptionsFFI {
    fn default() -> Self {
        Self {
            dist: AftDistributionFFI::Weibull,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-9,
            compute_inference: false,
            confidence_level: 0.95,
            priors: std::ptr::null(),
            priors_len: 0,
            vcov: VcovTypeFFI::Laplace,
        }
    }
}

/// Core results of an AFT fit. `coefficients` is owned by the caller and must be
/// released with `anofox_free_aft_result`.
#[repr(C)]
pub struct AftFitResultCore {
    pub coefficients: *mut f64,
    pub coefficients_len: usize,
    pub intercept: f64,
    pub scale: f64,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_observations: usize,
    pub n_events: usize,
    pub n_censored: usize,
    pub n_features: usize,
    pub iterations: u32,
    pub converged: bool,
}

/// Inference for an AFT fit. All arrays are caller-owned and released with
/// `anofox_free_aft_inference`.
#[repr(C)]
pub struct AftInferenceFFI {
    pub std_errors: *mut f64,
    pub z_values: *mut f64,
    pub p_values: *mut f64,
    pub ci_lower: *mut f64,
    pub ci_upper: *mut f64,
    pub len: usize,
    pub confidence_level: f64,
    /// NaN when no intercept was fitted.
    pub intercept_std_error: f64,
    /// NaN when the scale is fixed (exponential).
    pub log_scale_std_error: f64,
}

#[cfg(test)]
mod abi_tests {
    use super::*;

    /// The C header is maintained by hand, so nothing but a test stops a Rust-side
    /// enum from silently disagreeing with its C counterpart about width. A C enum
    /// is int-sized; these must be too, or every field after them shifts.
    #[test]
    fn ffi_enums_are_int_sized() {
        assert_eq!(std::mem::size_of::<PriorKindFFI>(), 4);
        assert_eq!(std::mem::size_of::<VcovTypeFFI>(), 4);
        assert_eq!(std::mem::size_of::<AftDistributionFFI>(), 4);
        assert_eq!(std::mem::size_of::<PoissonLinkFFI>(), 4);
        assert_eq!(std::mem::size_of::<BinomialLinkFFI>(), 4);
    }

    /// Offsets the C compiler will use for the equivalent struct.
    #[test]
    fn aft_options_layout_matches_the_c_struct() {
        use std::mem::{align_of, size_of};
        assert_eq!(align_of::<AftOptionsFFI>(), 8);
        // int, bool, uint32, double, bool, double, ptr, size_t, int
        //   0     4       8       16      24      32     40      48    56
        assert_eq!(size_of::<AftOptionsFFI>(), 64);

        let o = AftOptionsFFI::default();
        let base = &o as *const _ as usize;
        assert_eq!(&o.dist as *const _ as usize - base, 0);
        assert_eq!(&o.fit_intercept as *const _ as usize - base, 4);
        assert_eq!(&o.max_iterations as *const _ as usize - base, 8);
        assert_eq!(&o.tolerance as *const _ as usize - base, 16);
        assert_eq!(&o.compute_inference as *const _ as usize - base, 24);
        assert_eq!(&o.confidence_level as *const _ as usize - base, 32);
        assert_eq!(&o.priors as *const _ as usize - base, 40);
        assert_eq!(&o.priors_len as *const _ as usize - base, 48);
        assert_eq!(&o.vcov as *const _ as usize - base, 56);
    }

    #[test]
    fn prior_spec_layout_matches_the_c_struct() {
        use std::mem::size_of;
        // int + 4 pad + double + double
        assert_eq!(size_of::<PriorSpecFFI>(), 24);
        let p = PriorSpecFFI::default();
        let base = &p as *const _ as usize;
        assert_eq!(&p.kind as *const _ as usize - base, 0);
        assert_eq!(&p.loc as *const _ as usize - base, 8);
        assert_eq!(&p.scale as *const _ as usize - base, 16);
    }
}

// =============================================================================
// Empirical-Bayes shrinkage — issue #107
// =============================================================================

/// Between-group variance estimator.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TauMethodFFI {
    #[default]
    DerSimonianLaird = 0,
    /// Complete pooling.
    None = 1,
}

/// Options for empirical-Bayes shrinkage.
#[repr(C)]
pub struct EbShrinkOptionsFFI {
    pub method: TauMethodFFI,
    /// A fixed between-group variance; NaN means "estimate it".
    pub tau_squared: f64,
}

impl Default for EbShrinkOptionsFFI {
    fn default() -> Self {
        Self {
            method: TauMethodFFI::DerSimonianLaird,
            tau_squared: f64::NAN,
        }
    }
}

/// Result of a shrinkage pass. The five per-group arrays share `len` and are in
/// input order; all are caller-owned and released by `anofox_free_eb_shrink_result`.
#[repr(C)]
pub struct EbShrinkResultFFI {
    pub mu: f64,
    pub mu_se: f64,
    pub tau_squared: f64,
    pub i_squared: f64,
    pub q: f64,
    pub n_groups: usize,
    pub estimate: *mut f64,
    pub se: *mut f64,
    pub shrunken: *mut f64,
    pub shrunken_se: *mut f64,
    pub weight: *mut f64,
    pub len: usize,
}

// =============================================================================
// Mixed-effects GLMs — issue #107
// =============================================================================

/// Response family for a mixed-effects fit.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlmmFamilyFFI {
    #[default]
    Gaussian = 0,
    Poisson = 1,
    Binomial = 2,
    NegativeBinomial = 3,
    Gamma = 4,
    Tweedie = 5,
}

/// Options for a mixed-effects fit.
#[repr(C)]
pub struct GlmmOptionsFFI {
    pub family: GlmmFamilyFFI,
    pub fit_intercept: bool,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub compute_inference: bool,
    pub confidence_level: f64,
    pub reml: bool,
    /// Negative Binomial theta; ignored by other families.
    pub theta: f64,
    /// Tweedie variance power; ignored by other families.
    pub power: f64,
    /// 1-based index into `x` of an offset column; 0 means none.
    pub offset_column: usize,
    /// Pointer to `random_slopes_len` 0-based indices into `x` of columns that
    /// additionally carry a random slope. Null/zero-length = random intercept only.
    pub random_slopes: *const usize,
    pub random_slopes_len: usize,
}

impl Default for GlmmOptionsFFI {
    fn default() -> Self {
        Self {
            family: GlmmFamilyFFI::Gaussian,
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
            reml: true,
            theta: 1.0,
            power: 1.5,
            offset_column: 0,
            random_slopes: std::ptr::null(),
            random_slopes_len: 0,
        }
    }
}

/// Result of a mixed-effects fit. Every array is caller-owned and released by
/// `anofox_free_glmm_result`.
#[repr(C)]
pub struct GlmmResultFFI {
    pub coefficients: *mut f64,
    pub coefficients_len: usize,
    pub intercept: f64,
    pub std_errors: *mut f64,
    pub z_values: *mut f64,
    pub p_values: *mut f64,
    pub ci_lower: *mut f64,
    pub ci_upper: *mut f64,
    pub inference_len: usize,
    pub intercept_std_error: f64,
    pub confidence_level: f64,
    pub var_group: f64,
    pub var_residual: f64,
    pub icc: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub deviance: f64,
    pub n_observations: usize,
    pub n_groups: usize,
    pub n_features: usize,
    pub iterations: u32,
    pub converged: bool,
    /// Per-group random effects, all of length `ranef_len == n_groups`.
    pub ranef_group: *mut i32,
    pub ranef_value: *mut f64,
    pub ranef_se: *mut f64,
    pub ranef_n: *mut i64,
    pub ranef_len: usize,
    /// Random-effects covariance Σ, flattened row-major `random_dim × random_dim`.
    pub random_cov: *mut f64,
    /// `q = 1 + number of random slopes`.
    pub random_dim: usize,
    /// Per-group random-effect vectors `[intercept, slope_1, …]`, flattened
    /// row-major `ranef_len × random_dim`.
    pub ranef_effects: *mut f64,
    /// Per-factor variance components for crossed/nested fits (length
    /// `factor_len`). Empty for the single-factor path.
    pub factor_var: *mut f64,
    pub factor_n_levels: *mut i64,
    pub factor_len: usize,
}
