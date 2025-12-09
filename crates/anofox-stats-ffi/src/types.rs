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

/// OLS options for FFI
#[repr(C)]
pub struct OlsOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
}

impl Default for OlsOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
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
}

impl Default for RidgeOptionsFFI {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
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
}

impl Default for ElasticNetOptionsFFI {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            fit_intercept: true,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}
