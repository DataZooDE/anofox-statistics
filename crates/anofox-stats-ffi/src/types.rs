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

/// WLS (Weighted Least Squares) options for FFI
#[repr(C)]
pub struct WlsOptionsFFI {
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for CIs
    pub confidence_level: f64,
}

impl Default for WlsOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
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
        }
    }
}

/// GLM options for Negative Binomial regression
#[repr(C)]
pub struct NegBinomialOptionsFFI {
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
}

impl Default for NegBinomialOptionsFFI {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
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
