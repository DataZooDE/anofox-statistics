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
    /// Standard error of intercept (NaN if not computed or no intercept)
    pub intercept_std_error: f64,
    /// t-value for intercept (NaN if not computed or no intercept)
    pub intercept_t_value: f64,
    /// p-value for intercept (NaN if not computed or no intercept)
    pub intercept_p_value: f64,
    /// Lower bound of intercept confidence interval (NaN if not computed)
    pub intercept_ci_lower: f64,
    /// Upper bound of intercept confidence interval (NaN if not computed)
    pub intercept_ci_upper: f64,
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
            intercept_std_error: f64::NAN,
            intercept_t_value: f64::NAN,
            intercept_p_value: f64::NAN,
            intercept_ci_lower: f64::NAN,
            intercept_ci_upper: f64::NAN,
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
