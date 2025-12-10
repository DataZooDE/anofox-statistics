/// Core result from model fitting - always computed
#[derive(Debug, Clone)]
pub struct FitResultCore {
    /// Regression coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Intercept term (if fitted with intercept)
    pub intercept: Option<f64>,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Residual standard error
    pub residual_std_error: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of features (excluding intercept)
    pub n_features: usize,
}

/// Inference results - only computed if requested
#[derive(Debug, Clone)]
pub struct FitResultInference {
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// t-statistics for coefficients
    pub t_values: Vec<f64>,
    /// p-values for coefficients
    pub p_values: Vec<f64>,
    /// Lower bound of confidence intervals
    pub ci_lower: Vec<f64>,
    /// Upper bound of confidence intervals
    pub ci_upper: Vec<f64>,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,
    /// F-statistic for overall model significance
    pub f_statistic: Option<f64>,
    /// p-value for F-statistic
    pub f_pvalue: Option<f64>,
}

/// Diagnostic results
#[derive(Debug, Clone)]
pub struct FitResultDiagnostics {
    /// Variance Inflation Factors
    pub vif: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
}

/// Combined fit result
#[derive(Debug, Clone)]
pub struct FitResult {
    pub core: FitResultCore,
    pub inference: Option<FitResultInference>,
    pub diagnostics: Option<FitResultDiagnostics>,
}

/// Options for OLS fitting
#[derive(Debug, Clone)]
pub struct OlsOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Whether to compute inference statistics (std errors, p-values, etc.)
    pub compute_inference: bool,
    /// Confidence level for confidence intervals (default: 0.95)
    pub confidence_level: f64,
}

impl Default for OlsOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Options for Ridge regression
#[derive(Debug, Clone)]
pub struct RidgeOptions {
    /// L2 regularization parameter (must be >= 0)
    pub alpha: f64,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for RidgeOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Options for Elastic Net regression
#[derive(Debug, Clone)]
pub struct ElasticNetOptions {
    /// Regularization strength (must be >= 0)
    pub alpha: f64,
    /// L1 ratio: 0 = Ridge, 1 = Lasso (must be in [0, 1])
    pub l1_ratio: f64,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Maximum iterations for coordinate descent
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for ElasticNetOptions {
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

/// Convergence information for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether the algorithm converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: u32,
    /// Final tolerance achieved
    pub final_tolerance: f64,
}

/// Options for Weighted Least Squares
#[derive(Debug, Clone)]
pub struct WlsOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for WlsOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Policy for handling NULL values
#[derive(Debug, Clone, Copy, Default)]
pub enum NullPolicy {
    /// Skip rows with any NULL (default for fitting)
    #[default]
    DropNull,
    /// Treat NULL as NaN (propagates through computations)
    NullAsNaN,
    /// Error if any NULL encountered
    ErrorOnNull,
}

/// Policy for handling NaN values
#[derive(Debug, Clone, Copy, Default)]
pub enum NanPolicy {
    /// Skip rows with any NaN (default)
    #[default]
    DropNaN,
    /// Error if any NaN encountered (strict mode)
    ErrorOnNaN,
    /// Keep NaN (for predictions where input might be NaN)
    KeepNaN,
}
