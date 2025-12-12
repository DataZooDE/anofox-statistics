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

// ============================================================================
// GLM Types (Generalized Linear Models)
// ============================================================================

/// GLM Family - specifies the distribution and link function
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GlmFamily {
    /// Poisson regression (for count data)
    #[default]
    Poisson,
    /// Binomial/Logistic regression (for binary outcomes)
    Binomial,
    /// Negative Binomial regression (for overdispersed count data)
    NegativeBinomial,
    /// Tweedie regression (for zero-inflated continuous data)
    Tweedie,
}

/// Link function for Poisson regression
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PoissonLink {
    /// Log link (canonical) - most common
    #[default]
    Log,
    /// Identity link - linear mean
    Identity,
    /// Square root link - variance stabilizing
    Sqrt,
}

/// Link function for Binomial regression
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BinomialLink {
    /// Logit link (canonical) - logistic regression
    #[default]
    Logit,
    /// Probit link - normal CDF
    Probit,
    /// Complementary log-log link
    Cloglog,
    /// Cauchit link
    Cauchit,
    /// Log link
    Log,
}

/// Options for Poisson regression (GLM)
#[derive(Debug, Clone)]
pub struct PoissonOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Link function (Log, Identity, or Sqrt)
    pub link: PoissonLink,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for PoissonOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: PoissonLink::Log,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Options for Binomial regression (GLM)
#[derive(Debug, Clone)]
pub struct BinomialOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Link function (Logit, Probit, etc.)
    pub link: BinomialLink,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for BinomialOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            link: BinomialLink::Logit,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Options for Negative Binomial regression (GLM)
#[derive(Debug, Clone)]
pub struct NegBinomialOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Dispersion parameter (alpha). If None, it will be estimated.
    pub alpha: Option<f64>,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for NegBinomialOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            alpha: None,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Options for Tweedie regression (GLM)
#[derive(Debug, Clone)]
pub struct TweedieOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Tweedie power parameter (typically 1 < p < 2 for zero-inflated continuous)
    /// p=1 is Poisson, p=2 is Gamma, 1<p<2 is compound Poisson-Gamma
    pub power: f64,
    /// Maximum iterations for IRLS
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for TweedieOptions {
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

/// Result from GLM fitting - uses deviance-based metrics instead of R²
#[derive(Debug, Clone)]
pub struct GlmFitResult {
    /// Regression coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Intercept term (if fitted with intercept)
    pub intercept: Option<f64>,
    /// Null deviance (deviance of intercept-only model)
    pub null_deviance: f64,
    /// Residual deviance (deviance of fitted model)
    pub residual_deviance: f64,
    /// Pseudo R-squared (1 - residual_deviance/null_deviance)
    pub pseudo_r_squared: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of features (excluding intercept)
    pub n_features: usize,
    /// Number of iterations to convergence
    pub iterations: u32,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Dispersion parameter (for NegBinomial, Tweedie)
    pub dispersion: Option<f64>,
}

/// GLM inference results
#[derive(Debug, Clone)]
pub struct GlmInferenceResult {
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// z-statistics for coefficients (Wald test)
    pub z_values: Vec<f64>,
    /// p-values for coefficients
    pub p_values: Vec<f64>,
    /// Lower bound of confidence intervals
    pub ci_lower: Vec<f64>,
    /// Upper bound of confidence intervals
    pub ci_upper: Vec<f64>,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,
}

// ============================================================================
// ALM Types (Augmented Linear Models)
// ============================================================================

/// Distribution families for ALM (Augmented Linear Model)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlmDistribution {
    // Continuous distributions
    /// Normal (Gaussian) distribution - standard linear regression
    #[default]
    Normal,
    /// Laplace (double exponential) - robust to outliers, LAD regression
    Laplace,
    /// Student's t distribution - heavy-tailed, robust
    StudentT,
    /// Logistic distribution
    Logistic,
    /// Asymmetric Laplace - for quantile regression
    AsymmetricLaplace,
    /// Generalised Normal (Subbotin) distribution
    GeneralisedNormal,
    /// S distribution
    S,
    /// Log-Normal distribution - for positive skewed data
    LogNormal,
    /// Log-Laplace distribution
    LogLaplace,
    /// Log-S distribution
    LogS,
    /// Log-Generalised Normal distribution
    LogGeneralisedNormal,
    /// Folded Normal distribution - for positive data
    FoldedNormal,
    /// Rectified Normal distribution
    RectifiedNormal,
    /// Box-Cox Normal distribution
    BoxCoxNormal,
    /// Gamma distribution - for positive continuous data
    Gamma,
    /// Inverse Gaussian distribution
    InverseGaussian,
    /// Exponential distribution
    Exponential,
    /// Beta distribution - for data in (0, 1)
    Beta,
    /// Logit-Normal distribution
    LogitNormal,
    // Discrete distributions
    /// Poisson distribution - for count data
    Poisson,
    /// Negative Binomial - for overdispersed count data
    NegativeBinomial,
    /// Binomial distribution - for binary/proportion data
    Binomial,
    /// Geometric distribution
    Geometric,
    /// Cumulative Logistic (ordered logistic)
    CumulativeLogistic,
    /// Cumulative Normal (ordered probit)
    CumulativeNormal,
}

/// Loss functions for ALM model fitting
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlmLoss {
    /// Maximum Likelihood Estimation (default)
    #[default]
    Likelihood,
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Half Absolute Moment
    HAM,
    /// Robust Likelihood Estimator with trimming
    ROLE,
}

/// Options for ALM (Augmented Linear Model)
#[derive(Debug, Clone)]
pub struct AlmOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Error distribution family
    pub distribution: AlmDistribution,
    /// Loss function for fitting
    pub loss: AlmLoss,
    /// Trim fraction for ROLE loss (0.0 to 0.5)
    pub role_trim: f64,
    /// Quantile for asymmetric Laplace (0.0 to 1.0)
    pub quantile: f64,
    /// Maximum iterations for optimization
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Whether to compute inference statistics
    pub compute_inference: bool,
    /// Confidence level for confidence intervals
    pub confidence_level: f64,
}

impl Default for AlmOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            distribution: AlmDistribution::Normal,
            loss: AlmLoss::Likelihood,
            role_trim: 0.05,
            quantile: 0.5,
            max_iterations: 1000,
            tolerance: 1e-6,
            compute_inference: false,
            confidence_level: 0.95,
        }
    }
}

/// Result from ALM fitting
#[derive(Debug, Clone)]
pub struct AlmFitResult {
    /// Regression coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Intercept term (if fitted with intercept)
    pub intercept: Option<f64>,
    /// Log-likelihood of the fitted model
    pub log_likelihood: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
    /// Scale parameter (sigma for Normal, etc.)
    pub scale: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of features (excluding intercept)
    pub n_features: usize,
    /// Number of iterations to convergence
    pub iterations: u32,
    /// Whether the algorithm converged
    pub converged: bool,
}

// ============================================================================
// BLS Types (Bounded Least Squares)
// ============================================================================

/// Options for BLS (Bounded Least Squares)
#[derive(Debug, Clone)]
pub struct BlsOptions {
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Lower bounds for coefficients (None = -infinity)
    pub lower_bounds: Option<Vec<f64>>,
    /// Upper bounds for coefficients (None = +infinity)
    pub upper_bounds: Option<Vec<f64>>,
    /// Maximum iterations for active set algorithm
    pub max_iterations: u32,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for BlsOptions {
    fn default() -> Self {
        Self {
            fit_intercept: true,
            lower_bounds: None,
            upper_bounds: None,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl BlsOptions {
    /// Create options for Non-Negative Least Squares (NNLS)
    pub fn nnls() -> Self {
        Self {
            fit_intercept: false, // NNLS typically doesn't have intercept
            lower_bounds: None,   // Will be set to all zeros
            upper_bounds: None,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// Result from BLS fitting
#[derive(Debug, Clone)]
pub struct BlsFitResult {
    /// Regression coefficients (excluding intercept)
    pub coefficients: Vec<f64>,
    /// Intercept term (if fitted with intercept)
    pub intercept: Option<f64>,
    /// Sum of squared residuals
    pub ssr: f64,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
    /// Number of observations used
    pub n_observations: usize,
    /// Number of features (excluding intercept)
    pub n_features: usize,
    /// Number of active constraints (coefficients at bounds)
    pub n_active_constraints: usize,
    /// Which coefficients are at their lower bound
    pub at_lower_bound: Vec<bool>,
    /// Which coefficients are at their upper bound
    pub at_upper_bound: Vec<bool>,
}
