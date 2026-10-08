use thiserror::Error;

/// Errors that can occur during statistical computations
#[derive(Error, Debug)]
pub enum StatsError {
    // Input validation errors
    #[error("Invalid alpha parameter: {0} (must be >= 0)")]
    InvalidAlpha(f64),

    #[error("Invalid L1 ratio: {0} (must be in [0, 1])")]
    InvalidL1Ratio(f64),

    #[error("Insufficient data: {rows} rows, {cols} features (need rows > features)")]
    InsufficientData { rows: usize, cols: usize },

    #[error("All rows filtered due to NULL/NaN values")]
    NoValidData,

    #[error("Dimension mismatch: y has {y_len} elements, X has {x_rows} rows")]
    DimensionMismatch { y_len: usize, x_rows: usize },

    #[error("Empty input: {field} cannot be empty")]
    EmptyInput { field: &'static str },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid value for {field}: {message}")]
    InvalidValue {
        field: &'static str,
        message: String,
    },

    #[error("Dimension mismatch: {0}")]
    DimensionMismatchMsg(String),

    #[error("Insufficient data: {0}")]
    InsufficientDataMsg(String),

    // Numerical errors
    #[error("Matrix is singular or near-singular")]
    SingularMatrix,

    #[error("Cholesky decomposition failed: matrix not positive definite")]
    CholeskyFailed,

    #[error("QR decomposition failed")]
    QrFailed,

    #[error("Failed to converge after {iterations} iterations (tolerance: {tolerance})")]
    ConvergenceFailure { iterations: u32, tolerance: f64 },

    /// A data-dependent numerical breakdown (non-finite linear predictor or
    /// likelihood, etc.). Reported to SQL as a degenerate outcome (NULL), not as
    /// an input error.
    #[error("Numerical failure: {0}")]
    NumericalFailure(String),

    // Internal errors
    #[error("Memory allocation failed")]
    AllocationFailure,

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("regress-rs error: {0}")]
    RegressError(String),
}

/// Map upstream regression errors onto this crate's error type.
///
/// Call sites previously did `.map_err(|e| StatsError::RegressError(format!("{:?}", e)))`,
/// which collapsed every upstream variant into a `Debug` string and lost the
/// structure. Cases with a faithful local counterpart are translated; the rest keep
/// the string form.
impl From<anofox_regression::solvers::RegressionError> for StatsError {
    fn from(err: anofox_regression::solvers::RegressionError) -> Self {
        use anofox_regression::solvers::RegressionError as R;
        match err {
            R::DimensionMismatch { x_rows, y_len } => {
                StatsError::DimensionMismatch { y_len, x_rows }
            }
            R::InsufficientObservations { needed, got } => StatsError::InsufficientData {
                rows: got,
                cols: needed,
            },
            R::SingularMatrix => StatsError::SingularMatrix,
            R::ConvergenceFailed { iterations } => StatsError::ConvergenceFailure {
                iterations: iterations as u32,
                tolerance: f64::NAN,
            },
            R::AllFeaturesConstant => StatsError::SingularMatrix,
            R::InvalidOptions(e) => StatsError::InvalidInput(e.to_string()),
            R::InvalidWeights => StatsError::InvalidInput(
                "invalid weights: all weights must be non-negative".to_string(),
            ),
            R::NumericalError(msg) => classify_numerical_error(msg),
        }
    }
}

/// Map errors of the upstream penalized GLM engine and AFT model onto this
/// crate's error type. The upstream variants mirror `StatsError` 1:1, so every
/// FFI error code and the error-vs-NULL policy stay what they were.
impl From<anofox_regression::solvers::penalized_glm::GlmEngineError> for StatsError {
    fn from(err: anofox_regression::solvers::penalized_glm::GlmEngineError) -> Self {
        use anofox_regression::solvers::penalized_glm::GlmEngineError as G;
        match err {
            G::InsufficientData { rows, cols } => StatsError::InsufficientData { rows, cols },
            G::InsufficientDataMsg(msg) => StatsError::InsufficientDataMsg(msg),
            G::NoValidData => StatsError::NoValidData,
            G::DimensionMismatch { y_len, x_rows } => {
                StatsError::DimensionMismatch { y_len, x_rows }
            }
            G::EmptyInput { field } => StatsError::EmptyInput { field },
            G::InvalidInput(msg) => StatsError::InvalidInput(msg),
            G::InvalidValue { field, message } => StatsError::InvalidValue { field, message },
            G::SingularMatrix => StatsError::SingularMatrix,
            G::NumericalFailure(msg) => StatsError::NumericalFailure(msg),
            // `GlmEngineError` is `#[non_exhaustive]`; a variant added upstream is
            // surfaced as an error rather than silently degraded to NULL.
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// Map errors of the upstream moment-heuristic AID onto this crate's error type.
impl From<anofox_regression::solvers::aid::heuristic::AidError> for StatsError {
    fn from(err: anofox_regression::solvers::aid::heuristic::AidError) -> Self {
        use anofox_regression::solvers::aid::heuristic::AidError as A;
        match err {
            A::EmptyInput { field } => StatsError::EmptyInput { field },
            A::NoValidData => StatsError::NoValidData,
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// Map errors of the upstream empirical-Bayes shrinkage onto this crate's error
/// type, keeping the variants (and so the FFI error codes) used before.
impl From<anofox_regression::solvers::eb_shrink::EbShrinkError> for StatsError {
    fn from(err: anofox_regression::solvers::eb_shrink::EbShrinkError) -> Self {
        use anofox_regression::solvers::eb_shrink::EbShrinkError as E;
        match err {
            E::EmptyInput { field } => StatsError::EmptyInput { field },
            E::DimensionMismatch {
                estimates,
                standard_errors,
            } => StatsError::DimensionMismatch {
                y_len: estimates,
                x_rows: standard_errors,
            },
            E::InsufficientData { usable } => StatsError::InsufficientData {
                rows: usable,
                cols: 2,
            },
            E::InvalidValue { field, message } => StatsError::InvalidValue { field, message },
            other => StatsError::RegressError(other.to_string()),
        }
    }
}

/// Upstream `NumericalError` mixes three kinds of failure under one variant:
/// option/parameter validation ("tau must be between 0 and 1"), structurally
/// degenerate data ("needs at least two distinct groups") and genuine numerical
/// breakdown ("non-finite linear predictor"). Split them so the SQL layer can
/// raise the first and return NULL for the other two.
fn classify_numerical_error(msg: String) -> StatsError {
    let lower = msg.to_lowercase();
    if lower.contains("must be")
        || lower.contains("out of range")
        || lower.contains("requires exactly")
        || lower.contains("needs a random")
    {
        StatsError::InvalidInput(msg)
    } else if lower.contains("at least") {
        StatsError::InsufficientDataMsg(msg)
    } else {
        StatsError::NumericalFailure(msg)
    }
}

/// Result type for statistical operations
pub type StatsResult<T> = Result<T, StatsError>;
