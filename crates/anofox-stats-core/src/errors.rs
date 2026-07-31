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

    #[error(
        "Elastic Net failed to converge after {iterations} iterations (tolerance: {tolerance})"
    )]
    ConvergenceFailure { iterations: u32, tolerance: f64 },

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
            R::DimensionMismatch { x_rows, y_len } => StatsError::DimensionMismatch {
                y_len,
                x_rows,
            },
            R::InsufficientObservations { needed, got } => StatsError::InsufficientData {
                rows: got,
                cols: needed,
            },
            R::SingularMatrix => StatsError::SingularMatrix,
            R::ConvergenceFailed { iterations } => StatsError::ConvergenceFailure {
                iterations: iterations as u32,
                tolerance: f64::NAN,
            },
            other => StatsError::RegressError(format!("{other:?}")),
        }
    }
}

/// Result type for statistical operations
pub type StatsResult<T> = Result<T, StatsError>;
