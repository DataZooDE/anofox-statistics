use thiserror::Error;

#[derive(Error, Debug)]
pub enum StatError {
    #[error("empty input data")]
    EmptyData,
    #[error("insufficient data: need at least {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, StatError>;
