use crate::error::{Result, StatError};

/// Validate MCS numeric parameters.
pub(super) fn validate_mcs_parameters(alpha: f64, n_bootstrap: usize) -> Result<()> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(StatError::InvalidParameter(
            "alpha must be in (0, 1)".to_string(),
        ));
    }
    if n_bootstrap == 0 {
        return Err(StatError::InvalidParameter(
            "n_bootstrap must be positive".to_string(),
        ));
    }
    Ok(())
}

/// Validate model loss dimensions are consistent.
pub(super) fn validate_model_dimensions(losses: &[Vec<f64>]) -> Result<()> {
    let t = losses[0].len();
    if t == 0 {
        return Err(StatError::EmptyData);
    }

    for (i, model_losses) in losses.iter().enumerate() {
        if model_losses.len() != t {
            return Err(StatError::InvalidParameter(format!(
                "Model {} has {} observations, expected {}",
                i,
                model_losses.len(),
                t
            )));
        }
    }
    Ok(())
}

/// Validate MCS input parameters.
pub(super) fn validate_mcs_inputs(
    losses: &[Vec<f64>],
    alpha: f64,
    n_bootstrap: usize,
) -> Result<()> {
    if losses.is_empty() {
        return Err(StatError::InvalidParameter(
            "At least one model required".to_string(),
        ));
    }
    validate_mcs_parameters(alpha, n_bootstrap)?;
    validate_model_dimensions(losses)
}
