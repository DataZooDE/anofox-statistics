//! Equivalence bounds specification.

use crate::error::{Result, StatError};

/// Specification of equivalence bounds for TOST.
///
/// Equivalence bounds define the range within which an effect is considered
/// practically equivalent to zero (or the null value).
#[derive(Debug, Clone)]
pub enum EquivalenceBounds {
    /// Raw bounds in the original units of measurement.
    ///
    /// # Example
    /// Testing if a mean difference is within [-5, 5] points.
    Raw { lower: f64, upper: f64 },

    /// Symmetric bounds specified as ±delta in raw units.
    ///
    /// # Example
    /// `Symmetric { delta: 0.5 }` creates bounds [-0.5, 0.5]
    Symmetric { delta: f64 },

    /// Bounds specified as Cohen's d effect size.
    ///
    /// The bounds are ±d standard deviations. These are converted
    /// to raw units using the pooled standard deviation.
    ///
    /// Common values:
    /// - d = 0.2: small effect
    /// - d = 0.5: medium effect
    /// - d = 0.8: large effect
    ///
    /// For equivalence testing, d = 0.5 is a common default (SESOI).
    CohenD { d: f64 },
}

impl EquivalenceBounds {
    /// Create symmetric raw bounds.
    pub fn symmetric(delta: f64) -> Result<Self> {
        if delta <= 0.0 {
            return Err(StatError::InvalidParameter(
                "delta must be positive".to_string(),
            ));
        }
        Ok(Self::Symmetric { delta })
    }

    /// Create asymmetric raw bounds.
    pub fn raw(lower: f64, upper: f64) -> Result<Self> {
        if lower >= upper {
            return Err(StatError::InvalidParameter(format!(
                "lower bound ({}) must be less than upper bound ({})",
                lower, upper
            )));
        }
        Ok(Self::Raw { lower, upper })
    }

    /// Create Cohen's d based bounds.
    pub fn cohen_d(d: f64) -> Result<Self> {
        if d <= 0.0 {
            return Err(StatError::InvalidParameter(
                "Cohen's d must be positive".to_string(),
            ));
        }
        Ok(Self::CohenD { d })
    }

    /// Convert bounds to raw (lower, upper) values.
    ///
    /// For Cohen's d bounds, requires the pooled standard deviation.
    /// For correlation bounds, sd is ignored.
    pub fn to_raw(&self, sd: Option<f64>) -> Result<(f64, f64)> {
        match self {
            Self::Raw { lower, upper } => Ok((*lower, *upper)),
            Self::Symmetric { delta } => Ok((-*delta, *delta)),
            Self::CohenD { d } => {
                let sd = sd.ok_or_else(|| {
                    StatError::InvalidParameter(
                        "Standard deviation required for Cohen's d bounds".to_string(),
                    )
                })?;
                if sd <= 0.0 {
                    return Err(StatError::InvalidParameter(
                        "Standard deviation must be positive".to_string(),
                    ));
                }
                let delta = d * sd;
                Ok((-delta, delta))
            }
        }
    }

    /// Validate that the bounds are appropriate.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Raw { lower, upper } => {
                if *lower >= *upper {
                    return Err(StatError::InvalidParameter(format!(
                        "lower bound ({}) must be less than upper bound ({})",
                        lower, upper
                    )));
                }
            }
            Self::Symmetric { delta } => {
                if *delta <= 0.0 {
                    return Err(StatError::InvalidParameter(
                        "delta must be positive".to_string(),
                    ));
                }
            }
            Self::CohenD { d } => {
                if *d <= 0.0 {
                    return Err(StatError::InvalidParameter(
                        "Cohen's d must be positive".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for EquivalenceBounds {
    /// Default to Cohen's d = 0.5 (medium effect size).
    fn default() -> Self {
        Self::CohenD { d: 0.5 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_bounds() {
        let bounds = EquivalenceBounds::symmetric(0.5).unwrap();
        let (lower, upper) = bounds.to_raw(None).unwrap();
        assert!((lower - (-0.5)).abs() < 1e-10);
        assert!((upper - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_raw_bounds() {
        let bounds = EquivalenceBounds::raw(-0.3, 0.7).unwrap();
        let (lower, upper) = bounds.to_raw(None).unwrap();
        assert!((lower - (-0.3)).abs() < 1e-10);
        assert!((upper - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_d_bounds() {
        let bounds = EquivalenceBounds::cohen_d(0.5).unwrap();
        let (lower, upper) = bounds.to_raw(Some(2.0)).unwrap();
        assert!((lower - (-1.0)).abs() < 1e-10);
        assert!((upper - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_bounds() {
        assert!(EquivalenceBounds::symmetric(-0.5).is_err());
        assert!(EquivalenceBounds::raw(0.5, -0.5).is_err());
        assert!(EquivalenceBounds::cohen_d(0.0).is_err());
    }

    #[test]
    fn test_cohen_d_requires_sd() {
        let bounds = EquivalenceBounds::cohen_d(0.5).unwrap();
        assert!(bounds.to_raw(None).is_err());
    }
}
