//! Correlation tests
//!
//! - Pearson correlation
//! - Spearman rank correlation
//! - Kendall's tau
//! - Partial correlation
//! - Semi-partial correlation
//! - Distance correlation
//! - Intraclass correlation (ICC)

use super::{convert_error, CorrelationResult, TestResult};
use crate::{StatsError, StatsResult};
use anofox_tests::{
    distance_cor as lib_distance_cor, distance_cor_test as lib_distance_cor_test, icc as lib_icc,
    kendall as lib_kendall, partial_cor as lib_partial_cor, pearson as lib_pearson,
    semi_partial_cor as lib_semi_partial_cor, spearman as lib_spearman, Alternative, ICCType,
    KendallVariant,
};

/// Options for Pearson correlation
#[derive(Debug, Clone)]
pub struct PearsonOptions {
    /// Confidence level
    pub confidence_level: Option<f64>,
}

impl Default for PearsonOptions {
    fn default() -> Self {
        Self {
            confidence_level: Some(0.95),
        }
    }
}

/// Pearson product-moment correlation
pub fn pearson(x: &[f64], y: &[f64], options: &PearsonOptions) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Pearson correlation requires equal length vectors".into(),
        ));
    }

    // Filter paired values
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Pearson correlation requires at least 3 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result =
        lib_pearson(&x_filtered, &y_filtered, options.confidence_level).map_err(convert_error)?;

    Ok(CorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: x_filtered.len(),
        method: "Pearson correlation".into(),
    })
}

/// Options for Spearman correlation
#[derive(Debug, Clone)]
pub struct SpearmanOptions {
    /// Confidence level
    pub confidence_level: Option<f64>,
}

impl Default for SpearmanOptions {
    fn default() -> Self {
        Self {
            confidence_level: Some(0.95),
        }
    }
}

/// Spearman rank correlation
pub fn spearman(x: &[f64], y: &[f64], options: &SpearmanOptions) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Spearman correlation requires equal length vectors".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Spearman correlation requires at least 3 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result =
        lib_spearman(&x_filtered, &y_filtered, options.confidence_level).map_err(convert_error)?;

    Ok(CorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: options.confidence_level.unwrap_or(0.95),
        n: x_filtered.len(),
        method: "Spearman rank correlation".into(),
    })
}

/// Options for Kendall's tau
#[derive(Debug, Clone)]
pub struct KendallOptions {
    /// Tau variant (TauA, TauB, TauC)
    pub variant: KendallVariant,
}

impl Default for KendallOptions {
    fn default() -> Self {
        Self {
            variant: KendallVariant::TauB,
        }
    }
}

/// Kendall's tau correlation
pub fn kendall(x: &[f64], y: &[f64], options: &KendallOptions) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Kendall correlation requires equal length vectors".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Kendall correlation requires at least 3 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result = lib_kendall(&x_filtered, &y_filtered, options.variant).map_err(convert_error)?;

    Ok(CorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        ci_lower: result
            .conf_int
            .as_ref()
            .map(|ci| ci.lower)
            .unwrap_or(f64::NAN),
        ci_upper: result
            .conf_int
            .as_ref()
            .map(|ci| ci.upper)
            .unwrap_or(f64::NAN),
        confidence_level: f64::NAN,
        n: x_filtered.len(),
        method: format!("Kendall's {:?}", options.variant),
    })
}

/// Partial correlation
///
/// Correlation between x and y controlling for z variables.
pub fn partial_cor(x: &[f64], y: &[f64], z: &[Vec<f64>]) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Partial correlation requires equal length vectors".into(),
        ));
    }

    for (i, zi) in z.iter().enumerate() {
        if zi.len() != x.len() {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Control variable {} has different length",
                i
            )));
        }
    }

    let n = x.len();
    if n < z.len() + 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Partial correlation requires n > k + 2 where k is number of control variables".into(),
        ));
    }

    // Filter NaN across all variables
    let valid_indices: Vec<usize> = (0..n)
        .filter(|&i| !x[i].is_nan() && !y[i].is_nan() && z.iter().all(|zi| !zi[i].is_nan()))
        .collect();

    if valid_indices.len() < z.len() + 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Not enough valid observations after removing NaN".into(),
        ));
    }

    let x_filtered: Vec<f64> = valid_indices.iter().map(|&i| x[i]).collect();
    let y_filtered: Vec<f64> = valid_indices.iter().map(|&i| y[i]).collect();
    let z_filtered: Vec<Vec<f64>> = z
        .iter()
        .map(|zi| valid_indices.iter().map(|&i| zi[i]).collect())
        .collect();

    let z_refs: Vec<&[f64]> = z_filtered.iter().map(|v| v.as_slice()).collect();

    let result = lib_partial_cor(&x_filtered, &y_filtered, &z_refs).map_err(convert_error)?;

    Ok(CorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: x_filtered.len(),
        method: format!(
            "Partial correlation (controlling for {} variables)",
            z.len()
        ),
    })
}

/// Semi-partial (part) correlation
pub fn semi_partial_cor(x: &[f64], y: &[f64], z: &[Vec<f64>]) -> StatsResult<CorrelationResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Semi-partial correlation requires equal length vectors".into(),
        ));
    }

    for (i, zi) in z.iter().enumerate() {
        if zi.len() != x.len() {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Control variable {} has different length",
                i
            )));
        }
    }

    let n = x.len();
    let valid_indices: Vec<usize> = (0..n)
        .filter(|&i| !x[i].is_nan() && !y[i].is_nan() && z.iter().all(|zi| !zi[i].is_nan()))
        .collect();

    if valid_indices.len() < z.len() + 3 {
        return Err(StatsError::InsufficientDataMsg(
            "Not enough valid observations after removing NaN".into(),
        ));
    }

    let x_filtered: Vec<f64> = valid_indices.iter().map(|&i| x[i]).collect();
    let y_filtered: Vec<f64> = valid_indices.iter().map(|&i| y[i]).collect();
    let z_filtered: Vec<Vec<f64>> = z
        .iter()
        .map(|zi| valid_indices.iter().map(|&i| zi[i]).collect())
        .collect();

    let z_refs: Vec<&[f64]> = z_filtered.iter().map(|v| v.as_slice()).collect();

    let result = lib_semi_partial_cor(&x_filtered, &y_filtered, &z_refs).map_err(convert_error)?;

    Ok(CorrelationResult {
        r: result.estimate,
        statistic: result.statistic,
        p_value: result.p_value,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: x_filtered.len(),
        method: format!(
            "Semi-partial correlation (controlling for {} variables on y)",
            z.len()
        ),
    })
}

/// Distance correlation result
#[derive(Debug, Clone)]
pub struct DistanceCorResult {
    /// Distance correlation
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

/// Distance correlation
pub fn distance_cor(x: &[f64], y: &[f64]) -> StatsResult<DistanceCorResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Distance correlation requires equal length vectors".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "Distance correlation requires at least 4 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result = lib_distance_cor(&x_filtered, &y_filtered).map_err(convert_error)?;

    Ok(DistanceCorResult {
        dcor: result.dcor,
        dcov: result.dcov,
        dvar_x: result.dvar_x,
        dvar_y: result.dvar_y,
        n: x_filtered.len(),
    })
}

/// Options for distance correlation test
#[derive(Debug, Clone)]
pub struct DistanceCorTestOptions {
    /// Number of permutations
    pub n_permutations: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for DistanceCorTestOptions {
    fn default() -> Self {
        Self {
            n_permutations: 1000,
            seed: None,
        }
    }
}

/// Distance correlation significance test
pub fn distance_cor_test(
    x: &[f64],
    y: &[f64],
    options: &DistanceCorTestOptions,
) -> StatsResult<TestResult> {
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatchMsg(
            "Distance correlation test requires equal length vectors".into(),
        ));
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    if pairs.len() < 4 {
        return Err(StatsError::InsufficientDataMsg(
            "Distance correlation test requires at least 4 valid pairs".into(),
        ));
    }

    let (x_filtered, y_filtered): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let result = lib_distance_cor_test(
        &x_filtered,
        &y_filtered,
        options.n_permutations,
        options.seed,
    )
    .map_err(convert_error)?;

    Ok(TestResult {
        statistic: result.dcor,
        p_value: result.p_value.unwrap_or(f64::NAN),
        df: f64::NAN,
        effect_size: result.dcor,
        ci_lower: f64::NAN,
        ci_upper: f64::NAN,
        confidence_level: f64::NAN,
        n: x_filtered.len(),
        n1: 0,
        n2: 0,
        alternative: Alternative::TwoSided,
        method: format!(
            "Distance correlation test ({} permutations)",
            options.n_permutations
        ),
    })
}

/// ICC result
#[derive(Debug, Clone)]
pub struct ICCResult {
    /// ICC value
    pub icc: f64,
    /// F statistic
    pub f_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// ICC type
    pub icc_type: String,
}

/// Intraclass correlation coefficient
pub fn icc(data: &[Vec<f64>], icc_type: ICCType) -> StatsResult<ICCResult> {
    if data.is_empty() {
        return Err(StatsError::InsufficientDataMsg(
            "ICC requires at least one subject".into(),
        ));
    }

    let n_raters = data[0].len();
    if n_raters < 2 {
        return Err(StatsError::InsufficientDataMsg(
            "ICC requires at least 2 raters/measurements".into(),
        ));
    }

    for (i, row) in data.iter().enumerate() {
        if row.len() != n_raters {
            return Err(StatsError::DimensionMismatchMsg(format!(
                "Subject {} has different number of measurements",
                i
            )));
        }
    }

    let result = lib_icc(data, icc_type).map_err(convert_error)?;

    Ok(ICCResult {
        icc: result.icc,
        f_statistic: result.f_value,
        p_value: result.p_value,
        ci_lower: result.conf_int_lower,
        ci_upper: result.conf_int_upper,
        icc_type: format!("{:?}", icc_type),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let opts = PearsonOptions::default();
        let result = pearson(&x, &y, &opts).unwrap();

        assert_relative_eq!(result.r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let opts = SpearmanOptions::default();
        let result = spearman(&x, &y, &opts).unwrap();

        assert_relative_eq!(result.r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kendall() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let opts = KendallOptions::default();
        let result = kendall(&x, &y, &opts).unwrap();

        assert_relative_eq!(result.r, 1.0, epsilon = 1e-10);
    }
}
