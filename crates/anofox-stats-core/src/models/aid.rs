//! AID (Automatic Identification of Demand) implementation
//!
//! Provides demand classification and anomaly detection for time series data,
//! distinguishing between regular and intermittent demand patterns.

use crate::errors::{StatsError, StatsResult};
use crate::types::{AidAnomalyFlags, AidOptions, AidResult, OutlierMethod};

/// Compute AID (Automatic Identification of Demand) classification
///
/// Classifies demand patterns as regular or intermittent, identifies the best-fit
/// distribution, and counts various anomaly types.
///
/// # Arguments
/// * `y` - Time series of demand values (must preserve order)
/// * `options` - Classification options
///
/// # Returns
/// * `AidResult` containing classification and anomaly counts
pub fn compute_aid(y: &[f64], options: &AidOptions) -> StatsResult<AidResult> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }

    // Filter out NaN values but track valid indices
    let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid_values.len();

    if n == 0 {
        return Err(StatsError::NoValidData);
    }

    // Basic statistics
    let sum: f64 = valid_values.iter().sum();
    let mean = sum / n as f64;

    let variance = if n > 1 {
        let sq_sum: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum();
        sq_sum / (n - 1) as f64
    } else {
        0.0
    };

    // Zero proportion
    let zero_count = valid_values.iter().filter(|&&v| v == 0.0).count();
    let zero_proportion = zero_count as f64 / n as f64;

    // Demand type classification
    let is_intermittent = zero_proportion >= options.intermittent_threshold;
    let demand_type = if is_intermittent {
        "intermittent".to_string()
    } else {
        "regular".to_string()
    };

    // Compute anomaly flags for each observation
    let anomaly_flags = compute_anomaly_flags(y, options);

    // Count anomalies
    let mut stockout_count = 0usize;
    let mut new_product_count = 0usize;
    let mut obsolete_product_count = 0usize;
    let mut high_outlier_count = 0usize;
    let mut low_outlier_count = 0usize;

    for flags in &anomaly_flags {
        if flags.stockout {
            stockout_count += 1;
        }
        if flags.new_product {
            new_product_count += 1;
        }
        if flags.obsolete_product {
            obsolete_product_count += 1;
        }
        if flags.high_outlier {
            high_outlier_count += 1;
        }
        if flags.low_outlier {
            low_outlier_count += 1;
        }
    }

    // Determine pattern flags
    let has_stockouts = stockout_count > 0;
    let is_new_product = new_product_count > 0;
    let is_obsolete_product = obsolete_product_count > 0;

    // Select best-fit distribution
    let distribution = select_distribution(&valid_values, mean, variance, is_intermittent);

    Ok(AidResult {
        demand_type,
        is_intermittent,
        distribution,
        mean,
        variance,
        zero_proportion,
        n_observations: n,
        has_stockouts,
        is_new_product,
        is_obsolete_product,
        stockout_count,
        new_product_count,
        obsolete_product_count,
        high_outlier_count,
        low_outlier_count,
    })
}

/// Compute per-observation anomaly flags
///
/// Returns a vector of anomaly flags for each observation in the input,
/// maintaining the same order as the input.
///
/// # Arguments
/// * `y` - Time series of demand values (must preserve order)
/// * `options` - Classification options
///
/// # Returns
/// * Vector of `AidAnomalyFlags` for each input observation
pub fn compute_aid_anomalies(y: &[f64], options: &AidOptions) -> StatsResult<Vec<AidAnomalyFlags>> {
    if y.is_empty() {
        return Err(StatsError::EmptyInput { field: "y" });
    }

    Ok(compute_anomaly_flags(y, options))
}

/// Internal function to compute anomaly flags for each observation
fn compute_anomaly_flags(y: &[f64], options: &AidOptions) -> Vec<AidAnomalyFlags> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }

    // Compute statistics for outlier detection (excluding NaN)
    let valid_values: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    if valid_values.is_empty() {
        return y
            .iter()
            .map(|_| AidAnomalyFlags::default())
            .collect();
    }

    let mean: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    let std_dev = if valid_values.len() > 1 {
        let variance: f64 = valid_values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / (valid_values.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    // Compute outlier bounds based on method
    let (high_threshold, low_threshold) = match options.outlier_method {
        OutlierMethod::ZScore => {
            let high = mean + 3.0 * std_dev;
            let low = (mean - 3.0 * std_dev).max(0.0); // Don't flag zeros as low outliers
            (high, low)
        }
        OutlierMethod::Iqr => {
            let mut sorted = valid_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let q1_idx = sorted.len() / 4;
            let q3_idx = (3 * sorted.len()) / 4;
            let q1 = sorted.get(q1_idx).copied().unwrap_or(0.0);
            let q3 = sorted.get(q3_idx).copied().unwrap_or(mean);
            let iqr = q3 - q1;

            let high = q3 + 1.5 * iqr;
            let low = (q1 - 1.5 * iqr).max(0.0);
            (high, low)
        }
    };

    // Identify leading zeros (new product pattern)
    let mut first_nonzero_idx: Option<usize> = None;
    for (i, &val) in y.iter().enumerate() {
        if val.is_finite() && val != 0.0 {
            first_nonzero_idx = Some(i);
            break;
        }
    }

    // Identify trailing zeros (obsolete product pattern)
    let mut last_nonzero_idx: Option<usize> = None;
    for (i, &val) in y.iter().enumerate().rev() {
        if val.is_finite() && val != 0.0 {
            last_nonzero_idx = Some(i);
            break;
        }
    }

    // Build anomaly flags for each observation
    y.iter()
        .enumerate()
        .map(|(i, &val)| {
            if !val.is_finite() {
                return AidAnomalyFlags::default();
            }

            let is_zero = val == 0.0;

            // New product: leading zeros (before first non-zero)
            let new_product = match first_nonzero_idx {
                Some(first_idx) => is_zero && i < first_idx,
                None => false, // All zeros - not a new product pattern
            };

            // Obsolete product: trailing zeros (after last non-zero)
            let obsolete_product = match last_nonzero_idx {
                Some(last_idx) => is_zero && i > last_idx,
                None => false, // All zeros - not obsolete pattern
            };

            // Stockout: zero occurring between non-zeros (not leading or trailing)
            let stockout = match (first_nonzero_idx, last_nonzero_idx) {
                (Some(first_idx), Some(last_idx)) => {
                    is_zero && i > first_idx && i < last_idx
                }
                _ => false,
            };

            // Outlier detection (only for non-zero finite values)
            let high_outlier = !is_zero && val > high_threshold;
            let low_outlier = !is_zero && val < low_threshold && low_threshold > 0.0;

            AidAnomalyFlags {
                stockout,
                new_product,
                obsolete_product,
                high_outlier,
                low_outlier,
            }
        })
        .collect()
}

/// Select the best-fit distribution based on data characteristics
fn select_distribution(values: &[f64], mean: f64, variance: f64, is_intermittent: bool) -> String {
    if values.is_empty() {
        return "unknown".to_string();
    }

    // Check if data is count-like (non-negative integers)
    let is_count_data = values.iter().all(|&v| v >= 0.0 && v == v.floor());

    // Non-zero values for distribution fitting
    let nonzero_values: Vec<f64> = values.iter().copied().filter(|&v| v > 0.0).collect();

    if is_count_data {
        // Count data distributions
        if is_intermittent {
            // High zero proportion suggests Negative Binomial or Geometric
            // Check for overdispersion (variance > mean)
            if variance > mean * 1.5 {
                return "negative_binomial".to_string();
            } else {
                return "geometric".to_string();
            }
        } else {
            // Regular count data
            // Poisson if variance ≈ mean, otherwise Negative Binomial
            let dispersion_ratio = if mean > 0.0 { variance / mean } else { 1.0 };
            if dispersion_ratio < 1.5 {
                return "poisson".to_string();
            } else {
                return "negative_binomial".to_string();
            }
        }
    } else {
        // Continuous data distributions
        if nonzero_values.is_empty() {
            return "normal".to_string();
        }

        // Check for positive-only data
        let all_positive = values.iter().all(|&v| v >= 0.0);

        if all_positive {
            // Check skewness to decide between gamma/lognormal
            let skewness = compute_skewness(&nonzero_values, mean, variance.sqrt());

            if skewness > 1.0 {
                // Highly right-skewed -> lognormal
                return "lognormal".to_string();
            } else if skewness > 0.5 {
                // Moderately skewed -> gamma
                return "gamma".to_string();
            } else if is_intermittent {
                // Low skew but intermittent -> rectified_normal
                return "rectified_normal".to_string();
            } else {
                return "normal".to_string();
            }
        } else {
            // Can have negative values -> normal
            return "normal".to_string();
        }
    }
}

/// Compute sample skewness
fn compute_skewness(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if values.len() < 3 || std_dev == 0.0 {
        return 0.0;
    }

    let n = values.len() as f64;
    let m3: f64 = values.iter().map(|&v| ((v - mean) / std_dev).powi(3)).sum();

    // Sample skewness with bias correction
    let skewness = (m3 / n) * (n * (n - 1.0)).sqrt() / (n - 2.0);
    skewness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aid_regular_demand() {
        // Regular demand: low zero proportion
        let y = vec![10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 14.0, 10.0, 13.0, 11.0];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options).unwrap();

        assert_eq!(result.demand_type, "regular");
        assert!(!result.is_intermittent);
        assert_eq!(result.n_observations, 10);
        assert!((result.zero_proportion - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_aid_intermittent_demand() {
        // Intermittent demand: high zero proportion (40%)
        let y = vec![0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 0.0, 6.0];
        let options = AidOptions::default(); // threshold = 0.3

        let result = compute_aid(&y, &options).unwrap();

        assert_eq!(result.demand_type, "intermittent");
        assert!(result.is_intermittent);
        assert!((result.zero_proportion - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_aid_stockout_detection() {
        // Stockout: zeros in the middle
        let y = vec![5.0, 6.0, 0.0, 0.0, 7.0, 8.0];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options).unwrap();

        assert!(result.has_stockouts);
        assert_eq!(result.stockout_count, 2);
    }

    #[test]
    fn test_aid_new_product_detection() {
        // New product: leading zeros
        let y = vec![0.0, 0.0, 0.0, 5.0, 6.0, 7.0];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options).unwrap();

        assert!(result.is_new_product);
        assert_eq!(result.new_product_count, 3);
    }

    #[test]
    fn test_aid_obsolete_product_detection() {
        // Obsolete product: trailing zeros
        let y = vec![5.0, 6.0, 7.0, 0.0, 0.0, 0.0];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options).unwrap();

        assert!(result.is_obsolete_product);
        assert_eq!(result.obsolete_product_count, 3);
    }

    #[test]
    fn test_aid_outlier_detection_zscore() {
        // High outlier detection - need large sample and extreme outlier
        // With many similar values, std dev is small, so outlier is detected
        let mut y = vec![10.0; 20]; // 20 values of 10.0
        y.push(100.0); // Add extreme outlier

        let options = AidOptions {
            outlier_method: OutlierMethod::ZScore,
            ..Default::default()
        };

        let result = compute_aid(&y, &options).unwrap();

        assert!(result.high_outlier_count > 0);
    }

    #[test]
    fn test_aid_anomaly_flags() {
        let y = vec![0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0];
        let options = AidOptions::default();

        let flags = compute_aid_anomalies(&y, &options).unwrap();

        assert_eq!(flags.len(), 7);

        // First two zeros are new_product
        assert!(flags[0].new_product);
        assert!(flags[1].new_product);

        // Middle zero is stockout
        assert!(flags[3].stockout);

        // Last two zeros are obsolete_product
        assert!(flags[5].obsolete_product);
        assert!(flags[6].obsolete_product);
    }

    #[test]
    fn test_aid_count_data_distribution() {
        // Count-like data (integers)
        let y = vec![2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 5.0, 2.0, 3.0, 4.0];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options).unwrap();

        // Should select a count distribution
        assert!(
            result.distribution == "poisson" || result.distribution == "negative_binomial"
        );
    }

    #[test]
    fn test_aid_empty_input() {
        let y: Vec<f64> = vec![];
        let options = AidOptions::default();

        let result = compute_aid(&y, &options);
        assert!(result.is_err());
    }
}
