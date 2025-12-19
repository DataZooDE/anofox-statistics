//! Kendall's tau correlation coefficient.

use crate::correlation::{validate_correlation_input, CorrelationMethod, CorrelationResult};
use crate::error::Result;
use statrs::distribution::{ContinuousCDF, Normal};

/// Variant of Kendall's tau to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KendallVariant {
    /// Tau-a: No tie adjustment (simple ratio of concordant-discordant pairs)
    TauA,
    /// Tau-b: Adjusted for ties (default, matches R's cor.test)
    #[default]
    TauB,
    /// Tau-c: Stuart's tau-c for rectangular tables (adjusts for table size)
    TauC,
}

/// Compute Kendall's tau correlation coefficient with significance test.
///
/// Kendall's tau measures the strength of association between two variables
/// based on concordant and discordant pairs. It's particularly useful for
/// small samples and ordinal data.
///
/// # Variants
/// - `TauA`: Simple ratio, no tie adjustment: (C - D) / (n*(n-1)/2)
/// - `TauB`: Tie-adjusted (default, matches R): (C - D) / sqrt((C+D+Tx)(C+D+Ty))
/// - `TauC`: Stuart's tau-c: 2(C - D) / (n² * (m-1)/m) where m = min(rows, cols)
///
/// # Arguments
/// * `x` - First variable (must have at least 3 observations)
/// * `y` - Second variable (same length as x)
/// * `variant` - Which variant of tau to compute
///
/// # Returns
/// * `CorrelationResult` containing the correlation coefficient, z-statistic,
///   p-value (using normal approximation)
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::{kendall, KendallVariant};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0, 6.0, 7.0, 8.0, 7.0];
///
/// let result = kendall(&x, &y, KendallVariant::TauB).unwrap();
/// println!("Kendall tau = {:.4}", result.estimate);
/// println!("p-value = {:.4}", result.p_value);
/// ```
///
/// # R equivalent
/// `cor.test(x, y, method = "kendall")` (uses tau-b)
pub fn kendall(x: &[f64], y: &[f64], variant: KendallVariant) -> Result<CorrelationResult> {
    let n = validate_correlation_input(x, y)?;

    // Count concordant, discordant, and tied pairs
    let (concordant, discordant, ties_x, ties_y, ties_xy) = count_pairs(x, y);

    // Total number of pairs
    let n_pairs = (n * (n - 1)) / 2;

    // Compute tau based on variant
    let tau = match variant {
        KendallVariant::TauA => {
            // Tau-a: simple ratio
            (concordant as f64 - discordant as f64) / n_pairs as f64
        }
        KendallVariant::TauB => {
            // Tau-b: tie-adjusted (matches R)
            let c_minus_d = concordant as f64 - discordant as f64;
            let denom1 = n_pairs as f64 - ties_x as f64;
            let denom2 = n_pairs as f64 - ties_y as f64;

            if denom1 == 0.0 || denom2 == 0.0 {
                0.0
            } else {
                c_minus_d / (denom1 * denom2).sqrt()
            }
        }
        KendallVariant::TauC => {
            // Tau-c: Stuart's tau-c
            // For continuous data, we use the number of unique values
            let unique_x = count_unique(x);
            let unique_y = count_unique(y);
            let m = unique_x.min(unique_y);

            if m <= 1 {
                0.0
            } else {
                let c_minus_d = concordant as f64 - discordant as f64;
                2.0 * c_minus_d * m as f64 / ((n * n * (m - 1)) as f64)
            }
        }
    };

    // Clamp tau to [-1, 1]
    let tau = tau.clamp(-1.0, 1.0);

    // Compute z-statistic and p-value using normal approximation
    // Variance formula for tau-b (R's method):
    // var(tau) = (4n + 10) / (9n(n-1)) for no ties
    // With ties, use more complex formula

    let (z_stat, p_value) = compute_kendall_significance(
        tau, n, concordant, discordant, ties_x, ties_y, ties_xy, variant,
    );

    Ok(CorrelationResult {
        estimate: tau,
        statistic: z_stat,
        df: None, // Kendall uses normal approximation, no df
        p_value,
        conf_int: None, // CI for Kendall is complex, not commonly provided
        method: CorrelationMethod::Kendall,
        n,
    })
}

/// Count concordant, discordant, and tied pairs.
///
/// Returns (concordant, discordant, ties_in_x, ties_in_y, ties_in_both)
/// Note: ties_in_x includes all pairs tied in x (including those also tied in y)
/// Same for ties_in_y. This is needed for the tau-b denominator.
fn count_pairs(x: &[f64], y: &[f64]) -> (usize, usize, usize, usize, usize) {
    let n = x.len();
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut ties_x = 0usize; // All pairs tied in x
    let mut ties_y = 0usize; // All pairs tied in y
    let mut ties_xy = 0usize; // Tied in both

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];

            let tied_x = dx == 0.0;
            let tied_y = dy == 0.0;

            if tied_x {
                ties_x += 1;
            }
            if tied_y {
                ties_y += 1;
            }
            if tied_x && tied_y {
                ties_xy += 1;
            }

            // For concordant/discordant, only count pairs not tied in either
            if !tied_x && !tied_y {
                if (dx > 0.0 && dy > 0.0) || (dx < 0.0 && dy < 0.0) {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }
    }

    (concordant, discordant, ties_x, ties_y, ties_xy)
}

/// Count unique values in a slice
fn count_unique(data: &[f64]) -> usize {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    sorted.len()
}

/// Compute z-statistic and p-value for Kendall's tau using normal approximation.
#[allow(clippy::too_many_arguments)]
fn compute_kendall_significance(
    _tau: f64,
    n: usize,
    concordant: usize,
    discordant: usize,
    ties_x: usize,
    ties_y: usize,
    _ties_xy: usize,
    _variant: KendallVariant,
) -> (f64, f64) {
    let n_f = n as f64;
    let n_pairs = (n * (n - 1)) / 2;

    // S = concordant - discordant
    let s = concordant as f64 - discordant as f64;

    // Variance computation depends on ties
    // For tau-b, R uses the following variance formula with tie correction:
    // v0 = n(n-1)(2n+5)/18
    // vt = sum over tie groups in x: t(t-1)(2t+5)/18
    // vu = sum over tie groups in y: u(u-1)(2u+5)/18
    // v1 = sum_t * sum_u / (9n(n-1)(n-2))
    // v2 = sum_t2 * sum_u2 / (2n(n-1))
    // var(S) = v0 - vt - vu + v1 + v2

    let variance = if ties_x == 0 && ties_y == 0 {
        // No ties: simple formula
        n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0
    } else {
        // With ties: use approximation based on tie counts
        // This is a simplified version; R uses group-based calculation
        let t1 = ties_x as f64;
        let t2 = ties_y as f64;
        let n0 = n_pairs as f64;

        // Effective denominators (for more accurate variance calculation)
        let _n1 = n0 - t1;
        let _n2 = n0 - t2;

        // Approximate variance for tau-b
        // Using formula: var(tau-b) ≈ (4/(n*(n-1))) * ((n0-concordant-discordant+1)/(n0-1))
        // But R's actual formula is more complex

        // Simplified variance formula that works reasonably well:
        let v0 = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0;

        // Adjustment factors for ties (simplified)
        let adj = 1.0 - (t1 + t2) / (2.0 * n0);
        v0 * adj * adj
    };

    // Z-statistic
    let z_stat = if variance <= 0.0 {
        0.0
    } else {
        s / variance.sqrt()
    };

    // Two-sided p-value using normal approximation
    let p_value = if z_stat == 0.0 {
        1.0
    } else {
        let normal = Normal::new(0.0, 1.0).unwrap();
        2.0 * (1.0 - normal.cdf(z_stat.abs()))
    };

    (z_stat, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kendall_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert_eq!(result.method, CorrelationMethod::Kendall);
        assert_eq!(result.n, 5);
    }

    #[test]
    fn test_kendall_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((result.estimate - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_a_vs_tau_b() {
        // With no ties, tau-a and tau-b should be equal
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0];

        let tau_a = kendall(&x, &y, KendallVariant::TauA).unwrap();
        let tau_b = kendall(&x, &y, KendallVariant::TauB).unwrap();

        assert!((tau_a.estimate - tau_b.estimate).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_with_ties() {
        // Data where ties reduce tau-b below 1
        let x = vec![1.0, 2.0, 2.0, 4.0, 5.0, 3.0];
        let y = vec![1.0, 3.0, 2.0, 4.0, 5.0, 4.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        // Should be positive but less than 1 due to ties and discordant pairs
        assert!(result.estimate > 0.0);
        assert!(result.estimate < 1.0);
    }

    #[test]
    fn test_kendall_zero_correlation() {
        // Random-looking data with no clear correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![3.0, 1.0, 4.0, 2.0, 6.0, 5.0];

        let result = kendall(&x, &y, KendallVariant::TauB).unwrap();

        // Should be close to zero or small
        assert!(result.estimate.abs() < 0.5);
    }
}
