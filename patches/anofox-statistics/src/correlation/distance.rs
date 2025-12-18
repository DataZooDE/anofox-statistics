//! Distance correlation and distance covariance.
//!
//! Distance correlation is a measure of dependence between random vectors
//! that is zero if and only if the vectors are independent.

use crate::error::{Result, StatError};

/// Result of a distance correlation test
#[derive(Debug, Clone)]
pub struct DistanceCorResult {
    /// Distance correlation coefficient (0 to 1)
    pub dcor: f64,
    /// Distance covariance
    pub dcov: f64,
    /// Distance variance of X
    pub dvar_x: f64,
    /// Distance variance of Y
    pub dvar_y: f64,
    /// Test statistic (for permutation test)
    pub statistic: f64,
    /// p-value (from permutation test, if computed)
    pub p_value: Option<f64>,
    /// Number of observations
    pub n: usize,
    /// Method name
    pub method: String,
}

/// Compute distance correlation between two vectors.
///
/// Distance correlation is a measure of dependence between random variables
/// that is zero if and only if the variables are independent. Unlike Pearson
/// correlation, it can detect non-linear dependencies.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// * `DistanceCorResult` containing dCor, dCov, and dVar values
///
/// # Examples
/// ```
/// use anofox_statistics::correlation::distance_cor;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]; // y = x^2
///
/// let result = distance_cor(&x, &y).unwrap();
/// println!("Distance correlation = {:.4}", result.dcor);
/// ```
///
/// # R equivalent
/// `energy::dcor(x, y)`
pub fn distance_cor(x: &[f64], y: &[f64]) -> Result<DistanceCorResult> {
    validate_input(x, y)?;

    let n = x.len();

    // Compute distance matrices
    let a = distance_matrix(x);
    let b = distance_matrix(y);

    // Double-center the distance matrices
    let a_centered = double_center(&a);
    let b_centered = double_center(&b);

    // Compute distance covariance and variances
    let dcov_sq = distance_covariance_sq(&a_centered, &b_centered);
    let dvar_x_sq = distance_covariance_sq(&a_centered, &a_centered);
    let dvar_y_sq = distance_covariance_sq(&b_centered, &b_centered);

    let dcov = dcov_sq.max(0.0).sqrt();
    let dvar_x = dvar_x_sq.max(0.0).sqrt();
    let dvar_y = dvar_y_sq.max(0.0).sqrt();

    // Distance correlation
    let dcor = if dvar_x > 0.0 && dvar_y > 0.0 {
        (dcov_sq / (dvar_x_sq * dvar_y_sq).sqrt()).max(0.0).sqrt()
    } else {
        0.0
    };

    // Clamp to [0, 1]
    let dcor = dcor.clamp(0.0, 1.0);

    // Test statistic (n * dCov^2 is asymptotically chi-squared under independence)
    let statistic = n as f64 * dcov_sq;

    Ok(DistanceCorResult {
        dcor,
        dcov,
        dvar_x,
        dvar_y,
        statistic,
        p_value: None,
        n,
        method: "Distance correlation".to_string(),
    })
}

/// Compute distance correlation with permutation test for significance.
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
/// * `n_permutations` - Number of permutations for the test
/// * `seed` - Optional seed for reproducibility
///
/// # Returns
/// * `DistanceCorResult` with p-value from permutation test
///
/// # R equivalent
/// `energy::dcor.test(x, y, R = n_permutations)`
pub fn distance_cor_test(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<DistanceCorResult> {
    validate_input(x, y)?;

    let n = x.len();

    // Compute observed distance correlation
    let observed = distance_cor(x, y)?;
    let observed_stat = observed.statistic;

    // Permutation test
    let mut rng = SimpleRng::new(seed.unwrap_or(12345));
    let mut count_greater = 0usize;

    let mut y_perm: Vec<f64> = y.to_vec();

    for _ in 0..n_permutations {
        // Permute y
        fisher_yates_shuffle(&mut y_perm, &mut rng);

        // Compute test statistic for permuted data
        let perm_result = distance_cor(x, &y_perm)?;
        if perm_result.statistic >= observed_stat {
            count_greater += 1;
        }
    }

    let p_value = (count_greater as f64 + 1.0) / (n_permutations as f64 + 1.0);

    Ok(DistanceCorResult {
        dcor: observed.dcor,
        dcov: observed.dcov,
        dvar_x: observed.dvar_x,
        dvar_y: observed.dvar_y,
        statistic: observed_stat,
        p_value: Some(p_value),
        n,
        method: format!(
            "Distance correlation test ({} permutations)",
            n_permutations
        ),
    })
}

/// Validate input vectors for distance correlation.
fn validate_input(x: &[f64], y: &[f64]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    if x.len() != y.len() {
        return Err(StatError::InvalidParameter(format!(
            "x and y must have same length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.len() < 3 {
        return Err(StatError::InsufficientData {
            needed: 3,
            got: x.len(),
        });
    }

    // Check for non-finite values
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if !xi.is_finite() || !yi.is_finite() {
            return Err(StatError::InvalidParameter(format!(
                "Non-finite value at index {}: x={}, y={}",
                i, xi, yi
            )));
        }
    }

    Ok(())
}

/// Compute pairwise Euclidean distance matrix for a 1D vector.
fn distance_matrix(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut d = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            d[i][j] = (x[i] - x[j]).abs();
        }
    }

    d
}

/// Double-center a distance matrix.
/// A_ij = a_ij - mean(a_i.) - mean(a_.j) + mean(a_..)
fn double_center(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();

    // Row means
    let row_means: Vec<f64> = a
        .iter()
        .map(|row| row.iter().sum::<f64>() / n as f64)
        .collect();

    // Column means
    let col_means: Vec<f64> = (0..n)
        .map(|j| a.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();

    // Grand mean
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

    // Double-centered matrix
    let mut centered = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            centered[i][j] = a[i][j] - row_means[i] - col_means[j] + grand_mean;
        }
    }

    centered
}

/// Compute squared distance covariance from double-centered matrices.
fn distance_covariance_sq(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let n_sq = (n * n) as f64;

    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += a[i][j] * b[i][j];
        }
    }

    sum / n_sq
}

/// Simple random number generator (xorshift64).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// Fisher-Yates shuffle.
fn fisher_yates_shuffle(arr: &mut [f64], rng: &mut SimpleRng) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = rng.next_usize(i + 1);
        arr.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_cor_linear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = distance_cor(&x, &y).unwrap();

        // Perfect linear relationship should give dCor close to 1
        assert!(result.dcor > 0.99);
    }

    #[test]
    fn test_distance_cor_nonlinear() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = distance_cor(&x, &y).unwrap();

        // Quadratic relationship should still give high dCor
        assert!(result.dcor > 0.9);
    }

    #[test]
    fn test_distance_cor_weak() {
        // Data with weak relationship
        let x = vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 10.0];
        let y = vec![3.0, 7.0, 1.0, 6.0, 9.0, 2.0, 8.0, 4.0, 10.0, 5.0];

        let result = distance_cor(&x, &y).unwrap();

        // Distance correlation is always between 0 and 1
        assert!(result.dcor >= 0.0 && result.dcor <= 1.0);
        // With small samples, even "independent" data can show moderate dCor
        // Just verify it's less than perfect correlation
        assert!(result.dcor < 0.99);
    }

    #[test]
    fn test_distance_cor_symmetric() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result_xy = distance_cor(&x, &y).unwrap();
        let result_yx = distance_cor(&y, &x).unwrap();

        // Distance correlation is symmetric
        assert!((result_xy.dcor - result_yx.dcor).abs() < 1e-10);
    }

    #[test]
    fn test_distance_cor_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0).collect();

        let result = distance_cor_test(&x, &y, 99, Some(42)).unwrap();

        // Strong relationship should give small p-value
        assert!(result.p_value.unwrap() < 0.05);
        assert!(result.dcor > 0.9);
    }

    #[test]
    fn test_distance_cor_bounds() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 1.0, 4.0, 3.0, 5.0];

        let result = distance_cor(&x, &y).unwrap();

        // Distance correlation is always between 0 and 1
        assert!(result.dcor >= 0.0);
        assert!(result.dcor <= 1.0);
    }
}
