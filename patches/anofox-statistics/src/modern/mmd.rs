use crate::error::{Result, StatError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Kernel types supported by the MMD test
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    /// Gaussian (RBF) kernel: k(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))
    Gaussian { bandwidth: f64 },
    /// Linear kernel: k(x,y) = x·y
    Linear,
    /// Polynomial kernel: k(x,y) = (scale * x·y + offset)^degree
    Polynomial {
        degree: u32,
        scale: f64,
        offset: f64,
    },
    /// Laplacian kernel: k(x,y) = exp(-||x-y|| / bandwidth)
    Laplacian { bandwidth: f64 },
}

impl Default for Kernel {
    fn default() -> Self {
        Kernel::Gaussian { bandwidth: 1.0 }
    }
}

/// Result of the MMD test
#[derive(Debug, Clone)]
pub struct MMDResult {
    /// The MMD^2 test statistic (unbiased estimator)
    pub statistic: f64,
    /// The p-value (from permutation test)
    pub p_value: f64,
    /// Number of permutations used
    pub n_permutations: usize,
}

/// Compute the dot product between two vectors.
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the squared Euclidean distance between two vectors.
#[inline]
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Compute the L1 distance between two vectors.
#[inline]
fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Evaluate the kernel function for two points.
fn kernel_value(kernel: Kernel, a: &[f64], b: &[f64]) -> f64 {
    match kernel {
        Kernel::Gaussian { bandwidth } => {
            let sq_dist = squared_distance(a, b);
            (-sq_dist / (2.0 * bandwidth * bandwidth)).exp()
        }
        Kernel::Linear => dot_product(a, b),
        Kernel::Polynomial {
            degree,
            scale,
            offset,
        } => {
            let inner = scale * dot_product(a, b) + offset;
            inner.powi(degree as i32)
        }
        Kernel::Laplacian { bandwidth } => {
            let dist = l1_distance(a, b);
            (-dist / bandwidth).exp()
        }
    }
}

/// Sum kernel values for all pairs within a sample, excluding diagonal.
fn sum_kernel_within(samples: &[&[f64]], kernel: Kernel) -> f64 {
    let n = samples.len();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sum += kernel_value(kernel, samples[i], samples[j]);
            }
        }
    }
    sum
}

/// Sum kernel values for all pairs between two samples.
fn sum_kernel_cross(x: &[&[f64]], y: &[&[f64]], kernel: Kernel) -> f64 {
    let mut sum = 0.0;
    for xi in x.iter() {
        for yj in y.iter() {
            sum += kernel_value(kernel, xi, yj);
        }
    }
    sum
}

/// Compute the unbiased MMD^2 estimator.
///
/// MMD^2 = 1/(m(m-1)) * sum_{i≠j} k(x_i, x_j)
///       + 1/(n(n-1)) * sum_{i≠j} k(y_i, y_j)
///       - 2/(mn) * sum_{i,j} k(x_i, y_j)
fn mmd_squared(x: &[&[f64]], y: &[&[f64]], kernel: Kernel) -> f64 {
    let m = x.len();
    let n = y.len();

    if m < 2 || n < 2 {
        return 0.0;
    }

    let m_f = m as f64;
    let n_f = n as f64;

    let sum_xx = sum_kernel_within(x, kernel);
    let sum_yy = sum_kernel_within(y, kernel);
    let sum_xy = sum_kernel_cross(x, y, kernel);

    sum_xx / (m_f * (m_f - 1.0)) + sum_yy / (n_f * (n_f - 1.0)) - 2.0 * sum_xy / (m_f * n_f)
}

/// Perform the Maximum Mean Discrepancy (MMD) test for equality of distributions.
///
/// MMD is a kernel-based distance between probability distributions in a
/// reproducing kernel Hilbert space (RKHS). The test uses a permutation
/// approach to compute p-values.
///
/// # Arguments
/// * `x` - First sample (each element is a d-dimensional observation)
/// * `y` - Second sample (each element is a d-dimensional observation)
/// * `kernel` - The kernel function to use
/// * `n_permutations` - Number of permutations for p-value estimation
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `MMDResult` containing MMD^2 statistic and p-value
///
/// # References
/// * Gretton, A. et al. (2012). "A Kernel Two-Sample Test"
pub fn mmd_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    kernel: Kernel,
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<MMDResult> {
    // Validate inputs
    validate_mmd_inputs(x, y)?;

    // Convert to slices
    let x_slices: Vec<&[f64]> = x.iter().map(|v| v.as_slice()).collect();
    let y_slices: Vec<&[f64]> = y.iter().map(|v| v.as_slice()).collect();

    // Compute observed MMD^2
    let observed = mmd_squared(&x_slices, &y_slices, kernel);

    // Run permutation test
    let p_value = run_mmd_permutation_test(x, y, kernel, observed, n_permutations, seed);

    Ok(MMDResult {
        statistic: observed,
        p_value,
        n_permutations,
    })
}

/// Validate sample sizes for MMD test.
fn validate_sample_sizes(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }
    if x.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    if y.len() < 2 {
        return Err(StatError::InsufficientData {
            needed: 2,
            got: y.len(),
        });
    }
    Ok(())
}

/// Validate dimensions are consistent across all data points.
fn validate_dimensions(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<()> {
    let dim = x[0].len();
    if dim == 0 {
        return Err(StatError::InvalidParameter(
            "Data points must have at least one dimension".to_string(),
        ));
    }
    let all_same_dim = x.iter().chain(y.iter()).all(|v| v.len() == dim);
    if !all_same_dim {
        return Err(StatError::InvalidParameter(
            "All data points must have the same dimension".to_string(),
        ));
    }
    Ok(())
}

/// Validate MMD test inputs.
fn validate_mmd_inputs(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<()> {
    validate_sample_sizes(x, y)?;
    validate_dimensions(x, y)
}

/// Run the permutation test for MMD.
fn run_mmd_permutation_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    kernel: Kernel,
    observed: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> f64 {
    let combined: Vec<Vec<f64>> = x.iter().chain(y.iter()).cloned().collect();
    let n1 = x.len();

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut indices: Vec<usize> = (0..combined.len()).collect();
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        indices.shuffle(&mut rng);

        let perm_x: Vec<&[f64]> = indices[0..n1]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();
        let perm_y: Vec<&[f64]> = indices[n1..]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();

        let perm_stat = mmd_squared(&perm_x, &perm_y, kernel);

        if perm_stat >= observed {
            count_extreme += 1;
        }
    }

    (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0)
}

/// Convenience function for univariate MMD test with Gaussian kernel.
///
/// The bandwidth is automatically selected using the median heuristic.
pub fn mmd_test_1d(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<MMDResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    // Median heuristic for bandwidth selection
    let combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();
    let bandwidth = median_heuristic_1d(&combined);

    let x_vec: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
    let y_vec: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();

    mmd_test(
        &x_vec,
        &y_vec,
        Kernel::Gaussian { bandwidth },
        n_permutations,
        seed,
    )
}

/// Compute median heuristic bandwidth for 1D data.
fn median_heuristic_1d(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    // Compute all pairwise distances
    let mut distances: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push((data[i] - data[j]).abs());
        }
    }

    // Find median
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = distances.len() / 2;
    if distances.len() % 2 == 0 {
        (distances[mid - 1] + distances[mid]) / 2.0
    } else {
        distances[mid]
    }
    .max(0.01) // Ensure non-zero bandwidth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmd_different_distributions() {
        // Clearly different samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
        let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5];

        let result = mmd_test_1d(&x, &y, 499, Some(42)).unwrap();

        assert!(result.statistic > 0.0);
        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_mmd_similar_distributions() {
        // Similar samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];

        let result = mmd_test_1d(&x, &y, 499, Some(42)).unwrap();

        // Should not detect significant difference
        assert!(
            result.p_value > 0.1,
            "p_value {} should be > 0.1",
            result.p_value
        );
    }

    #[test]
    fn test_mmd_multivariate() {
        // 2D data with clearly different distributions
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![1.5, 1.5],
            vec![2.5, 2.5],
        ];
        let y = vec![
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![12.0, 12.0],
            vec![10.5, 10.5],
            vec![11.5, 11.5],
        ];

        let result = mmd_test(&x, &y, Kernel::Gaussian { bandwidth: 1.0 }, 499, Some(42)).unwrap();

        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_mmd_different_kernels() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        let x_vec: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
        let y_vec: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();

        // Test with different kernels
        let kernels = vec![
            Kernel::Gaussian { bandwidth: 1.0 },
            Kernel::Linear,
            Kernel::Polynomial {
                degree: 2,
                scale: 1.0,
                offset: 1.0,
            },
            Kernel::Laplacian { bandwidth: 1.0 },
        ];

        for kernel in kernels {
            let result = mmd_test(&x_vec, &y_vec, kernel, 199, Some(42)).unwrap();
            assert!(
                result.statistic > 0.0,
                "MMD should be positive for different distributions"
            );
        }
    }

    #[test]
    fn test_mmd_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(mmd_test_1d(&x, &y, 100, None).is_err());
    }

    #[test]
    fn test_mmd_insufficient() {
        let x = vec![1.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(mmd_test_1d(&x, &y, 100, None).is_err());
    }
}
