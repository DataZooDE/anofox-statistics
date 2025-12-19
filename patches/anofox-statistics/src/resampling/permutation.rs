use crate::error::{Result, StatError};
use crate::parametric::Alternative;
use crate::utils::math::{mean, variance};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Check if a permutation statistic is as extreme as the observed value.
#[inline]
fn is_extreme(perm_stat: f64, observed: f64, alternative: Alternative) -> bool {
    match alternative {
        Alternative::TwoSided => perm_stat.abs() >= observed.abs(),
        Alternative::Greater => perm_stat >= observed,
        Alternative::Less => perm_stat <= observed,
    }
}

/// Compute Welch's t-statistic for two samples.
fn welch_t_statistic(a: &[f64], b: &[f64]) -> f64 {
    let mean_a = mean(a).unwrap_or(0.0);
    let mean_b = mean(b).unwrap_or(0.0);
    let var_a = variance(a).unwrap_or(1.0);
    let var_b = variance(b).unwrap_or(1.0);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    let se = (var_a / n_a + var_b / n_b).sqrt();
    if se < 1e-14 {
        0.0
    } else {
        (mean_a - mean_b) / se
    }
}

/// Validate inputs for permutation t-test.
fn validate_permutation_inputs(x: &[f64], y: &[f64]) -> Result<()> {
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

/// Result of a permutation test
#[derive(Debug, Clone)]
pub struct PermutationResult {
    /// The observed test statistic
    pub statistic: f64,
    /// The p-value based on the permutation distribution
    pub p_value: f64,
    /// Number of permutations used
    pub n_permutations: usize,
}

/// Generic permutation engine for non-parametric hypothesis testing.
///
/// The engine performs permutation testing by shuffling the combined data
/// and computing a test statistic for each permutation.
pub struct PermutationEngine {
    /// Number of permutations to perform
    n_permutations: usize,
    /// Optional seed for reproducibility
    seed: Option<u64>,
}

impl PermutationEngine {
    /// Create a new PermutationEngine with the specified number of permutations.
    ///
    /// # Arguments
    /// * `n_permutations` - Number of permutations (typically 999 or 9999)
    pub fn new(n_permutations: usize) -> Self {
        Self {
            n_permutations,
            seed: None,
        }
    }

    /// Set a random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run the permutation test with a custom statistic function.
    ///
    /// # Arguments
    /// * `x` - First sample
    /// * `y` - Second sample
    /// * `statistic_fn` - Function that computes the test statistic given two samples
    /// * `alternative` - Alternative hypothesis
    ///
    /// # Returns
    /// * `PermutationResult` containing observed statistic and p-value
    pub fn run<F>(
        &self,
        x: &[f64],
        y: &[f64],
        statistic_fn: F,
        alternative: Alternative,
    ) -> Result<PermutationResult>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        if x.is_empty() || y.is_empty() {
            return Err(StatError::EmptyData);
        }

        let n1 = x.len();
        let n_total = x.len() + y.len();

        // Compute observed statistic
        let observed = statistic_fn(x, y);

        // Combine samples
        let mut combined: Vec<f64> = x.iter().chain(y.iter()).cloned().collect();

        // Initialize RNG
        let mut rng = match self.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        // Count how many permutation statistics are as extreme as observed
        let mut count_extreme = 0usize;

        for _ in 0..self.n_permutations {
            combined.shuffle(&mut rng);

            let perm_x = &combined[0..n1];
            let perm_y = &combined[n1..n_total];
            let perm_stat = statistic_fn(perm_x, perm_y);

            if is_extreme(perm_stat, observed, alternative) {
                count_extreme += 1;
            }
        }

        // P-value: (count + 1) / (n_permutations + 1)
        // The +1 accounts for the observed statistic itself
        let p_value = (count_extreme as f64 + 1.0) / (self.n_permutations as f64 + 1.0);

        Ok(PermutationResult {
            statistic: observed,
            p_value,
            n_permutations: self.n_permutations,
        })
    }
}

/// Perform a permutation t-test for comparing two independent samples.
///
/// This is a non-parametric alternative to the t-test that makes no
/// distributional assumptions. The null hypothesis is that the two samples
/// come from the same distribution.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Alternative hypothesis
/// * `n_permutations` - Number of permutations (default: 9999)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `PermutationResult` containing test statistic and p-value
pub fn permutation_t_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<PermutationResult> {
    validate_permutation_inputs(x, y)?;

    let mut engine = PermutationEngine::new(n_permutations);
    if let Some(s) = seed {
        engine = engine.with_seed(s);
    }

    engine.run(x, y, welch_t_statistic, alternative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_t_test_different_samples() {
        // Clearly different samples
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 11.0, 12.0, 13.0, 14.0];

        // Use seed for reproducibility
        let result = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(42)).unwrap();

        // Should have very low p-value for clearly different samples
        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_permutation_t_test_similar_samples() {
        // Similar samples
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];

        let result = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(42)).unwrap();

        // Should have high p-value for similar samples
        assert!(
            result.p_value > 0.1,
            "p_value {} should be > 0.1",
            result.p_value
        );
    }

    #[test]
    fn test_permutation_t_test_reproducibility() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        // Same seed should give same result
        let result1 = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(12345)).unwrap();
        let result2 = permutation_t_test(&x, &y, Alternative::TwoSided, 999, Some(12345)).unwrap();

        assert_eq!(result1.statistic, result2.statistic);
        assert_eq!(result1.p_value, result2.p_value);
    }

    #[test]
    fn test_permutation_t_test_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(permutation_t_test(&x, &y, Alternative::TwoSided, 100, None).is_err());
    }

    #[test]
    fn test_permutation_t_test_insufficient() {
        let x = vec![1.0];
        let y = vec![1.0, 2.0, 3.0];

        assert!(permutation_t_test(&x, &y, Alternative::TwoSided, 100, None).is_err());
    }

    #[test]
    fn test_permutation_engine_custom_statistic() {
        // Larger samples with clear separation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
        let y = vec![10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5];

        // Custom statistic: difference in medians
        let median_diff = |a: &[f64], b: &[f64]| -> f64 {
            let mut a_sorted: Vec<f64> = a.to_vec();
            let mut b_sorted: Vec<f64> = b.to_vec();
            a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let median_a = if a_sorted.len() % 2 == 1 {
                a_sorted[a_sorted.len() / 2]
            } else {
                (a_sorted[a_sorted.len() / 2 - 1] + a_sorted[a_sorted.len() / 2]) / 2.0
            };
            let median_b = if b_sorted.len() % 2 == 1 {
                b_sorted[b_sorted.len() / 2]
            } else {
                (b_sorted[b_sorted.len() / 2 - 1] + b_sorted[b_sorted.len() / 2]) / 2.0
            };
            median_a - median_b
        };

        let engine = PermutationEngine::new(999).with_seed(42);
        let result = engine
            .run(&x, &y, median_diff, Alternative::TwoSided)
            .unwrap();

        // Should detect significant difference with larger, clearly separated samples
        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }
}
