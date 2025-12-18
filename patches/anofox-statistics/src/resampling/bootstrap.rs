use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A stationary bootstrap engine for time series data.
///
/// The stationary bootstrap (Politis & Romano, 1994) generates bootstrap samples
/// by taking random blocks of varying lengths from the original data.
pub struct StationaryBootstrap {
    /// Expected block length (1/q where q is the probability of starting a new block)
    expected_block_length: f64,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl StationaryBootstrap {
    /// Create a new stationary bootstrap engine.
    ///
    /// # Arguments
    /// * `expected_block_length` - Expected length of blocks (typical: n^(1/3) where n is data length)
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(expected_block_length: f64, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            expected_block_length: expected_block_length.max(1.0),
            rng,
        }
    }

    /// Generate a bootstrap sample of the specified length.
    ///
    /// # Arguments
    /// * `data` - The original time series data
    /// * `length` - Length of the bootstrap sample to generate
    pub fn sample(&mut self, data: &[f64], length: usize) -> Vec<f64> {
        if data.is_empty() || length == 0 {
            return vec![];
        }

        let n = data.len();
        let q = 1.0 / self.expected_block_length;
        let mut result = Vec::with_capacity(length);

        // Start with a random index
        let mut current_idx = self.rng.gen_range(0..n);

        for _ in 0..length {
            result.push(data[current_idx]);

            // With probability q, start a new block (random jump)
            // Otherwise, continue to next index (wrap around)
            if self.rng.gen::<f64>() < q {
                current_idx = self.rng.gen_range(0..n);
            } else {
                current_idx = (current_idx + 1) % n;
            }
        }

        result
    }

    /// Generate multiple bootstrap samples.
    pub fn samples(&mut self, data: &[f64], length: usize, n_samples: usize) -> Vec<Vec<f64>> {
        (0..n_samples).map(|_| self.sample(data, length)).collect()
    }
}

/// A circular block bootstrap engine.
///
/// The circular block bootstrap uses fixed-length blocks with wrap-around.
pub struct CircularBlockBootstrap {
    /// Block length
    block_length: usize,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl CircularBlockBootstrap {
    /// Create a new circular block bootstrap engine.
    ///
    /// # Arguments
    /// * `block_length` - Length of each block
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(block_length: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            block_length: block_length.max(1),
            rng,
        }
    }

    /// Generate a bootstrap sample of the specified length.
    pub fn sample(&mut self, data: &[f64], length: usize) -> Vec<f64> {
        if data.is_empty() || length == 0 {
            return vec![];
        }

        let n = data.len();
        let mut result = Vec::with_capacity(length);

        while result.len() < length {
            // Random starting point for the block
            let start = self.rng.gen_range(0..n);

            // Add elements from the block (with wrap-around)
            for offset in 0..self.block_length {
                if result.len() >= length {
                    break;
                }
                let idx = (start + offset) % n;
                result.push(data[idx]);
            }
        }

        result
    }

    /// Generate multiple bootstrap samples.
    pub fn samples(&mut self, data: &[f64], length: usize, n_samples: usize) -> Vec<Vec<f64>> {
        (0..n_samples).map(|_| self.sample(data, length)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_bootstrap_length() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut bootstrap = StationaryBootstrap::new(5.0, Some(42));

        let sample = bootstrap.sample(&data, 50);
        assert_eq!(sample.len(), 50);
    }

    #[test]
    fn test_stationary_bootstrap_reproducibility() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        let sample1 = StationaryBootstrap::new(5.0, Some(42)).sample(&data, 50);
        let sample2 = StationaryBootstrap::new(5.0, Some(42)).sample(&data, 50);

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_stationary_bootstrap_empty_data() {
        let data: Vec<f64> = vec![];
        let mut bootstrap = StationaryBootstrap::new(5.0, Some(42));

        let sample = bootstrap.sample(&data, 10);
        assert!(sample.is_empty());
    }

    #[test]
    fn test_circular_block_bootstrap_length() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut bootstrap = CircularBlockBootstrap::new(10, Some(42));

        let sample = bootstrap.sample(&data, 50);
        assert_eq!(sample.len(), 50);
    }

    #[test]
    fn test_circular_block_bootstrap_multiple_samples() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let mut bootstrap = CircularBlockBootstrap::new(5, Some(42));

        let samples = bootstrap.samples(&data, 30, 100);
        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert_eq!(sample.len(), 30);
        }
    }
}
