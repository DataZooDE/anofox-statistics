use super::statistics::{compute_test_statistic, standardize_value};
use super::types::{MCSStatistic, PairwiseStats};
use crate::resampling::bootstrap::StationaryBootstrap;
use std::collections::HashMap;

/// Bootstrap p-value for MCS elimination test.
#[allow(clippy::too_many_arguments)]
pub(super) fn bootstrap_mcs_pvalue(
    d_ij: &HashMap<(usize, usize), Vec<f64>>,
    d_bars: &PairwiseStats,
    observed_stat: f64,
    statistic: MCSStatistic,
    bootstrap: &mut StationaryBootstrap,
    n_bootstrap: usize,
    t: usize,
    m: usize,
) -> f64 {
    let t_f = t as f64;
    let mut count_exceeds = 0usize;

    for _ in 0..n_bootstrap {
        let boot_t_stats = compute_bootstrap_tstats(d_ij, d_bars, bootstrap, t, t_f);
        let boot_stat = compute_test_statistic(statistic, &boot_t_stats, m);

        if boot_stat >= observed_stat {
            count_exceeds += 1;
        }
    }

    (count_exceeds as f64 + 1.0) / (n_bootstrap as f64 + 1.0)
}

/// Compute bootstrap t-statistics for one bootstrap iteration.
fn compute_bootstrap_tstats(
    d_ij: &HashMap<(usize, usize), Vec<f64>>,
    d_bars: &PairwiseStats,
    bootstrap: &mut StationaryBootstrap,
    t: usize,
    t_f: f64,
) -> HashMap<(usize, usize), f64> {
    let mut boot_t_stats: HashMap<(usize, usize), f64> = HashMap::with_capacity(d_ij.len());

    for ((i, j), d) in d_ij {
        let boot_sample = bootstrap.sample(d, t);
        let boot_mean: f64 = boot_sample.iter().sum::<f64>() / t_f;
        let original_mean = d_bars[&(*i, *j)];
        let centered = boot_mean - original_mean;

        let boot_var: f64 = boot_sample
            .iter()
            .map(|x| (x - boot_mean).powi(2))
            .sum::<f64>()
            / (t_f - 1.0)
            / t_f;

        let boot_t = standardize_value(centered, boot_var, t_f);
        boot_t_stats.insert((*i, *j), boot_t);
    }

    boot_t_stats
}
