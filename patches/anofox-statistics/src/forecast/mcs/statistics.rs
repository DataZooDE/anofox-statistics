use super::types::{MCSStatistic, PairwiseStats};
use std::collections::HashMap;

/// Compute pairwise loss differentials for remaining models.
pub(super) fn compute_pairwise_differentials(
    losses: &[Vec<f64>],
    remaining: &[usize],
    t: usize,
) -> HashMap<(usize, usize), Vec<f64>> {
    let m = remaining.len();
    let mut d_ij: HashMap<(usize, usize), Vec<f64>> = HashMap::with_capacity(m * (m - 1) / 2);

    for i in 0..m {
        for j in (i + 1)..m {
            let model_i = remaining[i];
            let model_j = remaining[j];
            let d: Vec<f64> = (0..t)
                .map(|idx| losses[model_i][idx] - losses[model_j][idx])
                .collect();
            d_ij.insert((i, j), d);
        }
    }

    d_ij
}

/// Compute pairwise t-statistics and means.
pub(super) fn compute_pairwise_tstats(
    d_ij: &HashMap<(usize, usize), Vec<f64>>,
    t: usize,
) -> (PairwiseStats, PairwiseStats) {
    let t_f = t as f64;
    let mut t_stats: HashMap<(usize, usize), f64> = HashMap::with_capacity(d_ij.len());
    let mut d_bars: HashMap<(usize, usize), f64> = HashMap::with_capacity(d_ij.len());

    for ((i, j), d) in d_ij {
        let d_bar: f64 = d.iter().sum::<f64>() / t_f;
        d_bars.insert((*i, *j), d_bar);

        let var_d: f64 = d.iter().map(|x| (x - d_bar).powi(2)).sum::<f64>() / (t_f - 1.0) / t_f;
        let t_stat = standardize_value(d_bar, var_d, t_f);
        t_stats.insert((*i, *j), t_stat);
    }

    (t_stats, d_bars)
}

/// Standardize a value, handling near-zero variance.
#[inline]
pub(super) fn standardize_value(mean: f64, var: f64, t_f: f64) -> f64 {
    if var > 1e-14 {
        mean * t_f.sqrt() / var.sqrt()
    } else {
        mean * t_f.sqrt() * 1e6
    }
}

/// Compute the test statistic based on pairwise t-statistics.
pub(super) fn compute_test_statistic(
    stat_type: MCSStatistic,
    t_stats: &HashMap<(usize, usize), f64>,
    m: usize,
) -> f64 {
    match stat_type {
        MCSStatistic::Range => {
            // T_R = max_{i,j} |t_{ij}|
            t_stats.values().map(|t| t.abs()).fold(0.0_f64, f64::max)
        }
        MCSStatistic::Max => {
            // T_max = max_i of average t-statistics
            let mut model_avgs: Vec<f64> = vec![0.0; m];
            let mut counts: Vec<usize> = vec![0; m];

            for ((i, j), t_val) in t_stats {
                model_avgs[*i] += *t_val;
                model_avgs[*j] -= *t_val;
                counts[*i] += 1;
                counts[*j] += 1;
            }

            for i in 0..m {
                if counts[i] > 0 {
                    model_avgs[i] /= counts[i] as f64;
                }
            }

            model_avgs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }
}

/// Identify the worst model based on average relative performance.
pub(super) fn identify_worst_model(t_stats: &HashMap<(usize, usize), f64>, m: usize) -> usize {
    let mut model_scores: Vec<f64> = vec![0.0; m];
    let mut counts: Vec<usize> = vec![0; m];

    for ((i, j), t_val) in t_stats {
        model_scores[*i] += *t_val;
        model_scores[*j] -= *t_val;
        counts[*i] += 1;
        counts[*j] += 1;
    }

    for i in 0..m {
        if counts[i] > 0 {
            model_scores[i] /= counts[i] as f64;
        }
    }

    model_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
