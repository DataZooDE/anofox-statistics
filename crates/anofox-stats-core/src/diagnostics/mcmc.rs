//! Convergence diagnostics for Markov-chain output.
//!
//! Rank-normalised split-R-hat and effective sample size, following Vehtari et al.
//! (2021) and Stan's reference implementation.
//!
//! These are statistics of a set of chains and nothing more: each takes
//! `&[Vec<f64>]`, one inner vector per chain, and returns a number. They carry no
//! opinion about where the draws came from, which is why they live here rather than
//! beside a particular sampler -- a Laplace approximation sampled at the mode, a
//! Metropolis run and a NUTS run are all judged by the same arithmetic.
//!
//! They were written for and are exercised by `anofox-bayes`, whose PyMC/ArviZ
//! golden-run parity suite is what makes the Stan fidelity below a real check rather
//! than a coincidence.

use statrs::distribution::{ContinuousCDF, Normal};

/// Replace values by their normal scores: rank, then map through the inverse normal
/// CDF with the Blom offset `(r - 3/8) / (n + 1/4)`.
///
/// Both R̂ and ESS are variance-based statistics, and a posterior without a finite
/// variance — several catalog families have heavy tails — makes them undefined. Rank
/// normalisation replaces the draws with a series that is guaranteed Gaussian-shaped
/// while preserving the ordering, and therefore preserving exactly the mixing
/// information the diagnostics are trying to measure.
///
/// Ties receive their average rank, which matters more than it looks: the tail-ESS
/// indicator series is all zeros and ones, so without tie handling every value would
/// map to one of two ranks and the statistic would be meaningless.
fn normal_scores(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .expect("callers filter non-finite values before ranking")
    });

    // Average ranks within tied runs (1-based).
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let avg = ((i + j) as f64) / 2.0 + 1.0;
        for k in i..=j {
            ranks[order[k]] = avg;
        }
        i = j + 1;
    }

    let normal = Normal::new(0.0, 1.0).expect("standard normal is always constructible");
    ranks
        .into_iter()
        .map(|r| {
            let p = (r - 0.375) / (n as f64 + 0.25);
            normal.inverse_cdf(p.clamp(1e-12, 1.0 - 1e-12))
        })
        .collect()
}

/// Split-R̂ over `chains`, or `None` when the statistic is not defined.
///
/// `None` is returned when there is only one chain, or when chains are too short to
/// split (fewer than 4 draws), or when every draw is identical. Returning `None`
/// rather than 1.0 is deliberate: an agent gating on `rhat <= 1.01` must not be told
/// "converged" by a statistic that was never computed.
pub fn rhat(chains: &[Vec<f64>]) -> Option<f64> {
    if chains.len() < 2 {
        return None;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return None;
    }

    // Rank-normalise across the pooled draws, then split.
    let pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scores = normal_scores(&pooled);

    let half = n / 2;
    let mut split: Vec<&[f64]> = Vec::with_capacity(chains.len() * 2);
    for (c, _) in chains.iter().enumerate() {
        let start = c * n;
        split.push(&scores[start..start + half]);
        split.push(&scores[start + half..start + 2 * half]);
    }

    rhat_of_splits(&split, half)
}

/// The variance-ratio statistic over already-split, already-normalised segments.
fn rhat_of_splits(split: &[&[f64]], n: usize) -> Option<f64> {
    let m = split.len();
    if m < 2 || n < 2 {
        return None;
    }

    let means: Vec<f64> = split
        .iter()
        .map(|s| s.iter().sum::<f64>() / n as f64)
        .collect();
    let vars: Vec<f64> = split
        .iter()
        .zip(&means)
        .map(|(s, &mu)| s.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n - 1) as f64)
        .collect();

    let grand_mean = means.iter().sum::<f64>() / m as f64;
    // Between-chain variance, scaled to the per-draw scale.
    let b = n as f64
        * means
            .iter()
            .map(|mu| (mu - grand_mean).powi(2))
            .sum::<f64>()
        / (m - 1) as f64;
    let w = vars.iter().sum::<f64>() / m as f64;

    if w <= 0.0 || !w.is_finite() || !b.is_finite() {
        // Every draw identical: the posterior is a point mass, not a converged
        // exploration of anything. No R-hat is defined.
        return None;
    }

    // var_plus is the marginal posterior variance estimate combining both sources.
    let var_plus = ((n - 1) as f64 * w + b) / n as f64;
    Some((var_plus / w).sqrt())
}

/// ESS of the rank-normalised, split draws: how far the posterior *mean* can be
/// trusted.
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    let Some(split) = split_chains(chains) else {
        return 0.0;
    };
    let pooled: Vec<f64> = split.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let n = split[0].len();
    let scores = normal_scores(&pooled);
    let normalised: Vec<Vec<f64>> = (0..split.len())
        .map(|c| scores[c * n..(c + 1) * n].to_vec())
        .collect();
    ess_of(&normalised)
}

/// The smaller of the ESS at the 5 % and 95 % quantiles: how far the posterior
/// *tails* can be trusted.
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    if split_chains(chains).is_none() {
        return 0.0;
    }
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    pooled.sort_by(|a, b| a.partial_cmp(b).expect("finiteness checked above"));

    // ESS of the indicator series `1[x <= q]`. The series is 0/1, and the rank
    // normalisation inside `ess_bulk` is what turns it into something the
    // autocorrelation estimator can work with.
    let at = |q: f64| -> f64 {
        let indicators: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&v| if v <= q { 1.0 } else { 0.0 }).collect())
            .collect();
        ess_bulk(&indicators)
    };

    at(quantile_sorted(&pooled, 0.05)).min(at(quantile_sorted(&pooled, 0.95)))
}

/// Halve every chain, as Stan does before computing R̂ and ESS.
///
/// Returns `None` when the input cannot be assessed: no chains, ragged chains, or
/// halves too short for a lag-2 autocorrelation.
fn split_chains(chains: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let m = chains.len();
    if m == 0 {
        return None;
    }
    let n = chains[0].len();
    if chains.iter().any(|c| c.len() != n) {
        return None;
    }
    let half = n / 2;
    if half < 4 {
        return None;
    }
    let mut out = Vec::with_capacity(m * 2);
    for c in chains {
        out.push(c[..half].to_vec());
        out.push(c[half..2 * half].to_vec());
    }
    Some(out)
}

/// Linear-interpolated quantile of an already-sorted slice.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

/// The autocorrelation-based ESS estimator, on chains that are already split and
/// rank-normalised.
///
/// Kept separate from the preprocessing so it can be tested directly against the
/// closed-form AR(1) answer, where `tau = (1 + rho) / (1 - rho)` is known exactly.
fn ess_of(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m == 0 {
        return 0.0;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return 0.0;
    }
    let total = (m * n) as f64;

    let means: Vec<f64> = chains
        .iter()
        .map(|c| c.iter().sum::<f64>() / n as f64)
        .collect();

    // Biased autocovariance (divisor n), matching Stan/ArviZ. Computed lazily by
    // lag: Geyer's rule truncates after a few dozen lags for any chain that mixes,
    // so the quadratic worst case is never reached in practice.
    let acov = |lag: usize| -> f64 {
        (0..m)
            .map(|c| {
                let x = &chains[c];
                let mu = means[c];
                let mut s = 0.0;
                for i in 0..(n - lag) {
                    s += (x[i] - mu) * (x[i + lag] - mu);
                }
                s / n as f64
            })
            .sum::<f64>()
            / m as f64
    };

    let chain_vars: Vec<f64> = (0..m)
        .map(|c| {
            let mu = means[c];
            chains[c].iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n - 1) as f64
        })
        .collect();
    let w = chain_vars.iter().sum::<f64>() / m as f64;

    // A parameter that never moves has no *defined* effective sample size: the
    // autocorrelation is 0/0. Zero is this estimator's "not assessable" signal, and
    // it is the honest answer -- reporting the raw count would say "well sampled"
    // about a series that never moved, and would let a degenerate parameter sail
    // through an `ess >= 400` gate. It also keeps ESS consistent with `rhat`, which
    // already returns `None` for a point mass.
    if w <= 0.0 || !w.is_finite() {
        return 0.0;
    }

    let var_plus = if m > 1 {
        let grand = means.iter().sum::<f64>() / m as f64;
        let b =
            n as f64 * means.iter().map(|mu| (mu - grand).powi(2)).sum::<f64>() / (m - 1) as f64;
        ((n - 1) as f64 * w + b) / n as f64
    } else {
        w
    };
    if var_plus <= 0.0 || !var_plus.is_finite() {
        return 0.0;
    }

    let rho = |lag: usize| -> f64 { 1.0 - (w - acov(lag)) / var_plus };

    // Geyer's initial positive sequence: walk forward in adjacent pairs, stopping as
    // soon as a pair sum goes non-positive.
    let mut rho_hat = vec![1.0, rho(1)];
    let mut t = 1usize;
    while t + 2 < n - 2 {
        let even = rho(t + 1);
        let odd = rho(t + 2);
        if even + odd <= 0.0 {
            break;
        }
        rho_hat.push(even);
        rho_hat.push(odd);
        t += 2;
    }
    // Stan keeps a trailing positive even term beyond the truncation point.
    let trailing = if t + 1 < n { rho(t + 1).max(0.0) } else { 0.0 };

    // Initial monotone sequence: a pair sum may not exceed its predecessor.
    let mut i = 3;
    while i + 1 < rho_hat.len() {
        let prev = rho_hat[i - 2] + rho_hat[i - 1];
        if rho_hat[i] + rho_hat[i + 1] > prev {
            rho_hat[i] = prev / 2.0;
            rho_hat[i + 1] = rho_hat[i];
        }
        i += 2;
    }

    let mut tau = -1.0 + 2.0 * rho_hat.iter().sum::<f64>() + trailing;
    // Stan's floor: without it, a strongly antithetic chain yields a tau at or below
    // zero and hence an infinite or negative ESS.
    tau = tau.max(1.0 / total.log10().max(1.0));

    total / tau
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic, roughly-normal stream.
    ///
    /// A plain counter-based hash rather than a dependency: these tests check the
    /// *statistic*, and what they need from a generator is reproducibility and rough
    /// normality, not cryptographic quality.
    fn normals(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut unit = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
        };
        (0..n)
            .map(|_| {
                let u1 = unit();
                let u2 = unit();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            })
            .collect()
    }

    fn iid(seed: u64, n: usize, mean: f64) -> Vec<f64> {
        normals(seed, n).into_iter().map(|z| z + mean).collect()
    }

    /// An AR(1) chain: each draw remembers the last one, which is what makes ESS
    /// smaller than the draw count.
    fn ar1(seed: u64, n: usize, rho: f64) -> Vec<f64> {
        let z = normals(seed, n);
        let mut out = Vec::with_capacity(n);
        let mut x = 0.0;
        for zi in z {
            x = rho * x + (1.0 - rho * rho).sqrt() * zi;
            out.push(x);
        }
        out
    }

    #[test]
    fn rhat_is_undefined_rather_than_one_when_it_cannot_be_computed() {
        // A statistic that was never computed must not read as "converged".
        assert_eq!(rhat(&[]), None, "no chains");
        assert_eq!(
            rhat(&[iid(1, 100, 0.0)]),
            None,
            "a single chain cannot split"
        );
        assert_eq!(
            rhat(&[vec![1.0; 2], vec![1.0; 2]]),
            None,
            "too short to split"
        );
        assert_eq!(
            rhat(&[vec![3.0; 500], vec![3.0; 500]]),
            None,
            "a constant has no between-chain variance to compare"
        );
        assert_eq!(
            rhat(&[iid(2, 100, 0.0), iid(3, 80, 0.0)]),
            None,
            "chains of different lengths"
        );
    }

    #[test]
    fn rhat_is_near_one_when_the_chains_agree() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| iid(10 + c, 2000, 0.0)).collect();
        let r = rhat(&chains).expect("defined for four long chains");
        assert!(
            r < 1.01,
            "chains from one distribution should agree, got {r}"
        );
    }

    #[test]
    fn rhat_rises_when_the_chains_have_not_met() {
        // Each chain sits on its own mean: the classic un-converged picture.
        let chains: Vec<Vec<f64>> = (0..4).map(|c| iid(20 + c, 2000, c as f64 * 5.0)).collect();
        let r = rhat(&chains).expect("defined");
        assert!(r > 1.5, "separated chains must be flagged, got {r}");
    }

    #[test]
    fn rhat_survives_a_posterior_without_finite_variance() {
        // The plain statistic is a ratio of variances and is undefined here; the
        // rank-normalised one is not, which is the whole reason for the ranking.
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                let z = normals(300 + c, 4096);
                (0..2000).map(|k| z[2 * k] / z[2 * k + 1]).collect()
            })
            .collect();
        let r = rhat(&chains).expect("defined");
        assert!(
            r.is_finite(),
            "Cauchy-like draws still yield a finite R-hat"
        );
        assert!(r < 1.05, "they are still the same distribution, got {r}");
    }

    #[test]
    fn ess_of_independent_draws_is_close_to_the_draw_count() {
        let chains: Vec<Vec<f64>> = (0..4).map(|c| iid(40 + c, 1000, 0.0)).collect();
        let bulk = ess_bulk(&chains);
        assert!(
            bulk > 2500.0 && bulk < 5000.0,
            "4000 independent draws should be worth most of themselves, got {bulk}"
        );
    }

    #[test]
    fn ess_falls_when_the_draws_are_autocorrelated() {
        let independent: Vec<Vec<f64>> = (0..4).map(|c| iid(50 + c, 2000, 0.0)).collect();
        let sticky: Vec<Vec<f64>> = (0..4).map(|c| ar1(60 + c, 2000, 0.9)).collect();
        let free = ess_bulk(&independent);
        let stuck = ess_bulk(&sticky);
        assert!(
            stuck < free / 3.0,
            "a chain that remembers its last draw carries less information: \
             {stuck} against {free}"
        );
        assert!(stuck > 0.0, "but it is not worthless");
    }

    #[test]
    fn tail_ess_is_reported_separately_from_bulk() {
        // They come apart in practice, and a decision that reads a quantile is
        // certified by the tail rather than by the mean.
        let chains: Vec<Vec<f64>> = (0..4).map(|c| ar1(70 + c, 2000, 0.8)).collect();
        let bulk = ess_bulk(&chains);
        let tail = ess_tail(&chains);
        assert!(bulk > 0.0 && tail > 0.0);
        assert!(bulk.is_finite() && tail.is_finite());
        assert!(
            (bulk - tail).abs() > 1.0,
            "bulk {bulk} and tail {tail} should not be the same statistic"
        );
    }

    #[test]
    fn ess_is_zero_rather_than_a_number_when_it_cannot_be_computed() {
        assert_eq!(ess_bulk(&[]), 0.0);
        assert_eq!(ess_bulk(&[vec![1.0; 2]]), 0.0, "too short to split");
        assert_eq!(
            ess_bulk(&[vec![7.0; 500], vec![7.0; 500]]),
            0.0,
            "a parameter that never moved has no effective sample"
        );
    }
}
