mod bootstrap;
mod statistics;
mod types;
mod validation;

pub use types::{MCSEliminationStep, MCSResult, MCSStatistic};

use bootstrap::bootstrap_mcs_pvalue;
use statistics::{
    compute_pairwise_differentials, compute_pairwise_tstats, compute_test_statistic,
    identify_worst_model,
};
use validation::validate_mcs_inputs;

use crate::error::Result;
use crate::resampling::bootstrap::StationaryBootstrap;

/// Perform the Model Confidence Set procedure.
///
/// Identifies a set of models that contains the best model with a given
/// confidence level. Uses sequential elimination based on bootstrap tests
/// of equal predictive ability.
///
/// # Arguments
/// * `losses` - Loss values for each model (K models, each with T observations)
/// * `alpha` - Significance level for elimination (e.g., 0.10)
/// * `statistic` - Which test statistic to use (Range or Max)
/// * `n_bootstrap` - Number of bootstrap samples
/// * `block_length` - Expected block length for stationary bootstrap
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `MCSResult` containing the model confidence set and elimination details
///
/// # References
/// * Hansen, P.R., Lunde, A., and Nason, J.M. (2011) "The Model Confidence Set"
pub fn model_confidence_set(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
) -> Result<MCSResult> {
    validate_mcs_inputs(losses, alpha, n_bootstrap)?;

    let k = losses.len();

    // Single model is trivially in the MCS
    if k == 1 {
        return Ok(MCSResult {
            included_models: vec![0],
            eliminated_models: vec![],
            mcs_p_value: 1.0,
            elimination_sequence: vec![],
            n_bootstrap,
            statistic_type: statistic,
        });
    }

    let t = losses[0].len();
    run_elimination_loop(losses, alpha, statistic, n_bootstrap, block_length, seed, t)
}

/// Run the sequential elimination loop for MCS.
fn run_elimination_loop(
    losses: &[Vec<f64>],
    alpha: f64,
    statistic: MCSStatistic,
    n_bootstrap: usize,
    block_length: f64,
    seed: Option<u64>,
    t: usize,
) -> Result<MCSResult> {
    let k = losses.len();
    let mut remaining_models: Vec<usize> = (0..k).collect();
    let mut eliminated_models: Vec<usize> = vec![];
    let mut elimination_sequence: Vec<MCSEliminationStep> = vec![];
    let mut bootstrap = StationaryBootstrap::new(block_length, seed);
    let mut mcs_p_value = 1.0;

    loop {
        if remaining_models.len() <= 1 {
            break;
        }

        let m = remaining_models.len();

        // Compute pairwise statistics
        let d_ij = compute_pairwise_differentials(losses, &remaining_models, t);
        let (t_stats, d_bars) = compute_pairwise_tstats(&d_ij, t);

        // Compute observed test statistic and identify worst model
        let observed_stat = compute_test_statistic(statistic, &t_stats, m);
        let worst_idx = identify_worst_model(&t_stats, m);

        // Bootstrap p-value
        let p_value = bootstrap_mcs_pvalue(
            &d_ij,
            &d_bars,
            observed_stat,
            statistic,
            &mut bootstrap,
            n_bootstrap,
            t,
            m,
        );

        // Record elimination step
        let step = MCSEliminationStep {
            model_idx: remaining_models[worst_idx],
            p_value,
            eliminated: p_value < alpha,
        };
        elimination_sequence.push(step);

        if p_value < alpha {
            eliminated_models.push(remaining_models[worst_idx]);
            remaining_models.remove(worst_idx);
        } else {
            mcs_p_value = p_value;
            break;
        }
    }

    // If we eliminated all but one, use the last p-value
    if remaining_models.len() == 1 && !elimination_sequence.is_empty() {
        mcs_p_value = elimination_sequence.last().unwrap().p_value;
    }

    Ok(MCSResult {
        included_models: remaining_models,
        eliminated_models,
        mcs_p_value,
        elimination_sequence,
        n_bootstrap,
        statistic_type: statistic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcs_single_model() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, Some(42)).unwrap();
        assert_eq!(result.included_models, vec![0]);
        assert!(result.eliminated_models.is_empty());
        assert_eq!(result.mcs_p_value, 1.0);
    }

    #[test]
    fn test_mcs_clearly_inferior_model() {
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert!(result.eliminated_models.contains(&1));
        assert_eq!(result.included_models.len(), 1);
    }

    #[test]
    fn test_mcs_equivalent_models() {
        let base: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin().abs() + 1.0)
            .collect();
        let losses = vec![base.clone(), base.clone(), base.clone()];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert_eq!(result.included_models.len(), 3);
        assert!(result.eliminated_models.is_empty());
    }

    #[test]
    fn test_mcs_multiple_inferior_models() {
        let losses = vec![vec![1.0; 100], vec![2.0; 100], vec![5.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert!(result.eliminated_models.contains(&2));
    }

    #[test]
    fn test_mcs_t_max_statistic() {
        let losses = vec![vec![1.0; 100], vec![10.0; 100]];
        let result =
            model_confidence_set(&losses, 0.10, MCSStatistic::Max, 500, 5.0, Some(42)).unwrap();

        assert!(result.included_models.contains(&0));
        assert_eq!(result.statistic_type, MCSStatistic::Max);
    }

    #[test]
    fn test_mcs_elimination_sequence() {
        let losses = vec![vec![1.0; 100], vec![3.0; 100], vec![5.0; 100]];
        let result =
            model_confidence_set(&losses, 0.25, MCSStatistic::Range, 500, 5.0, Some(42)).unwrap();

        assert!(!result.elimination_sequence.is_empty());

        for step in &result.elimination_sequence {
            assert!(step.p_value >= 0.0 && step.p_value <= 1.0);
        }
    }

    #[test]
    fn test_mcs_empty_error() {
        let losses: Vec<Vec<f64>> = vec![];
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_length_mismatch() {
        let losses = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_invalid_alpha() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        assert!(model_confidence_set(&losses, 0.0, MCSStatistic::Range, 100, 3.0, None).is_err());
        assert!(model_confidence_set(&losses, 1.0, MCSStatistic::Range, 100, 3.0, None).is_err());
        assert!(model_confidence_set(&losses, -0.1, MCSStatistic::Range, 100, 3.0, None).is_err());
    }

    #[test]
    fn test_mcs_zero_bootstrap() {
        let losses = vec![vec![1.0, 2.0, 3.0]];
        assert!(model_confidence_set(&losses, 0.10, MCSStatistic::Range, 0, 3.0, None).is_err());
    }
}
