use std::collections::HashMap;

/// Type alias for pairwise statistics maps.
pub(super) type PairwiseStats = HashMap<(usize, usize), f64>;

/// Test statistic type for Model Confidence Set
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MCSStatistic {
    /// T_R statistic: max_{i,j} |t_{ij}|
    /// Most powerful against models with one clearly inferior model
    Range,
    /// T_max statistic: max_i of average t-statistics across all pairs
    /// More balanced when multiple models may be inferior
    Max,
}

/// Information about a single elimination step
#[derive(Debug, Clone)]
pub struct MCSEliminationStep {
    /// The model index that was tested at this step
    pub model_idx: usize,
    /// The p-value for the equal predictive ability test at this step
    pub p_value: f64,
    /// Whether the model was eliminated (p_value < alpha)
    pub eliminated: bool,
}

/// Result of Model Confidence Set procedure
#[derive(Debug, Clone)]
pub struct MCSResult {
    /// Indices of models in the Model Confidence Set (not eliminated)
    pub included_models: Vec<usize>,
    /// Indices of models eliminated from the MCS
    pub eliminated_models: Vec<usize>,
    /// The MCS p-value (p-value when elimination stopped)
    pub mcs_p_value: f64,
    /// Full elimination sequence with p-values at each step
    pub elimination_sequence: Vec<MCSEliminationStep>,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
    /// The statistic type used
    pub statistic_type: MCSStatistic,
}
