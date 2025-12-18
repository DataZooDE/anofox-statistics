pub mod categorical;
pub mod correlation;
pub mod distributional;
pub mod equivalence;
pub mod error;
pub mod forecast;
pub mod modern;
pub mod nonparametric;
pub mod parametric;
pub mod resampling;
pub mod utils;

pub use correlation::{
    distance_cor, distance_cor_test, icc, kendall, partial_cor, pearson, semi_partial_cor,
    spearman, CorrelationConfInt, CorrelationMethod, CorrelationResult, DistanceCorResult,
    ICCResult, ICCType, KendallVariant, PartialCorResult,
};
pub use distributional::{dagostino_k_squared, shapiro_wilk, DAgostinoResult, ShapiroWilkResult};
pub use error::{Result, StatError};
pub use forecast::{
    clark_west, diebold_mariano, model_confidence_set, mspe_adjusted_spa, spa_test, CWResult,
    DMResult, LossFunction, MCSEliminationStep, MCSResult, MCSStatistic, MSPEAdjustedResult,
    SPAResult, VarEstimator,
};
pub use modern::{
    energy_distance_test, energy_distance_test_1d, mmd_test, mmd_test_1d, EnergyDistanceResult,
    Kernel, MMDResult,
};
pub use nonparametric::{
    brunner_munzel, kruskal_wallis, mann_whitney_u, rank, wilcoxon_signed_rank,
    BrunnerMunzelResult, KruskalResult, MannWhitneyResult, WilcoxonResult,
};
pub use parametric::{
    brown_forsythe, one_way_anova, repeated_measures_anova, t_test, two_way_anova, yuen_test,
    Alternative, AnovaKind, AnovaTableRow, CorrectedResult, LeveneResult, OneWayAnovaResult,
    RmAnovaResult, SphericityResult, TTestKind, TTestResult, TwoWayAnovaResult, YuenConfInt,
    YuenResult,
};
pub use resampling::{
    permutation_t_test, CircularBlockBootstrap, PermutationEngine, PermutationResult,
    StationaryBootstrap,
};

pub use categorical::{
    binom_test, chisq_goodness_of_fit, chisq_test, cohen_kappa, contingency_coef, cramers_v,
    fisher_exact, g_test, mcnemar_exact, mcnemar_test, phi_coefficient, prop_test_one,
    prop_test_two, AssociationResult, BinomTestResult, ChiSquareResult, FisherResult, KappaResult,
    McNemarkExactResult, McNemarkResult, PropTestResult,
};
pub use equivalence::{
    tost_bootstrap, tost_correlation, tost_prop_one, tost_prop_two, tost_t_test_one_sample,
    tost_t_test_paired, tost_t_test_two_sample, tost_wilcoxon_paired, tost_wilcoxon_two_sample,
    tost_yuen, CorrelationTostMethod, EquivalenceBounds, OneSidedTestResult, TostResult,
};
