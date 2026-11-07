#pragma once

#include "duckdb.hpp"

namespace duckdb {

struct CreateMacroInfo;

class ExtensionLoader;

// Forward declarations for function registration
void RegisterOlsFitFunction(ExtensionLoader &loader);
void RegisterRidgeFitFunction(ExtensionLoader &loader);
void RegisterElasticNetFitFunction(ExtensionLoader &loader);
void RegisterWlsFitFunction(ExtensionLoader &loader);
void RegisterHuberFitFunction(ExtensionLoader &loader);
void RegisterRansacFitFunction(ExtensionLoader &loader);
void RegisterTheilSenFitFunction(ExtensionLoader &loader);
void RegisterPredictFunction(ExtensionLoader &loader);
void RegisterTidyGlanceFunctions(ExtensionLoader &loader);
void RegisterOlsAggregateFunction(ExtensionLoader &loader);
void RegisterRidgeAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetAggregateFunction(ExtensionLoader &loader);
void RegisterLarsAggregateFunction(ExtensionLoader &loader);
void RegisterWlsAggregateFunction(ExtensionLoader &loader);
void RegisterHuberAggregateFunction(ExtensionLoader &loader);
void RegisterRansacAggregateFunction(ExtensionLoader &loader);
void RegisterTheilSenAggregateFunction(ExtensionLoader &loader);
void RegisterRlsAggregateFunction(ExtensionLoader &loader);
void RegisterRlsFitFunction(ExtensionLoader &loader);

// Window aggregate functions (fit_predict)
void RegisterOlsFitPredictFunction(ExtensionLoader &loader);
void RegisterHuberFitPredictFunction(ExtensionLoader &loader);
void RegisterRansacFitPredictFunction(ExtensionLoader &loader);
void RegisterTheilSenFitPredictFunction(ExtensionLoader &loader);
void RegisterRidgeFitPredictFunction(ExtensionLoader &loader);
void RegisterWlsFitPredictFunction(ExtensionLoader &loader);
void RegisterRlsFitPredictFunction(ExtensionLoader &loader);
void RegisterElasticNetFitPredictFunction(ExtensionLoader &loader);

// Fit + Predict aggregate functions (fit + predict all rows, with deprecated aliases)
void RegisterOlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterHuberFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRansacFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterTheilSenFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRidgeFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterWlsFitPredictAggregateFunction(ExtensionLoader &loader);
// ols_augment_agg / wls_augment_agg: per-row diagnostics (#151)
void RegisterAugmentAggregateFunctions(ExtensionLoader &loader);
void RegisterRlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterBlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterAlmFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterPoissonFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterPlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterIsotonicFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterQuantileFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterExtraFitAggregateFunctions(ExtensionLoader &loader);
// Bayesian Ridge / ARD fit and fit-predict aggregates (#61)
void RegisterBayesianAggregateFunctions(ExtensionLoader &loader);
// Passive-Aggressive (aggregates and window), lmDynamic and lowess_agg (#63)
void RegisterPassiveAggressiveFunctions(ExtensionLoader &loader);
void RegisterLsStateFunctions(ExtensionLoader &loader);
void RegisterLmDynamicFunctions(ExtensionLoader &loader);
void RegisterLowessFunction(ExtensionLoader &loader);
// P-spline smoother: pspline_fit_agg, pspline_fit_predict_agg (#316)
void RegisterPSplineAggregateFunctions(ExtensionLoader &loader);
void RegisterPathAggregateFunctions(ExtensionLoader &loader);

// GLM aggregate functions
void RegisterPoissonAggregateFunction(ExtensionLoader &loader);
void RegisterAftAggregateFunction(ExtensionLoader &loader);
void RegisterEbShrinkAggregateFunction(ExtensionLoader &loader);
void RegisterMediationAggregateFunction(ExtensionLoader &loader);
void RegisterOlsAnovaAggregateFunction(ExtensionLoader &loader);
void RegisterCollinearitySeparationAggregateFunctions(ExtensionLoader &loader);
void RegisterOlsAnovaTableAggregateFunction(ExtensionLoader &loader);
// Likelihood-ratio and score tests (#310)
void RegisterGlmScoreTestAggregateFunction(ExtensionLoader &loader);
void RegisterLrTestFunction(ExtensionLoader &loader);
void RegisterWaldAnovaFunctions(ExtensionLoader &loader);
void RegisterGlmmAggregateFunction(ExtensionLoader &loader);
void RegisterGlsFunctions(ExtensionLoader &loader);
void RegisterAftScalarFunctions(ExtensionLoader &loader);
void RegisterCoxFunctions(ExtensionLoader &loader);
void RegisterPentraceAggregateFunctions(ExtensionLoader &loader);
void RegisterBinomialAggregateFunction(ExtensionLoader &loader);
void RegisterNegBinomAggregateFunction(ExtensionLoader &loader);
void RegisterTweedieAggregateFunction(ExtensionLoader &loader);
void RegisterGammaAggregateFunction(ExtensionLoader &loader);
void RegisterLogisticAggregateFunction(ExtensionLoader &loader);

// ALM aggregate functions
void RegisterAlmAggregateFunction(ExtensionLoader &loader);

// BLS aggregate functions (includes NNLS)
void RegisterBlsAggregateFunction(ExtensionLoader &loader);

// AID aggregate functions (Automatic Identification of Demand)
void RegisterAidAggregateFunction(ExtensionLoader &loader);

// Diagnostic functions
void RegisterVifFunction(ExtensionLoader &loader);
void RegisterVifAggregateFunction(ExtensionLoader &loader);
void RegisterGvifAggregateFunction(ExtensionLoader &loader);
void RegisterAicBicFunctions(ExtensionLoader &loader);
void RegisterJarqueBeraFunction(ExtensionLoader &loader);
void RegisterNullDistributionFunctions(ExtensionLoader &loader);
void RegisterSplineFunctions(ExtensionLoader &loader);              // rcs, lsp, bspline, ... (#317)
void RegisterSplineKnotsAggregateFunctions(ExtensionLoader &loader); // rcs_knots_agg, bspline_knots_agg (#317)
void RegisterEffectsFunctions(ExtensionLoader &loader); // effects_summary_agg, contrast_agg, predict_grid_agg, ... (#319)
void RegisterJarqueBeraAggregateFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsAggregateFunction(ExtensionLoader &loader);

// Statistical hypothesis testing aggregate functions
void RegisterShapiroWilkAggregateFunction(ExtensionLoader &loader);
void RegisterTTestAggregateFunction(ExtensionLoader &loader);
void RegisterPearsonAggregateFunction(ExtensionLoader &loader);
void RegisterSpearmanAggregateFunction(ExtensionLoader &loader);
void RegisterPartialCorAggregateFunctions(ExtensionLoader &loader);
void RegisterMannWhitneyAggregateFunction(ExtensionLoader &loader);
void RegisterAnovaAggregateFunction(ExtensionLoader &loader);
void RegisterPairwisePosthocAggregateFunctions(ExtensionLoader &loader);
void RegisterPAdjustFunctions(ExtensionLoader &loader);
void RegisterKruskalWallisAggregateFunction(ExtensionLoader &loader);
void RegisterChiSquareAggregateFunction(ExtensionLoader &loader);

// Aggregates for existing FFI
void RegisterKendallAggregateFunction(ExtensionLoader &loader);
void RegisterFisherExactAggregateFunction(ExtensionLoader &loader);
void RegisterBrunnerMunzelAggregateFunction(ExtensionLoader &loader);
void RegisterDAgostinoK2AggregateFunction(ExtensionLoader &loader);
void RegisterEnergyDistanceAggregateFunction(ExtensionLoader &loader);
void RegisterDriftAggregateFunctions(ExtensionLoader &loader);
void RegisterEmbeddingAggregateFunctions(ExtensionLoader &loader);
void RegisterMmdAggregateFunction(ExtensionLoader &loader);
void RegisterTostTTestAggregateFunction(ExtensionLoader &loader);
void RegisterWilcoxonSignedRankAggregateFunction(ExtensionLoader &loader);
void RegisterDistanceCorAggregateFunction(ExtensionLoader &loader);
void RegisterYuenAggregateFunction(ExtensionLoader &loader);
void RegisterBrownForsytheAggregateFunction(ExtensionLoader &loader);
void RegisterDieboldMarianoAggregateFunction(ExtensionLoader &loader);
void RegisterTimeSeriesDiagnosticsAggregateFunctions(ExtensionLoader &loader);
void RegisterSurvivalAggregateFunctions(ExtensionLoader &loader);
void RegisterClarkWestAggregateFunction(ExtensionLoader &loader);
void RegisterForecastComparisonAggregateFunctions(ExtensionLoader &loader);
void RegisterPermutationTTestAggregateFunction(ExtensionLoader &loader);
void RegisterBootstrapMeanCiAggregateFunction(ExtensionLoader &loader);
void RegisterTostPairedAggregateFunction(ExtensionLoader &loader);
void RegisterTTestOneSamplePairedAggregateFunctions(ExtensionLoader &loader);
void RegisterTostCorrelationAggregateFunction(ExtensionLoader &loader);
void RegisterTostVariantAggregateFunctions(ExtensionLoader &loader);
void RegisterChisqGofAggregateFunction(ExtensionLoader &loader);
void RegisterPropTestOneAggregateFunction(ExtensionLoader &loader);
void RegisterPropTestTwoAggregateFunction(ExtensionLoader &loader);
void RegisterBinomTestAggregateFunction(ExtensionLoader &loader);
void RegisterCramersVAggregateFunction(ExtensionLoader &loader);
void RegisterCohenKappaAggregateFunction(ExtensionLoader &loader);
void RegisterIccAggregateFunction(ExtensionLoader &loader);
void RegisterFactorialAnovaAggregateFunctions(ExtensionLoader &loader);
void RegisterGTestAggregateFunction(ExtensionLoader &loader);
void RegisterMcNemarAggregateFunction(ExtensionLoader &loader);
void RegisterPhiCoefficientAggregateFunction(ExtensionLoader &loader);
void RegisterContingencyCoefAggregateFunction(ExtensionLoader &loader);

// Classifier evaluation: ROC/PR/calibration curves, AUC, Brier, log loss (#153)
void RegisterClassifierAggregateFunctions(ExtensionLoader &loader);

// Predictive accuracy indexes: concordance, somers2, rcorr.cens, lrm stats, val.prob, R2 measures (#308)
void RegisterAccuracyFunctions(ExtensionLoader &loader);

// Table Macros for fit_predict_by functions
void RegisterFitPredictTableMacros(ExtensionLoader &loader);
void RegisterModelOutputMacros(ExtensionLoader &loader);
void RegisterPosthocMacros(ExtensionLoader &loader);
void RegisterPathMacros(ExtensionLoader &loader);
void RegisterPredictionMacros(ExtensionLoader &loader);
//! Build a table macro (shared by the contract macro files).
unique_ptr<CreateMacroInfo> CreateContractTableMacro(const string &name, const vector<string> &parameters,
                                                     const vector<pair<string, string>> &named_parameters,
                                                     const string &sql, const string &description,
                                                     const string &example, const vector<string> &categories);
void RegisterCurveMacros(ExtensionLoader &loader);
// Ordinal regression: ordinal_fit_agg, continuation_ratio_fit_agg, ordinal_predict,
// ordinal_fit_predict_by, continuation_ratio_fit_predict_by (#313)
void RegisterOrdinalFunctions(ExtensionLoader &loader);
// Principal component analysis: pca_agg, pca_transform, pca_by (#303)
void RegisterPcaFunctions(ExtensionLoader &loader);
void RegisterImputationFunctions(ExtensionLoader &loader);
// Multinomial logistic regression: multinomial_fit_agg, multinomial_predict,
// multinomial_fit_predict_by (#314)
void RegisterMultinomialFunctions(ExtensionLoader &loader);
// Resampling validation / calibration: validate_agg, calibrate_agg, calibrate_cph_agg, bootcov_agg
// and their _by macros (#318)
void RegisterModelValidationFunctions(ExtensionLoader &loader);
// Data reduction and transforms: spearman2_agg, hoeffding_d_agg, box_cox_agg,
// box_cox_transform / box_cox_inverse, varclus_agg, redun_agg (#320)
void RegisterReductionFunctions(ExtensionLoader &loader);
// PCA reconstruction-error drift and alert bands: pca_reconstruction_*, alert_band_agg, alert_band_by (#366, #367)
void RegisterReconstructionFunctions(ExtensionLoader &loader);
// CBPE performance estimation: cbpe_fit_agg, cbpe_estimate_agg, cbpe_realized_agg, cbpe_by (#365)
void RegisterCbpeFunctions(ExtensionLoader &loader);
void RegisterClassifierCurveMacros(ExtensionLoader &loader);
// Metric pack: classification_metrics_agg, regression_metrics_agg, ranking_metrics_agg,
// their _by macros and data_quality_by (#368, #369)
void RegisterMetricsFunctions(ExtensionLoader &loader);

// Extension class required for static linking
class AnofoxStatisticsExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
