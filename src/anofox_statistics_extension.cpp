#define DUCKDB_EXTENSION_MAIN

#include "include/anofox_statistics_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "telemetry.hpp"
#include "anofox_statistics_banner.hpp"
#include "contract.hpp"

// The build stamps EXT_VERSION_ANOFOX_STATISTICS from the git tag; the fallback
// matches what SetProduct reports below.
#ifdef EXT_VERSION_ANOFOX_STATISTICS
#define ANOFOX_STATISTICS_BANNER_VERSION EXT_VERSION_ANOFOX_STATISTICS
#else
#define ANOFOX_STATISTICS_BANNER_VERSION "0.16.0"
#endif

// Deliberately outside namespace duckdb: the banner library is DuckDB-agnostic
// and the guard macro refers to this object from every guarded source file.
const datazoo::BannerInfo ANOFOX_STATISTICS_BANNER {
    "anofox_statistics", ANOFOX_STATISTICS_BANNER_VERSION,
    "https://github.com/DataZooDE/anofox-statistics"};

namespace duckdb {

#ifdef ANOFOX_TELEMETRY_ENABLED

namespace {

void OnTelemetryEnabled(ClientContext &context, SetScope scope, Value &parameter) {
    if (parameter.IsNull()) {
        throw InvalidInputException("anofox_telemetry_enabled cannot be NULL");
    }
    auto &telemetry = PostHogTelemetry::Instance();
    telemetry.SetEnabled(BooleanValue::Get(parameter));
}

void OnTelemetryKey(ClientContext &context, SetScope scope, Value &parameter) {
    if (parameter.IsNull()) {
        throw InvalidInputException("anofox_telemetry_key cannot be NULL");
    }
    auto &telemetry = PostHogTelemetry::Instance();
    telemetry.SetAPIKey(StringValue::Get(parameter));
}

} // anonymous namespace

static void RegisterTelemetryOptions(ExtensionLoader &loader) {
    auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

    config.AddExtensionOption("anofox_telemetry_enabled",
                              "Enable or disable anonymous usage telemetry",
                              LogicalType::BOOLEAN, Value::BOOLEAN(true), OnTelemetryEnabled);

    config.AddExtensionOption("anofox_telemetry_key",
                              "PostHog API key for telemetry",
                              LogicalType::VARCHAR,
                              Value("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t"),
                              OnTelemetryKey);
}

#endif // ANOFOX_TELEMETRY_ENABLED

void LoadInternal(ExtensionLoader &loader) {
#ifdef ANOFOX_TELEMETRY_ENABLED
    // Register telemetry options
    RegisterTelemetryOptions(loader);

    // Initialize and capture extension load event
    auto &telemetry = PostHogTelemetry::Instance();
    telemetry.SetAPIKey("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t");

    std::string version;
#ifdef EXT_VERSION_ANOFOX_STATISTICS
    version = EXT_VERSION_ANOFOX_STATISTICS;
#else
    version = "0.16.0";
#endif
    telemetry.SetProduct("anofox_statistics", version, "oss");
    telemetry.AssociateGroup("deployment", PostHogTelemetry::GetDistinctId());
    telemetry.CaptureExtensionLoad("anofox_statistics", version);
#endif // ANOFOX_TELEMETRY_ENABLED

    const auto preexisting = SnapshotSystemFunctionNames(loader);

    // Register scalar functions
    RegisterOlsFitFunction(loader);
    RegisterRidgeFitFunction(loader);
    RegisterElasticNetFitFunction(loader);
    RegisterWlsFitFunction(loader);
    RegisterHuberFitFunction(loader);
    RegisterRansacFitFunction(loader);
    RegisterTheilSenFitFunction(loader);
    RegisterPredictFunction(loader);
    RegisterTidyGlanceFunctions(loader);
    RegisterRlsFitFunction(loader);

    // Register aggregate functions
    RegisterOlsAggregateFunction(loader);
    RegisterRidgeAggregateFunction(loader);
    RegisterElasticNetAggregateFunction(loader);
    RegisterLarsAggregateFunction(loader);
    RegisterWlsAggregateFunction(loader);
    RegisterHuberAggregateFunction(loader);
    RegisterRansacAggregateFunction(loader);
    RegisterTheilSenAggregateFunction(loader);
    RegisterRlsAggregateFunction(loader);
    RegisterVifAggregateFunction(loader);
    RegisterGvifAggregateFunction(loader);
    RegisterJarqueBeraAggregateFunction(loader);
    RegisterResidualsDiagnosticsAggregateFunction(loader);

    // Register GLM aggregate functions
    RegisterPoissonAggregateFunction(loader);
    RegisterAftAggregateFunction(loader);
    RegisterEbShrinkAggregateFunction(loader);
    RegisterMediationAggregateFunction(loader);
    RegisterOlsAnovaAggregateFunction(loader);
    RegisterCollinearitySeparationAggregateFunctions(loader);
    RegisterOlsAnovaTableAggregateFunction(loader);
    RegisterGlmmAggregateFunction(loader);
    RegisterGlsFunctions(loader);
    RegisterAftScalarFunctions(loader);
    RegisterCoxFunctions(loader);
    RegisterPentraceAggregateFunctions(loader);
    RegisterBinomialAggregateFunction(loader);
    RegisterNegBinomAggregateFunction(loader);
    RegisterTweedieAggregateFunction(loader);
    RegisterGammaAggregateFunction(loader);
    RegisterLogisticAggregateFunction(loader);
    RegisterGlmScoreTestAggregateFunction(loader);
    RegisterLrTestFunction(loader);
    RegisterWaldAnovaFunctions(loader);

    // Register ALM aggregate functions
    RegisterAlmAggregateFunction(loader);

    // Register BLS aggregate functions (includes NNLS)
    RegisterBlsAggregateFunction(loader);

    // Register AID aggregate functions (Automatic Identification of Demand)
    RegisterAidAggregateFunction(loader);

    // Register statistical hypothesis testing aggregate functions
    RegisterShapiroWilkAggregateFunction(loader);
    RegisterTTestAggregateFunction(loader);
    RegisterTTestOneSamplePairedAggregateFunctions(loader);
    RegisterPearsonAggregateFunction(loader);
    RegisterSpearmanAggregateFunction(loader);
    RegisterPartialCorAggregateFunctions(loader);
    RegisterMannWhitneyAggregateFunction(loader);
    RegisterAnovaAggregateFunction(loader);
    RegisterPairwisePosthocAggregateFunctions(loader);
    RegisterPAdjustFunctions(loader);
    RegisterKruskalWallisAggregateFunction(loader);
    RegisterChiSquareAggregateFunction(loader);

    // Aggregates for existing FFI
    RegisterKendallAggregateFunction(loader);
    RegisterFisherExactAggregateFunction(loader);
    RegisterBrunnerMunzelAggregateFunction(loader);
    RegisterDAgostinoK2AggregateFunction(loader);
    RegisterEnergyDistanceAggregateFunction(loader);
    RegisterDriftAggregateFunctions(loader);
    RegisterEmbeddingAggregateFunctions(loader);
    RegisterMmdAggregateFunction(loader);
    RegisterTostTTestAggregateFunction(loader);

    // Wilcoxon signed-rank test
    RegisterWilcoxonSignedRankAggregateFunction(loader);

    // Distance correlation test
    RegisterDistanceCorAggregateFunction(loader);

    // Parametric tests
    RegisterYuenAggregateFunction(loader);
    RegisterBrownForsytheAggregateFunction(loader);

    // Forecast tests
    RegisterDieboldMarianoAggregateFunction(loader);
    RegisterTimeSeriesDiagnosticsAggregateFunctions(loader);
    RegisterSurvivalAggregateFunctions(loader);
    RegisterClarkWestAggregateFunction(loader);
    RegisterForecastComparisonAggregateFunctions(loader);

    // Resampling tests
    RegisterPermutationTTestAggregateFunction(loader);
    RegisterBootstrapMeanCiAggregateFunction(loader);

    // TOST equivalence test variants
    RegisterTostPairedAggregateFunction(loader);
    RegisterTostCorrelationAggregateFunction(loader);
    RegisterTostVariantAggregateFunctions(loader);

    // Categorical tests
    RegisterChisqGofAggregateFunction(loader);
    RegisterPropTestOneAggregateFunction(loader);
    RegisterPropTestTwoAggregateFunction(loader);
    RegisterBinomTestAggregateFunction(loader);
    RegisterCramersVAggregateFunction(loader);
    RegisterCohenKappaAggregateFunction(loader);
    RegisterIccAggregateFunction(loader);
    RegisterFactorialAnovaAggregateFunctions(loader); // two_way_anova_agg, repeated_measures_anova_agg (#297)
    RegisterGTestAggregateFunction(loader);
    RegisterMcNemarAggregateFunction(loader);
    RegisterPhiCoefficientAggregateFunction(loader);
    RegisterContingencyCoefAggregateFunction(loader);

    // Classifier evaluation (#153)
    RegisterClassifierAggregateFunctions(loader);
    // Predictive accuracy indexes (#308)
    RegisterAccuracyFunctions(loader);

    // Register window aggregate functions (fit_predict)
    RegisterOlsFitPredictFunction(loader);
    RegisterHuberFitPredictFunction(loader);
    RegisterRansacFitPredictFunction(loader);
    RegisterTheilSenFitPredictFunction(loader);
    RegisterRidgeFitPredictFunction(loader);
    RegisterWlsFitPredictFunction(loader);
    RegisterRlsFitPredictFunction(loader);
    RegisterElasticNetFitPredictFunction(loader);

    // Register fit + predict aggregate functions (with deprecated aliases)
    RegisterOlsFitPredictAggregateFunction(loader);
    RegisterHuberFitPredictAggregateFunction(loader);
    RegisterRansacFitPredictAggregateFunction(loader);
    RegisterTheilSenFitPredictAggregateFunction(loader);
    RegisterRidgeFitPredictAggregateFunction(loader);
    RegisterWlsFitPredictAggregateFunction(loader);
    RegisterAugmentAggregateFunctions(loader);
    RegisterRlsFitPredictAggregateFunction(loader);
    RegisterElasticNetFitPredictAggregateFunction(loader);
    RegisterBlsFitPredictAggregateFunction(loader);
    RegisterAlmFitPredictAggregateFunction(loader);
    RegisterPoissonFitPredictAggregateFunction(loader);
    RegisterPlsFitPredictAggregateFunction(loader);
    RegisterIsotonicFitPredictAggregateFunction(loader);
    RegisterQuantileFitPredictAggregateFunction(loader);
    RegisterExtraFitAggregateFunctions(loader);
    RegisterBayesianAggregateFunctions(loader);
    RegisterPassiveAggressiveFunctions(loader);
    RegisterLsStateFunctions(loader);
    RegisterLmDynamicFunctions(loader);
    RegisterLowessFunction(loader);
    // P-spline smoother: pspline_fit_agg, pspline_fit_predict_agg (#316)
    RegisterPSplineAggregateFunctions(loader);
    RegisterPathAggregateFunctions(loader);
    // Ordinal regression: ordinal_fit_agg, continuation_ratio_fit_agg, ordinal_predict (#313)
    RegisterOrdinalFunctions(loader);
    // Principal component analysis: pca_agg, pca_transform, pca_by (#303)
    RegisterPcaFunctions(loader);
    // Multiple imputation and Rubin pooling: areg_impute_agg/_by, pool_rubin_agg (#321)
    RegisterImputationFunctions(loader);
    // Multinomial logistic regression (#314)
    RegisterMultinomialFunctions(loader);
    // Resampling model validation and calibration (#318)
    RegisterModelValidationFunctions(loader);
    // Data reduction and transforms: spearman2, hoeffding_d, box_cox, varclus, redun (#320)
    RegisterReductionFunctions(loader);
    // PCA reconstruction-error drift and reference-band alert thresholds (#366, #367)
    RegisterReconstructionFunctions(loader);
    // CBPE performance estimation without labels (#365)
    RegisterCbpeFunctions(loader);

    // Register diagnostic functions
    RegisterVifFunction(loader);
    RegisterAicBicFunctions(loader);
    RegisterJarqueBeraFunction(loader);
    RegisterResidualsDiagnosticsFunction(loader);
    RegisterNullDistributionFunctions(loader);
    RegisterSplineFunctions(loader);              // #317
    RegisterSplineKnotsAggregateFunctions(loader); // #317
    RegisterEffectsFunctions(loader);              // #319

    // Register table macros for fit_predict_by functions
    RegisterFitPredictTableMacros(loader);
    // <model>_tidy_by / <model>_glance_by (#152)
    RegisterModelOutputMacros(loader);
    RegisterPosthocMacros(loader);
    // Paths, CV curves, the quantile process and partial residuals (#154, #157, #158)
    RegisterPathMacros(loader);
    // Contract `prediction` producers: fitted values with bands, marginal-effect grids (#183)
    RegisterPredictionMacros(loader);
    RegisterCurveMacros(loader);
    RegisterClassifierCurveMacros(loader);

    RegisterContractVersionFunction(loader);
    ApplyContractTags(loader, preexisting);

    datazoo::RegisterBannerOption(loader);
    // Last, so a load that fails earlier never advertises itself. Silent unless
    // stderr is a terminal and the ~/.duckdb stamp is over a day old.
    datazoo::ShowBanner(ANOFOX_STATISTICS_BANNER);
}

void AnofoxStatisticsExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
}

std::string AnofoxStatisticsExtension::Name() {
    return "anofox_statistics";
}

std::string AnofoxStatisticsExtension::Version() const {
#ifdef EXT_VERSION_ANOFOX_STATISTICS
    return EXT_VERSION_ANOFOX_STATISTICS;
#else
    return "0.16.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_statistics, loader) {
    duckdb::LoadInternal(loader);
}

DUCKDB_EXTENSION_API const char *anofox_statistics_version() {
#ifdef EXT_VERSION_ANOFOX_STATISTICS
    return EXT_VERSION_ANOFOX_STATISTICS;
#else
    return "0.16.0";
#endif
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
