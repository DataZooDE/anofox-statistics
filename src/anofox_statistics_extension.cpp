#define DUCKDB_EXTENSION_MAIN

#include "include/anofox_statistics_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "telemetry.hpp"

namespace duckdb {

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

void LoadInternal(ExtensionLoader &loader) {
    // Register telemetry options
    RegisterTelemetryOptions(loader);

    // Initialize and capture extension load event
    auto &telemetry = PostHogTelemetry::Instance();
    telemetry.SetAPIKey("phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t");

    std::string version;
#ifdef EXT_VERSION_ANOFOX_STATISTICS
    version = EXT_VERSION_ANOFOX_STATISTICS;
#else
    version = "0.1.0";
#endif
    telemetry.CaptureExtensionLoad("anofox_statistics", version);

    // Register scalar functions
    RegisterOlsFitFunction(loader);
    RegisterRidgeFitFunction(loader);
    RegisterElasticNetFitFunction(loader);
    RegisterWlsFitFunction(loader);
    RegisterPredictFunction(loader);
    RegisterRlsFitFunction(loader);

    // Register aggregate functions
    RegisterOlsAggregateFunction(loader);
    RegisterRidgeAggregateFunction(loader);
    RegisterElasticNetAggregateFunction(loader);
    RegisterWlsAggregateFunction(loader);
    RegisterRlsAggregateFunction(loader);
    RegisterVifAggregateFunction(loader);
    RegisterJarqueBeraAggregateFunction(loader);
    RegisterResidualsDiagnosticsAggregateFunction(loader);

    // Register GLM aggregate functions
    RegisterPoissonAggregateFunction(loader);

    // Register ALM aggregate functions
    RegisterAlmAggregateFunction(loader);

    // Register BLS aggregate functions (includes NNLS)
    RegisterBlsAggregateFunction(loader);

    // Register AID aggregate functions (Automatic Identification of Demand)
    RegisterAidAggregateFunction(loader);

    // Register statistical hypothesis testing aggregate functions
    RegisterShapiroWilkAggregateFunction(loader);
    RegisterTTestAggregateFunction(loader);
    RegisterPearsonAggregateFunction(loader);
    RegisterSpearmanAggregateFunction(loader);
    RegisterMannWhitneyAggregateFunction(loader);
    RegisterAnovaAggregateFunction(loader);
    RegisterKruskalWallisAggregateFunction(loader);
    RegisterChiSquareAggregateFunction(loader);

    // Phase 1: Aggregates for existing FFI
    RegisterKendallAggregateFunction(loader);
    RegisterFisherExactAggregateFunction(loader);
    RegisterBrunnerMunzelAggregateFunction(loader);
    RegisterDAgostinoK2AggregateFunction(loader);
    RegisterEnergyDistanceAggregateFunction(loader);
    RegisterMmdAggregateFunction(loader);
    RegisterTostTTestAggregateFunction(loader);

    // Phase 2: Wilcoxon signed-rank test
    RegisterWilcoxonSignedRankAggregateFunction(loader);

    // Phase 4: Distance correlation test
    RegisterDistanceCorAggregateFunction(loader);

    // Phase 5: Parametric tests
    RegisterYuenAggregateFunction(loader);
    RegisterBrownForsytheAggregateFunction(loader);

    // Phase 6: Forecast tests
    RegisterDieboldMarianoAggregateFunction(loader);
    RegisterClarkWestAggregateFunction(loader);

    // Phase 7: Resampling tests
    RegisterPermutationTTestAggregateFunction(loader);

    // Phase 8: TOST equivalence test variants
    RegisterTostPairedAggregateFunction(loader);
    RegisterTostCorrelationAggregateFunction(loader);

    // Phase 9: Categorical tests
    RegisterChisqGofAggregateFunction(loader);
    RegisterPropTestOneAggregateFunction(loader);
    RegisterPropTestTwoAggregateFunction(loader);
    RegisterBinomTestAggregateFunction(loader);
    RegisterCramersVAggregateFunction(loader);
    RegisterCohenKappaAggregateFunction(loader);
    RegisterIccAggregateFunction(loader);
    RegisterGTestAggregateFunction(loader);
    RegisterMcNemarAggregateFunction(loader);
    RegisterPhiCoefficientAggregateFunction(loader);
    RegisterContingencyCoefAggregateFunction(loader);

    // Register window aggregate functions (fit_predict)
    RegisterOlsFitPredictFunction(loader);
    RegisterRidgeFitPredictFunction(loader);
    RegisterWlsFitPredictFunction(loader);
    RegisterRlsFitPredictFunction(loader);
    RegisterElasticNetFitPredictFunction(loader);

    // Register fit + predict aggregate functions (with deprecated aliases)
    RegisterOlsFitPredictAggregateFunction(loader);
    RegisterRidgeFitPredictAggregateFunction(loader);
    RegisterWlsFitPredictAggregateFunction(loader);
    RegisterRlsFitPredictAggregateFunction(loader);
    RegisterElasticNetFitPredictAggregateFunction(loader);
    RegisterBlsFitPredictAggregateFunction(loader);
    RegisterAlmFitPredictAggregateFunction(loader);
    RegisterPoissonFitPredictAggregateFunction(loader);

    // Register diagnostic functions
    RegisterVifFunction(loader);
    RegisterAicBicFunctions(loader);
    RegisterJarqueBeraFunction(loader);
    RegisterResidualsDiagnosticsFunction(loader);

    // Register table macros for fit_predict_by functions
    RegisterFitPredictTableMacros(loader);
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
    return "0.1.0";
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
    return "0.1.0";
#endif
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
