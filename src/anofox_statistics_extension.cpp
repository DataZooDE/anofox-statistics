#define DUCKDB_EXTENSION_MAIN

#include "include/anofox_statistics_extension.hpp"

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

void AnofoxStatisticsExtension::Load(ExtensionLoader &loader) {
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

    // Register predict aggregate functions (non-rolling)
    RegisterOlsPredictAggregateFunction(loader);
    RegisterRidgePredictAggregateFunction(loader);
    RegisterWlsPredictAggregateFunction(loader);
    RegisterRlsPredictAggregateFunction(loader);
    RegisterElasticNetPredictAggregateFunction(loader);

    // Register diagnostic functions
    RegisterVifFunction(loader);
    RegisterAicBicFunctions(loader);
    RegisterJarqueBeraFunction(loader);
    RegisterResidualsDiagnosticsFunction(loader);
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
    // Register scalar functions
    duckdb::RegisterOlsFitFunction(loader);
    duckdb::RegisterRidgeFitFunction(loader);
    duckdb::RegisterElasticNetFitFunction(loader);
    duckdb::RegisterWlsFitFunction(loader);
    duckdb::RegisterPredictFunction(loader);
    duckdb::RegisterRlsFitFunction(loader);

    // Register aggregate functions
    duckdb::RegisterOlsAggregateFunction(loader);
    duckdb::RegisterRidgeAggregateFunction(loader);
    duckdb::RegisterElasticNetAggregateFunction(loader);
    duckdb::RegisterWlsAggregateFunction(loader);
    duckdb::RegisterRlsAggregateFunction(loader);
    duckdb::RegisterVifAggregateFunction(loader);
    duckdb::RegisterJarqueBeraAggregateFunction(loader);
    duckdb::RegisterResidualsDiagnosticsAggregateFunction(loader);

    // Register GLM aggregate functions
    duckdb::RegisterPoissonAggregateFunction(loader);

    // Register ALM aggregate functions
    duckdb::RegisterAlmAggregateFunction(loader);

    // Register BLS aggregate functions (includes NNLS)
    duckdb::RegisterBlsAggregateFunction(loader);

    // Register AID aggregate functions (Automatic Identification of Demand)
    duckdb::RegisterAidAggregateFunction(loader);

    // Register statistical hypothesis testing aggregate functions
    duckdb::RegisterShapiroWilkAggregateFunction(loader);
    duckdb::RegisterTTestAggregateFunction(loader);
    duckdb::RegisterPearsonAggregateFunction(loader);
    duckdb::RegisterSpearmanAggregateFunction(loader);
    duckdb::RegisterMannWhitneyAggregateFunction(loader);
    duckdb::RegisterAnovaAggregateFunction(loader);
    duckdb::RegisterKruskalWallisAggregateFunction(loader);
    duckdb::RegisterChiSquareAggregateFunction(loader);

    // Phase 1: Aggregates for existing FFI
    duckdb::RegisterKendallAggregateFunction(loader);
    duckdb::RegisterFisherExactAggregateFunction(loader);
    duckdb::RegisterBrunnerMunzelAggregateFunction(loader);
    duckdb::RegisterDAgostinoK2AggregateFunction(loader);
    duckdb::RegisterEnergyDistanceAggregateFunction(loader);
    duckdb::RegisterMmdAggregateFunction(loader);
    duckdb::RegisterTostTTestAggregateFunction(loader);

    // Phase 2: Wilcoxon signed-rank test
    duckdb::RegisterWilcoxonSignedRankAggregateFunction(loader);

    // Phase 4: Distance correlation test
    duckdb::RegisterDistanceCorAggregateFunction(loader);

    // Phase 5: Parametric tests
    duckdb::RegisterYuenAggregateFunction(loader);
    duckdb::RegisterBrownForsytheAggregateFunction(loader);

    // Phase 6: Forecast tests
    duckdb::RegisterDieboldMarianoAggregateFunction(loader);
    duckdb::RegisterClarkWestAggregateFunction(loader);

    // Phase 7: Resampling tests
    duckdb::RegisterPermutationTTestAggregateFunction(loader);

    // Phase 8: TOST equivalence test variants
    duckdb::RegisterTostPairedAggregateFunction(loader);
    duckdb::RegisterTostCorrelationAggregateFunction(loader);

    // Phase 9: Categorical tests
    duckdb::RegisterChisqGofAggregateFunction(loader);
    duckdb::RegisterPropTestOneAggregateFunction(loader);
    duckdb::RegisterPropTestTwoAggregateFunction(loader);
    duckdb::RegisterBinomTestAggregateFunction(loader);
    duckdb::RegisterCramersVAggregateFunction(loader);
    duckdb::RegisterCohenKappaAggregateFunction(loader);
    duckdb::RegisterIccAggregateFunction(loader);
    duckdb::RegisterGTestAggregateFunction(loader);
    duckdb::RegisterMcNemarAggregateFunction(loader);
    duckdb::RegisterPhiCoefficientAggregateFunction(loader);
    duckdb::RegisterContingencyCoefAggregateFunction(loader);

    // Register window aggregate functions (fit_predict)
    duckdb::RegisterOlsFitPredictFunction(loader);
    duckdb::RegisterRidgeFitPredictFunction(loader);
    duckdb::RegisterWlsFitPredictFunction(loader);
    duckdb::RegisterRlsFitPredictFunction(loader);
    duckdb::RegisterElasticNetFitPredictFunction(loader);

    // Register predict aggregate functions (non-rolling)
    duckdb::RegisterOlsPredictAggregateFunction(loader);

    // Register diagnostic functions
    duckdb::RegisterVifFunction(loader);
    duckdb::RegisterAicBicFunctions(loader);
    duckdb::RegisterJarqueBeraFunction(loader);
    duckdb::RegisterResidualsDiagnosticsFunction(loader);
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
