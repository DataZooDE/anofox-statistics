#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

// Forward declarations for function registration
void RegisterOlsFitFunction(ExtensionLoader &loader);
void RegisterRidgeFitFunction(ExtensionLoader &loader);
void RegisterElasticNetFitFunction(ExtensionLoader &loader);
void RegisterWlsFitFunction(ExtensionLoader &loader);
void RegisterPredictFunction(ExtensionLoader &loader);
void RegisterOlsAggregateFunction(ExtensionLoader &loader);
void RegisterRidgeAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetAggregateFunction(ExtensionLoader &loader);
void RegisterWlsAggregateFunction(ExtensionLoader &loader);
void RegisterRlsAggregateFunction(ExtensionLoader &loader);
void RegisterRlsFitFunction(ExtensionLoader &loader);

// Window aggregate functions (fit_predict)
void RegisterOlsFitPredictFunction(ExtensionLoader &loader);
void RegisterRidgeFitPredictFunction(ExtensionLoader &loader);
void RegisterWlsFitPredictFunction(ExtensionLoader &loader);
void RegisterRlsFitPredictFunction(ExtensionLoader &loader);
void RegisterElasticNetFitPredictFunction(ExtensionLoader &loader);

// Fit + Predict aggregate functions (fit + predict all rows, with deprecated aliases)
void RegisterOlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRidgeFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterWlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterBlsFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterAlmFitPredictAggregateFunction(ExtensionLoader &loader);
void RegisterPoissonFitPredictAggregateFunction(ExtensionLoader &loader);

// GLM aggregate functions
void RegisterPoissonAggregateFunction(ExtensionLoader &loader);

// ALM aggregate functions
void RegisterAlmAggregateFunction(ExtensionLoader &loader);

// BLS aggregate functions (includes NNLS)
void RegisterBlsAggregateFunction(ExtensionLoader &loader);

// AID aggregate functions (Automatic Identification of Demand)
void RegisterAidAggregateFunction(ExtensionLoader &loader);

// Diagnostic functions
void RegisterVifFunction(ExtensionLoader &loader);
void RegisterVifAggregateFunction(ExtensionLoader &loader);
void RegisterAicBicFunctions(ExtensionLoader &loader);
void RegisterJarqueBeraFunction(ExtensionLoader &loader);
void RegisterJarqueBeraAggregateFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsAggregateFunction(ExtensionLoader &loader);

// Statistical hypothesis testing aggregate functions
void RegisterShapiroWilkAggregateFunction(ExtensionLoader &loader);
void RegisterTTestAggregateFunction(ExtensionLoader &loader);
void RegisterPearsonAggregateFunction(ExtensionLoader &loader);
void RegisterSpearmanAggregateFunction(ExtensionLoader &loader);
void RegisterMannWhitneyAggregateFunction(ExtensionLoader &loader);
void RegisterAnovaAggregateFunction(ExtensionLoader &loader);
void RegisterKruskalWallisAggregateFunction(ExtensionLoader &loader);
void RegisterChiSquareAggregateFunction(ExtensionLoader &loader);

// Phase 1: Aggregates for existing FFI
void RegisterKendallAggregateFunction(ExtensionLoader &loader);
void RegisterFisherExactAggregateFunction(ExtensionLoader &loader);
void RegisterBrunnerMunzelAggregateFunction(ExtensionLoader &loader);
void RegisterDAgostinoK2AggregateFunction(ExtensionLoader &loader);
void RegisterEnergyDistanceAggregateFunction(ExtensionLoader &loader);
void RegisterMmdAggregateFunction(ExtensionLoader &loader);
void RegisterTostTTestAggregateFunction(ExtensionLoader &loader);
void RegisterWilcoxonSignedRankAggregateFunction(ExtensionLoader &loader);
void RegisterDistanceCorAggregateFunction(ExtensionLoader &loader);
void RegisterYuenAggregateFunction(ExtensionLoader &loader);
void RegisterBrownForsytheAggregateFunction(ExtensionLoader &loader);
void RegisterDieboldMarianoAggregateFunction(ExtensionLoader &loader);
void RegisterClarkWestAggregateFunction(ExtensionLoader &loader);
void RegisterPermutationTTestAggregateFunction(ExtensionLoader &loader);
void RegisterTostPairedAggregateFunction(ExtensionLoader &loader);
void RegisterTostCorrelationAggregateFunction(ExtensionLoader &loader);
void RegisterChisqGofAggregateFunction(ExtensionLoader &loader);
void RegisterPropTestOneAggregateFunction(ExtensionLoader &loader);
void RegisterPropTestTwoAggregateFunction(ExtensionLoader &loader);
void RegisterBinomTestAggregateFunction(ExtensionLoader &loader);
void RegisterCramersVAggregateFunction(ExtensionLoader &loader);
void RegisterCohenKappaAggregateFunction(ExtensionLoader &loader);
void RegisterIccAggregateFunction(ExtensionLoader &loader);
void RegisterGTestAggregateFunction(ExtensionLoader &loader);
void RegisterMcNemarAggregateFunction(ExtensionLoader &loader);
void RegisterPhiCoefficientAggregateFunction(ExtensionLoader &loader);
void RegisterContingencyCoefAggregateFunction(ExtensionLoader &loader);

// Extension class required for static linking
class AnofoxStatisticsExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
