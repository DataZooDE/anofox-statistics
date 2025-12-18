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

// Predict aggregate functions (non-rolling fit + predict all rows)
void RegisterOlsPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRidgePredictAggregateFunction(ExtensionLoader &loader);
void RegisterWlsPredictAggregateFunction(ExtensionLoader &loader);
void RegisterRlsPredictAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetPredictAggregateFunction(ExtensionLoader &loader);

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

// Extension class required for static linking
class AnofoxStatisticsExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
