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

// Diagnostic functions
void RegisterVifFunction(ExtensionLoader &loader);
void RegisterVifAggregateFunction(ExtensionLoader &loader);
void RegisterAicBicFunctions(ExtensionLoader &loader);
void RegisterJarqueBeraFunction(ExtensionLoader &loader);
void RegisterJarqueBeraAggregateFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsFunction(ExtensionLoader &loader);
void RegisterResidualsDiagnosticsAggregateFunction(ExtensionLoader &loader);

// Extension class required for static linking
class AnofoxStatsExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
    std::string Version() const override;
};

} // namespace duckdb
