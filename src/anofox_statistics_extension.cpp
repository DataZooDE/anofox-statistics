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
