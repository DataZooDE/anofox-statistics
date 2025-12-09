#define DUCKDB_EXTENSION_MAIN

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "include/anofox_stats_extension.hpp"

namespace duckdb {

// Forward declarations are in header

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(anofox_stats, loader) {
    // Register scalar functions
    duckdb::RegisterOlsFitFunction(loader);
    duckdb::RegisterRidgeFitFunction(loader);
    duckdb::RegisterElasticNetFitFunction(loader);

    // Register aggregate functions
    duckdb::RegisterOlsAggregateFunction(loader);
    duckdb::RegisterRidgeAggregateFunction(loader);
    duckdb::RegisterElasticNetAggregateFunction(loader);
}

DUCKDB_EXTENSION_API const char *anofox_stats_version() {
#ifdef EXT_VERSION_ANOFOX_STATS
    return EXT_VERSION_ANOFOX_STATS;
#else
    return "0.1.0";
#endif
}

}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
