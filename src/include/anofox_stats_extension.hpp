#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

// Forward declarations for function registration
void RegisterOlsFitFunction(ExtensionLoader &loader);
void RegisterRidgeFitFunction(ExtensionLoader &loader);
void RegisterElasticNetFitFunction(ExtensionLoader &loader);
void RegisterOlsAggregateFunction(ExtensionLoader &loader);
void RegisterRidgeAggregateFunction(ExtensionLoader &loader);
void RegisterElasticNetAggregateFunction(ExtensionLoader &loader);

} // namespace duckdb
