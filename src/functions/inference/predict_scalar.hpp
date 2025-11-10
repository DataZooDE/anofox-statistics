#pragma once

#include "duckdb.hpp"

namespace duckdb {
namespace anofox_statistics {

/**
 * Scalar prediction functions that accept model structs from aggregates
 */
class PredictScalarFunctions {
public:
	/**
	 * Register all scalar predict functions
	 */
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
