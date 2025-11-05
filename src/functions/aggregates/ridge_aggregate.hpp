#pragma once

namespace duckdb {
class ExtensionLoader;

namespace anofox_statistics {

struct RidgeAggregateFunction {
	static void Register(ExtensionLoader &loader);
};

} // namespace anofox_statistics
} // namespace duckdb
