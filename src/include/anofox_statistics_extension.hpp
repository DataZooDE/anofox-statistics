#pragma once

#include "duckdb.hpp"

namespace duckdb {

/**
 * @brief AnofoxStatistics DuckDB Extension
 *
 * Provides SQL bindings for the AnofoxStatistics C++ library,
 * enabling statistical regression analysis directly in DuckDB.
 *
 * Functions provided (Phase 1):
 * - anofox_ols_fit: Ordinary Least Squares regression
 *
 * Environment variables:
 * - ANOFOX_LOG_LEVEL: Control logging (trace, debug, info, warn, error)
 */
class AnofoxStatisticsExtension : public Extension {
public:
	/**
	 * @brief Load all extension functions into DuckDB
	 *
	 * @param loader Extension loader context
	 */
	void Load(ExtensionLoader &loader) override;

	/**
	 * @brief Get extension name
	 *
	 * @return "anofox-statistics"
	 */
	std::string Name() override;

	/**
	 * @brief Get extension version
	 *
	 * @return Version string
	 */
	std::string Version() const override;
};

} // namespace duckdb
