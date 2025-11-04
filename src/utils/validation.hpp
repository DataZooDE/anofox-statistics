#pragma once

#include "duckdb.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Input validation utilities for the bridge layer
 *
 * Provides validation for:
 * - Column existence and naming
 * - Data types (numeric, non-null constraints)
 * - Data chunk integrity
 * - Array bounds and dimensions
 */
class ValidationUtils {
public:
	/**
	 * @brief Validate that a column index is within bounds
	 *
	 * @param chunk DataChunk to validate against
	 * @param col_index Column index to check
	 * @param operation_name Name of operation for error message
	 * @throws DuckDBException if index out of bounds
	 */
	static void ValidateColumnIndex(const DataChunk &chunk, idx_t col_index,
	                                const std::string &operation_name = "column access");

	/**
	 * @brief Validate that all column indices are within bounds
	 *
	 * @param chunk DataChunk to validate against
	 * @param col_indices Vector of column indices to check
	 * @param operation_name Name of operation for error message
	 * @throws DuckDBException if any index out of bounds
	 */
	static void ValidateColumnIndices(const DataChunk &chunk, const std::vector<idx_t> &col_indices,
	                                  const std::string &operation_name = "matrix extraction");

	/**
	 * @brief Validate that column exists by name (v1.4.1 API)
	 *
	 * @param column_names Vector of column names to search
	 * @param col_name Column name to find
	 * @return Column index if found
	 * @throws DuckDBException if not found
	 */
	static idx_t FindColumnByName(const std::vector<std::string> &column_names, const std::string &col_name);

	/**
	 * @brief Validate that all named columns exist (v1.4.1 API)
	 *
	 * @param column_names Vector of column names to search
	 * @param col_names Vector of column names to find
	 * @return Vector of column indices in same order
	 * @throws DuckDBException if any not found
	 */
	static std::vector<idx_t> FindColumnsByNames(const std::vector<std::string> &column_names,
	                                             const std::vector<std::string> &col_names);

	/**
	 * @brief Validate that column is numeric type
	 *
	 * @param vector Vector to check
	 * @param col_name Optional column name for error message
	 * @throws DuckDBException if not numeric
	 */
	static void ValidateNumericType(const Vector &vector, const std::string &col_name = "");

	/**
	 * @brief Validate that column does not contain NULLs (v1.4.1 API)
	 *
	 * @param vector Vector to check
	 * @param size Number of rows to check
	 * @param col_name Optional column name for error message
	 * @throws DuckDBException if NULLs found
	 */
	static void ValidateNoNulls(const Vector &vector, idx_t size, const std::string &col_name = "");

	/**
	 * @brief Validate that all rows have valid values (v1.4.1 API)
	 *
	 * Checks that:
	 * - All values are numeric
	 * - No NULLs present
	 * - Vector is properly formed
	 *
	 * @param vector Vector to validate
	 * @param size Number of rows to check
	 * @param col_name Optional column name for error message
	 * @throws DuckDBException on validation failure
	 */
	static void ValidateColumn(const Vector &vector, idx_t size, const std::string &col_name = "");

	/**
	 * @brief Validate that DataChunk is well-formed
	 *
	 * @param chunk DataChunk to check
	 * @throws DuckDBException if malformed
	 */
	static void ValidateDataChunk(const DataChunk &chunk);

	/**
	 * @brief Validate dimensions match expectations
	 *
	 * @param rows Expected number of rows
	 * @param cols Expected number of columns
	 * @param actual_rows Actual number of rows
	 * @param actual_cols Actual number of columns
	 * @throws DuckDBException if mismatch
	 */
	static void ValidateDimensions(idx_t rows, idx_t cols, idx_t actual_rows, idx_t actual_cols);

	/**
	 * @brief Check if column contains any non-finite values (v1.4.1 API)
	 *
	 * @param vector Vector to check
	 * @param size Number of rows to check
	 * @param col_name Optional column name for error message
	 * @return true if any NaN or Inf found
	 */
	static bool HasNonFiniteValues(const Vector &vector, idx_t size, const std::string &col_name = "");

	/**
	 * @brief Count number of NULL values in column (v1.4.1 API)
	 *
	 * @param vector Vector to check
	 * @param size Number of rows to check
	 * @return Number of NULL values
	 */
	static idx_t CountNullValues(const Vector &vector, idx_t size);

	/**
	 * @brief Get human-readable type name
	 *
	 * @param type LogicalType to describe
	 * @return String description of type
	 */
	static std::string GetTypeName(const LogicalType &type);

	/**
	 * @brief Validate that array of column names is not empty
	 *
	 * @param col_names Vector of names to check
	 * @param operation_name Name of operation for error message
	 * @throws DuckDBException if empty
	 */
	static void ValidateNonEmptyColumnList(const std::vector<std::string> &col_names,
	                                       const std::string &operation_name = "operation");

	/**
	 * @brief Validate that all required columns exist (v1.4.1 API)
	 *
	 * @param column_names Vector of available column names
	 * @param required_cols Vector of required column names
	 * @throws DuckDBException if any required column missing
	 */
	static void ValidateRequiredColumns(const std::vector<std::string> &column_names,
	                                    const std::vector<std::string> &required_cols);
};

} // namespace anofox_statistics
} // namespace duckdb
