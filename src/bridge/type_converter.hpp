#pragma once

#include "duckdb.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Bridge layer for converting between DuckDB and Eigen/AnofoxStatistics types
 *
 * Handles conversion of:
 * - DuckDB Vector → Eigen::VectorXd
 * - DuckDB vectors → Eigen::MatrixXd
 * - Eigen::VectorXd → DuckDB Vector
 * - NULL/NaN semantics
 * - Type safety and validation
 */
class TypeConverter {
public:
	/**
	 * @brief Extract a single column from DataChunk as Eigen::VectorXd
	 *
	 * @param chunk Input DataChunk
	 * @param col_index Column index
	 * @return Eigen::VectorXd with values, NaN for NULLs
	 * @throws DuckDBException if column index invalid
	 */
	static Eigen::VectorXd ExtractDoubleColumn(const DataChunk &chunk, idx_t col_index);

	/**
	 * @brief Extract multiple columns from DataChunk as Eigen::MatrixXd
	 *
	 * @param chunk Input DataChunk
	 * @param col_indices Column indices to extract
	 * @return Eigen::MatrixXd with shape (chunk.size, col_indices.size)
	 * @throws DuckDBException if any column index invalid or type mismatch
	 */
	static Eigen::MatrixXd ExtractDoubleMatrix(const DataChunk &chunk, const std::vector<idx_t> &col_indices);

	/**
	 * @brief Extract column names from DataChunk
	 *
	 * @param chunk Input DataChunk
	 * @return Vector of column names
	 */
	static std::vector<std::string> ExtractColumnNames(const DataChunk &chunk);

	/**
	 * @brief Convert Eigen vector to DuckDB Vector
	 *
	 * @param result Output vector to populate
	 * @param eigen_vec Input Eigen vector
	 * @param size Number of elements to convert
	 * @throws DuckDBException on conversion error
	 */
	static void SetDoubleVector(Vector &result, const Eigen::VectorXd &eigen_vec, idx_t size);

	/**
	 * @brief Create a validity mask for NULL values in a column
	 *
	 * @param vector DuckDB vector
	 * @return Validity mask (true = valid, false = NULL)
	 */
	static std::vector<bool> CreateValidityMask(const Vector &vector);

	/**
	 * @brief Count non-NULL values in column
	 *
	 * @param vector DuckDB vector
	 * @param start Start index
	 * @param end End index (0 means use vector size)
	 * @return Number of valid (non-NULL) values
	 */
	static idx_t CountValidValues(const Vector &vector, idx_t start = 0, idx_t end = 0);

	/**
	 * @brief Check if column has any NULL values
	 *
	 * @param vector DuckDB vector
	 * @return true if there are any NULLs, false otherwise
	 */
	static bool HasNullValues(const Vector &vector);

	/**
	 * @brief Get logical type of vector
	 *
	 * @param vector DuckDB vector
	 * @return LogicalType of the vector
	 */
	static const LogicalType &GetVectorType(const Vector &vector);

private:
	/**
	 * @brief Helper to safely get double value from vector
	 *
	 * @param vector Source vector
	 * @param index Element index
	 * @param is_null Output parameter - set to true if value is NULL
	 * @return double value (undefined if is_null is true)
	 */
	static double GetDoubleValue(const Vector &vector, idx_t index, bool &is_null);

	/**
	 * @brief Validate that all values in vector are numeric
	 *
	 * @param vector Vector to validate
	 * @throws DuckDBException if non-numeric values found
	 */
	static void ValidateNumericColumn(const Vector &vector);
};

} // namespace anofox_statistics
} // namespace duckdb
