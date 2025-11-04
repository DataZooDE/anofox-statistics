#include "type_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include <cmath>
#include <limits>

namespace duckdb {
namespace anofox_statistics {

Eigen::VectorXd TypeConverter::ExtractDoubleColumn(const DataChunk &chunk, idx_t col_index) {

	if (col_index >= chunk.ColumnCount()) {
		throw InvalidInputException("Column index %d out of bounds (max %d)", col_index, chunk.ColumnCount() - 1);
	}

	auto &vector = chunk.data[col_index];
	ValidateNumericColumn(vector);

	Eigen::VectorXd result(chunk.size());

	for (idx_t i = 0; i < chunk.size(); i++) {
		bool is_null = false;
		double val = GetDoubleValue(vector, i, is_null);
		if (!is_null) {
			result[i] = val;
		} else {
			// NULL → NaN
			result[i] = std::numeric_limits<double>::quiet_NaN();
		}
	}

	return result;
}

Eigen::MatrixXd TypeConverter::ExtractDoubleMatrix(const DataChunk &chunk, const std::vector<idx_t> &col_indices) {

	if (col_indices.empty()) {
		throw InvalidInputException("No columns specified for matrix extraction");
	}

	// Validate all column indices
	for (auto idx : col_indices) {
		if (idx >= chunk.ColumnCount()) {
			throw InvalidInputException("Column index %d out of bounds", idx);
		}
	}

	// Create matrix with rows = chunk.size(), cols = col_indices.size()
	Eigen::MatrixXd result(chunk.size(), col_indices.size());

	// Extract each column
	for (size_t col = 0; col < col_indices.size(); col++) {
		auto &vector = chunk.data[col_indices[col]];
		ValidateNumericColumn(vector);

		for (idx_t row = 0; row < chunk.size(); row++) {
			bool is_null = false;
			double val = GetDoubleValue(vector, row, is_null);
			if (!is_null) {
				result(row, col) = val;
			} else {
				// NULL → NaN
				result(row, col) = std::numeric_limits<double>::quiet_NaN();
			}
		}
	}

	return result;
}

std::vector<std::string> TypeConverter::ExtractColumnNames(const DataChunk &chunk) {
	std::vector<std::string> names;
	names.reserve(chunk.ColumnCount());

	// DataChunk uses ColumnCount() for columns, but we don't have direct column names access
	// Fall back to generating column names based on index
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		names.push_back("col_" + std::to_string(i));
	}

	return names;
}

void TypeConverter::SetDoubleVector(Vector &result, const Eigen::VectorXd &eigen_vec, idx_t size) {

	if ((idx_t)eigen_vec.size() < size) {
		throw InvalidInputException("Eigen vector size (%d) less than required size (%d)", eigen_vec.size(), size);
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto flat = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);

	for (idx_t i = 0; i < size; i++) {
		double val = eigen_vec[i];
		if (std::isnan(val) || std::isinf(val)) {
			// NaN/Inf → NULL
			mask.SetInvalid(i);
		} else {
			flat[i] = val;
			mask.SetValid(i);
		}
	}
}

std::vector<bool> TypeConverter::CreateValidityMask(const Vector &vector) {
	// NOTE: This function needs the actual size from calling context
	// For now, return empty vector - should be called with size parameter
	std::vector<bool> mask;
	return mask;
}

idx_t TypeConverter::CountValidValues(const Vector &vector, idx_t start, idx_t end) {

	// NOTE: This function needs the actual size from calling context
	// For now, return 0 - should be called with proper size parameter
	if (end <= start) {
		return 0;
	}

	auto &validity = FlatVector::Validity(vector);
	idx_t count = 0;

	for (idx_t i = start; i < end; i++) {
		if (validity.RowIsValid(i)) {
			count++;
		}
	}

	return count;
}

bool TypeConverter::HasNullValues(const Vector &vector) {
	// NOTE: This function needs the actual size from calling context
	// For now, conservatively return false
	// In practice, this should be called with size parameter from DataChunk
	return false;
}

const LogicalType &TypeConverter::GetVectorType(const Vector &vector) {
	return vector.GetType();
}

double TypeConverter::GetDoubleValue(const Vector &vector, idx_t index, bool &is_null) {

	auto &validity = FlatVector::Validity(vector);

	if (!validity.RowIsValid(index)) {
		is_null = true;
		return 0.0; // Return dummy value
	}

	is_null = false;
	auto flat = FlatVector::GetData<double>(vector);
	return flat[index];
}

void TypeConverter::ValidateNumericColumn(const Vector &vector) {
	const auto &type = vector.GetType();

	// Check if type is numeric (DOUBLE, FLOAT, INTEGER, BIGINT, etc.)
	if (type.id() != LogicalTypeId::DOUBLE && type.id() != LogicalTypeId::FLOAT &&
	    type.id() != LogicalTypeId::INTEGER && type.id() != LogicalTypeId::BIGINT &&
	    type.id() != LogicalTypeId::SMALLINT && type.id() != LogicalTypeId::TINYINT &&
	    type.id() != LogicalTypeId::HUGEINT && type.id() != LogicalTypeId::UTINYINT &&
	    type.id() != LogicalTypeId::USMALLINT && type.id() != LogicalTypeId::UBIGINT) {

		throw InvalidInputException("Column type %s is not numeric (required for regression)", type.ToString().c_str());
	}
}

} // namespace anofox_statistics
} // namespace duckdb
