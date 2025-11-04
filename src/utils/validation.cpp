#include "validation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include <sstream>
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

void ValidationUtils::ValidateColumnIndex(const DataChunk &chunk, idx_t col_index, const std::string &operation_name) {
	if (col_index >= chunk.ColumnCount()) {
		throw InvalidInputException("Column index %d out of bounds for %s (max index: %d)", col_index,
		                            operation_name.c_str(), chunk.ColumnCount() - 1);
	}
}

void ValidationUtils::ValidateColumnIndices(const DataChunk &chunk, const std::vector<idx_t> &col_indices,
                                            const std::string &operation_name) {
	for (size_t i = 0; i < col_indices.size(); i++) {
		idx_t idx = col_indices[i];
		if (idx >= chunk.ColumnCount()) {
			throw InvalidInputException("Column index %d (position %zu) out of bounds for %s (max index: %d)", idx, i,
			                            operation_name.c_str(), chunk.ColumnCount() - 1);
		}
	}
}

idx_t ValidationUtils::FindColumnByName(const std::vector<std::string> &column_names, const std::string &col_name) {
	for (idx_t i = 0; i < column_names.size(); i++) {
		if (column_names[i] == col_name) {
			return i;
		}
	}
	throw InvalidInputException("Column '%s' not found in table", col_name.c_str());
}

std::vector<idx_t> ValidationUtils::FindColumnsByNames(const std::vector<std::string> &column_names,
                                                       const std::vector<std::string> &col_names) {
	std::vector<idx_t> result;
	result.reserve(col_names.size());

	for (const auto &name : col_names) {
		result.push_back(FindColumnByName(column_names, name));
	}

	return result;
}

void ValidationUtils::ValidateNumericType(const Vector &vector, const std::string &col_name) {
	const auto &type = vector.GetType();

	bool is_numeric = (type.id() == LogicalTypeId::DOUBLE || type.id() == LogicalTypeId::FLOAT ||
	                   type.id() == LogicalTypeId::INTEGER || type.id() == LogicalTypeId::BIGINT ||
	                   type.id() == LogicalTypeId::SMALLINT || type.id() == LogicalTypeId::TINYINT ||
	                   type.id() == LogicalTypeId::HUGEINT || type.id() == LogicalTypeId::UTINYINT ||
	                   type.id() == LogicalTypeId::USMALLINT || type.id() == LogicalTypeId::UINTEGER ||
	                   type.id() == LogicalTypeId::UBIGINT);

	if (!is_numeric) {
		std::string col_info = col_name.empty() ? "" : ("column '" + col_name + "': ");
		throw InvalidInputException("Invalid type for %s%s (required numeric type)", col_info.c_str(),
		                            type.ToString().c_str());
	}
}

void ValidationUtils::ValidateNoNulls(const Vector &vector, idx_t size, const std::string &col_name) {
	auto &validity = FlatVector::Validity(vector);

	for (idx_t i = 0; i < size; i++) {
		if (!validity.RowIsValid(i)) {
			std::string col_info = col_name.empty() ? "" : ("'" + col_name + "': ");
			throw InvalidInputException("NULL value found at row %d in %sNULL values not allowed for regression", i,
			                            col_info.c_str());
		}
	}
}

void ValidationUtils::ValidateColumn(const Vector &vector, idx_t size, const std::string &col_name) {
	ValidateNumericType(vector, col_name);
	ValidateNoNulls(vector, size, col_name);
}

void ValidationUtils::ValidateDataChunk(const DataChunk &chunk) {
	if (chunk.ColumnCount() == 0) {
		throw InvalidInputException("DataChunk has no columns");
	}

	if (chunk.size() == 0) {
		throw InvalidInputException("DataChunk is empty (no rows)");
	}

	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		auto &vec = chunk.data[i];
		if (vec.GetType().id() == LogicalTypeId::INVALID) {
			throw InvalidInputException("Column %d has invalid type", i);
		}
	}
}

void ValidationUtils::ValidateDimensions(idx_t rows, idx_t cols, idx_t actual_rows, idx_t actual_cols) {
	if (actual_rows != rows) {
		throw InvalidInputException("Row count mismatch: expected %d, got %d", rows, actual_rows);
	}

	if (actual_cols != cols) {
		throw InvalidInputException("Column count mismatch: expected %d, got %d", cols, actual_cols);
	}
}

bool ValidationUtils::HasNonFiniteValues(const Vector &vector, idx_t size, const std::string &col_name) {
	ValidateNumericType(vector, col_name);

	auto data = FlatVector::GetData<double>(vector);
	auto &validity = FlatVector::Validity(vector);

	for (idx_t i = 0; i < size; i++) {
		if (validity.RowIsValid(i)) {
			double val = data[i];
			if (std::isnan(val) || std::isinf(val)) {
				return true;
			}
		}
	}

	return false;
}

idx_t ValidationUtils::CountNullValues(const Vector &vector, idx_t size) {
	auto &validity = FlatVector::Validity(vector);
	idx_t count = 0;

	for (idx_t i = 0; i < size; i++) {
		if (!validity.RowIsValid(i)) {
			count++;
		}
	}

	return count;
}

std::string ValidationUtils::GetTypeName(const LogicalType &type) {
	return type.ToString();
}

void ValidationUtils::ValidateNonEmptyColumnList(const std::vector<std::string> &col_names,
                                                 const std::string &operation_name) {
	if (col_names.empty()) {
		throw InvalidInputException("Empty column list provided for %s (at least one column required)",
		                            operation_name.c_str());
	}
}

void ValidationUtils::ValidateRequiredColumns(const std::vector<std::string> &column_names,
                                              const std::vector<std::string> &required_cols) {
	ValidateNonEmptyColumnList(required_cols, "required columns check");

	for (const auto &col_name : required_cols) {
		try {
			FindColumnByName(column_names, col_name);
		} catch (const InvalidInputException &) {
			throw InvalidInputException("Required column '%s' not found in table", col_name.c_str());
		}
	}
}

} // namespace anofox_statistics
} // namespace duckdb
