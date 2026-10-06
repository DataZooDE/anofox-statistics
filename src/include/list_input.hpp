#pragma once

#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/vector.hpp"

#include <limits>

namespace duckdb {

//! A LIST that contains a NULL element is itself valid, so the list-level validity
//! mask says nothing about its elements. These helpers read the child mask so a
//! NULL element is never consumed as whatever bytes sit in the child buffer.

//! True when any element of the list entry is NULL.
inline bool ListHasNullElement(const ValidityMask &child_validity, const list_entry_t &entry) {
	if (child_validity.AllValid()) {
		return false;
	}
	for (idx_t j = 0; j < entry.length; j++) {
		if (!child_validity.RowIsValid(entry.offset + j)) {
			return true;
		}
	}
	return false;
}

//! The element at child_idx, or NaN when it is NULL.
inline double ListChildValue(const double *child_data, const ValidityMask &child_validity, idx_t child_idx) {
	return child_validity.RowIsValid(child_idx) ? child_data[child_idx] : std::numeric_limits<double>::quiet_NaN();
}

} // namespace duckdb
