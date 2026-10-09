#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Helpers for filling STRUCT result children in aggregate Finalize functions.

//! Set a VARCHAR child; a null pointer stores NULL.
inline void SetResultString(Vector &child, idx_t row, const char *value) {
	if (!value) {
		FlatVector::SetNull(child, row, true);
		return;
	}
	FlatVector::GetData<string_t>(child)[row] = StringVector::AddString(child, value);
}

//! Store NULL in a child of any type.
inline void SetResultNull(Vector &child, idx_t row) {
	FlatVector::SetNull(child, row, true);
}

} // namespace duckdb
