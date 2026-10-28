#pragma once

#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor_state.hpp"

namespace duckdb {

//! Runs `impl` only on the rows where none of the first `n_data_args` arguments is NULL;
//! the other rows of `result` are NULL. `args` must be flat.
//!
//! The list-input scalar fits (ols_fit, ...) register their options overload with
//! SPECIAL_HANDLING so a NULL options argument means "no options" (as for the aggregates
//! and table macros) instead of turning the whole call into NULL (#276). DuckDB then no
//! longer skips NULL data rows, so this wrapper does: a NULL row's list entry is never read.
template <class IMPL>
void ExecuteOnNonNullRows(DataChunk &args, ExpressionState &state, Vector &result, idx_t n_data_args, IMPL impl) {
	const idx_t count = args.size();
	SelectionVector valid_sel(count);
	idx_t n_valid = 0;
	for (idx_t row = 0; row < count; row++) {
		bool valid = true;
		for (idx_t c = 0; c < n_data_args && valid; c++) {
			valid = FlatVector::Validity(args.data[c]).RowIsValid(row);
		}
		if (valid) {
			valid_sel.set_index(n_valid++, row);
		}
	}
	if (n_valid == count) {
		impl(args, state, result);
		return;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	if (n_valid > 0) {
		DataChunk valid_args;
		valid_args.InitializeEmpty(args.GetTypes());
		valid_args.Slice(args, valid_sel, n_valid);
		valid_args.Flatten();
		Vector valid_result(result.GetType(), n_valid);
		impl(valid_args, state, valid_result);
		// Scatter back: output row valid_sel[i] takes valid_result[i]; NULL rows read row 0
		// and are overwritten with NULL below.
		SelectionVector back(count);
		for (idx_t row = 0; row < count; row++) {
			back.set_index(row, 0);
		}
		for (idx_t i = 0; i < n_valid; i++) {
			back.set_index(valid_sel.get_index(i), i);
		}
		VectorOperations::Copy(valid_result, result, back, count, 0, 0);
	}
	for (idx_t row = 0, i = 0; row < count; row++) {
		if (i < n_valid && valid_sel.get_index(i) == row) {
			i++;
		} else {
			FlatVector::SetNull(result, row, true);
		}
	}
}

} // namespace duckdb
