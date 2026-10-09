#pragma once

#include "duckdb/function/aggregate_state.hpp"

#include <utility>

namespace duckdb {

//! Hand a source-state member to a Combine target.
//!
//! DuckDB only lets Combine consume its source when the caller passes
//! AggregateCombineType::ALLOW_DESTRUCTIVE. The window segment tree combines with
//! the default PRESERVE_INPUT and reuses its internal nodes for many frames, so
//! moving out of the source there silently empties a node that later frames still
//! read. Copy unless destruction was explicitly allowed.
template <class T>
inline T CombineTake(T &source_member, const AggregateInputData &aggr_input_data) {
	if (aggr_input_data.combine_type == AggregateCombineType::ALLOW_DESTRUCTIVE) {
		return std::move(source_member);
	}
	return source_member;
}

} // namespace duckdb
