#pragma once

#include "duckdb/function/aggregate_function.hpp"

namespace duckdb {

//! Exception-safe wrapper for an aggregate Finalize callback.
//!
//! When a Finalize throws (e.g. an InvalidInputException for bad options or a
//! degenerate fit), DuckDB skips the state destructors for that chunk: the
//! single-scan GROUP BY path (RadixPartitionedHashTable, DESTROY_AFTER_DONE)
//! destroys states only after FinalizeStates returns, and its global-state
//! cleanup then assumes they were already destroyed. Every state in the chunk
//! leaks (seen as LeakSanitizer reports for the vectors held in our states).
//!
//! On exception this destroys the chunk's states and re-initializes them to an
//! empty state before rethrowing. An empty state owns no heap memory, so the
//! states are freed on the leaking path, and a caller that still destroys them
//! later (multi-scan paths) destroys a valid, empty state rather than freeing
//! twice.
template <aggregate_finalize_t FINALIZE, aggregate_destructor_t DESTROY, aggregate_initialize_t INIT>
struct GuardedFinalize {
	static void Call(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
	                 idx_t offset) {
		try {
			FINALIZE(state_vector, aggr_input_data, result, count, offset);
		} catch (...) {
			DESTROY(state_vector, aggr_input_data, count);
			UnifiedVectorFormat sdata;
			state_vector.ToUnifiedFormat(count, sdata);
			auto states = reinterpret_cast<data_ptr_t *>(sdata.data);
			// Our Initialize callbacks only placement-new the state and ignore the
			// AggregateFunction argument; pass a placeholder.
			static const AggregateFunction placeholder(
			    "anofox_finalize_guard", vector<LogicalType> {}, LogicalType(LogicalTypeId::ANY), aggregate_size_t(nullptr),
			    aggregate_initialize_t(nullptr), aggregate_update_t(nullptr), aggregate_combine_t(nullptr),
			    aggregate_finalize_t(nullptr), FunctionNullHandling::DEFAULT_NULL_HANDLING);
			for (idx_t i = 0; i < count; i++) {
				INIT(placeholder, states[sdata.sel->get_index(i)]);
			}
			throw;
		}
	}
};

} // namespace duckdb

//! Use in place of the Finalize argument of an AggregateFunction registration.
#define ANOFOX_GUARDED_FINALIZE(FIN, DESTROY, INIT) (&::duckdb::GuardedFinalize<FIN, DESTROY, INIT>::Call)
