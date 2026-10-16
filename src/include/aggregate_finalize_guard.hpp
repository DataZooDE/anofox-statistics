#pragma once

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb_compat.hpp"

#include <unordered_set>

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
template <anofox_aggregate_finalize_t FINALIZE, aggregate_destructor_t DESTROY, anofox_aggregate_init_t INIT>
struct GuardedFinalize {
	static void Call(Vector &state_vector, AggregateFinalizeInput &aggr_input_data, Vector &result, idx_t count,
	                 idx_t offset) {
		try {
			FINALIZE(state_vector, aggr_input_data, result, count, offset);
		} catch (...) {
			DESTROY(state_vector, aggr_input_data, count);
			UnifiedVectorFormat sdata;
			state_vector.ToUnifiedFormat(count, sdata);
			auto states = UnifiedVectorFormat::GetData<data_ptr_t>(sdata);
			for (idx_t i = 0; i < count; i++) {
				INIT(states[sdata.sel->get_index(i)]);
			}
			throw;
		}
	}
};

//! Exception-safe wrapper for an aggregate Update callback.
//!
//! An Update that throws (e.g. a third group label in a two-sample test) aborts
//! the hash-aggregate sink; DuckDB then never destroys the states that the sink
//! had already created, so their vectors leak. On exception, destroy every
//! distinct state touched by this chunk and re-initialize it empty before
//! rethrowing (the states vector may repeat a pointer, hence the dedupe).
template <aggregate_update_t UPDATE, aggregate_destructor_t DESTROY, anofox_aggregate_init_t INIT>
struct GuardedUpdate {
	static void Call(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
	                 idx_t count) {
		try {
			UPDATE(inputs, aggr_input_data, input_count, state_vector, count);
		} catch (...) {
			UnifiedVectorFormat sdata;
			state_vector.ToUnifiedFormat(count, sdata);
			auto states = UnifiedVectorFormat::GetData<data_ptr_t>(sdata);
			std::unordered_set<data_ptr_t> seen;
			for (idx_t i = 0; i < count; i++) {
				auto state = states[sdata.sel->get_index(i)];
				if (!seen.insert(state).second) {
					continue;
				}
				Vector one(LogicalType::POINTER, 1);
				FlatDataMut<data_ptr_t>(one)[0] = state;
				DESTROY(one, aggr_input_data, 1);
				INIT(state);
			}
			throw;
		}
	}
};

} // namespace duckdb

//! Use in place of the Finalize argument of an AggregateFunction registration.
#define ANOFOX_GUARDED_UPDATE(UPD, DESTROY, INIT) (&::duckdb::GuardedUpdate<UPD, DESTROY, INIT>::Call)
#define ANOFOX_GUARDED_FINALIZE(FIN, DESTROY, INIT) (&::duckdb::GuardedFinalize<FIN, DESTROY, INIT>::Call)
