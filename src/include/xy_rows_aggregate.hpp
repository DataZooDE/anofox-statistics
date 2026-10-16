#pragma once

// Aggregate state that collects the complete (y, x) rows of a group, shared by
// the coefficient-returning aggregates of extra_fit_aggregates.cpp and the
// path / CV / quantile-process aggregates of path_aggregates.cpp.
//
// Rows with a NULL y, a NULL x list or a NULL list element are skipped. An
// optional per-row `extra` value (a CV fold id) travels with the row; rows
// whose extra value is NULL are skipped too.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb_compat.hpp"

#include "aggregate_combine.hpp"
#include "anofox_stats_ffi.h"
#include "list_input.hpp"

namespace duckdb {

struct XyRowsState {
	vector<double> y_values;
	vector<vector<double>> x_columns;
	vector<double> extra;
	idx_t n_features;
	bool initialized;

	XyRowsState() : n_features(0), initialized(false) {
	}
};

//! State pointers of a state vector, whatever its vector type.
struct XyRowsStatePointers {
	UnifiedVectorFormat format;
	XyRowsStatePointers(Vector &v, idx_t count) {
		v.ToUnifiedFormat(count, format);
	}
	XyRowsState &operator[](idx_t i) const {
		return *reinterpret_cast<XyRowsState *const *>(format.data)[format.sel->get_index(i)];
	}
};

inline void XyRowsInitialize(data_ptr_t state_p) {
	new (state_p) XyRowsState();
}

inline void XyRowsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	XyRowsStatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		states[i].~XyRowsState();
	}
}

//! Update for (y DOUBLE, x LIST(DOUBLE)[, extra BIGINT][, options]). With
//! `with_extra`, inputs[2] is the per-row extra value.
inline void XyRowsUpdateImpl(Vector inputs[], idx_t count, Vector &state_vector, bool with_extra) {
	UnifiedVectorFormat y_data, x_data, e_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	if (with_extra) {
		inputs[2].ToUnifiedFormat(count, e_data);
	}
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto states = (XyRowsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		double extra = 0.0;
		if (with_extra) {
			auto e_idx = e_data.sel->get_index(i);
			if (!e_data.validity.RowIsValid(e_idx)) {
				continue;
			}
			extra = static_cast<double>(UnifiedVectorFormat::GetData<int64_t>(e_data)[e_idx]);
		}
		auto list_entry = x_list_data[x_idx];
		if (ListHasNullElement(x_child_validity, list_entry)) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = list_entry.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (list_entry.length != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            list_entry.length);
		}
		state.y_values.push_back(y_values[y_idx]);
		for (idx_t j = 0; j < state.n_features; j++) {
			state.x_columns[j].push_back(x_child_data[list_entry.offset + j]);
		}
		if (with_extra) {
			state.extra.push_back(extra);
		}
	}
}

inline void XyRowsUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	XyRowsUpdateImpl(inputs, count, state_vector, false);
}

inline void XyRowsExtraUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	XyRowsUpdateImpl(inputs, count, state_vector, true);
}

inline void XyRowsCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                          idx_t count) {
	XyRowsStatePointers sources(source_vector, count);
	XyRowsStatePointers targets(target_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &source = sources[i];
		auto &target = targets[i];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.extra = CombineTake(source.extra, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", target.n_features,
			                            source.n_features);
		}
		target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
		target.extra.insert(target.extra.end(), source.extra.begin(), source.extra.end());
	}
}

inline vector<AnofoxDataArray> XyRowsColumnArrays(XyRowsState &state) {
	vector<AnofoxDataArray> arrays;
	for (auto &col : state.x_columns) {
		arrays.push_back({col.data(), nullptr, col.size()});
	}
	return arrays;
}

//! Write a LIST(DOUBLE) value; non-finite elements become NULL.
inline void XyRowsSetDoubleList(Vector &list_vec, idx_t row, const double *values, idx_t n) {
	auto list_data = FlatDataMut<list_entry_t>(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + n);
	auto child_data = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	auto &child_validity = FlatValidityMut(ListVector::GetEntry(list_vec));
	for (idx_t j = 0; j < n; j++) {
		if (std::isfinite(values[j])) {
			child_data[offset + j] = values[j];
		} else {
			child_validity.SetInvalid(offset + j);
		}
	}
	list_data[row].offset = offset;
	list_data[row].length = n;
	ListVector::SetListSize(list_vec, offset + n);
}

//! Write a DOUBLE; non-finite becomes NULL.
inline void XyRowsSetDouble(Vector &vec, idx_t row, double value) {
	if (std::isfinite(value)) {
		FlatDataMut<double>(vec)[row] = value;
	} else {
		FlatVector::SetNull(vec, row, true);
	}
}

} // namespace duckdb
