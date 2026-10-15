#pragma once

// Shared state of the *_fit_predict_agg aggregates whose model is fitted
// through one FFI call in Finalize (Bayesian Ridge / ARD, Passive-Aggressive,
// lmDynamic): every input row is kept for the output, the training rows also
// in column form.
//
//   y_all / y_is_null / is_training / x_all  one entry per input row
//   y_train / x_train                        the rows used to fit
//
// A training row has a non-NULL, finite y, a complete x and, with a split
// column, split 'train' / 'training'. A row whose whole x is NULL is kept as a
// non-training row (fit_predict_rows.hpp) so positions line up.

#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"

#include "aggregate_combine.hpp"
#include "anofox_stats_ffi.h"
#include "fit_predict_rows.hpp"
#include "list_input.hpp"
#include "map_options_parser.hpp"

namespace duckdb {

struct FitPredictRowsState {
	vector<double> y_train;
	vector<vector<double>> x_train;
	vector<double> y_all;
	vector<bool> y_is_null;
	vector<bool> is_training;
	vector<vector<double>> x_all;
	idx_t n_features = 0;
	bool initialized = false;
};

//! Bind data of these aggregates derives from this: Update reads the split flag.
struct FitPredictRowsBindBase : public FunctionData {
	bool use_split_col = false;
	//! 'drop_y_zero_x' also keeps rows with a zero feature out of the fit.
	NullPolicy null_policy = NullPolicy::DROP;
};

inline void FitPredictRowsInitialize(const AggregateFunction &, data_ptr_t state_p) {
	new (state_p) FitPredictRowsState();
}

inline void FitPredictRowsDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (FitPredictRowsState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~FitPredictRowsState();
	}
}

inline bool FitPredictIsTrainSplit(const string_t &split) {
	auto value = StringUtil::Lower(split.GetString());
	return value == "train" || value == "training";
}

//! Update for (y DOUBLE, x LIST(DOUBLE)[, split VARCHAR][, options]).
inline void FitPredictRowsUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                 Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<FitPredictRowsBindBase>();
	UnifiedVectorFormat y_data, x_data, split_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	const bool with_split = bind.use_split_col && input_count >= 3;
	if (with_split) {
		inputs[2].ToUnifiedFormat(count, split_data);
	}
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatVector::GetData<double>(x_child);
	auto &x_child_validity = FlatVector::Validity(x_child);
	auto states = (FitPredictRowsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		bool y_valid = y_data.validity.RowIsValid(y_idx);
		double y_val = y_valid ? y_values[y_idx] : std::nan("");

		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			PushNullXRow(state, y_valid, y_val);
			continue;
		}
		auto entry = x_lists[x_idx];
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_train.resize(state.n_features);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %llu, got %llu",
			                            static_cast<unsigned long long>(state.n_features),
			                            static_cast<unsigned long long>(entry.length));
		}
		vector<double> x_row(state.n_features);
		bool complete = true;
		for (idx_t j = 0; j < state.n_features; j++) {
			x_row[j] = ListChildValue(x_child_data, x_child_validity, entry.offset + j);
			complete = complete && std::isfinite(x_row[j]);
		}

		bool training = y_valid && std::isfinite(y_val) && complete;
		if (training && bind.null_policy == NullPolicy::DROP_Y_ZERO_X) {
			for (auto v : x_row) {
				if (v == 0.0) {
					training = false;
					break;
				}
			}
		}
		if (with_split) {
			auto s_idx = split_data.sel->get_index(i);
			training = training && split_data.validity.RowIsValid(s_idx) &&
			           FitPredictIsTrainSplit(UnifiedVectorFormat::GetData<string_t>(split_data)[s_idx]);
		}

		state.y_all.push_back(y_val);
		state.y_is_null.push_back(!y_valid);
		state.is_training.push_back(training);
		if (training) {
			state.y_train.push_back(y_val);
			for (idx_t j = 0; j < state.n_features; j++) {
				state.x_train[j].push_back(x_row[j]);
			}
		}
		state.x_all.push_back(std::move(x_row));
	}
}

//! Combine appends the source rows after the target's, so ordered input stays ordered.
inline void FitPredictRowsCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                                  idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (FitPredictRowsState **)source_data.data;
	auto targets = (FitPredictRowsState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			// Only NULL-x rows; keep them.
			AppendOutputRows(target, source);
			continue;
		}
		if (!target.initialized) {
			auto pending = TakeOutputRows(target);
			target.y_train = CombineTake(source.y_train, aggr_input_data);
			target.x_train = CombineTake(source.x_train, aggr_input_data);
			target.y_all = CombineTake(source.y_all, aggr_input_data);
			target.y_is_null = CombineTake(source.y_is_null, aggr_input_data);
			target.is_training = CombineTake(source.is_training, aggr_input_data);
			target.x_all = CombineTake(source.x_all, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			PrependOutputRows(target, std::move(pending));
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Cannot combine states with different feature counts");
		}
		AppendTo(target.y_train, source.y_train);
		for (idx_t j = 0; j < target.n_features; j++) {
			AppendTo(target.x_train[j], source.x_train[j]);
		}
		AppendOutputRows(target, source);
	}
}

//! The training columns of a state as FFI arrays.
inline vector<AnofoxDataArray> FitPredictTrainColumns(FitPredictRowsState &state) {
	vector<AnofoxDataArray> x;
	for (auto &col : state.x_train) {
		x.push_back({col.data(), nullptr, col.size()});
	}
	return x;
}

//! The LIST(STRUCT(y, yhat, yhat_lower, yhat_upper, is_training)) result type.
inline LogicalType FitPredictRowsResultType() {
	child_list_t<LogicalType> row;
	row.push_back(make_pair("y", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
	row.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
	row.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

//! Write one state's rows into the LIST result; non-finite values become NULL.
//! `lower` / `upper` may be null (no interval).
inline void WriteFitPredictRows(Vector &result, idx_t result_idx, const FitPredictRowsState &state, const double *yhat,
                                const double *lower, const double *upper) {
	const idx_t n_rows = state.y_all.size();
	auto list_data = ListVector::GetData(result);
	auto list_offset = ListVector::GetListSize(result);
	ListVector::Reserve(result, list_offset + n_rows);
	list_data[result_idx].offset = list_offset;
	list_data[result_idx].length = n_rows;
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
	auto write = [](Vector &vec, idx_t idx, double value) {
		if (std::isfinite(value)) {
			FlatVector::GetData<double>(vec)[idx] = value;
		} else {
			FlatVector::SetNull(vec, idx, true);
		}
	};
	for (idx_t r = 0; r < n_rows; r++) {
		idx_t child = list_offset + r;
		if (state.y_is_null[r]) {
			FlatVector::SetNull(*entries[0], child, true);
		} else {
			FlatVector::GetData<double>(*entries[0])[child] = state.y_all[r];
		}
		write(*entries[1], child, yhat[r]);
		write(*entries[2], child, lower ? lower[r] : std::nan(""));
		write(*entries[3], child, upper ? upper[r] : std::nan(""));
		FlatVector::GetData<bool>(*entries[4])[child] = state.is_training[r];
	}
	ListVector::SetListSize(result, list_offset + n_rows);
}

} // namespace duckdb
