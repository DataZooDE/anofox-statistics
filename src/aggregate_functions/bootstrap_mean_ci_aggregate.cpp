// Percentile bootstrap confidence interval of the mean (#294):
//
//   bootstrap_mean_ci_agg(x [, options] [ORDER BY t])
//     -> {estimate, std_error, conf_low, conf_high, conf_level, n, n_bootstrap,
//         block_length, method}
//
// options: method 'iid' (default) | 'stationary' | 'circular_block' |
// 'moving_block', n_bootstrap (10000), block_length (required for the block
// methods: the expected block length >= 1 of 'stationary', a whole number >= 2
// otherwise), confidence_level (0.95), seed (ANOFOX_DEFAULT_MC_SEED). The block bootstrap resamples the series in
// the aggregate's ORDER BY order, as acf_agg does; the IID bootstrap sorts the
// values first so a seeded result does not depend on row order. NULL / NaN
// values are skipped. The resampling and the interval come from
// anofox-statistics (resampling::bootstrap_mean_ci_with_options).

#include <algorithm>
#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

static constexpr const char *BOOTSTRAP_MEAN_CI_AGG = "bootstrap_mean_ci_agg";

// Six doubles, then two sizes (no padding on 64-bit or wasm32).
static_assert(offsetof(AnofoxBootstrapMeanResult, n) == 6 * sizeof(double) &&
                  sizeof(AnofoxBootstrapMeanResult) == 6 * sizeof(double) + 2 * sizeof(size_t),
              "AnofoxBootstrapMeanResult must match BootstrapMeanResultFFI in crates/anofox-stats-ffi/src/bootstrap_ffi.rs");

//! FFI method codes of anofox_bootstrap_mean_ci.
static constexpr int32_t BOOTSTRAP_IID = 0;
static constexpr int32_t BOOTSTRAP_CIRCULAR_BLOCK = 1;
static constexpr int32_t BOOTSTRAP_STATIONARY = 2;
static constexpr int32_t BOOTSTRAP_MOVING_BLOCK = 3;

static const char *BootstrapMethodName(int32_t method) {
	switch (method) {
	case BOOTSTRAP_CIRCULAR_BLOCK:
		return "circular_block";
	case BOOTSTRAP_STATIONARY:
		return "stationary";
	case BOOTSTRAP_MOVING_BLOCK:
		return "moving_block";
	default:
		return "iid";
	}
}

struct BootstrapMeanState {
	vector<double> values;
};

struct BootstrapMeanBindData : public FunctionData {
	int32_t method = BOOTSTRAP_IID;
	size_t n_bootstrap = 10000;
	double block_length = 0;
	double confidence_level = 0.95;
	uint64_t seed = ANOFOX_DEFAULT_MC_SEED;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<BootstrapMeanBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<BootstrapMeanBindData>();
		return method == o.method && n_bootstrap == o.n_bootstrap && block_length == o.block_length &&
		       confidence_level == o.confidence_level && seed == o.seed;
	}
};

static LogicalType BootstrapMeanResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("estimate", LogicalType::DOUBLE));
	children.push_back(make_pair("std_error", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_level", LogicalType::DOUBLE));
	children.push_back(make_pair("n", LogicalType::BIGINT));
	children.push_back(make_pair("n_bootstrap", LogicalType::BIGINT));
	children.push_back(make_pair("block_length", LogicalType::DOUBLE));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(children));
}

static void BootstrapMeanInitialize(data_ptr_t state_p) {
	new (state_p) BootstrapMeanState();
}

static void BootstrapMeanDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (BootstrapMeanState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~BootstrapMeanState();
	}
}

static void BootstrapMeanUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat value_data, sdata;
	inputs[0].ToUnifiedFormat(count, value_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);
	auto states = (BootstrapMeanState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = value_data.sel->get_index(i);
		if (!value_data.validity.RowIsValid(idx) || std::isnan(values[idx])) {
			continue;
		}
		states[sdata.sel->get_index(i)]->values.push_back(values[idx]);
	}
}

static void BootstrapMeanCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                                 idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (BootstrapMeanState **)source_data.data;
	auto targets = (BootstrapMeanState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (target.values.empty()) {
			target.values = CombineTake(source.values, aggr_input_data);
		} else {
			target.values.insert(target.values.end(), source.values.begin(), source.values.end());
		}
	}
}

static void BootstrapMeanFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                  idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (BootstrapMeanState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<BootstrapMeanBindData>();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Seeded IID resampling: make the result independent of the row order
		// threads delivered. The block bootstrap keeps the ORDER BY order.
		vector<double> values = state.values;
		if (bind.method == BOOTSTRAP_IID) {
			std::sort(values.begin(), values.end());
		}
		AnofoxDataArray x;
		x.data = values.data();
		x.validity = nullptr;
		x.len = static_cast<size_t>(values.size());

		AnofoxBootstrapMeanResult r;
		AnofoxError error;
		if (!anofox_bootstrap_mean_ci(x, bind.method, bind.n_bootstrap, bind.confidence_level, bind.block_length,
		                              bind.seed, true, &r, &error)) {
			ThrowUnlessDegenerate(BOOTSTRAP_MEAN_CI_AGG, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto set_double = [&](idx_t field, double v) {
			auto &vec = StructEntry(entries[field]);
			if (std::isnan(v)) {
				FlatVector::SetNull(vec, row, true);
			} else {
				FlatDataMut<double>(vec)[row] = v;
			}
		};
		set_double(0, r.estimate);
		set_double(1, r.std_error);
		set_double(2, r.conf_low);
		set_double(3, r.conf_high);
		set_double(4, r.conf_level);
		FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(r.n);
		FlatDataMut<int64_t>(StructEntry(entries[6]))[row] = static_cast<int64_t>(r.n_bootstrap);
		auto &block_vec = StructEntry(entries[7]);
		if (bind.method == BOOTSTRAP_IID) {
			FlatVector::SetNull(block_vec, row, true);
		} else {
			FlatDataMut<double>(block_vec)[row] = r.block_length;
		}
		auto &method_vec = StructEntry(entries[8]);
		FlatDataMut<string_t>(method_vec)[row] = StringVector::AddString(method_vec, BootstrapMethodName(bind.method));
	}
}

static unique_ptr<FunctionData> BootstrapMeanBind(ClientContext &context, AggregateBindFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, BootstrapMeanResultType());
	auto bind = make_uniq<BootstrapMeanBindData>();
	if (arguments.size() >= 2) {
		Value options_val = EvaluateConstantOptions(context, *arguments[1], BOOTSTRAP_MEAN_CI_AGG);
		auto opts = BootstrapMeanMapOptions::ParseFromValue(options_val, BOOTSTRAP_MEAN_CI_AGG);
		const string method = opts.method.value_or("iid");
		if (method == "iid") {
			if (opts.block_length.has_value()) {
				throw InvalidInputException("%s: block_length applies only to the block methods ('stationary', "
				                            "'circular_block', 'moving_block')",
				                            BOOTSTRAP_MEAN_CI_AGG);
			}
		} else {
			bind->method = method == "stationary"       ? BOOTSTRAP_STATIONARY
			               : method == "circular_block" ? BOOTSTRAP_CIRCULAR_BLOCK
			                                            : BOOTSTRAP_MOVING_BLOCK;
			if (!opts.block_length.has_value()) {
				throw InvalidInputException("%s: method '%s' requires block_length", BOOTSTRAP_MEAN_CI_AGG, method);
			}
			const double l = opts.block_length.value();
			if (bind->method == BOOTSTRAP_STATIONARY) {
				if (!std::isfinite(l) || l < 1) {
					throw InvalidInputException(
					    "%s: block_length (the expected block length) must be >= 1 for 'stationary', got %s",
					    BOOTSTRAP_MEAN_CI_AGG, std::to_string(l));
				}
			} else if (!std::isfinite(l) || l < 2 || l != std::floor(l)) {
				throw InvalidInputException("%s: block_length must be a whole number >= 2 for '%s', got %s",
				                            BOOTSTRAP_MEAN_CI_AGG, method, std::to_string(l));
			}
			bind->block_length = l;
		}
		if (opts.n_bootstrap.has_value()) {
			if (opts.n_bootstrap.value() < 2) {
				throw InvalidInputException("%s: n_bootstrap must be >= 2, got %u", BOOTSTRAP_MEAN_CI_AGG,
				                            opts.n_bootstrap.value());
			}
			bind->n_bootstrap = opts.n_bootstrap.value();
		}
		if (opts.confidence_level.has_value()) {
			bind->confidence_level = opts.confidence_level.value();
		}
		if (opts.seed.has_value()) {
			bind->seed = opts.seed.value();
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(BOOTSTRAP_MEAN_CI_AGG);
	return std::move(bind);
}

static AggregateFunction BootstrapMeanFunction(vector<LogicalType> args) {
	AggregateFunction fn(BOOTSTRAP_MEAN_CI_AGG, std::move(args), LogicalType::ANY,
	                     AggregateFunction::StateSize<BootstrapMeanState>, ANOFOX_AGG_INIT(BootstrapMeanInitialize),
	                     ANOFOX_GUARDED_UPDATE(BootstrapMeanUpdate, BootstrapMeanDestroy, BootstrapMeanInitialize),
	                     BootstrapMeanCombine,
	                     ANOFOX_GUARDED_FINALIZE(BootstrapMeanFinalize, BootstrapMeanDestroy, BootstrapMeanInitialize),
	                     nullptr, ANOFOX_AGG_BIND(BootstrapMeanBind), BootstrapMeanDestroy);
	// The block bootstrap resamples the series: honour `agg(... ORDER BY t)`.
	SetOrderDependentOf(fn, AggregateOrderDependent::ORDER_DEPENDENT); // compat helper works on DuckDB 1.5 and 2.0
	return fn;
}

void RegisterBootstrapMeanCiAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet set(BOOTSTRAP_MEAN_CI_AGG);
	set.AddFunction(BootstrapMeanFunction({LogicalType::DOUBLE}));
	set.AddFunction(BootstrapMeanFunction({LogicalType::DOUBLE, LogicalType::ANY}));

	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const vector<string> categories = {"resampling", "confidence-interval"};
	FunctionDescription d1;
	d1.description = "Percentile bootstrap confidence interval of the mean of x (IID bootstrap, 10000 replicates, "
	                 "95%, seeded).";
	d1.examples = {"bootstrap_mean_ci_agg(x)"};
	d1.categories = categories;
	d1.parameter_names = {"x"};
	d1.parameter_types = {LogicalType::DOUBLE};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = "Percentile bootstrap confidence interval of the mean of x. options: method 'iid' | "
	                 "'stationary' (geometric blocks of mean block_length, R tsboot(sim = 'geom')) | 'circular_block' "
	                 "(blocks of block_length consecutive values, wrapping around, R tsboot(sim = 'fixed')) | "
	                 "'moving_block' (non-wrapping blocks), resampling the series in the aggregate's ORDER BY order; "
	                 "n_bootstrap, block_length, confidence_level, seed.";
	d2.examples = {"bootstrap_mean_ci_agg(x, {'method': 'stationary', 'block_length': 5} ORDER BY t)",
	               "bootstrap_mean_ci_agg(x, {'method': 'circular_block', 'block_length': 5} ORDER BY t)"};
	d2.categories = categories;
	d2.parameter_names = {"x", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
