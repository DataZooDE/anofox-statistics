#include <limits>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Mixed-effects GLM with a random intercept over one grouping factor.
//
//   anofox_stats_glmm_fit_agg(y DOUBLE, x LIST(DOUBLE), group ANY [, options])
//
// The grouping key is dictionary-encoded here, so any DuckDB type works and the
// FFI boundary only ever sees dense int32 indices.
//===--------------------------------------------------------------------===//
struct GlmmAggregateState {
	vector<double> y_values;
	vector<vector<double>> x_columns;
	vector<int32_t> group_ids;
	//! Group key -> dense index, in first-seen order.
	unordered_map<string, int32_t> group_index;
	//! Dense index -> the original key, for labelling the random effects.
	vector<string> group_labels;
	idx_t n_features;
	bool initialized;

	AnofoxGlmmFamily family;
	bool fit_intercept;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;
	bool reml;
	double theta;
	double power;
	idx_t offset_column;
	//! 0-based indices into x of columns that also carry a random slope.
	vector<idx_t> random_slopes;
	//! 0-based indices into x of additional crossed grouping-factor columns.
	vector<idx_t> group_columns;

	GlmmAggregateState()
	    : n_features(0), initialized(false), family(ANOFOX_GLMM_GAUSSIAN), fit_intercept(true), max_iterations(100),
	      tolerance(1e-8), compute_inference(false), confidence_level(0.95), reml(true), theta(1.0), power(1.5),
	      offset_column(0) {
	}

	int32_t Intern(const string &key) {
		auto it = group_index.find(key);
		if (it != group_index.end()) {
			return it->second;
		}
		auto id = (int32_t)group_labels.size();
		group_index.emplace(key, id);
		group_labels.push_back(key);
		return id;
	}

	void Reset() {
		y_values.clear();
		x_columns.clear();
		group_ids.clear();
		group_index.clear();
		group_labels.clear();
		n_features = 0;
		initialized = false;
	}
};

struct GlmmAggregateBindData : public FunctionData {
	AnofoxGlmmFamily family = ANOFOX_GLMM_GAUSSIAN;
	bool fit_intercept = true;
	uint32_t max_iterations = 100;
	double tolerance = 1e-8;
	bool compute_inference = false;
	double confidence_level = 0.95;
	bool reml = true;
	double theta = 1.0;
	double power = 1.5;
	idx_t offset_column = 0;
	vector<idx_t> random_slopes;
	vector<idx_t> group_columns;

	unique_ptr<FunctionData> Copy() const override {
		auto r = make_uniq<GlmmAggregateBindData>();
		r->family = family;
		r->fit_intercept = fit_intercept;
		r->max_iterations = max_iterations;
		r->tolerance = tolerance;
		r->compute_inference = compute_inference;
		r->confidence_level = confidence_level;
		r->reml = reml;
		r->theta = theta;
		r->power = power;
		r->offset_column = offset_column;
		r->random_slopes = random_slopes;
		r->group_columns = group_columns;
		return std::move(r);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<GlmmAggregateBindData>();
		return family == o.family && fit_intercept == o.fit_intercept && max_iterations == o.max_iterations &&
		       tolerance == o.tolerance && compute_inference == o.compute_inference &&
		       confidence_level == o.confidence_level && reml == o.reml && theta == o.theta && power == o.power &&
		       offset_column == o.offset_column && random_slopes == o.random_slopes &&
		       group_columns == o.group_columns;
	}
};

static LogicalType GetGlmmResultType(bool compute_inference) {
	child_list_t<LogicalType> ranef_children;
	ranef_children.push_back(make_pair("group", LogicalType::VARCHAR));
	ranef_children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	ranef_children.push_back(make_pair("se", LogicalType::DOUBLE));
	ranef_children.push_back(make_pair("n", LogicalType::BIGINT));
	auto ranef_type = LogicalType::STRUCT(std::move(ranef_children));

	child_list_t<LogicalType> children;
	children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	children.push_back(make_pair("var_group", LogicalType::DOUBLE));
	children.push_back(make_pair("var_residual", LogicalType::DOUBLE));
	children.push_back(make_pair("icc", LogicalType::DOUBLE));
	children.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
	children.push_back(make_pair("aic", LogicalType::DOUBLE));
	children.push_back(make_pair("bic", LogicalType::DOUBLE));
	children.push_back(make_pair("deviance", LogicalType::DOUBLE));
	children.push_back(make_pair("n_observations", LogicalType::BIGINT));
	children.push_back(make_pair("n_groups", LogicalType::BIGINT));
	children.push_back(make_pair("n_features", LogicalType::BIGINT));
	children.push_back(make_pair("iterations", LogicalType::INTEGER));
	children.push_back(make_pair("converged", LogicalType::BOOLEAN));
	// Random-effects covariance Sigma, flattened row-major (random_dim x random_dim).
	children.push_back(make_pair("random_cov", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("random_dim", LogicalType::INTEGER));
	// Per-factor variance components for crossed/nested fits (empty otherwise).
	child_list_t<LogicalType> factor_children;
	factor_children.push_back(make_pair("n_levels", LogicalType::BIGINT));
	factor_children.push_back(make_pair("var", LogicalType::DOUBLE));
	children.push_back(make_pair("factors", LogicalType::LIST(LogicalType::STRUCT(std::move(factor_children)))));

	if (compute_inference) {
		children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("z_values", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
		children.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));
	}

	children.push_back(make_pair("ranef", LogicalType::LIST(ranef_type)));
	return LogicalType::STRUCT(std::move(children));
}

static void GlmmAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
	new (state_p) GlmmAggregateState();
}

static void GlmmAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~GlmmAggregateState();
	}
}

static void GlmmAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                          idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<GlmmAggregateBindData>();

	UnifiedVectorFormat y_data, x_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = ListVector::GetData(inputs[1]);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatVector::GetData<double>(x_child);
	// A LIST containing NULLs is not itself NULL, so the list-level validity mask
	// says nothing about the elements. Read the child mask too and pass NaN for a
	// NULL element; the Rust side drops any row that is not finite throughout.
	auto &x_child_validity = FlatVector::Validity(x_child);

	// The grouping key can be any type, so it is read through Vector::GetValue and
	// rendered to a string for the dictionary. That is slower than a typed fast
	// path, but the alternative -- casting the whole vector -- needs a
	// ClientContext that an aggregate update does not have, and the cost is
	// negligible beside the fit itself.
	auto &group_vec = inputs[2];

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];

		state.family = bind_data.family;
		state.fit_intercept = bind_data.fit_intercept;
		state.max_iterations = bind_data.max_iterations;
		state.tolerance = bind_data.tolerance;
		state.compute_inference = bind_data.compute_inference;
		state.confidence_level = bind_data.confidence_level;
		state.reml = bind_data.reml;
		state.theta = bind_data.theta;
		state.power = bind_data.power;
		state.offset_column = bind_data.offset_column;
		state.random_slopes = bind_data.random_slopes;
		state.group_columns = bind_data.group_columns;

		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		auto group_value = group_vec.GetValue(i);
		if (group_value.IsNull()) {
			continue;
		}

		auto list_entry = x_list_data[x_idx];
		idx_t n_features = list_entry.length;
		if (!state.initialized) {
			state.n_features = n_features;
			state.x_columns.resize(n_features);
			state.initialized = true;
		}
		if (n_features != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            n_features);
		}

		state.y_values.push_back(y_values[y_idx]);
		state.group_ids.push_back(state.Intern(group_value.ToString()));
		for (idx_t j = 0; j < n_features; j++) {
			const idx_t child_idx = list_entry.offset + j;
			state.x_columns[j].push_back(x_child_validity.RowIsValid(child_idx)
			                                 ? x_child_data[child_idx]
			                                 : std::numeric_limits<double>::quiet_NaN());
		}
	}
}

static void GlmmAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (GlmmAggregateState **)source_data.data;
	auto targets = (GlmmAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}

		if (!target.initialized) {
			target.n_features = source.n_features;
			target.x_columns.resize(source.n_features);
			target.initialized = true;
			target.family = source.family;
			target.fit_intercept = source.fit_intercept;
			target.max_iterations = source.max_iterations;
			target.tolerance = source.tolerance;
			target.compute_inference = source.compute_inference;
			target.confidence_level = source.confidence_level;
			target.reml = source.reml;
			target.theta = source.theta;
			target.power = source.power;
			target.offset_column = source.offset_column;
			target.random_slopes = source.random_slopes;
			target.group_columns = source.group_columns;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count during combine");
		}

		// The two states interned groups independently, so remap the source's
		// dense ids into the target's dictionary rather than concatenating them.
		for (idx_t r = 0; r < source.y_values.size(); r++) {
			target.y_values.push_back(source.y_values[r]);
			target.group_ids.push_back(target.Intern(source.group_labels[source.group_ids[r]]));
			for (idx_t j = 0; j < target.n_features; j++) {
				target.x_columns[j].push_back(source.x_columns[j][r]);
			}
		}
	}
}

static void SetDoubleListG(Vector &target, idx_t row, const double *values, idx_t len) {
	auto list_data = FlatVector::GetData<list_entry_t>(target);
	auto child_offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, child_offset + len);
	auto child_values = FlatVector::GetData<double>(ListVector::GetEntry(target));
	for (idx_t j = 0; j < len; j++) {
		child_values[child_offset + j] = values[j];
	}
	ListVector::SetListSize(target, child_offset + len);
	list_data[row].offset = child_offset;
	list_data[row].length = len;
}

static void GlmmAggFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAggregateState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;

		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		AnofoxDataArray y_array {state.y_values.data(), nullptr, state.y_values.size()};

		// Additional crossed grouping factors are dictionary-encoded here and
		// removed from the design (like the offset column).
		vector<bool> is_group_col(state.n_features, false);
		for (auto gc : state.group_columns) {
			if (gc < state.n_features) {
				is_group_col[gc] = true;
			}
		}
		vector<AnofoxDataArray> x_arrays;
		for (idx_t j = 0; j < state.n_features; j++) {
			if (!is_group_col[j]) {
				x_arrays.push_back(AnofoxDataArray {state.x_columns[j].data(), nullptr, state.x_columns[j].size()});
			}
		}
		vector<vector<int32_t>> extra_factor_ids;
		for (auto gc : state.group_columns) {
			if (gc >= state.n_features) {
				continue;
			}
			unordered_map<double, int32_t> levels;
			vector<int32_t> ids;
			ids.reserve(state.x_columns[gc].size());
			for (double v : state.x_columns[gc]) {
				auto it = levels.find(v);
				int32_t id = (it == levels.end()) ? (int32_t)levels.size() : it->second;
				if (it == levels.end()) {
					levels.emplace(v, id);
				}
				ids.push_back(id);
			}
			extra_factor_ids.push_back(std::move(ids));
		}
		vector<const int32_t *> extra_ptrs;
		for (auto &f : extra_factor_ids) {
			extra_ptrs.push_back(f.data());
		}

		AnofoxGlmmOptions options {};
		options.family = state.family;
		options.fit_intercept = state.fit_intercept;
		options.max_iterations = state.max_iterations;
		options.tolerance = state.tolerance;
		options.compute_inference = state.compute_inference;
		options.confidence_level = state.confidence_level;
		options.reml = state.reml;
		options.theta = state.theta;
		options.power = state.power;
		options.offset_column = state.offset_column;
		options.random_slopes = state.random_slopes.empty() ? nullptr : state.random_slopes.data();
		options.random_slopes_len = state.random_slopes.size();

		AnofoxGlmmResult res {};
		AnofoxError error;
		bool ok = anofox_glmm_fit(y_array, x_arrays.data(), x_arrays.size(), state.group_ids.data(),
		                          state.group_ids.size(), extra_ptrs.empty() ? nullptr : extra_ptrs.data(),
		                          extra_ptrs.size(), options, &res, &error);
		if (!ok) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}

		idx_t c = 0;
		SetDoubleListG(*struct_entries[c++], row, res.coefficients, res.coefficients_len);
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.intercept;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.var_group;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.var_residual;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.icc;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.log_likelihood;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.aic;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.bic;
		FlatVector::GetData<double>(*struct_entries[c++])[row] = res.deviance;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)res.n_observations;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)res.n_groups;
		FlatVector::GetData<int64_t>(*struct_entries[c++])[row] = (int64_t)res.n_features;
		FlatVector::GetData<int32_t>(*struct_entries[c++])[row] = (int32_t)res.iterations;
		FlatVector::GetData<bool>(*struct_entries[c++])[row] = res.converged;
		SetDoubleListG(*struct_entries[c++], row, res.random_cov, res.random_dim * res.random_dim);
		FlatVector::GetData<int32_t>(*struct_entries[c++])[row] = (int32_t)res.random_dim;

		// Per-factor variance components LIST(STRUCT(n_levels, var)).
		{
			auto &fac_vec = *struct_entries[c++];
			auto fac_list = FlatVector::GetData<list_entry_t>(fac_vec);
			auto fac_off = ListVector::GetListSize(fac_vec);
			ListVector::Reserve(fac_vec, fac_off + res.factor_len);
			auto &fac_struct = ListVector::GetEntry(fac_vec);
			auto &fac_children = StructVector::GetEntries(fac_struct);
			auto f_levels = FlatVector::GetData<int64_t>(*fac_children[0]);
			auto f_var = FlatVector::GetData<double>(*fac_children[1]);
			for (idx_t j = 0; j < res.factor_len; j++) {
				f_levels[fac_off + j] = (int64_t)res.factor_n_levels[j];
				f_var[fac_off + j] = res.factor_var[j];
			}
			ListVector::SetListSize(fac_vec, fac_off + res.factor_len);
			fac_list[row].offset = fac_off;
			fac_list[row].length = res.factor_len;
		}

		if (state.compute_inference) {
			SetDoubleListG(*struct_entries[c++], row, res.std_errors, res.inference_len);
			SetDoubleListG(*struct_entries[c++], row, res.z_values, res.inference_len);
			SetDoubleListG(*struct_entries[c++], row, res.p_values, res.inference_len);
			SetDoubleListG(*struct_entries[c++], row, res.ci_lower, res.inference_len);
			SetDoubleListG(*struct_entries[c++], row, res.ci_upper, res.inference_len);
			FlatVector::GetData<double>(*struct_entries[c++])[row] = res.intercept_std_error;
		}

		// The random-effects LIST(STRUCT(group, intercept, se, n)).
		auto &ranef_vec = *struct_entries[c];
		auto ranef_list = FlatVector::GetData<list_entry_t>(ranef_vec);
		auto ranef_off = ListVector::GetListSize(ranef_vec);
		ListVector::Reserve(ranef_vec, ranef_off + res.ranef_len);
		auto &ranef_struct = ListVector::GetEntry(ranef_vec);
		auto &ranef_children = StructVector::GetEntries(ranef_struct);
		auto r_group = FlatVector::GetData<string_t>(*ranef_children[0]);
		auto r_value = FlatVector::GetData<double>(*ranef_children[1]);
		auto r_se = FlatVector::GetData<double>(*ranef_children[2]);
		auto r_n = FlatVector::GetData<int64_t>(*ranef_children[3]);

		for (idx_t j = 0; j < res.ranef_len; j++) {
			auto id = res.ranef_group[j];
			const string &label =
			    (id >= 0 && (idx_t)id < state.group_labels.size()) ? state.group_labels[id] : string();
			r_group[ranef_off + j] = StringVector::AddString(*ranef_children[0], label);
			r_value[ranef_off + j] = res.ranef_value[j];
			r_se[ranef_off + j] = res.ranef_se[j];
			r_n[ranef_off + j] = res.ranef_n[j];
		}
		ListVector::SetListSize(ranef_vec, ranef_off + res.ranef_len);
		ranef_list[row].offset = ranef_off;
		ranef_list[row].length = res.ranef_len;

		anofox_free_glmm_result(&res);
		state.Reset();
	}
}

static unique_ptr<FunctionData> GlmmAggBind(ClientContext &context, AggregateFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<GlmmAggregateBindData>();

	if (arguments.size() >= 4 && arguments[3]->IsFoldable()) {
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[3]);
		if (opts.glmm_family.has_value()) {
			result->family = (AnofoxGlmmFamily)opts.glmm_family.value();
		}
		if (opts.fit_intercept.has_value()) {
			result->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.compute_inference.has_value()) {
			result->compute_inference = opts.compute_inference.value();
		}
		if (opts.confidence_level.has_value()) {
			result->confidence_level = opts.confidence_level.value();
		}
		if (opts.max_iterations.has_value()) {
			result->max_iterations = opts.max_iterations.value();
		}
		if (opts.tolerance.has_value()) {
			result->tolerance = opts.tolerance.value();
		}
		if (opts.reml.has_value()) {
			result->reml = opts.reml.value();
		}
		if (opts.nb_theta.has_value()) {
			result->theta = opts.nb_theta.value();
		}
		if (opts.tweedie_power.has_value()) {
			result->power = opts.tweedie_power.value();
		}
		if (opts.offset_column.has_value()) {
			result->offset_column = opts.offset_column.value();
		}
		if (opts.random_slopes.has_value()) {
			// Options carry 1-based indices; the core/FFI want 0-based.
			for (auto idx1 : opts.random_slopes.value()) {
				result->random_slopes.push_back(idx1 - 1);
			}
		}
		if (opts.group_columns.has_value()) {
			for (auto idx1 : opts.group_columns.value()) {
				result->group_columns.push_back(idx1 - 1);
			}
		}
	}

	function.return_type = GetGlmmResultType(result->compute_inference);
	PostHogTelemetry::Instance().RecordFunctionCall("glmm_fit_agg");
	return std::move(result);
}

void RegisterGlmmAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("anofox_stats_glmm_fit_agg");

	auto basic = AggregateFunction(
	    "anofox_stats_glmm_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<GlmmAggregateState>, GlmmAggInitialize, GlmmAggUpdate,
	    GlmmAggCombine, GlmmAggFinalize, nullptr, GlmmAggBind, GlmmAggDestroy);
	func_set.AddFunction(basic);

	auto with_opts = AggregateFunction(
	    "anofox_stats_glmm_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<GlmmAggregateState>, GlmmAggInitialize, GlmmAggUpdate,
	    GlmmAggCombine, GlmmAggFinalize, nullptr, GlmmAggBind, GlmmAggDestroy);
	func_set.AddFunction(with_opts);

	CreateAggregateFunctionInfo info(func_set);
	loader.RegisterFunction(info);

	AggregateFunctionSet alias_set("glmm_fit_agg");
	alias_set.AddFunction(basic);
	alias_set.AddFunction(with_opts);
	CreateAggregateFunctionInfo alias_info(alias_set);
	alias_info.alias_of = "anofox_stats_glmm_fit_agg";
	loader.RegisterFunction(alias_info);
}

} // namespace duckdb
