#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/glm_weights_arg.hpp"
#include "../include/glm_prior_options.hpp"
#include "../include/cluster_ids.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/model_struct.hpp"
#include "../include/profile_ci.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Poisson Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct PoissonAggregateState {
	GlmPriorState prior_state;
	vector<double> y_values;
	vector<double> weights; // prior weights (#305); empty unless weighted
	ClusterRows cluster;    // cluster ids of the cluster form; empty otherwise
	vector<vector<double>> x_columns;
	idx_t n_features;
	bool initialized;

	// Options
	bool fit_intercept;
	AnofoxPoissonLink link;
	uint32_t max_iterations;
	double tolerance;
	bool compute_inference;
	double confidence_level;
	double glm_lambda;
	idx_t offset_column;
	bool profile_ci = false;

	PoissonAggregateState()
	    : n_features(0), initialized(false), fit_intercept(true), link(ANOFOX_POISSON_LINK_LOG), max_iterations(100),
	      tolerance(1e-8), compute_inference(false), confidence_level(0.95), glm_lambda(0.0), offset_column(0) {
	}

	void Reset() {
		prior_state.Clear();
		y_values.clear();
		weights.clear();
		cluster.Clear();
		x_columns.clear();
		n_features = 0;
		initialized = false;
	}
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct PoissonAggregateBindData : public FunctionData {
	GlmPriorBindData prior_opts;
	bool fit_intercept = true;
	PoissonLink link = PoissonLink::LOG;
	uint32_t max_iterations = 100;
	double tolerance = 1e-8;
	bool compute_inference = false;
	double confidence_level = 0.95;
	double glm_lambda = 0.0;
	idx_t offset_column = 0;
	bool weighted = false; // f(y, x, w [, options]) (#305)
	bool profile_ci = false;
	ClusterBind cluster; // f(y, x, cluster, {'vcov': 'cluster', ...})

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<PoissonAggregateBindData>();
		result->prior_opts = prior_opts;
		result->fit_intercept = fit_intercept;
		result->link = link;
		result->max_iterations = max_iterations;
		result->tolerance = tolerance;
		result->compute_inference = compute_inference;
		result->confidence_level = confidence_level;
		result->glm_lambda = glm_lambda;
		result->offset_column = offset_column;
		result->weighted = weighted;
		result->profile_ci = profile_ci;
		result->cluster = cluster;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<PoissonAggregateBindData>();
		return fit_intercept == other.fit_intercept && link == other.link && max_iterations == other.max_iterations &&
		       tolerance == other.tolerance && compute_inference == other.compute_inference &&
		       confidence_level == other.confidence_level && glm_lambda == other.glm_lambda &&
		       offset_column == other.offset_column && weighted == other.weighted && profile_ci == other.profile_ci &&
		       cluster.Equals(other.cluster) && prior_opts.Equals(other.prior_opts);
	}
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetPoissonAggResultType(bool clustered = false) {
	child_list_t<LogicalType> children;

	children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	children.push_back(make_pair("deviance", LogicalType::DOUBLE));
	children.push_back(make_pair("null_deviance", LogicalType::DOUBLE));
	children.push_back(make_pair("pseudo_r_squared", LogicalType::DOUBLE));
	children.push_back(make_pair("aic", LogicalType::DOUBLE));
	children.push_back(make_pair("dispersion", LogicalType::DOUBLE));
	children.push_back(make_pair("n_observations", LogicalType::BIGINT));
	children.push_back(make_pair("n_features", LogicalType::BIGINT));
	children.push_back(make_pair("iterations", LogicalType::INTEGER));
	children.push_back(make_pair("converged", LogicalType::BOOLEAN));

	// Stable shape (#152): the inference fields are always present, NULL
	// without compute_inference.
	AppendCoefficientInferenceFields(children, "z_values", false);

	// Appended last so existing field positions are unchanged. predict(model, x)
	// reads them to apply the inverse link.
	children.push_back(make_pair("family", LogicalType::VARCHAR));
	children.push_back(make_pair("link", LogicalType::VARCHAR));

	AppendModelSummaryFields(children);
	AppendCovarianceFields(children);
	if (clustered) {
		AppendClusterFields(children);
	}

	return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PoissonAggInitialize(data_ptr_t state_p) {
	new (state_p) PoissonAggregateState();
}

static void PoissonAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoissonAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.~PoissonAggregateState();
	}
}

static AnofoxPoissonLink ConvertPoissonLink(PoissonLink link) {
	switch (link) {
	case PoissonLink::LOG:
		return ANOFOX_POISSON_LINK_LOG;
	case PoissonLink::IDENTITY:
		return ANOFOX_POISSON_LINK_IDENTITY;
	case PoissonLink::SQRT:
		return ANOFOX_POISSON_LINK_SQRT;
	default:
		return ANOFOX_POISSON_LINK_LOG;
	}
}

static void PoissonAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                             Vector &state_vector, idx_t count) {
	auto &bind_data = aggr_input_data.bind_data->Cast<PoissonAggregateBindData>();

	UnifiedVectorFormat y_data;
	UnifiedVectorFormat x_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	GlmWeightsInput w_input;
	if (bind_data.weighted) {
		w_input.Load(inputs[2], count);
	}
	vector<string> cluster_ids;

	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	// A LIST containing NULLs is not itself NULL, so the list-level validity mask
	// says nothing about the elements. Read the child mask too and pass NaN for a
	// NULL element; the Rust side drops any row that is not finite throughout.
	auto &x_child_validity = FlatValidityMut(x_child);

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoissonAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];

		state.fit_intercept = bind_data.fit_intercept;
		state.link = ConvertPoissonLink(bind_data.link);
		state.max_iterations = bind_data.max_iterations;
		state.tolerance = bind_data.tolerance;
		state.compute_inference = bind_data.compute_inference;
		state.confidence_level = bind_data.confidence_level;
		state.glm_lambda = bind_data.glm_lambda;
		state.offset_column = bind_data.offset_column;
		state.profile_ci = bind_data.profile_ci;

		auto y_idx = y_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx)) {
			continue;
		}
		double y_val = y_values[y_idx];

		auto x_idx = x_data.sel->get_index(i);
		if (!x_data.validity.RowIsValid(x_idx)) {
			continue;
		}

		auto list_entry = x_list_data[x_idx];
		idx_t n_features = list_entry.length;
		if (bind_data.cluster.active && !ReadClusterIds(inputs[2].GetValue(i), cluster_ids)) {
			continue; // NULL cluster id: the row is left out
		}

		if (!state.initialized) {
			state.n_features = n_features;
			state.prior_state.Materialize(bind_data.prior_opts, n_features, state.fit_intercept);
			state.x_columns.resize(n_features);
			state.initialized = true;
		}

		if (n_features != state.n_features) {
			throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
			                            n_features);
		}

		if (bind_data.weighted) {
			double w_val;
			if (!w_input.Get(i, w_val)) {
				continue; // NULL weight: the row is left out, as wls_fit_agg
			}
			state.weights.push_back(w_val);
		}
		if (bind_data.cluster.active) {
			state.cluster.Push(cluster_ids, "poisson_fit_agg");
		}
		state.y_values.push_back(y_val);

		for (idx_t j = 0; j < n_features; j++) {
			const idx_t child_idx = list_entry.offset + j;
			state.x_columns[j].push_back(x_child_validity.RowIsValid(child_idx)
			                                 ? x_child_data[child_idx]
			                                 : std::numeric_limits<double>::quiet_NaN());
		}
	}
}

static void PoissonAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);

	auto sources = (PoissonAggregateState **)source_data.data;
	auto targets = (PoissonAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];

		if (!source.initialized) {
			continue;
		}

		target.cluster.Append(source.cluster, "poisson_fit_agg");
		if (!target.initialized) {
			target.y_values = CombineTake(source.y_values, aggr_input_data);
			target.weights = CombineTake(source.weights, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			target.fit_intercept = source.fit_intercept;
			// Resolved priors are part of the option set and must travel with it,
			// otherwise a parallel aggregation finalises a state that has none.
			target.prior_state = CombineTake(source.prior_state, aggr_input_data);
			target.link = source.link;
			target.max_iterations = source.max_iterations;
			target.tolerance = source.tolerance;
			target.compute_inference = source.compute_inference;
			target.confidence_level = source.confidence_level;
			target.glm_lambda = source.glm_lambda;
			target.offset_column = source.offset_column;
			target.profile_ci = source.profile_ci;
			continue;
		}

		if (source.n_features != target.n_features) {
			throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
			                            source.n_features, target.n_features);
		}

		target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
		target.weights.insert(target.weights.end(), source.weights.begin(), source.weights.end());

		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
	auto &child = ListVector::GetEntry(list_vec);
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	ListVector::SetListSize(list_vec, offset + len);
	auto vec_data = FlatDataMut<double>(child);
	for (size_t i = 0; i < len; i++) {
		vec_data[offset + i] = data[i];
	}
	ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void PoissonAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                               idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PoissonAggregateBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (PoissonAggregateState **)sdata.data;

	auto &struct_entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;

		if (!state.initialized || state.y_values.size() < 2) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		// Note: Detailed min_obs validation including zero-variance column handling is done in Rust
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		const bool weighted = !state.weights.empty();
		const bool clustered = !state.cluster.dims.empty();
		if (clustered) {
			CanonicalRows rows;
			rows.Keys(state.x_columns).Key(state.y_values);
			state.cluster.AddTo(rows);
			rows.Sort();
		} else if (weighted) {
			CanonicalRows().Keys(state.x_columns).Key(state.y_values).Key(state.weights).Sort();
		} else {
			SortRowsCanonically(state.y_values, state.x_columns);
		}
		AnofoxDataArray y_array;
		y_array.data = state.y_values.data();
		y_array.validity = nullptr;
		y_array.len = state.y_values.size();

		vector<AnofoxDataArray> x_arrays;
		for (auto &col : state.x_columns) {
			AnofoxDataArray arr;
			arr.data = col.data();
			arr.validity = nullptr;
			arr.len = col.size();
			x_arrays.push_back(arr);
		}

		AnofoxPoissonOptions options {}; // value-initialised: these are hand-mirrored C structs, so a
		                                 // field added to the header must not be left as garbage
		options.fit_intercept = state.fit_intercept;
		options.link = state.link;
		options.max_iterations = state.max_iterations;
		options.tolerance = state.tolerance;
		options.compute_inference = state.compute_inference;
		options.confidence_level = state.confidence_level;
		options.lambda = state.glm_lambda;
		options.offset_column = state.offset_column;
		state.prior_state.Apply(options);

		AnofoxGlmFitResultCore core_result;
		AnofoxFitResultInference inference_result;
		AnofoxError error;

		// Prior weights (#305): upstream's weighted Poisson fit; the unweighted fit is unchanged.
		AnofoxDataArray w_array {state.weights.data(), nullptr, state.weights.size()};
		AnofoxFitResultInference *inference_out = state.compute_inference ? &inference_result : nullptr;
		auto cluster_ptrs = state.cluster.Pointers();
		vector<int64_t> n_clusters(cluster_ptrs.size(), 0);
		bool success = clustered ? anofox_poisson_cluster_fit(y_array, x_arrays.data(), x_arrays.size(), options,
		                                                      cluster_ptrs.data(), cluster_ptrs.size(),
		                                                      ConvertHcType(bind.cluster.hc_type),
		                                                      bind.cluster.cadjust, &core_result, inference_out,
		                                                      n_clusters.data(), &error)
		               : weighted ? anofox_poisson_fit_weighted(y_array, x_arrays.data(), x_arrays.size(), w_array, options,
		                                                 &core_result, inference_out, &error)
		                        : anofox_poisson_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
		                                          inference_out, &error);

		if (!success) {
			ThrowUnlessDegenerate("poisson_fit_agg", error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		if (state.profile_ci) {
			if (clustered) {
				throw InvalidInputException("poisson_fit_agg: profile_ci cannot be combined with the cluster-robust "
				                            "covariance");
			}
			if (weighted) {
				throw InvalidInputException("poisson_fit_agg: profile_ci cannot be combined with prior weights");
			}
			ApplyProfileConfint("poisson_fit_agg", AnofoxLikelihoodFamily {1, static_cast<int32_t>(state.link), std::nan("")}, y_array, x_arrays,
			                    state.fit_intercept, state.offset_column, state.confidence_level, core_result,
			                    inference_result);
		}

		idx_t struct_idx = 0;

		SetListInResult(StructEntry(struct_entries[struct_idx++]), result_idx, core_result.coefficients,
		                core_result.coefficients_len);

		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.intercept;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.deviance;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.null_deviance;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.pseudo_r_squared;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.aic;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.dispersion;
		FlatDataMut<int64_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.n_observations;
		FlatDataMut<int64_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.n_features;
		FlatDataMut<int32_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.iterations;
		FlatDataMut<bool>(StructEntry(struct_entries[struct_idx++]))[result_idx] = core_result.converged;

		// Inference (NULL unless requested), CI fields and the model summary
		// (family and link come from upstream).
		writer.WriteInference(result_idx, state.compute_inference ? &inference_result : nullptr);
		writer.WriteSummary(result_idx, core_result.summary);
		writer.WriteCovariance(result_idx, state.compute_inference ? core_result.vcov : nullptr, core_result.vcov_dim,
		                       ResidualDf(core_result.n_observations, core_result.coefficients,
		                                  core_result.coefficients_len, std::isfinite(core_result.intercept)));
		if (bind.cluster.active) {
			WriteClusterFields(result, result_idx, state.compute_inference, bind.cluster.VcovTypeName(), n_clusters);
		}
		if (state.compute_inference) {
			anofox_free_result_inference(&inference_result);
		}

		anofox_free_glm_result(&core_result);

		state.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PoissonAggBind(ClientContext &context, AggregateBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
	auto result = make_uniq<PoissonAggregateBindData>();

	// f(y, x, w) / f(y, x, w, options): prior weights (#305); f(y, x, cluster, {'vcov': 'cluster'}):
	// the cluster-robust covariance.
	result->weighted = GlmWeightsArgument(arguments);
	const idx_t options_idx = result->weighted ? 3 : 2;
	if (arguments.size() > options_idx) {
		auto opts = RegressionMapOptions::ParseFromExpression(
            context, *arguments[options_idx], "poisson_fit_agg",
            {"fit_intercept", "compute_inference", "confidence_level", "max_iterations", "tolerance", "poisson_link", "offset", "glm_lambda", "feature_names", "prior", "vcov", "profile_ci", "hc_type", "cadjust"});
		result->cluster.Load(opts, GlmClusterForm(opts, arguments), true, "poisson_fit_agg",
		                     "poisson_fit_agg(y, x, cluster, {'vcov': 'cluster'})");
		if (result->cluster.active) {
			result->weighted = false;
			result->compute_inference = true; // the default of the cluster form, as ols_fit_agg's
		}
		CheckGlmWeightsArgument(arguments, result->weighted, "poisson_fit_agg");
		result->prior_opts.LoadFrom(opts);
		if (opts.fit_intercept.has_value()) {
			result->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.compute_inference.has_value()) {
			result->compute_inference = opts.compute_inference.value();
		}
		if (opts.confidence_level.has_value()) {
			result->confidence_level = opts.confidence_level.value();
		}
		if (opts.poisson_link.has_value()) {
			result->link = opts.poisson_link.value();
		}
		if (opts.max_iterations.has_value()) {
			result->max_iterations = opts.max_iterations.value();
		}
		if (opts.tolerance.has_value()) {
			result->tolerance = opts.tolerance.value();
		}
		if (opts.glm_lambda.has_value()) {
			result->glm_lambda = opts.glm_lambda.value();
		}
		if (opts.offset_column.has_value()) {
			result->offset_column = opts.offset_column.value();
		}
		result->profile_ci = ProfileCiOption(opts, "poisson_fit_agg");
		if (result->profile_ci) {
			result->compute_inference = true;
		}
	}

	SetReturnTypeOf(function, GetPoissonAggResultType(result->cluster.active));

	PostHogTelemetry::Instance().RecordFunctionCall(result->cluster.active ? "poisson_fit_agg_cluster"
	                                                                       : "poisson_fit_agg");
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPoissonAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("poisson_fit_agg");

	auto basic_func = AggregateFunction(
	    "poisson_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}, LogicalType::ANY,
	    AggregateFunction::StateSize<PoissonAggregateState>, ANOFOX_AGG_INIT(PoissonAggInitialize), ANOFOX_GUARDED_UPDATE(PoissonAggUpdate, PoissonAggDestroy, PoissonAggInitialize), PoissonAggCombine,
	    ANOFOX_GUARDED_FINALIZE(PoissonAggFinalize, PoissonAggDestroy, PoissonAggInitialize), nullptr, ANOFOX_AGG_BIND(PoissonAggBind), PoissonAggDestroy);
	func_set.AddFunction(basic_func);

	auto map_func = AggregateFunction(
	    "poisson_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<PoissonAggregateState>, ANOFOX_AGG_INIT(PoissonAggInitialize), ANOFOX_GUARDED_UPDATE(PoissonAggUpdate, PoissonAggDestroy, PoissonAggInitialize),
	    PoissonAggCombine, ANOFOX_GUARDED_FINALIZE(PoissonAggFinalize, PoissonAggDestroy, PoissonAggInitialize), nullptr, ANOFOX_AGG_BIND(PoissonAggBind), PoissonAggDestroy);
	func_set.AddFunction(map_func);

	// (y, x, w, options): prior weights (#305). (y, x, w) is the ANY overload above
	// with a numeric third argument.
	auto weighted_func = AggregateFunction(
	    "poisson_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<PoissonAggregateState>, ANOFOX_AGG_INIT(PoissonAggInitialize),
	    ANOFOX_GUARDED_UPDATE(PoissonAggUpdate, PoissonAggDestroy, PoissonAggInitialize), PoissonAggCombine,
	    ANOFOX_GUARDED_FINALIZE(PoissonAggFinalize, PoissonAggDestroy, PoissonAggInitialize), nullptr,
	    ANOFOX_AGG_BIND(PoissonAggBind), PoissonAggDestroy);
	func_set.AddFunction(weighted_func);
	// (y, x, cluster, {'vcov': 'cluster'}): the cluster-robust covariance; also (y, x, w, options) with a
	// non-DOUBLE numeric w, which binds here rather than to the DOUBLE overload above.
	func_set.AddFunction(AggregateFunction(
	    "poisson_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<PoissonAggregateState>, ANOFOX_AGG_INIT(PoissonAggInitialize),
	    ANOFOX_GUARDED_UPDATE(PoissonAggUpdate, PoissonAggDestroy, PoissonAggInitialize), PoissonAggCombine,
	    ANOFOX_GUARDED_FINALIZE(PoissonAggFinalize, PoissonAggDestroy, PoissonAggInitialize), nullptr,
	    ANOFOX_AGG_BIND(PoissonAggBind), PoissonAggDestroy));

	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	FunctionDescription d1;
	d1.description =
	    "Fits a Poisson regression (GLM with log link) and returns coefficients, deviance, AIC, and fit statistics.";
	d1.examples = {"poisson_fit_agg(y, x, {'fit_intercept': true})"};
	d1.categories = {"regression"};
	d1.parameter_names = {"y", "x", "options"};
	d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description =
	    "Fits a Poisson regression (GLM with log link) and returns coefficients, deviance, AIC, and fit statistics.";
	d2.examples = {"poisson_fit_agg(y, x)"};
	d2.categories = {"regression"};
	d2.parameter_names = {"y", "x"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)};
	info.descriptions.push_back(std::move(d2));
	FunctionDescription d3;
	d3.description = "Fits the same model with prior (case) weights w, as R's glm(weights = w); a NULL weight drops "
	                 "the row.";
	d3.examples = {"poisson_fit_agg(y, x, w)", "poisson_fit_agg(y, x, w, {'compute_inference': true})"};
	d3.categories = {"regression"};
	d3.parameter_names = {"y", "x", "weights", "options"};
	d3.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE,
	                      LogicalType::ANY};
	info.descriptions.push_back(std::move(d3));
	info.descriptions.push_back(GlmClusterDescription("poisson_fit_agg"));
	loader.RegisterFunction(std::move(info));

}

} // namespace duckdb
