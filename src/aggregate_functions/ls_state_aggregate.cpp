// Compact, mergeable least-squares states (#266).
//
//   ls_state_agg(y, x[, weight])    -> BLOB   state of a group's rows
//   ls_state_combine_agg(state)     -> BLOB   merge stored states
//   ls_state_info(state)            -> STRUCT(n_observations, n_features, weighted)
//   ols_fit_from_state(state[, options]), wls_fit_from_state(...), ridge_fit_from_state(...)
//                                   -> the STRUCT of ols_fit_agg / wls_fit_agg / ridge_fit_agg
//
// A state is upstream's SufficientStats in its versioned byte encoding (counts,
// weighted means, centred co-moments: O(p^2) bytes). Store per-partition states in a
// table, combine them later and fit without the rows. The statistics are upstream's
// (anofox-regression); this file only moves bytes and results across the FFI.
//
// Determinism (#226): the state is built from the rows in a canonical order and
// states are merged in the order of their bytes, so neither the row order nor the
// thread count changes a state or a fit from it. A fit from merged states equals the
// direct fit up to floating-point summation order (~1e-12 relative), not bitwise.

#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/aggregate_finalize_guard.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "../include/null_rows.hpp"
#include "../include/xy_rows_aggregate.hpp"
#include "anofox_statistics_banner.hpp"
#include "telemetry.hpp"

namespace duckdb {

namespace {

//! Copy a library-owned byte buffer into a BLOB result row and free it.
void WriteBlob(Vector &result, idx_t row, uint8_t *bytes, size_t len) {
	FlatDataMut<string_t>(result)[row] =
	    StringVector::AddStringOrBlob(result, const_char_ptr_cast(bytes), static_cast<idx_t>(len));
	anofox_free_bytes(bytes);
}

//===--------------------------------------------------------------------===//
// ls_state_agg(y, x[, weight])
//===--------------------------------------------------------------------===//

void LsStateWeightedUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	XyRowsUpdateImpl(inputs, count, state_vector, &inputs[2]);
}

template <bool WEIGHTED>
void LsStateFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	XyRowsStatePointers states(state_vector, count);
	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray y {state.y_values.data(), nullptr, state.y_values.size()};
		auto x = XyRowsColumnArrays(state);
		AnofoxDataArray w {WEIGHTED ? state.extra.data() : nullptr, nullptr, WEIGHTED ? state.extra.size() : 0};
		uint8_t *bytes = nullptr;
		size_t len = 0;
		AnofoxError error;
		if (!anofox_ls_state_from_rows(y, x.data(), x.size(), w, &bytes, &len, &error)) {
			ThrowUnlessDegenerate("ls_state_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteBlob(result, row, bytes, len);
	}
}

//===--------------------------------------------------------------------===//
// ls_state_combine_agg(state)
//===--------------------------------------------------------------------===//

struct LsCombineState {
	vector<string> states;
};

void LsCombineInitialize(data_ptr_t state_p) {
	new (state_p) LsCombineState();
}

void LsCombineDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LsCombineState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~LsCombineState();
	}
}

void LsCombineUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat bdata, sdata;
	inputs[0].ToUnifiedFormat(count, bdata);
	state_vector.ToUnifiedFormat(count, sdata);
	auto blobs = UnifiedVectorFormat::GetData<string_t>(bdata);
	auto states = (LsCombineState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto b_idx = bdata.sel->get_index(i);
		if (!bdata.validity.RowIsValid(b_idx)) {
			continue;
		}
		states[sdata.sel->get_index(i)]->states.emplace_back(blobs[b_idx].GetData(), blobs[b_idx].GetSize());
	}
}

void LsCombineCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat src, tgt;
	source_vector.ToUnifiedFormat(count, src);
	target_vector.ToUnifiedFormat(count, tgt);
	auto sources = (LsCombineState **)src.data;
	auto targets = (LsCombineState **)tgt.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[src.sel->get_index(i)];
		auto &target = *targets[tgt.sel->get_index(i)];
		target.states.insert(target.states.end(), source.states.begin(), source.states.end());
	}
}

void LsCombineFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (LsCombineState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.states.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<const uint8_t *> ptrs;
		vector<size_t> lens;
		for (auto &s : state.states) {
			ptrs.push_back(reinterpret_cast<const uint8_t *>(s.data()));
			lens.push_back(s.size());
		}
		uint8_t *bytes = nullptr;
		size_t len = 0;
		AnofoxError error;
		if (!anofox_ls_state_merge(ptrs.data(), lens.data(), ptrs.size(), &bytes, &len, &error)) {
			ThrowUnlessDegenerate("ls_state_combine_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteBlob(result, row, bytes, len);
	}
}

//===--------------------------------------------------------------------===//
// ls_state_info(state)
//===--------------------------------------------------------------------===//

LogicalType LsStateInfoType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	c.push_back(make_pair("weighted", LogicalType::BOOLEAN));
	return LogicalType::STRUCT(std::move(c));
}

void LsStateInfoImpl(DataChunk &args, ExpressionState &, Vector &result) {
	auto blobs = FlatVector::GetData<string_t>(args.data[0]);
	auto &entries = StructVector::GetEntries(result);
	for (idx_t row = 0; row < args.size(); row++) {
		size_t n_obs = 0, n_features = 0;
		bool weighted = false;
		AnofoxError error;
		if (!anofox_ls_state_info(reinterpret_cast<const uint8_t *>(blobs[row].GetData()), blobs[row].GetSize(),
		                          &n_obs, &n_features, &weighted, &error)) {
			throw InvalidInputException("ls_state_info: %s", error.message);
		}
		FlatDataMut<int64_t>(StructEntry(entries[0]))[row] = static_cast<int64_t>(n_obs);
		FlatDataMut<int64_t>(StructEntry(entries[1]))[row] = static_cast<int64_t>(n_features);
		FlatDataMut<bool>(StructEntry(entries[2]))[row] = weighted;
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

void LsStateInfoFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	args.Flatten();
	ExecuteOnNonNullRows(args, state, result, 1, LsStateInfoImpl);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// ols_fit_from_state / wls_fit_from_state / ridge_fit_from_state
//===--------------------------------------------------------------------===//

enum class StateModel : uint8_t { OLS, WLS, RIDGE };

const char *ModelFunctionName(StateModel m) {
	switch (m) {
	case StateModel::OLS:
		return "ols_fit_from_state";
	case StateModel::WLS:
		return "wls_fit_from_state";
	default:
		return "ridge_fit_from_state";
	}
}

struct FitFromStateBindData : public FunctionData {
	StateModel model = StateModel::OLS;
	bool fit_intercept = true;
	bool compute_inference = false;
	double confidence_level = 0.95;
	SolverType solver = SolverType::SVD;
	HcType hc_type = HcType::NONE;
	double alias_tolerance = 0.0;
	bool compute_loo = false;
	double alpha = 1.0;
	LambdaScaling lambda_scaling = LambdaScaling::RAW;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FitFromStateBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<FitFromStateBindData>();
		return model == o.model && fit_intercept == o.fit_intercept && compute_inference == o.compute_inference &&
		       confidence_level == o.confidence_level && solver == o.solver && hc_type == o.hc_type &&
		       alias_tolerance == o.alias_tolerance && compute_loo == o.compute_loo && alpha == o.alpha &&
		       lambda_scaling == o.lambda_scaling;
	}
};

LogicalType FitFromStateResultType() {
	child_list_t<LogicalType> c;
	c.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(make_pair("intercept", LogicalType::DOUBLE));
	c.push_back(make_pair("r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("adj_r_squared", LogicalType::DOUBLE));
	c.push_back(make_pair("residual_std_error", LogicalType::DOUBLE));
	c.push_back(make_pair("n_observations", LogicalType::BIGINT));
	c.push_back(make_pair("n_features", LogicalType::BIGINT));
	// Same shape as ols_fit_agg / wls_fit_agg / ridge_fit_agg (#152).
	AppendCoefficientInferenceFields(c);
	AppendModelSummaryFields(c);
	AppendCovarianceFields(c);
	AppendLooFields(c);
	return LogicalType::STRUCT(std::move(c));
}

template <StateModel MODEL>
unique_ptr<FunctionData> FitFromStateBind(ClientContext &context, ScalarBindFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<FitFromStateBindData>();
	bind->model = MODEL;
	const char *name = ModelFunctionName(MODEL);
	if (arguments.size() >= 2) {
		vector<string> keys = {"fit_intercept", "compute_inference", "confidence_level", "solver", "compute_loo"};
		if (MODEL == StateModel::RIDGE) {
			keys.insert(keys.end(), {"alpha", "lambda", "lambda_scaling"});
		} else {
			// hc_type is accepted so that asking for it gives the "needs the rows" error.
			keys.insert(keys.end(), {"hc_type", "alias_tolerance"});
		}
		auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[1], name, keys);
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.compute_inference.has_value()) {
			bind->compute_inference = opts.compute_inference.value();
		}
		if (opts.confidence_level.has_value()) {
			bind->confidence_level = opts.confidence_level.value();
		}
		if (opts.solver.has_value()) {
			bind->solver = opts.solver.value();
		}
		if (opts.compute_loo.has_value()) {
			bind->compute_loo = opts.compute_loo.value();
		}
		if (opts.hc_type.has_value()) {
			bind->hc_type = opts.hc_type.value();
		}
		if (opts.alias_tolerance.has_value()) {
			bind->alias_tolerance = opts.alias_tolerance.value();
		}
		auto reg_strength = opts.GetRegularizationStrength();
		if (reg_strength.has_value()) {
			bind->alpha = reg_strength.value();
		}
		if (opts.lambda_scaling.has_value()) {
			bind->lambda_scaling = opts.lambda_scaling.value();
		}
	}
	SetReturnTypeOf(bound_function, FitFromStateResultType());
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

// Named binds: DATAZOO_GUARD takes a function name, not a template-dependent one.
unique_ptr<FunctionData> OlsFromStateBind(ClientContext &context, ScalarBindFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	return FitFromStateBind<StateModel::OLS>(context, bound_function, arguments);
}
unique_ptr<FunctionData> WlsFromStateBind(ClientContext &context, ScalarBindFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	return FitFromStateBind<StateModel::WLS>(context, bound_function, arguments);
}
unique_ptr<FunctionData> RidgeFromStateBind(ClientContext &context, ScalarBindFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	return FitFromStateBind<StateModel::RIDGE>(context, bound_function, arguments);
}

bool CallFitFromState(const FitFromStateBindData &b, const string_t &blob, AnofoxFitResultCore &core,
                      AnofoxFitResultInference *inference, double **vcov, size_t *dim, AnofoxError &error) {
	auto bytes = reinterpret_cast<const uint8_t *>(blob.GetData());
	size_t len = blob.GetSize();
	switch (b.model) {
	case StateModel::OLS: {
		AnofoxOlsOptions o {};
		o.fit_intercept = b.fit_intercept;
		o.compute_inference = b.compute_inference;
		o.confidence_level = b.confidence_level;
		o.solver = ConvertSolverType(b.solver);
		o.hc_type = ConvertHcType(b.hc_type);
		o.alias_tolerance = b.alias_tolerance;
		o.compute_loo = b.compute_loo;
		return anofox_ols_fit_from_state(bytes, len, o, &core, inference, vcov, dim, &error);
	}
	case StateModel::WLS: {
		AnofoxWlsOptions o {};
		o.fit_intercept = b.fit_intercept;
		o.compute_inference = b.compute_inference;
		o.confidence_level = b.confidence_level;
		o.solver = ConvertSolverType(b.solver);
		o.hc_type = ConvertHcType(b.hc_type);
		o.alias_tolerance = b.alias_tolerance;
		o.compute_loo = b.compute_loo;
		return anofox_wls_fit_from_state(bytes, len, o, &core, inference, vcov, dim, &error);
	}
	default: {
		AnofoxRidgeOptions o {};
		o.alpha = b.alpha;
		o.fit_intercept = b.fit_intercept;
		o.compute_inference = b.compute_inference;
		o.confidence_level = b.confidence_level;
		o.solver = ConvertSolverType(b.solver);
		o.lambda_scaling = ConvertLambdaScaling(b.lambda_scaling);
		o.compute_loo = b.compute_loo;
		return anofox_ridge_fit_from_state(bytes, len, o, &core, inference, vcov, dim, &error);
	}
	}
}

void FitFromStateImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &b = BindInfoOf(state.expr.Cast<BoundFunctionExpression>())->Cast<FitFromStateBindData>();
	const char *name = ModelFunctionName(b.model);
	auto blobs = FlatVector::GetData<string_t>(args.data[0]);
	auto &entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);
	// ridge_fit_agg reports coefficient inference only without a penalty.
	const bool coef_inference = b.compute_inference && (b.model != StateModel::RIDGE || b.alpha == 0.0);
	for (idx_t row = 0; row < args.size(); row++) {
		AnofoxFitResultCore core;
		AnofoxFitResultInference inference;
		double *vcov = nullptr;
		size_t dim = 0;
		AnofoxError error;
		if (!CallFitFromState(b, blobs[row], core, b.compute_inference ? &inference : nullptr, &vcov, &dim, error)) {
			ThrowUnlessDegenerate(name, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t f = 0;
		auto &coef = StructEntry(entries[f++]);
		auto &child = ListVector::GetEntry(coef);
		auto list_offset = ListVector::GetListSize(coef);
		ListVector::Reserve(coef, list_offset + core.coefficients_len);
		ListVector::SetListSize(coef, list_offset + core.coefficients_len);
		auto child_data = FlatDataMut<double>(child);
		for (size_t j = 0; j < core.coefficients_len; j++) {
			child_data[list_offset + j] = core.coefficients[j];
		}
		ListVector::GetData(coef)[row] = {list_offset, static_cast<idx_t>(core.coefficients_len)};
		FlatDataMut<double>(StructEntry(entries[f++]))[row] = core.intercept;
		FlatDataMut<double>(StructEntry(entries[f++]))[row] = core.r_squared;
		FlatDataMut<double>(StructEntry(entries[f++]))[row] = core.adj_r_squared;
		FlatDataMut<double>(StructEntry(entries[f++]))[row] = core.residual_std_error;
		FlatDataMut<int64_t>(StructEntry(entries[f++]))[row] = static_cast<int64_t>(core.n_observations);
		FlatDataMut<int64_t>(StructEntry(entries[f++]))[row] = static_cast<int64_t>(core.n_features);
		writer.WriteInference(row, coef_inference ? &inference : nullptr);
		writer.WriteSummary(row, core.summary);
		writer.WriteCovariance(row, vcov, dim,
		                       vcov ? ResidualDf(core.n_observations, core.coefficients, core.coefficients_len,
		                                         b.fit_intercept)
		                            : 0.0);
		anofox_free_interval_matrix(vcov);
		if (b.compute_inference) {
			anofox_free_result_inference(&inference);
		}
		anofox_free_result_core(&core);
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

void FitFromStateFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	args.Flatten();
	// A NULL state gives a NULL row; a NULL options argument means no options (#276).
	ExecuteOnNonNullRows(args, state, result, 1, FitFromStateImpl);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

FunctionDescription Describe(const string &description, const string &example, vector<string> names,
                             vector<LogicalType> types, vector<string> categories = {"regression", "state"}) {
	FunctionDescription d;
	d.description = description;
	d.examples = {example};
	d.categories = std::move(categories);
	d.parameter_names = std::move(names);
	d.parameter_types = std::move(types);
	return d;
}

template <StateModel MODEL>
void RegisterFitFromState(ExtensionLoader &loader, const string &description, const string &example,
                          const string &options_example) {
	const char *name = ModelFunctionName(MODEL);
	ScalarFunctionSet set(name);
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::BLOB};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		auto bind = [] {
			if constexpr (MODEL == StateModel::OLS) {
				return ANOFOX_SCALAR_BIND(DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, OlsFromStateBind));
			} else if constexpr (MODEL == StateModel::WLS) {
				return ANOFOX_SCALAR_BIND(DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, WlsFromStateBind));
			} else {
				return ANOFOX_SCALAR_BIND(DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, RidgeFromStateBind));
			}
		}();
		ScalarFunction fn(args, LogicalType::ANY, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, FitFromStateFunction), bind);
		MarkFallible(fn);
		if (with_options) {
			SetNullHandlingOf(fn, FunctionNullHandling::SPECIAL_HANDLING);
		}
		set.AddFunction(fn);
	}
	CreateScalarFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	info.descriptions.push_back(Describe(description, example, {"state"}, {LogicalType::BLOB}));
	info.descriptions.push_back(
	    Describe(description, options_example, {"state", "options"}, {LogicalType::BLOB, LogicalType::ANY}));
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterLsStateFunctions(ExtensionLoader &loader) {
	const auto x_type = LogicalType::LIST(LogicalType::DOUBLE);

	// ls_state_agg
	{
		AggregateFunctionSet set("ls_state_agg");
		AggregateFunction plain("ls_state_agg", {LogicalType::DOUBLE, x_type}, LogicalType::BLOB,
		                        AggregateFunction::StateSize<XyRowsState>, ANOFOX_AGG_INIT(XyRowsInitialize),
		                        XyRowsUpdate, XyRowsCombine,
		                        ANOFOX_GUARDED_FINALIZE(LsStateFinalize<false>, XyRowsDestroy, XyRowsInitialize),
		                        nullptr, nullptr, XyRowsDestroy);
		AggregateFunction weighted("ls_state_agg", {LogicalType::DOUBLE, x_type, LogicalType::DOUBLE},
		                           LogicalType::BLOB, AggregateFunction::StateSize<XyRowsState>,
		                           ANOFOX_AGG_INIT(XyRowsInitialize), LsStateWeightedUpdate, XyRowsCombine,
		                           ANOFOX_GUARDED_FINALIZE(LsStateFinalize<true>, XyRowsDestroy, XyRowsInitialize),
		                           nullptr, nullptr, XyRowsDestroy);
		set.AddFunction(plain);
		set.AddFunction(weighted);
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const string description =
		    "Compact, mergeable least-squares state of a group's rows (counts, weighted means and centred co-moments, "
		    "O(p^2) bytes) as a BLOB. Store it, merge states with ls_state_combine_agg, and fit with "
		    "ols_fit_from_state / wls_fit_from_state / ridge_fit_from_state without the rows. Rows with a NULL y, x "
		    "element or weight are skipped; the bytes do not depend on the row order.";
		info.descriptions.push_back(
		    Describe(description, "ls_state_agg(y, [x1, x2])", {"y", "x"}, {LogicalType::DOUBLE, x_type}));
		info.descriptions.push_back(Describe(description, "ls_state_agg(y, [x1, x2], weight)", {"y", "x", "weight"},
		                                     {LogicalType::DOUBLE, x_type, LogicalType::DOUBLE}));
		loader.RegisterFunction(std::move(info));
	}

	// ls_state_combine_agg
	{
		AggregateFunction fn("ls_state_combine_agg", {LogicalType::BLOB}, LogicalType::BLOB,
		                     AggregateFunction::StateSize<LsCombineState>, ANOFOX_AGG_INIT(LsCombineInitialize),
		                     LsCombineUpdate, LsCombineCombine,
		                     ANOFOX_GUARDED_FINALIZE(LsCombineFinalize, LsCombineDestroy, LsCombineInitialize),
		                     nullptr, nullptr, LsCombineDestroy);
		AggregateFunctionSet set("ls_state_combine_agg");
		set.AddFunction(fn);
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		info.descriptions.push_back(Describe(
		    "Merges least-squares states from ls_state_agg into one state, as if ls_state_agg had seen all their "
		    "rows (NULL states are skipped). States must have the same number of features; the result does not "
		    "depend on the order of the states.",
		    "ls_state_combine_agg(state)", {"state"}, {LogicalType::BLOB}));
		loader.RegisterFunction(std::move(info));
	}

	// ls_state_info
	{
		ScalarFunction fn("ls_state_info", {LogicalType::BLOB}, LsStateInfoType(),
		                  DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, LsStateInfoFunction));
		MarkFallible(fn);
		CreateScalarFunctionInfo info(fn);
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		info.descriptions.push_back(Describe("Row count (zero-weight rows included), feature count and whether "
		                                     "the rows were weighted, of a least-squares state.",
		                                     "ls_state_info(state)", {"state"}, {LogicalType::BLOB}));
		loader.RegisterFunction(std::move(info));
	}

	const string row_note = " Row-level options (hc_type, compute_loo) need the rows and raise an error.";
	RegisterFitFromState<StateModel::OLS>(
	    loader,
	    "Fits OLS from a least-squares state (ls_state_agg / ls_state_combine_agg): the same STRUCT as ols_fit_agg "
	    "on the rows (coefficients, inference with compute_inference, vcov), up to floating-point summation order. "
	    "A weighted state gives the WLS fit." +
	        row_note,
	    "ols_fit_from_state(state)", "ols_fit_from_state(state, {'compute_inference': true})");
	RegisterFitFromState<StateModel::WLS>(
	    loader,
	    "Fits WLS from a weighted least-squares state (ls_state_agg(y, x, weight)): the same STRUCT as wls_fit_agg "
	    "on the rows." +
	        row_note,
	    "wls_fit_from_state(state)", "wls_fit_from_state(state, {'compute_inference': true})");
	RegisterFitFromState<StateModel::RIDGE>(
	    loader,
	    "Fits ridge regression from a least-squares state: the same STRUCT as ridge_fit_agg on the rows. The "
	    "penalty (alpha, lambda_scaling) is applied here, so one stored state serves any alpha." +
	        row_note,
	    "ridge_fit_from_state(state, {'alpha': 1.0})", "ridge_fit_from_state(state, {'alpha': 0.5})");
}

} // namespace duckdb
