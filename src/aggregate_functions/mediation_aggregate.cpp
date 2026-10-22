#include <cmath>
#include <limits>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "aggregate_finalize_guard.hpp"
#include "canonical_order.hpp"
#include "list_input.hpp"
#include "telemetry.hpp"
#include "xy_rows_aggregate.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Causal mediation analysis (issue #223).
//
//   mediation_agg(y DOUBLE, m DOUBLE, x DOUBLE [, covariates DOUBLE[] [, options]])
//
// Fits the mediator model m ~ x + covariates and the outcome model
// y ~ x + m + covariates (OLS, upstream anofox-regression) and returns one
// `terms`-like row per effect: acme, ade, total, prop_mediated. Rows with a NULL
// in y, m, x, the covariate list or one of its elements are dropped from both
// models.
//
// The rows are collected in an XyRowsState with x_columns = [x, m, covariates...]
// and sorted canonically before the seeded bootstrap, so the result does not
// depend on the order DuckDB's threads delivered them.
//===--------------------------------------------------------------------===//

namespace {

constexpr const char *kMediationName = "mediation_agg";

struct MediationBindData : public FunctionData {
	AnofoxMediationMethod method = ANOFOX_MEDIATION_BOOTSTRAP;
	AnofoxMediationCi ci_type = ANOFOX_MEDIATION_CI_PERCENTILE;
	uint32_t n_sims = 1000;
	uint64_t seed = 42;
	double conf_level = 0.95;
	double treat_value = 1.0;
	double control_value = 0.0;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MediationBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<MediationBindData>();
		return method == o.method && ci_type == o.ci_type && n_sims == o.n_sims && seed == o.seed &&
		       conf_level == o.conf_level && treat_value == o.treat_value && control_value == o.control_value;
	}

	string MethodName() const {
		switch (method) {
		case ANOFOX_MEDIATION_SOBEL:
			return "sobel";
		case ANOFOX_MEDIATION_QUASI_BAYESIAN:
			return "quasi_bayesian";
		default:
			return ci_type == ANOFOX_MEDIATION_CI_BCA ? "bootstrap_bca" : "bootstrap_percentile";
		}
	}
};

LogicalType MediationRowType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("term", LogicalType::VARCHAR));
	children.push_back(make_pair("estimate", LogicalType::DOUBLE));
	children.push_back(make_pair("std_error", LogicalType::DOUBLE));
	children.push_back(make_pair("statistic", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_level", LogicalType::DOUBLE));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	children.push_back(make_pair("n_obs", LogicalType::BIGINT));
	children.push_back(make_pair("n_sims", LogicalType::BIGINT));
	return LogicalType::STRUCT(std::move(children));
}

void MediationUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector, idx_t count) {
	const bool has_covariates = input_count >= 4;
	UnifiedVectorFormat y_data, m_data, x_data, c_data, sdata;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, m_data);
	inputs[2].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto m_values = UnifiedVectorFormat::GetData<double>(m_data);
	auto x_values = UnifiedVectorFormat::GetData<double>(x_data);
	const list_entry_t *c_lists = nullptr;
	const double *c_child = nullptr;
	const ValidityMask *c_child_validity = nullptr;
	if (has_covariates) {
		inputs[3].ToUnifiedFormat(count, c_data);
		c_lists = UnifiedVectorFormat::GetData<list_entry_t>(c_data);
		auto &child = ListVector::GetEntry(inputs[3]);
		c_child = FlatDataMut<double>(child);
		c_child_validity = &FlatValidityMut(child);
	}
	auto states = (XyRowsState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto m_idx = m_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !m_data.validity.RowIsValid(m_idx) ||
		    !x_data.validity.RowIsValid(x_idx)) {
			continue;
		}
		list_entry_t covs {0, 0};
		if (has_covariates) {
			auto c_idx = c_data.sel->get_index(i);
			if (!c_data.validity.RowIsValid(c_idx)) {
				continue;
			}
			covs = c_lists[c_idx];
			if (ListHasNullElement(*c_child_validity, covs)) {
				continue;
			}
		}
		if (!state.initialized) {
			state.n_features = 2 + covs.length;
			state.x_columns.resize(state.n_features);
			state.initialized = true;
		}
		if (2 + covs.length != state.n_features) {
			throw InvalidInputException("%s: inconsistent covariate count: expected %lu, got %lu", kMediationName,
			                            state.n_features - 2, covs.length);
		}
		state.y_values.push_back(y_values[y_idx]);
		state.x_columns[0].push_back(x_values[x_idx]);
		state.x_columns[1].push_back(m_values[m_idx]);
		for (idx_t j = 0; j < covs.length; j++) {
			state.x_columns[2 + j].push_back(c_child[covs.offset + j]);
		}
	}
}

void WriteDouble(Vector &vec, idx_t row, double value) {
	XyRowsSetDouble(vec, row, value);
}

void MediationFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                       idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<MediationBindData>();
	XyRowsStatePointers states(state_vector, count);
	const string method_name = bind.MethodName();

	for (idx_t i = 0; i < count; i++) {
		auto &state = states[i];
		const idx_t row = i + offset;
		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		SortRowsCanonically(state.y_values, state.x_columns);

		const idx_t n = state.y_values.size();
		AnofoxDataArray y_arr {state.y_values.data(), nullptr, n};
		AnofoxDataArray x_arr {state.x_columns[0].data(), nullptr, n};
		AnofoxDataArray m_arr {state.x_columns[1].data(), nullptr, n};
		vector<AnofoxDataArray> cov_arrays;
		for (idx_t j = 2; j < state.n_features; j++) {
			cov_arrays.push_back({state.x_columns[j].data(), nullptr, n});
		}

		AnofoxMediationOptions options {};
		options.method = bind.method;
		options.ci_type = bind.ci_type;
		options.n_sims = bind.n_sims;
		options.seed = bind.seed;
		options.conf_level = bind.conf_level;
		options.treat_value = bind.treat_value;
		options.control_value = bind.control_value;

		AnofoxMediationResult res {};
		AnofoxError error;
		if (!anofox_mediation(y_arr, m_arr, x_arr, cov_arrays.empty() ? nullptr : cov_arrays.data(),
		                      cov_arrays.size(), options, &res, &error)) {
			ThrowUnlessDegenerate(kMediationName, error);
			FlatVector::SetNull(result, row, true);
			continue;
		}

		const struct {
			const char *term;
			const AnofoxMediationEffect &effect;
		} effects[] = {{"acme", res.acme}, {"ade", res.ade}, {"total", res.total}, {"prop_mediated", res.prop_mediated}};
		const idx_t n_effects = sizeof(effects) / sizeof(effects[0]);

		auto list_data = FlatDataMut<list_entry_t>(result);
		auto child_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, child_offset + n_effects);
		auto &child = ListVector::GetEntry(result);
		auto &fields = StructVector::GetEntries(child);
		for (idx_t k = 0; k < n_effects; k++) {
			const idx_t out = child_offset + k;
			const auto &e = effects[k].effect;
			auto &term_vec = StructEntry(fields[0]);
			FlatDataMut<string_t>(term_vec)[out] = StringVector::AddString(term_vec, effects[k].term);
			WriteDouble(StructEntry(fields[1]), out, e.estimate);
			WriteDouble(StructEntry(fields[2]), out, e.std_error);
			WriteDouble(StructEntry(fields[3]), out, e.statistic);
			WriteDouble(StructEntry(fields[4]), out, e.p_value);
			WriteDouble(StructEntry(fields[5]), out, e.conf_low);
			WriteDouble(StructEntry(fields[6]), out, e.conf_high);
			FlatDataMut<double>(StructEntry(fields[7]))[out] = bind.conf_level;
			auto &method_vec = StructEntry(fields[8]);
			FlatDataMut<string_t>(method_vec)[out] = StringVector::AddString(method_vec, method_name);
			FlatDataMut<int64_t>(StructEntry(fields[9]))[out] = (int64_t)res.n_obs;
			FlatDataMut<int64_t>(StructEntry(fields[10]))[out] = (int64_t)res.n_sims;
		}
		ListVector::SetListSize(result, child_offset + n_effects);
		list_data[row].offset = child_offset;
		list_data[row].length = n_effects;
	}
}

unique_ptr<FunctionData> MediationBind(ClientContext &context, AggregateBindFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<MediationBindData>();
	if (arguments.size() >= 5) {
		Value options_val = EvaluateConstantOptions(context, *arguments[4], kMediationName);
		auto opts = MediationMapOptions::ParseFromValue(options_val, kMediationName);
		if (opts.method.has_value()) {
			const auto &m = opts.method.value();
			bind->method = m == "sobel"            ? ANOFOX_MEDIATION_SOBEL
			               : m == "quasi_bayesian" ? ANOFOX_MEDIATION_QUASI_BAYESIAN
			                                       : ANOFOX_MEDIATION_BOOTSTRAP;
		}
		if (opts.ci_type.has_value()) {
			bind->ci_type = opts.ci_type.value() == "bca" ? ANOFOX_MEDIATION_CI_BCA : ANOFOX_MEDIATION_CI_PERCENTILE;
		}
		if (opts.n_bootstrap.has_value()) {
			bind->n_sims = opts.n_bootstrap.value();
		}
		if (opts.seed.has_value()) {
			bind->seed = opts.seed.value();
		}
		if (opts.confidence_level.has_value()) {
			bind->conf_level = opts.confidence_level.value();
		}
		if (opts.treat_value.has_value()) {
			bind->treat_value = opts.treat_value.value();
		}
		if (opts.control_value.has_value()) {
			bind->control_value = opts.control_value.value();
		}
	}
	if (bind->method != ANOFOX_MEDIATION_SOBEL && bind->n_sims < 2) {
		throw InvalidInputException("%s: n_bootstrap must be at least 2, got %u", kMediationName, bind->n_sims);
	}
	if (bind->treat_value == bind->control_value) {
		throw InvalidInputException("%s: treat_value and control_value must differ", kMediationName);
	}
	SetReturnTypeOf(function, LogicalType::LIST(MediationRowType()));
	PostHogTelemetry::Instance().RecordFunctionCall(kMediationName);
	return std::move(bind);
}

} // namespace

void RegisterMediationAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set(kMediationName);
	const auto list_double = LogicalType::LIST(LogicalType::DOUBLE);
	const vector<vector<LogicalType>> signatures = {
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, list_double},
	    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, list_double, LogicalType::ANY}};
	for (auto &args : signatures) {
		AggregateFunction fn(kMediationName, args, LogicalType::ANY, AggregateFunction::StateSize<XyRowsState>,
		                     ANOFOX_AGG_INIT(XyRowsInitialize),
		                     ANOFOX_GUARDED_UPDATE(MediationUpdate, XyRowsDestroy, XyRowsInitialize), XyRowsCombine,
		                     ANOFOX_GUARDED_FINALIZE(MediationFinalize, XyRowsDestroy, XyRowsInitialize), nullptr,
		                     ANOFOX_AGG_BIND(MediationBind), XyRowsDestroy);
		func_set.AddFunction(fn);
	}

	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	const string description =
	    "Causal mediation analysis (linear models, as R mediation::mediate): fits m ~ x + covariates and "
	    "y ~ x + m + covariates and returns one row per effect (acme, ade, total, prop_mediated) with estimate, "
	    "std_error, statistic, p_value, conf_low, conf_high, conf_level, method, n_obs and n_sims. Inference by "
	    "nonparametric bootstrap (default), Sobel or quasi-Bayesian Monte Carlo.";
	const vector<vector<string>> names = {
	    {"y", "m", "x"}, {"y", "m", "x", "covariates"}, {"y", "m", "x", "covariates", "options"}};
	const vector<string> examples = {"mediation_agg(y, m, treat)", "mediation_agg(y, m, treat, [age, baseline])",
	                                 "mediation_agg(y, m, treat, [age], {'n_bootstrap': 2000, 'seed': 7})"};
	for (idx_t k = 0; k < signatures.size(); k++) {
		FunctionDescription d;
		d.description = description;
		d.examples = {examples[k]};
		d.categories = {"mediation"};
		d.parameter_names = names[k];
		d.parameter_types = signatures[k];
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

} // namespace duckdb
