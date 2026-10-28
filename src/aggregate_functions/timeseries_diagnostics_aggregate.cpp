// Time-series regression diagnostics (#155):
//
//   acf_agg(x [, max_lag] ORDER BY t)            -> [{lag, acf, conf_low, conf_high}]
//   pacf_agg(x [, max_lag] ORDER BY t)           -> [{lag, pacf, conf_low, conf_high}]
//   ljung_box_agg(resid [, lag [, fitdf]] ORDER BY t)   / box_pierce_agg(...)
//   durbin_watson_agg(resid ORDER BY t)
//   breusch_pagan_agg(resid, [x1, ...] [, studentize]) / white_test_agg(resid, [x1, ...])
//
// The order-sensitive aggregates read the rows in the aggregate's ORDER BY order
// (constant arguments go before the ORDER BY: `acf_agg(x, 12 ORDER BY t)`)
// and skip NULL / NaN values. The tests return the unified test shape (#159);
// the band of acf/pacf is R's 95% white-noise band +-qnorm(0.975)/sqrt(n). The
// statistics come from anofox-statistics (acf, pacf, Box tests, Durbin-Watson)
// and anofox-regression (Breusch-Pagan, White).

#include <cmath>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/test_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "list_input.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

// Must match AcfResultFFI (crates/anofox-stats-ffi/src/types.rs): a double, then
// the pointer and three sizes (no padding on 64-bit or wasm32).
static_assert(offsetof(AnofoxAcfResult, values) == sizeof(double) &&
                  offsetof(AnofoxAcfResult, n) == sizeof(double) + sizeof(double *) + 2 * sizeof(size_t),
              "AnofoxAcfResult must match AcfResultFFI in crates/anofox-stats-ffi/src/types.rs");

enum class SeriesKind : uint8_t { ACF, PACF, LJUNG_BOX, BOX_PIERCE, DURBIN_WATSON };

static const char *SeriesName(SeriesKind kind) {
	switch (kind) {
	case SeriesKind::ACF:
		return "acf_agg";
	case SeriesKind::PACF:
		return "pacf_agg";
	case SeriesKind::LJUNG_BOX:
		return "ljung_box_agg";
	case SeriesKind::BOX_PIERCE:
		return "box_pierce_agg";
	case SeriesKind::DURBIN_WATSON:
	default:
		return "durbin_watson_agg";
	}
}

//! A constant argument (lag, fitdf, studentize, ...), evaluated at bind time.
static Value ConstantArgument(ClientContext &context, Expression &expr, const string &function_name, const char *what) {
	if (!expr.IsFoldable()) {
		throw InvalidInputException("%s: %s must be a constant", function_name, what);
	}
	return ExpressionExecutor::EvaluateScalar(context, expr);
}

static int64_t ConstantCount(ClientContext &context, Expression &expr, const string &function_name, const char *what,
                             int64_t minimum) {
	auto value = ConstantArgument(context, expr, function_name, what);
	if (value.IsNull()) {
		throw InvalidInputException("%s: %s must not be NULL", function_name, what);
	}
	auto v = value.GetValue<int64_t>();
	if (v < minimum) {
		throw InvalidInputException("%s: %s must be >= %lld, got %lld", function_name, what, (long long)minimum,
		                            (long long)v);
	}
	return v;
}

//===--------------------------------------------------------------------===//
// Series aggregates: acf, pacf, ljung_box, box_pierce, durbin_watson
//===--------------------------------------------------------------------===//
struct SeriesState {
	vector<double> values;
};

struct SeriesBindData : public FunctionData {
	SeriesKind kind = SeriesKind::ACF;
	//! acf/pacf max_lag (-1: R's default) or the portmanteau lag.
	int64_t lag = -1;
	int64_t fitdf = 0;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<SeriesBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<SeriesBindData>();
		return kind == o.kind && lag == o.lag && fitdf == o.fitdf;
	}
};

static LogicalType AcfRowType(SeriesKind kind) {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("lag", LogicalType::BIGINT));
	children.push_back(make_pair(kind == SeriesKind::PACF ? "pacf" : "acf", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_low", LogicalType::DOUBLE));
	children.push_back(make_pair("conf_high", LogicalType::DOUBLE));
	return LogicalType::STRUCT(std::move(children));
}

static LogicalType SeriesResultType(SeriesKind kind) {
	if (kind == SeriesKind::ACF || kind == SeriesKind::PACF) {
		return LogicalType::LIST(AcfRowType(kind));
	}
	child_list_t<LogicalType> children;
	if (kind == SeriesKind::LJUNG_BOX || kind == SeriesKind::BOX_PIERCE) {
		children.push_back(make_pair("lag", LogicalType::BIGINT));
		children.push_back(make_pair("fitdf", LogicalType::BIGINT));
	}
	AppendTestResultFields(children);
	return LogicalType::STRUCT(std::move(children));
}

static void SeriesInitialize(data_ptr_t state_p) {
	new (state_p) SeriesState();
}

static void SeriesDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (SeriesState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~SeriesState();
	}
}

static void SeriesUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat value_data, sdata;
	inputs[0].ToUnifiedFormat(count, value_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto values = UnifiedVectorFormat::GetData<double>(value_data);
	auto states = (SeriesState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = value_data.sel->get_index(i);
		if (!value_data.validity.RowIsValid(idx) || std::isnan(values[idx])) {
			continue;
		}
		states[sdata.sel->get_index(i)]->values.push_back(values[idx]);
	}
}

static void SeriesCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                          idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (SeriesState **)source_data.data;
	auto targets = (SeriesState **)target_data.data;
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

static void SetNumberOrNull(Vector &vec, idx_t row, double value) {
	if (std::isnan(value)) {
		FlatVector::SetNull(vec, row, true);
	} else {
		FlatDataMut<double>(vec)[row] = value;
	}
}

static void SeriesFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (SeriesState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<SeriesBindData>();
	const string fn_name = SeriesName(bind.kind);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (state.values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		AnofoxDataArray x;
		x.data = state.values.data();
		x.validity = nullptr;
		x.len = static_cast<size_t>(state.values.size());
		AnofoxError error;

		if (bind.kind == SeriesKind::ACF || bind.kind == SeriesKind::PACF) {
			AnofoxAcfResult acf;
			if (!anofox_autocorrelation(x, bind.lag, bind.kind == SeriesKind::PACF, &acf, &error)) {
				ThrowUnlessDegenerate(fn_name.c_str(), error);
				FlatVector::SetNull(result, row, true);
				continue;
			}
			auto len = static_cast<idx_t>(acf.len);
			auto list_offset = ListVector::GetListSize(result);
			ListVector::Reserve(result, list_offset + len);
			auto &fields = StructVector::GetEntries(ListVector::GetEntry(result));
			for (idx_t k = 0; k < len; k++) {
				auto out = list_offset + k;
				FlatDataMut<int64_t>(StructEntry(fields[0]))[out] = static_cast<int64_t>(acf.first_lag + k);
				SetNumberOrNull(StructEntry(fields[1]), out, acf.values[k]);
				FlatDataMut<double>(StructEntry(fields[2]))[out] = -acf.conf_bound;
				FlatDataMut<double>(StructEntry(fields[3]))[out] = acf.conf_bound;
			}
			ListVector::SetListSize(result, list_offset + len);
			ListVector::GetData(result)[row] = list_entry_t(list_offset, len);
			anofox_free_acf_result(&acf);
			continue;
		}

		AnofoxTestSummaryFFI summary;
		bool ok;
		if (bind.kind == SeriesKind::DURBIN_WATSON) {
			ok = anofox_durbin_watson(x, &summary, &error);
		} else {
			ok = anofox_portmanteau(x, static_cast<size_t>(bind.lag), static_cast<size_t>(bind.fitdf),
			                        bind.kind == SeriesKind::BOX_PIERCE, &summary, &error);
		}
		if (!ok) {
			ThrowUnlessDegenerate(fn_name.c_str(), error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		idx_t struct_idx = 0;
		auto &entries = StructVector::GetEntries(result);
		if (bind.kind != SeriesKind::DURBIN_WATSON) {
			FlatDataMut<int64_t>(StructEntry(entries[struct_idx++]))[row] = bind.lag;
			FlatDataMut<int64_t>(StructEntry(entries[struct_idx++]))[row] = bind.fitdf;
		}
		WriteTestResultFields(result, row, struct_idx, summary);
	}
}

template <SeriesKind KIND>
static unique_ptr<FunctionData> SeriesBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	const string fn_name = SeriesName(KIND);
	auto bind = make_uniq<SeriesBindData>();
	bind->kind = KIND;
	if (KIND == SeriesKind::LJUNG_BOX || KIND == SeriesKind::BOX_PIERCE) {
		bind->lag = 1; // R Box.test default
	}
	if (arguments.size() >= 2) {
		if (KIND == SeriesKind::ACF || KIND == SeriesKind::PACF) {
			// A NULL max_lag keeps R's default (lets table macros pass it through).
			if (!ConstantArgument(context, *arguments[1], fn_name, "max_lag").IsNull()) {
				bind->lag = ConstantCount(context, *arguments[1], fn_name, "max_lag", KIND == SeriesKind::ACF ? 0 : 1);
			}
		} else {
			bind->lag = ConstantCount(context, *arguments[1], fn_name, "lag", 1);
		}
	}
	if (arguments.size() >= 3) {
		bind->fitdf = ConstantCount(context, *arguments[2], fn_name, "fitdf", 0);
		if (bind->fitdf >= bind->lag) {
			throw InvalidInputException("%s: fitdf must be < lag (%lld >= %lld)", fn_name, (long long)bind->fitdf,
			                            (long long)bind->lag);
		}
	}
	SetReturnTypeOf(function, SeriesResultType(KIND));
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(bind);
}

template <SeriesKind KIND>
static AggregateFunction SeriesFunction(vector<LogicalType> args) {
	AggregateFunction fn(SeriesName(KIND), std::move(args), LogicalType::ANY, AggregateFunction::StateSize<SeriesState>,
	                     ANOFOX_AGG_INIT(SeriesInitialize), ANOFOX_GUARDED_UPDATE(SeriesUpdate, SeriesDestroy, SeriesInitialize),
	                     SeriesCombine, ANOFOX_GUARDED_FINALIZE(SeriesFinalize, SeriesDestroy, SeriesInitialize),
	                     nullptr, ANOFOX_AGG_BIND(SeriesBind<KIND>), SeriesDestroy);
	// Row order is part of the input: declare it so DuckDB honours `agg(... ORDER BY t)`.
	SetOrderDependentOf(fn, AggregateOrderDependent::ORDER_DEPENDENT); // compat helper works on DuckDB 1.5 and 2.0
	return fn;
}

template <SeriesKind KIND>
static void RegisterSeries(ExtensionLoader &loader, const vector<vector<string>> &overloads, const string &description,
                           const string &example, const vector<string> &categories) {
	const string name = SeriesName(KIND);
	AggregateFunctionSet set(AsDuckName(name));
	for (auto &params : overloads) {
		vector<LogicalType> args = {LogicalType::DOUBLE};
		for (idx_t p = 1; p < params.size(); p++) {
			args.push_back(LogicalType::BIGINT);
		}
		set.AddFunction(SeriesFunction<KIND>(args));
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	for (auto &params : overloads) {
		FunctionDescription d;
		d.description = description;
		d.examples = {example};
		d.categories = categories;
		d.parameter_names = params;
		d.parameter_types = {LogicalType::DOUBLE};
		for (idx_t p = 1; p < params.size(); p++) {
			d.parameter_types.push_back(LogicalType::BIGINT);
		}
		info.descriptions.push_back(std::move(d));
	}
	loader.RegisterFunction(std::move(info));
}

//===--------------------------------------------------------------------===//
// Heteroskedasticity tests: breusch_pagan, white_test
//===--------------------------------------------------------------------===//
struct HeteroState {
	vector<double> residuals;
	vector<vector<double>> x_columns;
	idx_t n_features = 0;
	bool initialized = false;
};

//! breusch_pagan_agg, white_test_agg, and durbin_watson_agg(resid, x ...) (the test with a p-value).
enum class HeteroKind : uint8_t { BREUSCH_PAGAN, WHITE, DURBIN_WATSON };

struct HeteroBindData : public FunctionData {
	HeteroKind kind = HeteroKind::BREUSCH_PAGAN;
	bool studentize = true;
	AnofoxAlternative alternative = ANOFOX_ALTERNATIVE_GREATER;
	bool fit_intercept = true;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<HeteroBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<HeteroBindData>();
		return kind == o.kind && studentize == o.studentize && alternative == o.alternative &&
		       fit_intercept == o.fit_intercept;
	}
};

static const char *HeteroName(HeteroKind kind) {
	switch (kind) {
	case HeteroKind::WHITE:
		return "white_test_agg";
	case HeteroKind::DURBIN_WATSON:
		return "durbin_watson_agg";
	default:
		return "breusch_pagan_agg";
	}
}

static void HeteroInitialize(data_ptr_t state_p) {
	new (state_p) HeteroState();
}

static void HeteroDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (HeteroState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~HeteroState();
	}
}

static void HeteroUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                         idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<HeteroBindData>();
	UnifiedVectorFormat resid_data, x_data, sdata;
	inputs[0].ToUnifiedFormat(count, resid_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	state_vector.ToUnifiedFormat(count, sdata);
	auto resid = UnifiedVectorFormat::GetData<double>(resid_data);
	auto x_lists = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto states = (HeteroState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto r_idx = resid_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!resid_data.validity.RowIsValid(r_idx) || !x_data.validity.RowIsValid(x_idx) || std::isnan(resid[r_idx])) {
			continue;
		}
		auto entry = x_lists[x_idx];
		if (ListHasNullElement(x_child_validity, entry)) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(entry.length);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("%s: inconsistent feature count: expected %llu, got %llu",
			                            HeteroName(bind.kind), (unsigned long long)state.n_features,
			                            (unsigned long long)entry.length);
		}
		state.residuals.push_back(resid[r_idx]);
		for (idx_t j = 0; j < entry.length; j++) {
			state.x_columns[j].push_back(x_child_data[entry.offset + j]);
		}
	}
}

static void HeteroCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                          idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<HeteroBindData>();
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (HeteroState **)source_data.data;
	auto targets = (HeteroState **)target_data.data;
	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.residuals = CombineTake(source.residuals, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("%s: inconsistent feature count during combine", HeteroName(bind.kind));
		}
		target.residuals.insert(target.residuals.end(), source.residuals.begin(), source.residuals.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void HeteroFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (HeteroState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<HeteroBindData>();

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		const bool dw = bind.kind == HeteroKind::DURBIN_WATSON;
		if (!state.initialized || state.residuals.empty() || (state.n_features == 0 && !(dw && bind.fit_intercept))) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (!dw) {
			// Breusch-Pagan and White do not depend on the row order (Durbin-Watson
			// does): sort for bitwise-identical results whatever the thread count (#226).
			CanonicalRows().Keys(state.x_columns).Key(state.residuals).Sort();
		}
		AnofoxDataArray resid;
		resid.data = state.residuals.data();
		resid.validity = nullptr;
		resid.len = static_cast<size_t>(state.residuals.size());
		vector<AnofoxDataArray> x(state.n_features);
		for (idx_t j = 0; j < state.n_features; j++) {
			x[j].data = state.x_columns[j].data();
			x[j].validity = nullptr;
			x[j].len = static_cast<size_t>(state.x_columns[j].size());
		}
		AnofoxTestSummaryFFI summary;
		AnofoxError error;
		bool ok;
		switch (bind.kind) {
		case HeteroKind::WHITE:
			ok = anofox_white_test(resid, x.data(), x.size(), &summary, &error);
			break;
		case HeteroKind::DURBIN_WATSON:
			ok = anofox_durbin_watson_test(resid, x.data(), x.size(), bind.fit_intercept, bind.alternative, -1, &summary,
			                               &error);
			break;
		default:
			ok = anofox_breusch_pagan(resid, x.data(), x.size(), bind.studentize, &summary, &error);
			break;
		}
		if (!ok) {
			ThrowUnlessDegenerate(HeteroName(bind.kind), error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		WriteTestResultFields(result, row, 0, summary);
	}
}

template <HeteroKind KIND>
static unique_ptr<FunctionData> HeteroBind(ClientContext &context, AggregateBindFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<HeteroBindData>();
	bind->kind = KIND;
	const string name = HeteroName(KIND);
	if (KIND == HeteroKind::BREUSCH_PAGAN && arguments.size() >= 3) {
		auto v = ConstantArgument(context, *arguments[2], name.c_str(), "studentize");
		if (v.IsNull()) {
			throw InvalidInputException("%s: studentize must not be NULL", name);
		}
		bind->studentize = v.GetValue<bool>();
	}
	if (KIND == HeteroKind::DURBIN_WATSON && arguments.size() >= 3) {
		auto v = ConstantArgument(context, *arguments[2], name.c_str(), "alternative");
		if (v.IsNull()) {
			throw InvalidInputException("%s: alternative must not be NULL", name);
		}
		auto alt = StringUtil::Lower(v.ToString());
		if (alt == "greater" || alt == "right") {
			bind->alternative = ANOFOX_ALTERNATIVE_GREATER;
		} else if (alt == "less" || alt == "left") {
			bind->alternative = ANOFOX_ALTERNATIVE_LESS;
		} else if (alt == "two_sided" || alt == "two-sided" || alt == "two.sided" || alt == "twosided") {
			bind->alternative = ANOFOX_ALTERNATIVE_TWO_SIDED;
		} else {
			throw InvalidInputException("%s: invalid alternative '%s' (expected 'greater', 'less' or 'two_sided')", name,
			                            alt);
		}
	}
	if (KIND == HeteroKind::DURBIN_WATSON && arguments.size() >= 4) {
		auto v = ConstantArgument(context, *arguments[3], name.c_str(), "fit_intercept");
		if (v.IsNull()) {
			throw InvalidInputException("%s: fit_intercept must not be NULL", name);
		}
		bind->fit_intercept = v.GetValue<bool>();
	}
	child_list_t<LogicalType> children;
	AppendTestResultFields(children);
	SetReturnTypeOf(function, LogicalType::STRUCT(std::move(children)));
	PostHogTelemetry::Instance().RecordFunctionCall(name);
	return std::move(bind);
}

template <HeteroKind KIND>
static AggregateFunction HeteroFunction(vector<LogicalType> args) {
	AggregateFunction fn(HeteroName(KIND), std::move(args), LogicalType::ANY,
	                     AggregateFunction::StateSize<HeteroState>, ANOFOX_AGG_INIT(HeteroInitialize),
	                     ANOFOX_GUARDED_UPDATE(HeteroUpdate, HeteroDestroy, HeteroInitialize), HeteroCombine,
	                     ANOFOX_GUARDED_FINALIZE(HeteroFinalize, HeteroDestroy, HeteroInitialize), nullptr,
	                     ANOFOX_AGG_BIND(HeteroBind<KIND>), HeteroDestroy);
	if (KIND == HeteroKind::DURBIN_WATSON) {
		// Residual order is part of the input: honour `agg(... ORDER BY t)`.
		SetOrderDependentOf(fn, AggregateOrderDependent::ORDER_DEPENDENT);
	}
	return fn;
}

void RegisterTimeSeriesDiagnosticsAggregateFunctions(ExtensionLoader &loader) {
	const vector<string> ts = {"time-series", "diagnostics"};
	RegisterSeries<SeriesKind::ACF>(
	    loader, {{"x"}, {"x", "max_lag"}},
	    "Sample autocorrelation function of x in the aggregate's ORDER BY order (R acf): one row per lag 0..max_lag "
	    "with the 95% white-noise band conf_low/conf_high = -/+ qnorm(0.975)/sqrt(n). max_lag defaults to R's "
	    "floor(10 log10 n). NULL values are skipped; NULL when n <= max_lag.",
	    "acf_agg(resid, 10 ORDER BY t)", ts);
	RegisterSeries<SeriesKind::PACF>(
	    loader, {{"x"}, {"x", "max_lag"}},
	    "Sample partial autocorrelation function of x in the aggregate's ORDER BY order (R pacf): one row per lag "
	    "1..max_lag with the 95% white-noise band. max_lag defaults to R's floor(10 log10 n).",
	    "pacf_agg(resid, 10 ORDER BY t)", ts);
	RegisterSeries<SeriesKind::LJUNG_BOX>(
	    loader, {{"x"}, {"x", "lag"}, {"x", "lag", "fitdf"}},
	    "Ljung-Box portmanteau test for autocorrelation of x in the aggregate's ORDER BY order (R Box.test(x, lag, "
	    "type = 'Ljung-Box', fitdf)): statistic Q, df1 = lag - fitdf, p_value. lag defaults to 1, fitdf to 0.",
	    "ljung_box_agg(resid, 10, 2 ORDER BY t)", ts);
	RegisterSeries<SeriesKind::BOX_PIERCE>(
	    loader, {{"x"}, {"x", "lag"}, {"x", "lag", "fitdf"}},
	    "Box-Pierce portmanteau test for autocorrelation of x in the aggregate's ORDER BY order (R Box.test(x, lag, "
	    "type = 'Box-Pierce', fitdf)). lag defaults to 1, fitdf to 0.",
	    "box_pierce_agg(resid, 10 ORDER BY t)", ts);

	const auto x_type = LogicalType::LIST(LogicalType::DOUBLE);
	{
		AggregateFunctionSet set("breusch_pagan_agg");
		set.AddFunction(HeteroFunction<HeteroKind::BREUSCH_PAGAN>({LogicalType::DOUBLE, x_type}));
		set.AddFunction(HeteroFunction<HeteroKind::BREUSCH_PAGAN>({LogicalType::DOUBLE, x_type, LogicalType::BOOLEAN}));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		for (bool with_flag : {false, true}) {
			FunctionDescription d;
			d.description = "Breusch-Pagan test for heteroskedasticity of regression residuals against the "
			                "regressors x (R lmtest::bptest): statistic, df1, p_value. studentize (default true) is "
			                "Koenker's robust variant, bptest's default; false gives the original 1979 statistic.";
			d.examples = {"breusch_pagan_agg(resid, [x1, x2])"};
			d.categories = {"regression", "diagnostics"};
			d.parameter_names = {"resid", "x"};
			d.parameter_types = {LogicalType::DOUBLE, x_type};
			if (with_flag) {
				d.examples = {"breusch_pagan_agg(resid, [x1, x2], false)"};
				d.parameter_names.push_back("studentize");
				d.parameter_types.push_back(LogicalType::BOOLEAN);
			}
			info.descriptions.push_back(std::move(d));
		}
		loader.RegisterFunction(std::move(info));
	}
	{
		AggregateFunctionSet set("white_test_agg");
		set.AddFunction(HeteroFunction<HeteroKind::WHITE>({LogicalType::DOUBLE, x_type}));
		CreateAggregateFunctionInfo info(std::move(set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription d;
		d.description = "White's test for heteroskedasticity: the studentized Breusch-Pagan test against x, its "
		                "squares and pairwise cross-products (R bptest(fit, ~ x1 * x2 + I(x1^2) + I(x2^2))).";
		d.examples = {"white_test_agg(resid, [x1, x2])"};
		d.categories = {"regression", "diagnostics"};
		d.parameter_names = {"resid", "x"};
		d.parameter_types = {LogicalType::DOUBLE, x_type};
		info.descriptions.push_back(std::move(d));
		loader.RegisterFunction(std::move(info));
	}
	{
		// durbin_watson_agg(resid, x [, alternative [, fit_intercept]] ORDER BY t): the test with a p-value.
		const vector<vector<LogicalType>> arg_lists = {
		    {LogicalType::DOUBLE, x_type},
		    {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR},
		    {LogicalType::DOUBLE, x_type, LogicalType::VARCHAR, LogicalType::BOOLEAN}};
		AggregateFunctionSet set("durbin_watson_agg");
		set.AddFunction(SeriesFunction<SeriesKind::DURBIN_WATSON>({LogicalType::DOUBLE}));
		for (auto &args : arg_lists) {
			set.AddFunction(HeteroFunction<HeteroKind::DURBIN_WATSON>(args));
		}
		CreateAggregateFunctionInfo info(std::move(set));
		{
			FunctionDescription d;
			d.description =
			    "Durbin-Watson statistic sum((e_t - e_{t-1})^2) / sum(e_t^2) of residuals in the aggregate's ORDER BY "
			    "order (the statistic of R lmtest::dwtest). p_value is NULL here: pass the regressors, "
			    "durbin_watson_agg(resid, x ORDER BY t), for the test with a p-value.";
			d.examples = {"durbin_watson_agg(resid ORDER BY t)"};
			d.categories = ts;
			d.parameter_names = {"resid"};
			d.parameter_types = {LogicalType::DOUBLE};
			info.descriptions.push_back(std::move(d));
		}
		for (auto &args : arg_lists) {
			FunctionDescription d;
			d.description =
			    "Durbin-Watson test for first-order autocorrelation of regression residuals in the aggregate's ORDER "
			    "BY order, with the p-value of R lmtest::dwtest: x are the model's regressors (an intercept column is "
			    "added unless fit_intercept is false). The p-value is exact (Imhof) for n < 100, else the normal "
			    "approximation. alternative: 'greater' (default, positive autocorrelation), 'less' or 'two_sided'.";
			d.examples = {"durbin_watson_agg(resid, [x1, x2] ORDER BY t)"};
			d.categories = ts;
			d.parameter_names = {"resid", "x", "alternative", "fit_intercept"};
			d.parameter_names.resize(args.size());
			d.parameter_types = args;
			info.descriptions.push_back(std::move(d));
		}
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		loader.RegisterFunction(std::move(info));
	}
}

} // namespace duckdb
