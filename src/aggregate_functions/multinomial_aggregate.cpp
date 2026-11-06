// Multinomial logistic regression (#314): multinomial_fit_agg, multinomial_predict
// and multinomial_fit_predict_by; predict(model, x) dispatches multinomial models here.
// The fit is upstream anofox-regression MultinomialLogisticRegressor (scikit-learn's
// LogisticRegression objective with lambda = 1 / C, l1_ratio, sample weights); this
// file buffers rows, maps the class labels onto their sorted order and marshals the
// model STRUCT.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/glm_weights_arg.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/canonical_order.hpp"
#include "anofox_statistics_banner.hpp"
#include "anofox_statistics_extension.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "list_input.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

static_assert(offsetof(AnofoxMultinomialOptions, max_iterations) == 3 * sizeof(double) &&
                  offsetof(AnofoxMultinomialOptions, fit_intercept) == 3 * sizeof(double) + sizeof(uint32_t),
              "AnofoxMultinomialOptions must match MultinomialOptionsFFI");
static_assert(offsetof(AnofoxMultinomialFitResult, n_rows) == 2 * sizeof(double *) &&
                  offsetof(AnofoxMultinomialFitResult, log_likelihood) ==
                      ((2 * sizeof(double *) + 4 * sizeof(size_t) + alignof(double) - 1) / alignof(double)) *
                          alignof(double) &&
                  offsetof(AnofoxMultinomialFitResult, iterations) ==
                      offsetof(AnofoxMultinomialFitResult, log_likelihood) + 3 * sizeof(double) &&
                  offsetof(AnofoxMultinomialFitResult, converged) ==
                      offsetof(AnofoxMultinomialFitResult, iterations) + sizeof(uint32_t),
              "AnofoxMultinomialFitResult must match MultinomialFitResultFFI");

//===--------------------------------------------------------------------===//
// multinomial_fit_agg(y VARCHAR | BIGINT, x DOUBLE[] [, weight] [, options])
//===--------------------------------------------------------------------===//
struct MultinomialAggregateState {
	vector<string> y_strings; // VARCHAR labels
	vector<int64_t> y_ints;   // integer labels
	vector<vector<double>> x_columns;
	vector<double> weights; // empty unless weighted
	idx_t n_features;
	bool initialized;

	MultinomialAggregateState() : n_features(0), initialized(false) {
	}

	idx_t Rows() const {
		return x_columns.empty() ? MaxValue<idx_t>(y_strings.size(), y_ints.size()) : x_columns[0].size();
	}

	void Reset() {
		y_strings.clear();
		y_ints.clear();
		x_columns.clear();
		weights.clear();
		n_features = 0;
		initialized = false;
	}
};

struct MultinomialBindData : public FunctionData {
	bool varchar_labels = false;
	bool weighted = false;
	double lambda = 1.0; // sklearn's default C = 1
	double l1_ratio = 0.0;
	bool fit_intercept = true;
	uint32_t max_iterations = 100;
	double tolerance = 1e-8;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MultinomialBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<MultinomialBindData>();
		return varchar_labels == o.varchar_labels && weighted == o.weighted && lambda == o.lambda &&
		       l1_ratio == o.l1_ratio && fit_intercept == o.fit_intercept && max_iterations == o.max_iterations &&
		       tolerance == o.tolerance;
	}
};

static LogicalType MultinomialModelType(const LogicalType &label_type) {
	child_list_t<LogicalType> c;
	c.push_back(MakeChild("classes", LogicalType::LIST(label_type)));
	c.push_back(MakeChild("coefficients", LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE))));
	c.push_back(MakeChild("intercepts", LogicalType::LIST(LogicalType::DOUBLE)));
	c.push_back(MakeChild("n_classes", LogicalType::BIGINT));
	c.push_back(MakeChild("accuracy", LogicalType::DOUBLE));
	c.push_back(MakeChild("log_likelihood", LogicalType::DOUBLE));
	c.push_back(MakeChild("objective", LogicalType::DOUBLE));
	c.push_back(MakeChild("lambda", LogicalType::DOUBLE));
	c.push_back(MakeChild("l1_ratio", LogicalType::DOUBLE));
	c.push_back(MakeChild("n_observations", LogicalType::BIGINT));
	c.push_back(MakeChild("n_features", LogicalType::BIGINT));
	c.push_back(MakeChild("iterations", LogicalType::INTEGER));
	c.push_back(MakeChild("converged", LogicalType::BOOLEAN));
	c.push_back(MakeChild("model_type", LogicalType::VARCHAR));
	c.push_back(MakeChild("family", LogicalType::VARCHAR));
	c.push_back(MakeChild("link", LogicalType::VARCHAR));
	return LogicalType::STRUCT(std::move(c));
}

static void MultinomialAggInitialize(data_ptr_t state_p) {
	new (state_p) MultinomialAggregateState();
}

static void MultinomialAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (MultinomialAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~MultinomialAggregateState();
	}
}

static void MultinomialAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                                 idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<MultinomialBindData>();
	UnifiedVectorFormat y_data, x_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	GlmWeightsInput w_input;
	if (bind.weighted) {
		w_input.Load(inputs[2], count);
	}

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (MultinomialAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		auto x_idx = x_data.sel->get_index(i);
		if (!y_data.validity.RowIsValid(y_idx) || !x_data.validity.RowIsValid(x_idx)) {
			continue; // NULL label or NULL x: the row is left out
		}
		double w = 1.0;
		if (bind.weighted && !w_input.Get(i, w)) {
			continue; // NULL weight: the row is left out, as wls_fit_agg
		}
		auto entry = x_list_data[x_idx];
		// A row with a NULL, NaN or infinite feature is left out, as a NULL row.
		if (ListHasNullElement(x_child_validity, entry)) {
			continue;
		}
		bool finite = true;
		for (idx_t j = 0; j < entry.length; j++) {
			finite = finite && std::isfinite(x_child_data[entry.offset + j]);
		}
		if (!finite) {
			continue;
		}
		if (!state.initialized) {
			state.n_features = entry.length;
			state.x_columns.resize(entry.length);
			state.initialized = true;
		}
		if (entry.length != state.n_features) {
			throw InvalidInputException("multinomial_fit_agg: inconsistent feature count: expected %llu, got %llu",
			                            (unsigned long long)state.n_features, (unsigned long long)entry.length);
		}
		if (bind.varchar_labels) {
			state.y_strings.push_back(UnifiedVectorFormat::GetData<string_t>(y_data)[y_idx].GetString());
		} else {
			state.y_ints.push_back(UnifiedVectorFormat::GetData<int64_t>(y_data)[y_idx]);
		}
		for (idx_t j = 0; j < entry.length; j++) {
			state.x_columns[j].push_back(x_child_data[entry.offset + j]);
		}
		if (bind.weighted) {
			state.weights.push_back(w);
		}
	}
}

static void MultinomialAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                                  idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (MultinomialAggregateState **)source_data.data;
	auto targets = (MultinomialAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		if (!target.initialized) {
			target.y_strings = CombineTake(source.y_strings, aggr_input_data);
			target.y_ints = CombineTake(source.y_ints, aggr_input_data);
			target.x_columns = CombineTake(source.x_columns, aggr_input_data);
			target.weights = CombineTake(source.weights, aggr_input_data);
			target.n_features = source.n_features;
			target.initialized = true;
			continue;
		}
		if (source.n_features != target.n_features) {
			throw InvalidInputException("multinomial_fit_agg: inconsistent feature count: expected %llu, got %llu",
			                            (unsigned long long)target.n_features, (unsigned long long)source.n_features);
		}
		target.y_strings.insert(target.y_strings.end(), source.y_strings.begin(), source.y_strings.end());
		target.y_ints.insert(target.y_ints.end(), source.y_ints.begin(), source.y_ints.end());
		target.weights.insert(target.weights.end(), source.weights.begin(), source.weights.end());
		for (idx_t j = 0; j < target.n_features; j++) {
			target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
			                           source.x_columns[j].end());
		}
	}
}

static void AppendDoubles(Vector &list_vec, idx_t row, const double *data, idx_t len) {
	auto offset = ListVector::GetListSize(list_vec);
	ListVector::Reserve(list_vec, offset + len);
	auto child_data = FlatDataMut<double>(ListVector::GetEntry(list_vec));
	for (idx_t i = 0; i < len; i++) {
		child_data[offset + i] = data[i];
	}
	ListVector::SetListSize(list_vec, offset + len);
	ListVector::GetData(list_vec)[row] = {offset, len};
}

//! The sorted distinct labels and each row's class index (as a double for the FFI).
template <class T>
static vector<T> EncodeClasses(const vector<T> &labels, vector<double> &codes) {
	vector<T> classes(labels);
	std::sort(classes.begin(), classes.end());
	classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
	codes.resize(labels.size());
	for (idx_t i = 0; i < labels.size(); i++) {
		codes[i] = static_cast<double>(std::lower_bound(classes.begin(), classes.end(), labels[i]) - classes.begin());
	}
	return classes;
}

static void MultinomialAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                   idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (MultinomialAggregateState **)sdata.data;
	auto &bind = aggr_input_data.bind_data->Cast<MultinomialBindData>();
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.Rows() < 2) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		vector<double> codes;
		vector<string> string_classes;
		vector<int64_t> int_classes;
		if (bind.varchar_labels) {
			string_classes = EncodeClasses(state.y_strings, codes);
		} else {
			int_classes = EncodeClasses(state.y_ints, codes);
		}
		const idx_t n_classes = bind.varchar_labels ? string_classes.size() : int_classes.size();
		// Canonical row order: bitwise-identical results whatever the thread count (#226).
		// The class codes depend only on the set of labels, not on the row order.
		if (bind.weighted) {
			CanonicalRows().Keys(state.x_columns).Key(codes).Key(state.weights).Sort();
		} else {
			CanonicalRows().Keys(state.x_columns).Key(codes).Sort();
		}
		vector<AnofoxDataArray> x_arrays;
		for (auto &col : state.x_columns) {
			x_arrays.push_back({col.data(), nullptr, col.size()});
		}
		AnofoxMultinomialOptions options {};
		options.lambda = bind.lambda;
		options.l1_ratio = bind.l1_ratio;
		options.tolerance = bind.tolerance;
		options.max_iterations = bind.max_iterations;
		options.fit_intercept = bind.fit_intercept;

		AnofoxMultinomialFitResult fit;
		AnofoxError error;
		if (!anofox_multinomial_fit(codes.data(), codes.size(), x_arrays.data(), x_arrays.size(),
		                            bind.weighted ? state.weights.data() : nullptr, options, &fit, &error)) {
			ThrowUnlessDegenerate("multinomial_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		if (fit.n_classes != n_classes) {
			anofox_free_multinomial_result(&fit);
			throw InvalidInputException("multinomial_fit_agg: internal error: %llu classes fitted, %llu labelled",
			                            (unsigned long long)fit.n_classes, (unsigned long long)n_classes);
		}

		// classes
		auto &classes_vec = StructEntry(entries[0]);
		auto classes_offset = ListVector::GetListSize(classes_vec);
		ListVector::Reserve(classes_vec, classes_offset + n_classes);
		auto &classes_child = ListVector::GetEntry(classes_vec);
		for (idx_t k = 0; k < n_classes; k++) {
			if (bind.varchar_labels) {
				FlatDataMut<string_t>(classes_child)[classes_offset + k] =
				    StringVector::AddString(classes_child, string_classes[k]);
			} else {
				FlatDataMut<int64_t>(classes_child)[classes_offset + k] = int_classes[k];
			}
		}
		ListVector::SetListSize(classes_vec, classes_offset + n_classes);
		ListVector::GetData(classes_vec)[row] = {classes_offset, n_classes};

		// coefficients: one list of n_features values per coefficient row
		auto &coef_vec = StructEntry(entries[1]);
		auto outer_offset = ListVector::GetListSize(coef_vec);
		ListVector::Reserve(coef_vec, outer_offset + fit.n_rows);
		auto &inner = ListVector::GetEntry(coef_vec);
		for (idx_t k = 0; k < fit.n_rows; k++) {
			AppendDoubles(inner, outer_offset + k, fit.coefficients + k * fit.n_features, fit.n_features);
		}
		ListVector::SetListSize(coef_vec, outer_offset + fit.n_rows);
		ListVector::GetData(coef_vec)[row] = {outer_offset, fit.n_rows};

		AppendDoubles(StructEntry(entries[2]), row, fit.intercepts, fit.n_rows);
		FlatDataMut<int64_t>(StructEntry(entries[3]))[row] = static_cast<int64_t>(fit.n_classes);
		FlatDataMut<double>(StructEntry(entries[4]))[row] = fit.accuracy;
		FlatDataMut<double>(StructEntry(entries[5]))[row] = fit.log_likelihood;
		FlatDataMut<double>(StructEntry(entries[6]))[row] = fit.objective;
		FlatDataMut<double>(StructEntry(entries[7]))[row] = bind.lambda;
		FlatDataMut<double>(StructEntry(entries[8]))[row] = bind.l1_ratio;
		FlatDataMut<int64_t>(StructEntry(entries[9]))[row] = static_cast<int64_t>(fit.n_observations);
		FlatDataMut<int64_t>(StructEntry(entries[10]))[row] = static_cast<int64_t>(fit.n_features);
		FlatDataMut<int32_t>(StructEntry(entries[11]))[row] = static_cast<int32_t>(fit.iterations);
		FlatDataMut<bool>(StructEntry(entries[12]))[row] = fit.converged;
		auto set_string = [&](idx_t field, const char *value) {
			auto &vec = StructEntry(entries[field]);
			FlatDataMut<string_t>(vec)[row] = StringVector::AddString(vec, value);
		};
		set_string(13, "multinomial_logistic");
		set_string(14, "multinomial");
		// Two classes: one linear predictor, the log-odds of the second class.
		set_string(15, fit.n_rows == 1 ? "logit" : "softmax");
		anofox_free_multinomial_result(&fit);
		state.Reset();
	}
}

static unique_ptr<FunctionData> MultinomialAggBind(ClientContext &context, AggregateBindFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	auto bind = make_uniq<MultinomialBindData>();
	bind->varchar_labels = ArgumentsOf(function)[0].id() == LogicalTypeId::VARCHAR;
	// f(y, x, w) / f(y, x, w, options): sample weights, as poisson_fit_agg (#305).
	bind->weighted = GlmWeightsArgument(arguments);
	const idx_t options_idx = bind->weighted ? 3 : 2;
	if (arguments.size() > options_idx) {
		auto opts = RegressionMapOptions::ParseFromExpression(
		    context, *arguments[options_idx], "multinomial_fit_agg",
		    {"fit_intercept", "c", "lambda", "l1_ratio", "max_iterations", "tolerance"});
		if (opts.pa_c.has_value() && opts.lambda.has_value()) {
			throw InvalidInputException("multinomial_fit_agg: give either 'C' or 'lambda' (= 1 / C), not both");
		}
		if (opts.pa_c.has_value()) {
			double c = opts.pa_c.value();
			if (!(c > 0.0)) {
				throw InvalidInputException("multinomial_fit_agg: option 'C' must be > 0, got %g", c);
			}
			bind->lambda = 1.0 / c; // C = inf: no penalty
		}
		if (opts.lambda.has_value()) {
			bind->lambda = opts.lambda.value();
		}
		if (!(std::isfinite(bind->lambda) && bind->lambda >= 0.0)) {
			throw InvalidInputException("multinomial_fit_agg: option 'lambda' must be finite and >= 0, got %g",
			                            bind->lambda);
		}
		if (opts.l1_ratio.has_value()) {
			bind->l1_ratio = opts.l1_ratio.value();
			if (!(bind->l1_ratio >= 0.0 && bind->l1_ratio <= 1.0)) {
				throw InvalidInputException("multinomial_fit_agg: option 'l1_ratio' must be in [0, 1], got %g",
				                            bind->l1_ratio);
			}
		}
		if (opts.fit_intercept.has_value()) {
			bind->fit_intercept = opts.fit_intercept.value();
		}
		if (opts.max_iterations.has_value()) {
			if (opts.max_iterations.value() == 0) {
				throw InvalidInputException("multinomial_fit_agg: option 'max_iterations' must be at least 1");
			}
			bind->max_iterations = opts.max_iterations.value();
		}
		if (opts.tolerance.has_value()) {
			bind->tolerance = opts.tolerance.value();
			if (!(std::isfinite(bind->tolerance) && bind->tolerance > 0.0)) {
				throw InvalidInputException("multinomial_fit_agg: option 'tolerance' must be finite and > 0, got %g",
				                            bind->tolerance);
			}
		}
	}
	SetReturnTypeOf(function, MultinomialModelType(bind->varchar_labels ? LogicalType::VARCHAR : LogicalType::BIGINT));
	PostHogTelemetry::Instance().RecordFunctionCall("multinomial_fit_agg");
	return std::move(bind);
}

//===--------------------------------------------------------------------===//
// multinomial_predict(model, x DOUBLE[] [, options]) and predict(model, x [, options])
//===--------------------------------------------------------------------===//
struct MultinomialPredictBindData : public FunctionData {
	bool probabilities = false;
	bool varchar_labels = false;
	idx_t classes_idx = DConstants::INVALID_INDEX;
	idx_t coef_idx = DConstants::INVALID_INDEX;
	idx_t intercepts_idx = DConstants::INVALID_INDEX;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<MultinomialPredictBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<MultinomialPredictBindData>();
		return probabilities == o.probabilities && varchar_labels == o.varchar_labels && classes_idx == o.classes_idx &&
		       coef_idx == o.coef_idx && intercepts_idx == o.intercepts_idx;
	}
};

bool IsMultinomialModelType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	for (auto &child : StructType::GetChildTypes(type)) {
		if (NameString(child.first) == "classes" && child.second.id() == LogicalTypeId::LIST) {
			return true;
		}
	}
	return false;
}

static void MultinomialPredictFunction(DataChunk &args, ExpressionState &state, Vector &result);

unique_ptr<FunctionData> MultinomialPredictBind(ClientContext &context, ScalarBindFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments, const char *fn_name) {
	auto &in_type = ReturnTypeOf(*arguments[0]);
	if (in_type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("%s(model, x): model must be the STRUCT returned by multinomial_fit_agg, got %s",
		                            fn_name, in_type.ToString());
	}
	auto data = make_uniq<MultinomialPredictBindData>();
	// Cast the model's fields to the multinomial_fit_agg types (a hand-written literal may hold
	// DECIMALs or INTEGER labels).
	child_list_t<LogicalType> cast_children;
	auto &children = StructType::GetChildTypes(in_type);
	for (idx_t i = 0; i < children.size(); i++) {
		auto &name = NameString(children[i].first);
		auto type = children[i].second;
		if (name == "classes" && type.id() == LogicalTypeId::LIST) {
			data->classes_idx = i;
			auto label = ListType::GetChildType(type);
			if (label.id() == LogicalTypeId::VARCHAR) {
				data->varchar_labels = true;
			} else if (label.IsIntegral() || label.id() == LogicalTypeId::BOOLEAN) {
				type = LogicalType::LIST(LogicalType::BIGINT);
			} else {
				throw InvalidInputException("%s: the model's classes must be VARCHAR or integer labels, got %s",
				                            fn_name, type.ToString());
			}
		} else if (name == "coefficients") {
			data->coef_idx = i;
			type = LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE));
		} else if (name == "intercepts") {
			data->intercepts_idx = i;
			type = LogicalType::LIST(LogicalType::DOUBLE);
		}
		cast_children.push_back(MakeChild(NameString(children[i].first), type));
	}
	if (data->classes_idx == DConstants::INVALID_INDEX || data->coef_idx == DConstants::INVALID_INDEX ||
	    data->intercepts_idx == DConstants::INVALID_INDEX) {
		throw InvalidInputException("%s(model, x): model must have the fields classes, coefficients and intercepts "
		                            "(the STRUCT returned by multinomial_fit_agg), got %s",
		                            fn_name, in_type.ToString());
	}
	auto target = LogicalType::STRUCT(std::move(cast_children));
	if (target != in_type) {
		arguments[0] = BoundCastExpression::AddCastToType(context, std::move(arguments[0]), target);
	}
	if (arguments.size() >= 3) {
		Value options = UnwrapVariant(EvaluateConstantOptions(context, *arguments[2], fn_name));
		auto visit = [&](const string &key, const Value &val) {
			if (key != "type") {
				throw InvalidInputException("%s: unknown option '%s' for a multinomial model (supported: type)",
				                            fn_name, key);
			}
			auto v = val.IsNull() ? string("class") : StringUtil::Lower(val.ToString());
			if (v == "class") {
				data->probabilities = false;
			} else if (v == "prob") {
				data->probabilities = true;
			} else {
				throw InvalidInputException(
				    "%s: option 'type' must be 'class' or 'prob' for a multinomial model, got '%s'", fn_name,
				    val.ToString());
			}
		};
		if (options.IsNull()) {
			// no options
		} else if (options.type().id() == LogicalTypeId::STRUCT) {
			auto &vals = StructValue::GetChildren(options);
			for (idx_t i = 0; i < vals.size(); i++) {
				visit(StringUtil::Lower(NameString(StructType::GetChildName(options.type(), i))),
				      UnwrapVariant(vals[i]));
			}
		} else if (options.type().id() == LogicalTypeId::MAP) {
			for (auto &entry : MapValue::GetChildren(options)) {
				auto &kv = StructValue::GetChildren(entry);
				visit(StringUtil::Lower(kv[0].ToString()), UnwrapVariant(kv[1]));
			}
		} else {
			throw InvalidInputException("%s: options must be a MAP or STRUCT", fn_name);
		}
	}
	SetReturnTypeOf(bound_function, data->probabilities
	                                    ? LogicalType::LIST(LogicalType::DOUBLE)
	                                    : (data->varchar_labels ? LogicalType::VARCHAR : LogicalType::BIGINT));
	SetFunctionCallbackOf(bound_function, DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, MultinomialPredictFunction));
	PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
	return std::move(data);
}

static unique_ptr<FunctionData> MultinomialPredictOwnBind(ClientContext &context, ScalarBindFunction &bound_function,
                                                          vector<unique_ptr<Expression>> &arguments) {
	return MultinomialPredictBind(context, bound_function, arguments, "multinomial_predict");
}

static void MultinomialPredictFunctionImpl(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind = BindInfoOf(func_expr)->Cast<MultinomialPredictBindData>();
	idx_t count = args.size();
	args.Flatten();
	auto &model_vec = args.data[0];
	auto &x_vec = args.data[1];
	auto &entries = StructVector::GetEntries(model_vec);
	auto &classes_vec = StructEntry(entries[bind.classes_idx]);
	auto &coef_vec = StructEntry(entries[bind.coef_idx]);
	auto &icpt_vec = StructEntry(entries[bind.intercepts_idx]);
	auto &classes_child = ListVector::GetEntry(classes_vec);
	auto &coef_inner = ListVector::GetEntry(coef_vec);
	auto &coef_child = ListVector::GetEntry(coef_inner);
	auto &icpt_child = ListVector::GetEntry(icpt_vec);
	auto x_lists = FlatDataMut<list_entry_t>(x_vec);
	auto &x_child = ListVector::GetEntry(x_vec);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);

	for (idx_t row = 0; row < count; row++) {
		if (FlatVector::IsNull(model_vec, row) || FlatVector::IsNull(x_vec, row) ||
		    FlatVector::IsNull(classes_vec, row) || FlatVector::IsNull(coef_vec, row) ||
		    FlatVector::IsNull(icpt_vec, row)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto x_entry = x_lists[row];
		if (ListHasNullElement(x_child_validity, x_entry)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto cl = FlatDataMut<list_entry_t>(classes_vec)[row];
		auto co = FlatDataMut<list_entry_t>(coef_vec)[row];
		auto ic = FlatDataMut<list_entry_t>(icpt_vec)[row];
		if (ListHasNullElement(FlatValidityMut(classes_child), cl) ||
		    ListHasNullElement(FlatValidityMut(coef_inner), co) ||
		    ListHasNullElement(FlatValidityMut(icpt_child), ic)) {
			throw InvalidInputException("multinomial_predict: the model's classes, coefficients and intercepts must "
			                            "not hold NULLs");
		}
		const idx_t n_features = x_entry.length;
		vector<double> coefficients;
		coefficients.reserve(co.length * n_features);
		auto inner_lists = FlatDataMut<list_entry_t>(coef_inner);
		auto coef_data = FlatDataMut<double>(coef_child);
		for (idx_t k = 0; k < co.length; k++) {
			auto e = inner_lists[co.offset + k];
			if (ListHasNullElement(FlatValidityMut(coef_child), e)) {
				throw InvalidInputException("multinomial_predict: the model's coefficients must not hold NULLs");
			}
			if (e.length != n_features) {
				throw InvalidInputException("multinomial_predict: the model has %llu coefficients per class but x "
				                            "has %llu values",
				                            (unsigned long long)e.length, (unsigned long long)n_features);
			}
			coefficients.insert(coefficients.end(), coef_data + e.offset, coef_data + e.offset + e.length);
		}
		vector<double> proba(cl.length);
		size_t best = 0;
		AnofoxError error;
		if (!anofox_multinomial_predict_proba(coefficients.data(), FlatDataMut<double>(icpt_child) + ic.offset,
		                                      co.length, n_features, cl.length, x_child_data + x_entry.offset,
		                                      x_entry.length, proba.data(), &best, &error)) {
			ThrowUnlessDegenerate("multinomial_predict", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (bind.probabilities) {
			AppendDoubles(result, row, proba.data(), proba.size());
		} else if (bind.varchar_labels) {
			FlatDataMut<string_t>(result)[row] =
			    StringVector::AddString(result, FlatDataMut<string_t>(classes_child)[cl.offset + best]);
		} else {
			FlatDataMut<int64_t>(result)[row] = FlatDataMut<int64_t>(classes_child)[cl.offset + best];
		}
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
}

// Constant inputs must yield a CONSTANT_VECTOR (DuckDB constant folding asserts this in debug builds).
static void MultinomialPredictFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const bool all_constant = args.AllConstant();
	MultinomialPredictFunctionImpl(args, state, result);
	if (all_constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

//===--------------------------------------------------------------------===//
// multinomial_fit_predict_by(source, group_col, y_col, x_cols, options, split)
//===--------------------------------------------------------------------===//
// The training-row test is SPLIT_IS_TRAIN of fit_predict_macros.cpp: 'train' / 'training'
// in any case; with split omitted (the NULL literal default) every row trains.
static const char *const MULTINOMIAL_FIT_PREDICT_BY_SQL = R"(
WITH _anofox_src AS (
    SELECT *,
        coalesce(lower(split::VARCHAR) IN ('train', 'training'), replace(typeof(split), '"', '') = 'NULL') AS _anofox_train
    FROM query_table(source::VARCHAR)
),
_anofox_fits AS (
    SELECT group_col AS _anofox_group, multinomial_fit_agg(y_col, x_cols, options) AS _anofox_model
    FROM _anofox_src
    WHERE y_col IS NOT NULL AND _anofox_train
    GROUP BY group_col
)
SELECT
    * EXCLUDE (_anofox_group, _anofox_model, _anofox_train),
    multinomial_predict(_anofox_model, x_cols) AS yhat,
    multinomial_predict(_anofox_model, x_cols, {'type': 'prob'}) AS probabilities,
    (y_col IS NOT NULL AND _anofox_train) AS is_training
FROM _anofox_src
LEFT JOIN _anofox_fits ON group_col IS NOT DISTINCT FROM _anofox_group
ORDER BY group_col
)";

void RegisterMultinomialFunctions(ExtensionLoader &loader) {
	{
		AggregateFunctionSet func_set("multinomial_fit_agg");
		for (auto &label_type : {LogicalType::VARCHAR, LogicalType::BIGINT}) {
			vector<vector<LogicalType>> signatures = {
			    {label_type, LogicalType::LIST(LogicalType::DOUBLE)},
			    {label_type, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
			    {label_type, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY}};
			for (auto &args : signatures) {
				func_set.AddFunction(AggregateFunction(
				    "multinomial_fit_agg", args, LogicalType::ANY,
				    AggregateFunction::StateSize<MultinomialAggregateState>, ANOFOX_AGG_INIT(MultinomialAggInitialize),
				    ANOFOX_GUARDED_UPDATE(MultinomialAggUpdate, MultinomialAggDestroy, MultinomialAggInitialize),
				    MultinomialAggCombine,
				    ANOFOX_GUARDED_FINALIZE(MultinomialAggFinalize, MultinomialAggDestroy, MultinomialAggInitialize),
				    nullptr, ANOFOX_AGG_BIND(MultinomialAggBind), MultinomialAggDestroy));
			}
		}
		CreateAggregateFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		const char *fit_description =
		    "Fits a multinomial (softmax) logistic regression with an elastic-net penalty, the objective of "
		    "scikit-learn's LogisticRegression: y is a VARCHAR or integer class label (the classes are its sorted "
		    "distinct values, at least two). Returns STRUCT(classes, coefficients (one list of n_features values per "
		    "class; a single row, the log-odds of the second class, for two classes), intercepts, n_classes, "
		    "accuracy, log_likelihood, objective, lambda, l1_ratio, n_observations, n_features, iterations, "
		    "converged, model_type, family, link). Options: C (default 1) or lambda (= 1 / C, 0: no penalty), "
		    "l1_ratio (0 ridge, 1 lasso), fit_intercept, max_iterations (100), tolerance (1e-8). Rows with a NULL "
		    "label, a NULL x or a NULL / non-finite feature are left out.";
		const char *weighted_description =
		    "Fits the same multinomial logistic regression with non-negative sample weights (scikit-learn's "
		    "sample_weight; a numeric third argument is the weight); a NULL weight leaves the row out.";
		// One description per overload (VARCHAR and BIGINT labels).
		for (auto &label_type : {LogicalType::VARCHAR, LogicalType::BIGINT}) {
			const bool varchar = label_type == LogicalTypeId::VARCHAR;
			const string y = varchar ? "species" : "class_id";
			struct {
				vector<string> names;
				vector<LogicalType> types;
				string example;
				const char *text;
			} docs[] = {{{"y", "x"},
			             {label_type, LogicalType::LIST(LogicalType::DOUBLE)},
			             "multinomial_fit_agg(" + y + ", [sepal_length, petal_length])",
			             fit_description},
			            {{"y", "x", "options"},
			             {label_type, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
			             "multinomial_fit_agg(" + y + ", [sepal_length, petal_length], {'C': 1.0, 'l1_ratio': 0.5})",
			             fit_description},
			            {{"y", "x", "weights", "options"},
			             {label_type, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::DOUBLE, LogicalType::ANY},
			             "multinomial_fit_agg(" + y + ", [sepal_length, petal_length], w, {'C': 10})",
			             weighted_description}};
			for (auto &doc : docs) {
				FunctionDescription d;
				d.description = doc.text;
				d.examples = {doc.example};
				d.categories = {"regression", "classification"};
				d.parameter_names = doc.names;
				d.parameter_types = doc.types;
				info.descriptions.push_back(std::move(d));
			}
		}
		loader.RegisterFunction(std::move(info));
	}
	{
		ScalarFunctionSet func_set("multinomial_predict");
		for (bool with_options : {false, true}) {
			vector<LogicalType> args = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE)};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			ScalarFunction func(args, LogicalType::ANY,
			                    DATAZOO_GUARD(ANOFOX_STATISTICS_BANNER, MultinomialPredictFunction),
			                    ANOFOX_SCALAR_BIND(MultinomialPredictOwnBind));
			MarkFallible(func);
			SetNullHandlingOf(func, FunctionNullHandling::SPECIAL_HANDLING);
			func_set.AddFunction(func);
		}
		CreateScalarFunctionInfo info(std::move(func_set));
		info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
		FunctionDescription desc;
		desc.description = "Predicts one row from a multinomial_fit_agg model: the most probable class (the type of "
		                   "the model's classes), or with {'type': 'prob'} the class probabilities (DOUBLE[], in the "
		                   "order of the model's classes; scikit-learn's predict / predict_proba). NULL when the model "
		                   "or x is NULL or x holds a NULL. predict(model, x [, options]) does the same.";
		desc.examples = {"multinomial_predict(model, [5.1, 1.4])",
		                 "multinomial_predict(model, [5.1, 1.4], {'type': 'prob'})"};
		desc.categories = {"regression", "classification", "prediction"};
		desc.parameter_names = {"model", "x", "options"};
		desc.parameter_types = {LogicalType::ANY, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
		info.descriptions.push_back(std::move(desc));
		loader.RegisterFunction(std::move(info));
	}
	{
		auto macro = CreateContractTableMacro(
		    "multinomial_fit_predict_by", {"source", "group_col", "y_col", "x_cols"},
		    {{"options", "NULL"}, {"split", "NULL"}}, MULTINOMIAL_FIT_PREDICT_BY_SQL,
		    "Table macro: fits a multinomial logistic regression (multinomial_fit_agg, options as there) per group of "
		    "source (training rows: y_col NOT NULL and split omitted or 'train' / 'training' in any case) and returns "
		    "every source row with yhat (the most probable class), probabilities (DOUBLE[] in the order of the "
		    "model's classes) and is_training appended.",
		    "multinomial_fit_predict_by('flowers', site, species, [sepal_length, petal_length], options := {'C': 1.0})",
		    {"classification", "table-macro"});
		loader.RegisterFunction(*macro);
	}
}

} // namespace duckdb
