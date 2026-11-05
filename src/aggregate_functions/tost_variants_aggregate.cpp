// TOST equivalence variants (#293): tost_one_sample_agg, tost_prop_one_agg,
// tost_prop_two_agg, tost_wilcoxon_paired_agg, tost_wilcoxon_agg,
// tost_yuen_agg and tost_bootstrap_agg. The statistics are upstream
// (anofox-statistics `equivalence`); this file buffers the rows, resolves the
// options and writes the tost_t_test_agg result STRUCT.

#include <algorithm>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/test_struct.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/result_fields.hpp"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "two_group.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {

enum TostVariant : int {
	TOST_ONE_SAMPLE = 0,
	TOST_WILCOXON_PAIRED = 1,
	TOST_WILCOXON = 2,
	TOST_YUEN = 3,
	TOST_BOOTSTRAP = 4,
	TOST_PROP_ONE = 5,
	TOST_PROP_TWO = 6,
};

struct TostVariantSpec {
	const char *name;
	//! Option keys beyond the bounds / alpha / confidence_level.
	vector<string> extra_keys;
	//! Smallest sample (per group / pairs) that the core accepts; smaller is NULL.
	idx_t min_n;
	const char *description;
	const char *example;
};

const TostVariantSpec &Spec(int kind) {
	static const TostVariantSpec specs[] = {
	    {"tost_one_sample_agg", {"mu"}, 2,
	     "Tests whether a mean is equivalent to mu using the Two One-Sided Tests (TOST) procedure with a one-sample "
	     "t-test.",
	     "tost_one_sample_agg(value, {'mu': 0.0, 'delta': 0.5})"},
	    {"tost_wilcoxon_paired_agg", {}, 3,
	     "Tests whether the pseudo-median of paired differences is equivalent to zero using TOST with shifted "
	     "Wilcoxon signed-rank tests.",
	     "tost_wilcoxon_paired_agg(x, y, {'delta': 0.5})"},
	    {"tost_wilcoxon_agg", {}, 3,
	     "Tests whether the location shift of two groups is equivalent to zero using TOST with shifted Wilcoxon "
	     "rank-sum tests.",
	     "tost_wilcoxon_agg(value, group_id, {'delta': 0.5})"},
	    {"tost_yuen_agg", {"trim"}, 4,
	     "Tests whether two trimmed means are equivalent using TOST with Yuen's test.",
	     "tost_yuen_agg(value, group_id, {'delta': 0.5, 'trim': 0.2})"},
	    {"tost_bootstrap_agg", {"n_bootstrap", "seed"}, 3,
	     "Tests whether two means are equivalent using a percentile-bootstrap TOST.",
	     "tost_bootstrap_agg(value, group_id, {'delta': 0.5, 'n_bootstrap': 2000, 'seed': 42})"},
	    {"tost_prop_one_agg", {"p0"}, 1,
	     "Tests whether a proportion is equivalent to p0 using TOST with z-tests (normal approximation).",
	     "tost_prop_one_agg(success, {'p0': 0.5, 'delta': 0.1})"},
	    {"tost_prop_two_agg", {}, 1,
	     "Tests whether two proportions are equivalent using TOST with z-tests (normal approximation).",
	     "tost_prop_two_agg(success, group_id, {'delta': 0.1})"},
	};
	return specs[kind];
}

//===--------------------------------------------------------------------===//
// State: one layout for every variant; each uses only its own members.
//===--------------------------------------------------------------------===//
struct TostVariantState {
	TwoGroupSamples samples; // two-group variants (prop_two: 1.0 = success)
	vector<double> x;        // one-sample values / paired x
	vector<double> y;        // paired y
	uint64_t successes;      // prop_one
	uint64_t trials;
	bool initialized;

	TostVariantState() : successes(0), trials(0), initialized(false) {
	}

	void Reset() {
		samples.Clear();
		x.clear();
		y.clear();
		successes = 0;
		trials = 0;
		initialized = false;
	}
};

//===--------------------------------------------------------------------===//
// Result type: the STRUCT of tost_t_test_agg. t_lower / t_upper are the
// statistics of the two one-sided tests (t, z, Wilcoxon V / U, or the
// bootstrap point estimate, as the method dictates).
//===--------------------------------------------------------------------===//
LogicalType GetTostVariantResultType() {
	child_list_t<LogicalType> children;
	children.push_back(make_pair("t_lower", LogicalType::DOUBLE));
	children.push_back(make_pair("t_upper", LogicalType::DOUBLE));
	children.push_back(make_pair("p_lower", LogicalType::DOUBLE));
	children.push_back(make_pair("p_upper", LogicalType::DOUBLE));
	children.push_back(make_pair("p_value", LogicalType::DOUBLE));
	children.push_back(make_pair("df", LogicalType::DOUBLE));
	children.push_back(make_pair("estimate", LogicalType::DOUBLE));
	children.push_back(make_pair("ci_lower", LogicalType::DOUBLE));
	children.push_back(make_pair("ci_upper", LogicalType::DOUBLE));
	children.push_back(make_pair("bound_lower", LogicalType::DOUBLE));
	children.push_back(make_pair("bound_upper", LogicalType::DOUBLE));
	children.push_back(make_pair("equivalent", LogicalType::BOOLEAN));
	children.push_back(make_pair("n", LogicalType::BIGINT));
	children.push_back(make_pair("method", LogicalType::VARCHAR));
	children.push_back(make_pair("statistic", LogicalType::DOUBLE));
	children.push_back(make_pair("alternative", LogicalType::VARCHAR));
	AppendTestResultFields(children);
	return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Bind data
//===--------------------------------------------------------------------===//
struct TostVariantBindData : public FunctionData {
	double bound_lower = -0.5;
	double bound_upper = 0.5;
	double alpha = 0.05;
	double mu = 0.0; // one-sample mu, prop_one p0
	double trim = 0.2;
	uint32_t n_bootstrap = 10000;
	uint64_t seed = ANOFOX_DEFAULT_MC_SEED;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<TostVariantBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<TostVariantBindData>();
		return bound_lower == o.bound_lower && bound_upper == o.bound_upper && alpha == o.alpha && mu == o.mu &&
		       trim == o.trim && n_bootstrap == o.n_bootstrap && seed == o.seed;
	}
};

template <int KIND>
unique_ptr<FunctionData> TostVariantBind(ClientContext &context, AggregateBindFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	const auto &spec = Spec(KIND);
	SetReturnTypeOf(function, GetTostVariantResultType());
	auto bind_data = make_uniq<TostVariantBindData>();
	if (KIND == TOST_PROP_ONE) {
		bind_data->mu = 0.5;
	}

	// The options are the last argument: (value [, group | y], options).
	const idx_t n_data_args = (KIND == TOST_ONE_SAMPLE || KIND == TOST_PROP_ONE) ? 1 : 2;
	// Equivalence bounds are a substantive choice: no default (#293).
	const auto require_bounds = [&]() {
		throw InvalidInputException(
		    "%s: equivalence bounds are required: set 'delta' or 'bound_lower'/'bound_upper'", spec.name);
	};
	if (arguments.size() <= n_data_args) {
		require_bounds();
	}
	{
		Value options_val = EvaluateConstantOptions(context, *arguments[n_data_args], spec.name);
		auto opts = TostVariantMapOptions::ParseFromValue(options_val, spec.name, spec.extra_keys);
		if (!opts.delta.has_value() && !(opts.bound_lower.has_value() && opts.bound_upper.has_value())) {
			require_bounds();
		}
		if (opts.bound_lower.has_value()) {
			bind_data->bound_lower = opts.bound_lower.value();
		}
		if (opts.bound_upper.has_value()) {
			bind_data->bound_upper = opts.bound_upper.value();
		}
		// A symmetric delta takes precedence over explicit bounds, as in tost_t_test_agg.
		if (opts.delta.has_value()) {
			bind_data->bound_lower = -opts.delta.value();
			bind_data->bound_upper = opts.delta.value();
		}
		if (!(bind_data->bound_lower < bind_data->bound_upper)) {
			throw InvalidInputException("%s: bound_lower (%g) must be less than bound_upper (%g)", spec.name,
			                            bind_data->bound_lower, bind_data->bound_upper);
		}
		bind_data->alpha = ResolveTostAlpha(spec.name, opts.alpha, opts.confidence_level);
		if (opts.mu.has_value()) {
			bind_data->mu = opts.mu.value();
		}
		if (opts.p0.has_value()) {
			bind_data->mu = opts.p0.value();
		}
		if (opts.trim.has_value()) {
			bind_data->trim = opts.trim.value();
		}
		if (opts.n_bootstrap.has_value()) {
			bind_data->n_bootstrap = opts.n_bootstrap.value();
		}
		if (opts.seed.has_value()) {
			bind_data->seed = opts.seed.value();
		}
	}

	PostHogTelemetry::Instance().RecordFunctionCall(spec.name);
	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// Aggregate operations
//===--------------------------------------------------------------------===//
void TostVariantInitialize(data_ptr_t state_p) {
	new (state_p) TostVariantState();
}

void TostVariantDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (TostVariantState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~TostVariantState();
	}
}

template <int KIND>
void TostVariantUpdate(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector,
                       idx_t count) {
	const char *name = Spec(KIND).name;
	UnifiedVectorFormat a_data, b_data;
	inputs[0].ToUnifiedFormat(count, a_data);
	const bool has_second = KIND != TOST_ONE_SAMPLE && KIND != TOST_PROP_ONE;
	if (has_second) {
		inputs[1].ToUnifiedFormat(count, b_data);
	}
	const bool group_is_string = has_second && inputs[1].GetType().id() == LogicalTypeId::VARCHAR;

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (TostVariantState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;

		auto a_idx = a_data.sel->get_index(i);
		if (!a_data.validity.RowIsValid(a_idx)) {
			continue;
		}
		idx_t b_idx = 0;
		if (has_second) {
			b_idx = b_data.sel->get_index(i);
			if (!b_data.validity.RowIsValid(b_idx)) {
				continue;
			}
		}

		if (KIND == TOST_PROP_ONE || KIND == TOST_PROP_TWO) {
			// Non-zero is a success, as in prop_test_one_agg / prop_test_two_agg.
			const bool success = UnifiedVectorFormat::GetData<int64_t>(a_data)[a_idx] != 0;
			if (KIND == TOST_PROP_ONE) {
				state.successes += success ? 1 : 0;
				state.trials++;
			} else {
				state.samples.Add(ReadGroupLabel(b_data, b_idx, group_is_string), success ? 1.0 : 0.0, name);
			}
			continue;
		}

		const double a = UnifiedVectorFormat::GetData<double>(a_data)[a_idx];
		if (std::isnan(a)) {
			continue;
		}
		if (KIND == TOST_ONE_SAMPLE) {
			state.x.push_back(a);
		} else if (KIND == TOST_WILCOXON_PAIRED) {
			const double b = UnifiedVectorFormat::GetData<double>(b_data)[b_idx];
			if (std::isnan(b)) {
				continue;
			}
			state.x.push_back(a);
			state.y.push_back(b);
		} else {
			state.samples.Add(ReadGroupLabel(b_data, b_idx, group_is_string), a, name);
		}
	}
}

template <int KIND>
void TostVariantCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                        idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (TostVariantState **)source_data.data;
	auto targets = (TostVariantState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		target.samples.Merge(source.samples, aggr_input_data, Spec(KIND).name);
		if (target.x.empty()) {
			target.x = CombineTake(source.x, aggr_input_data);
			target.y = CombineTake(source.y, aggr_input_data);
		} else {
			target.x.insert(target.x.end(), source.x.begin(), source.x.end());
			target.y.insert(target.y.end(), source.y.begin(), source.y.end());
		}
		target.successes += source.successes;
		target.trials += source.trials;
		target.initialized = true;
	}
}

template <class V>
AnofoxDataArray AsArray(const V &v) {
	AnofoxDataArray a;
	a.data = v.data();
	a.validity = nullptr;
	a.len = v.size();
	return a;
}

template <class V>
uint64_t CountSuccesses(const V &v) {
	return static_cast<uint64_t>(std::count(v.begin(), v.end(), 1.0));
}

template <int KIND>
void TostVariantFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	const auto &spec = Spec(KIND);
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (TostVariantState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	auto &bind_data = aggr_input_data.bind_data->Cast<TostVariantBindData>();

	AnofoxTostVariantOptions options;
	options.bound_lower = bind_data.bound_lower;
	options.bound_upper = bind_data.bound_upper;
	options.alpha = bind_data.alpha;
	options.mu = bind_data.mu;
	options.trim = bind_data.trim;
	options.n_bootstrap = bind_data.n_bootstrap;
	options.seed = bind_data.seed;
	options.has_seed = true;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t result_idx = i + offset;

		AnofoxTostResult tost_result;
		AnofoxError error;
		bool too_small = !state.initialized;
		bool success = false;

		if (!too_small) {
			// Canonical row order: bitwise-identical results (including the seeded
			// bootstrap draw) whatever the thread count or row order (#214, #226).
			if (KIND == TOST_ONE_SAMPLE) {
				too_small = state.x.size() < spec.min_n;
				if (!too_small) {
					std::sort(state.x.begin(), state.x.end());
					vector<double> none;
					success = anofox_tost_variant(KIND, AsArray(state.x), AsArray(none), options, &tost_result,
					                              &error);
				}
			} else if (KIND == TOST_WILCOXON_PAIRED) {
				too_small = state.x.size() < spec.min_n;
				if (!too_small) {
					CanonicalRows().Key(state.x).Key(state.y).Sort();
					success = anofox_tost_variant(KIND, AsArray(state.x), AsArray(state.y), options, &tost_result,
					                              &error);
				}
			} else if (KIND == TOST_PROP_ONE) {
				too_small = state.trials < spec.min_n;
				if (!too_small) {
					success = anofox_tost_prop(state.successes, state.trials, 0, 0, false, options, &tost_result,
					                           &error);
				}
			} else {
				auto &g1 = state.samples.Group1();
				auto &g2 = state.samples.Group2();
				too_small = g1.size() < spec.min_n || g2.size() < spec.min_n;
				if (!too_small && KIND == TOST_PROP_TWO) {
					success = anofox_tost_prop(CountSuccesses(g1), g1.size(), CountSuccesses(g2), g2.size(), true,
					                           options, &tost_result, &error);
				} else if (!too_small) {
					state.samples.SortValues();
					success = anofox_tost_variant(KIND, AsArray(state.samples.Group1()),
					                              AsArray(state.samples.Group2()), options, &tost_result, &error);
				}
			}
		}

		if (too_small) {
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}
		if (!success) {
			ThrowUnlessDegenerate(spec.name, error);
			FlatVector::SetNull(result, result_idx, true);
			continue;
		}

		idx_t struct_idx = 0;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.t_lower;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.t_upper;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.p_lower;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.p_upper;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.p_value;
		auto &df_vector = StructEntry(struct_entries[struct_idx++]);
		if (std::isnan(tost_result.df)) {
			SetResultNull(df_vector, result_idx); // z, rank and bootstrap tests have no df
		} else {
			FlatDataMut<double>(df_vector)[result_idx] = tost_result.df;
		}
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.estimate;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.ci_lower;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.ci_upper;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.bound_lower;
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.bound_upper;
		FlatDataMut<bool>(StructEntry(struct_entries[struct_idx++]))[result_idx] = tost_result.equivalent;
		FlatDataMut<int64_t>(StructEntry(struct_entries[struct_idx++]))[result_idx] =
		    static_cast<int64_t>(tost_result.n);
		auto &method_vector = StructEntry(struct_entries[struct_idx++]);
		FlatDataMut<string_t>(method_vector)[result_idx] =
		    StringVector::AddString(method_vector, tost_result.method ? tost_result.method : spec.name);
		// statistic of the one-sided test that determines the TOST p-value
		FlatDataMut<double>(StructEntry(struct_entries[struct_idx++]))[result_idx] =
		    tost_result.p_lower >= tost_result.p_upper ? tost_result.t_lower : tost_result.t_upper;
		SetResultNull(StructEntry(struct_entries[struct_idx++]), result_idx); // alternative: not applicable
		WriteTestResultFields(result, result_idx, struct_idx, tost_result.unified);

		anofox_free_tost_result(&tost_result);
		state.Reset();
	}
}

template <int KIND>
AggregateFunction MakeTostVariant(vector<LogicalType> args) {
	return AggregateFunction(
	    Spec(KIND).name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<TostVariantState>,
	    ANOFOX_AGG_INIT(TostVariantInitialize),
	    ANOFOX_GUARDED_UPDATE(TostVariantUpdate<KIND>, TostVariantDestroy, TostVariantInitialize),
	    TostVariantCombine<KIND>,
	    ANOFOX_GUARDED_FINALIZE(TostVariantFinalize<KIND>, TostVariantDestroy, TostVariantInitialize), nullptr,
	    ANOFOX_AGG_BIND(TostVariantBind<KIND>), TostVariantDestroy);
}

//! Register one variant. `data_args` are the data arguments; a BIGINT group
//! argument (`group_arg`) is also registered as VARCHAR.
template <int KIND>
void RegisterTostVariant(ExtensionLoader &loader, const vector<LogicalType> &data_args,
                         const vector<string> &data_names, int group_arg) {
	const auto &spec = Spec(KIND);
	AggregateFunctionSet func_set(spec.name);
	for (bool with_options : {true, false}) {
		auto args = data_args;
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		// The overload without options exists only to raise the "bounds are
		// required" error instead of a generic binder error.
		func_set.AddFunction(MakeTostVariant<KIND>(args));
		if (group_arg >= 0) {
			args[group_arg] = LogicalType::VARCHAR;
			func_set.AddFunction(MakeTostVariant<KIND>(args));
		}
	}

	CreateAggregateFunctionInfo info(std::move(func_set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	auto doc_types = data_args;
	if (group_arg >= 0) {
		doc_types[group_arg] = LogicalType::ANY;
	}
	FunctionDescription d1;
	d1.description = spec.description;
	d1.examples = {spec.example};
	d1.categories = {"hypothesis-testing", "equivalence"};
	d1.parameter_names = data_names;
	d1.parameter_names.push_back("options");
	d1.parameter_types = doc_types;
	d1.parameter_types.push_back(LogicalType::ANY);
	info.descriptions.push_back(std::move(d1));
	loader.RegisterFunction(std::move(info));
}

} // namespace

void RegisterTostVariantAggregateFunctions(ExtensionLoader &loader) {
	const auto D = LogicalType::DOUBLE;
	const auto I = LogicalType::BIGINT;
	RegisterTostVariant<TOST_ONE_SAMPLE>(loader, {D}, {"value"}, -1);
	RegisterTostVariant<TOST_WILCOXON_PAIRED>(loader, {D, D}, {"x", "y"}, -1);
	RegisterTostVariant<TOST_WILCOXON>(loader, {D, I}, {"value", "group_id"}, 1);
	RegisterTostVariant<TOST_YUEN>(loader, {D, I}, {"value", "group_id"}, 1);
	RegisterTostVariant<TOST_BOOTSTRAP>(loader, {D, I}, {"value", "group_id"}, 1);
	RegisterTostVariant<TOST_PROP_ONE>(loader, {I}, {"success"}, -1);
	RegisterTostVariant<TOST_PROP_TWO>(loader, {I, I}, {"success", "group_id"}, 1);
}

} // namespace duckdb
