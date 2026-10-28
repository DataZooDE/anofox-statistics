// Data-drift aggregates (#245-#255): evidently's drift statistics from upstream
// anofox-statistics 0.5, exposed as
//   ks_test_agg, cvm_test_agg, epps_singleton_agg, wasserstein_agg,
//   psi_agg, kl_divergence_agg, jensen_shannon_agg, hellinger_agg, tvd_agg
//     (value, group [, options]) -- two samples, the reference is the smaller
//     group label unless options.reference names it;
//   ad_test_agg(value, group [, options]) -- k samples.
// The divergences and tvd_agg also take VARCHAR values (one bin per category).

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/test_struct.hpp"
#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "two_group.hpp"

namespace duckdb {

static constexpr int DRIFT_AD = -1; // ad_test_agg; the others are AnofoxDriftKind

static bool IsDivergence(int kind) {
	return kind == ANOFOX_DRIFT_PSI || kind == ANOFOX_DRIFT_KL || kind == ANOFOX_DRIFT_JENSEN_SHANNON ||
	       kind == ANOFOX_DRIFT_HELLINGER;
}

//===--------------------------------------------------------------------===//
// State: the values of each group label. VARCHAR values are stored as dense
// category codes (first-seen order, renumbered by label before the FFI call).
// Relocatable: only vectors and shared pointers.
//===--------------------------------------------------------------------===//
struct DriftAggregateState {
	bool initialized;
	GroupCoder groups;
	vector<vector<double>> values; // per group code
	GroupCoder categories;         // VARCHAR values

	DriftAggregateState() : initialized(false) {
	}

	vector<double> &Slot(const GroupLabel &label) {
		auto code = static_cast<idx_t>(groups.Code(label));
		if (code >= values.size()) {
			values.resize(code + 1);
		}
		return values[code];
	}

	void Reset() {
		initialized = false;
		groups.labels.clear();
		values.clear();
		categories.labels.clear();
	}
};

struct DriftBindData : public FunctionData {
	int kind = 0;
	string name;
	DriftMapOptions options;
	string options_text; // for Equals

	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<DriftBindData>();
		copy->kind = kind;
		copy->name = name;
		copy->options = options;
		copy->options_text = options_text;
		return std::move(copy);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<DriftBindData>();
		return kind == other.kind && options_text == other.options_text;
	}
};

//===--------------------------------------------------------------------===//
// Result types
//===--------------------------------------------------------------------===//
static LogicalType DriftResultType(int kind) {
	child_list_t<LogicalType> c;
	auto add = [&](const char *name, const LogicalType &type) { c.push_back(make_pair(name, type)); };
	switch (kind) {
	case DRIFT_AD:
		add("statistic", LogicalType::DOUBLE);
		add("ad", LogicalType::DOUBLE);
		add("p_value", LogicalType::DOUBLE);
		add("k", LogicalType::BIGINT);
		add("n", LogicalType::BIGINT);
		add("method", LogicalType::VARCHAR);
		break;
	case ANOFOX_DRIFT_KS:
	case ANOFOX_DRIFT_CVM:
		add("statistic", LogicalType::DOUBLE);
		add("p_value", LogicalType::DOUBLE);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("method", LogicalType::VARCHAR);
		add("exact", LogicalType::BOOLEAN);
		break;
	case ANOFOX_DRIFT_EPPS_SINGLETON:
		add("statistic", LogicalType::DOUBLE);
		add("p_value", LogicalType::DOUBLE);
		add("df", LogicalType::BIGINT);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("method", LogicalType::VARCHAR);
		break;
	case ANOFOX_DRIFT_WASSERSTEIN:
		add("distance", LogicalType::DOUBLE);
		add("distance_normed", LogicalType::DOUBLE);
		add("p_value", LogicalType::DOUBLE);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("method", LogicalType::VARCHAR);
		break;
	case ANOFOX_DRIFT_TVD:
		add("statistic", LogicalType::DOUBLE);
		add("p_value", LogicalType::DOUBLE);
		add("n_categories", LogicalType::BIGINT);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("method", LogicalType::VARCHAR);
		break;
	default: // divergences
		add(kind == ANOFOX_DRIFT_PSI  ? "psi"
		    : kind == ANOFOX_DRIFT_KL ? "kl"
		                              : "distance",
		    LogicalType::DOUBLE);
		if (kind == ANOFOX_DRIFT_JENSEN_SHANNON) {
			add("divergence", LogicalType::DOUBLE);
		}
		add("n_bins", LogicalType::BIGINT);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("drift", LogicalType::BOOLEAN);
		add("method", LogicalType::VARCHAR);
		break;
	}
	AppendTestResultFields(c);
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// Aggregate operations
//===--------------------------------------------------------------------===//
static void DriftInitialize(data_ptr_t state_p) {
	new (state_p) DriftAggregateState();
}

static void DriftDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (DriftAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~DriftAggregateState();
	}
}

static void DriftUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                        Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<DriftBindData>();
	UnifiedVectorFormat value_data, group_data;
	inputs[0].ToUnifiedFormat(count, value_data);
	inputs[1].ToUnifiedFormat(count, group_data);
	const bool value_is_string = inputs[0].GetType().id() == LogicalTypeId::VARCHAR;
	const bool group_is_string = inputs[1].GetType().id() == LogicalTypeId::VARCHAR;

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (DriftAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto val_idx = value_data.sel->get_index(i);
		auto grp_idx = group_data.sel->get_index(i);
		if (!value_data.validity.RowIsValid(val_idx) || !group_data.validity.RowIsValid(grp_idx)) {
			continue;
		}
		double v;
		if (value_is_string) {
			v = state.categories.Code(
			    GroupLabel::FromString(UnifiedVectorFormat::GetData<string_t>(value_data)[val_idx].GetString()));
		} else {
			v = UnifiedVectorFormat::GetData<double>(value_data)[val_idx];
			if (std::isnan(v)) {
				continue;
			}
		}
		auto label = ReadGroupLabel(group_data, grp_idx, group_is_string);
		auto &slot = state.Slot(label);
		if (bind.kind != DRIFT_AD && state.groups.labels.size() > 2) {
			throw InvalidInputException("%s compares exactly two groups, but found a third group label %s (already "
			                            "saw %s and %s)",
			                            bind.name, label.ToString(), state.groups.labels[0].ToString(),
			                            state.groups.labels[1].ToString());
		}
		slot.push_back(v);
	}
}

static void DriftCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                         idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<DriftBindData>();
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (DriftAggregateState **)source_data.data;
	auto targets = (DriftAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		target.initialized = true;
		// Category codes of the source in the target's numbering.
		auto category_map = target.categories.Remap(source.categories);
		const bool remap = !source.categories.labels.empty();
		for (idx_t g = 0; g < source.groups.labels.size(); g++) {
			auto &target_values = target.Slot(source.groups.labels[g]);
			if (bind.kind != DRIFT_AD && target.groups.labels.size() > 2) {
				throw InvalidInputException("%s compares exactly two groups, but found a third group label %s",
				                            bind.name, source.groups.labels[g].ToString());
			}
			if (g >= source.values.size()) {
				continue;
			}
			auto &source_values = source.values[g];
			if (remap) {
				for (auto v : source_values) {
					target_values.push_back(category_map[static_cast<idx_t>(v)]);
				}
			} else if (target_values.empty()) {
				target_values = CombineTake(source_values, aggr_input_data);
			} else {
				target_values.insert(target_values.end(), source_values.begin(), source_values.end());
			}
		}
	}
}

//! Renumber the category codes so code k is the k-th smallest category, making
//! the FFI input (and a seeded permutation draw) independent of the row order.
static void CanonicalizeCategories(DriftAggregateState &state) {
	auto &labels = state.categories.labels;
	if (labels.empty()) {
		return;
	}
	vector<idx_t> order(labels.size());
	std::iota(order.begin(), order.end(), idx_t(0));
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) { return labels[a] < labels[b]; });
	vector<double> rank(labels.size());
	for (idx_t r = 0; r < order.size(); r++) {
		rank[order[r]] = static_cast<double>(r);
	}
	for (auto &group : state.values) {
		for (auto &v : group) {
			v = rank[static_cast<idx_t>(v)];
		}
	}
}

//! Index of the reference group among the two labels.
static idx_t ReferenceSlot(const DriftAggregateState &state, const DriftBindData &bind) {
	auto &labels = state.groups.labels;
	if (!bind.options.reference) {
		return labels[1] < labels[0] ? 1 : 0;
	}
	auto &ref = *bind.options.reference;
	for (idx_t k = 0; k < labels.size(); k++) {
		bool match = labels[k].is_string ? ref.ToString() == *labels[k].str_value
		                                 : ref.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>() ==
		                                       labels[k].int_value;
		if (match) {
			return k;
		}
	}
	throw InvalidInputException("%s: reference group %s not found (groups: %s and %s)", bind.name, ref.ToString(),
	                            labels[0].ToString(), labels[1].ToString());
}

static AnofoxDataArray AsArray(const vector<double> &v) {
	AnofoxDataArray a;
	a.data = v.data();
	a.validity = nullptr;
	a.len = v.size();
	return a;
}

static int32_t BinningCode(const DriftBindData &bind, bool categorical_values) {
	auto binning = bind.options.binning.value_or(categorical_values ? "categorical" : "auto");
	if (categorical_values && binning != "categorical" && binning != "auto") {
		throw InvalidInputException("%s: binning '%s' needs numeric values; VARCHAR values are binned by category",
		                            bind.name, binning);
	}
	if (categorical_values || binning == "categorical") {
		return 2;
	}
	return binning == "sturges" ? 1 : binning == "quantile" ? 3 : 0;
}

static void DriftFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<DriftBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (DriftAggregateState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		const bool categorical = !state.categories.labels.empty();
		CanonicalizeCategories(state);
		for (auto &group : state.values) {
			std::sort(group.begin(), group.end()); // row-order independent (#214)
		}

		AnofoxDriftResult r;
		AnofoxError error;
		bool success;
		if (bind.kind == DRIFT_AD) {
			if (!state.initialized || state.groups.labels.size() < 2) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			vector<double> codes = {};
			for (idx_t g = 0; g < state.values.size(); g++) {
				codes.push_back(static_cast<double>(g));
			}
			state.groups.Canonicalize(codes);
			vector<double> values, groups;
			for (idx_t g = 0; g < state.values.size(); g++) {
				values.insert(values.end(), state.values[g].begin(), state.values[g].end());
				groups.insert(groups.end(), state.values[g].size(), codes[g]);
			}
			success = anofox_ad_test_ksamp(AsArray(values), AsArray(groups), bind.options.midrank.value_or(true), &r,
			                               &error);
		} else {
			if (!state.initialized || state.groups.labels.size() < 2 || state.values.size() < 2 ||
			    state.values[0].empty() || state.values[1].empty()) {
				FlatVector::SetNull(result, row, true);
				continue;
			}
			if (categorical && !IsDivergence(bind.kind) && bind.kind != ANOFOX_DRIFT_TVD) {
				throw InvalidInputException("%s needs numeric values", bind.name);
			}
			auto ref = ReferenceSlot(state, bind);
			AnofoxDriftOptions o;
			auto &t = bind.options.t;
			o.t = t ? t->data() : nullptr;
			o.t_len = t ? t->size() : 0;
			o.p = bind.options.p.value_or(1.0);
			o.base = bind.options.base.value_or(std::nan(""));
			o.epsilon = bind.options.epsilon.value_or(std::nan(""));
			o.n_permutations = bind.options.n_permutations.value_or(bind.kind == ANOFOX_DRIFT_TVD ? 1000 : 0);
			o.seed = bind.options.seed.value_or(ANOFOX_DEFAULT_MC_SEED);
			o.has_seed = true;
			o.n_bins = bind.options.n_bins.value_or(10);
			auto alternative = bind.options.alternative.value_or(Alternative::TWO_SIDED);
			o.alternative = alternative == Alternative::TWO_SIDED ? ANOFOX_ALTERNATIVE_TWO_SIDED
			                : alternative == Alternative::LESS    ? ANOFOX_ALTERNATIVE_LESS
			                                                      : ANOFOX_ALTERNATIVE_GREATER;
			o.exact = bind.options.exact ? (*bind.options.exact ? 1 : 0) : -1;
			auto method = bind.options.method.value_or("auto");
			o.cvm_method = method == "exact" ? 1 : method == "asymptotic" ? 2 : 0;
			o.binning = IsDivergence(bind.kind) ? BinningCode(bind, categorical) : 0;
			success = anofox_drift_test(bind.kind, AsArray(state.values[ref]), AsArray(state.values[1 - ref]), o, &r,
			                            &error);
		}
		if (!success) {
			ThrowUnlessDegenerate(bind.name.c_str(), error);
			FlatVector::SetNull(result, row, true);
			continue;
		}

		idx_t k = 0;
		auto put_double = [&](double v) {
			auto &vec = StructEntry(entries[k++]);
			if (std::isnan(v)) {
				SetResultNull(vec, row);
			} else {
				FlatDataMut<double>(vec)[row] = v;
			}
		};
		auto put_int = [&](int64_t v) { FlatDataMut<int64_t>(StructEntry(entries[k++]))[row] = v; };
		auto put_bool = [&](bool v) { FlatDataMut<bool>(StructEntry(entries[k++]))[row] = v; };
		auto put_method = [&]() { SetResultString(StructEntry(entries[k++]), row, r.unified.method); };
		auto n1 = static_cast<int64_t>(r.n1), n2 = static_cast<int64_t>(r.n2);
		switch (bind.kind) {
		case DRIFT_AD:
			put_double(r.statistic);
			put_double(r.statistic2);
			put_double(r.p_value);
			put_int(n2);
			put_int(n1);
			put_method();
			break;
		case ANOFOX_DRIFT_KS:
		case ANOFOX_DRIFT_CVM:
			put_double(r.statistic);
			put_double(r.p_value);
			put_int(n1);
			put_int(n2);
			put_method();
			put_bool(r.exact);
			break;
		case ANOFOX_DRIFT_EPPS_SINGLETON:
			put_double(r.statistic);
			put_double(r.p_value);
			put_int(static_cast<int64_t>(r.df));
			put_int(n1);
			put_int(n2);
			put_method();
			break;
		case ANOFOX_DRIFT_WASSERSTEIN:
			put_double(r.statistic);
			put_double(r.statistic2);
			put_double(r.p_value);
			put_int(n1);
			put_int(n2);
			put_method();
			break;
		case ANOFOX_DRIFT_TVD:
			put_double(r.statistic);
			put_double(r.p_value);
			put_int(static_cast<int64_t>(r.n_bins));
			put_int(n1);
			put_int(n2);
			put_method();
			break;
		default:
			put_double(r.statistic);
			if (bind.kind == ANOFOX_DRIFT_JENSEN_SHANNON) {
				put_double(r.statistic2);
			}
			put_int(static_cast<int64_t>(r.n_bins));
			put_int(n1);
			put_int(n2);
			put_bool(r.statistic >= bind.options.threshold.value_or(0.1));
			put_method();
			break;
		}
		WriteTestResultFields(result, row, k, r.unified);
		state.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Bind and registration
//===--------------------------------------------------------------------===//
static const char *DriftName(int kind) {
	switch (kind) {
	case DRIFT_AD:
		return "ad_test_agg";
	case ANOFOX_DRIFT_KS:
		return "ks_test_agg";
	case ANOFOX_DRIFT_CVM:
		return "cvm_test_agg";
	case ANOFOX_DRIFT_EPPS_SINGLETON:
		return "epps_singleton_agg";
	case ANOFOX_DRIFT_WASSERSTEIN:
		return "wasserstein_agg";
	case ANOFOX_DRIFT_PSI:
		return "psi_agg";
	case ANOFOX_DRIFT_KL:
		return "kl_divergence_agg";
	case ANOFOX_DRIFT_JENSEN_SHANNON:
		return "jensen_shannon_agg";
	case ANOFOX_DRIFT_HELLINGER:
		return "hellinger_agg";
	default:
		return "tvd_agg";
	}
}

template <int KIND>
static unique_ptr<FunctionData> DriftBind(ClientContext &context, AggregateBindFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, DriftResultType(KIND));
	auto bind = make_uniq<DriftBindData>();
	bind->kind = KIND;
	bind->name = DriftName(KIND);
	if (arguments.size() >= 3) {
		Value options = EvaluateConstantOptions(context, *arguments[2], bind->name);
		bind->options = DriftMapOptions::ParseFromValue(options, bind->name, KIND);
		bind->options_text = options.ToString();
	}
	PostHogTelemetry::Instance().RecordFunctionCall(bind->name);
	return std::move(bind);
}

struct DriftSpec {
	const char *name;
	const char *description;
	const char *options_example;
	bool categorical_values;
};

template <int KIND>
static void RegisterDrift(ExtensionLoader &loader, const DriftSpec &spec) {
	AggregateFunctionSet set(spec.name);
	vector<LogicalType> value_types = {LogicalType::DOUBLE};
	if (spec.categorical_values) {
		value_types.push_back(LogicalType::VARCHAR);
	}
	for (auto &value_type : value_types) {
		for (auto &group_type : {LogicalType::BIGINT, LogicalType::VARCHAR}) {
			for (bool with_options : {true, false}) {
				vector<LogicalType> args = {value_type, group_type};
				if (with_options) {
					args.push_back(LogicalType::ANY);
				}
				AggregateFunction f(
				    spec.name, args, LogicalType::ANY, AggregateFunction::StateSize<DriftAggregateState>,
				    ANOFOX_AGG_INIT(DriftInitialize), ANOFOX_GUARDED_UPDATE(DriftUpdate, DriftDestroy, DriftInitialize),
				    DriftCombine, ANOFOX_GUARDED_FINALIZE(DriftFinalize, DriftDestroy, DriftInitialize), nullptr,
				    ANOFOX_AGG_BIND(DriftBind<KIND>), DriftDestroy);
				set.AddFunction(f);
			}
		}
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	auto value_type = spec.categorical_values ? LogicalType::ANY : LogicalType::DOUBLE;
	FunctionDescription d1;
	d1.description = spec.description;
	d1.examples = {string(spec.name) + "(value, group_id, " + spec.options_example + ")"};
	d1.categories = {"distribution-comparison", "drift"};
	d1.parameter_names = {"value", "group_id", "options"};
	d1.parameter_types = {value_type, LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = string(spec.description) + " Default options.";
	d2.examples = {string(spec.name) + "(value, group_id)"};
	d2.categories = {"distribution-comparison", "drift"};
	d2.parameter_names = {"value", "group_id"};
	d2.parameter_types = {value_type, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

void RegisterDriftAggregateFunctions(ExtensionLoader &loader) {
	RegisterDrift<ANOFOX_DRIFT_KS>(
	    loader, {"ks_test_agg",
	             "Two-sample Kolmogorov-Smirnov test (R ks.test): exact p-value when n1*n2 < 10000, otherwise the "
	             "limiting Kolmogorov distribution.",
	             "{'alternative': 'two_sided', 'exact': true}", false});
	RegisterDrift<DRIFT_AD>(loader, {"ad_test_agg",
	                                 "k-sample Anderson-Darling test (Scholz and Stephens; kSamples::ad.test, "
	                                 "asymptotic p-value) of two or more groups.",
	                                 "{'midrank': true}", false});
	RegisterDrift<ANOFOX_DRIFT_CVM>(
	    loader, {"cvm_test_agg",
	             "Two-sample Cramer-von Mises test (Anderson's T): exact permutation p-value when max(n1, n2) <= 20, "
	             "otherwise the limiting distribution.",
	             "{'method': 'auto'}", false});
	RegisterDrift<ANOFOX_DRIFT_EPPS_SINGLETON>(
	    loader, {"epps_singleton_agg",
	             "Epps-Singleton two-sample test based on the empirical characteristic function (scipy "
	             "epps_singleton_2samp).",
	             "{'t': [0.4, 0.8]}", false});
	RegisterDrift<ANOFOX_DRIFT_WASSERSTEIN>(
	    loader, {"wasserstein_agg",
	             "1-D Wasserstein (earth mover's) distance, raw and normed by the reference's population SD "
	             "(evidently), with an optional permutation p-value.",
	             "{'p': 1, 'n_permutations': 1000}", false});
	RegisterDrift<ANOFOX_DRIFT_PSI>(
	    loader, {"psi_agg",
	             "Population Stability Index of the current group against the reference on shared bins (evidently "
	             "'psi'); drift when psi >= threshold.",
	             "{'binning': 'quantile', 'n_bins': 10}", true});
	RegisterDrift<ANOFOX_DRIFT_KL>(
	    loader, {"kl_divergence_agg",
	             "Kullback-Leibler divergence KL(reference || current) on shared bins (evidently 'kl_div').",
	             "{'binning': 'auto', 'base': 2}", true});
	RegisterDrift<ANOFOX_DRIFT_JENSEN_SHANNON>(
	    loader, {"jensen_shannon_agg",
	             "Jensen-Shannon distance (square root of the divergence) on shared bins (evidently "
	             "'jensenshannon').",
	             "{'binning': 'auto', 'base': 2}", true});
	RegisterDrift<ANOFOX_DRIFT_HELLINGER>(
	    loader, {"hellinger_agg", "Hellinger distance in [0, 1] on shared bins (evidently 'hellinger').",
	             "{'binning': 'auto', 'threshold': 0.1}", true});
	RegisterDrift<ANOFOX_DRIFT_TVD>(
	    loader, {"tvd_agg",
	             "Total variation distance between the category distributions of two groups, with a seeded "
	             "permutation p-value (evidently 'TVD').",
	             "{'n_permutations': 1000, 'seed': 42}", true});
}

} // namespace duckdb
