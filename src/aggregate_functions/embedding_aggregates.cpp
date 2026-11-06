// Embedding drift and multivariate two-sample tests (#302), from upstream
// anofox-statistics 0.7:
//   embedding_drift_agg(embedding DOUBLE[], group [, options]) -- evidently's
//     embedding drift methods 'distance', 'ratio' and 'mmd' (fixed or
//     bootstrapped threshold). group is a BIGINT / VARCHAR label (the reference
//     is the smaller label unless options.reference names it) or a BOOLEAN
//     is_reference flag (TRUE rows are the reference).
//   energy_distance_test_agg(x DOUBLE[], group [, options]),
//   mmd_test_agg(x DOUBLE[], group [, options]) -- multivariate energy distance
//     and MMD permutation tests of the two groups (group 1: the smaller label).
// Every embedding of a call must have the same length (an error otherwise).
// Rows with a NULL embedding, a NULL group, or a NULL / NaN element are skipped.

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
#include "../include/list_input.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/result_fields.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"
#include "two_group.hpp"

namespace duckdb {

static constexpr int EMB_DRIFT = 0;  // embedding_drift_agg
static constexpr int EMB_ENERGY = 1; // energy_distance_test_agg (FFI kind 0)
static constexpr int EMB_MMD = 2;    // mmd_test_agg (FFI kind 1)

//===--------------------------------------------------------------------===//
// State: the embeddings of each group label, row-major. Relocatable: only
// vectors and shared pointers.
//===--------------------------------------------------------------------===//
struct EmbeddingAggregateState {
	bool initialized;
	bool has_dim;
	idx_t dim;
	GroupCoder groups;
	vector<vector<double>> values; // per group code, row-major

	EmbeddingAggregateState() : initialized(false), has_dim(false), dim(0) {
	}

	vector<double> &Slot(const GroupLabel &label) {
		auto code = static_cast<idx_t>(groups.Code(label));
		if (code >= values.size()) {
			values.resize(code + 1);
		}
		return values[code];
	}

	idx_t Rows(idx_t slot) const {
		return dim == 0 || slot >= values.size() ? 0 : values[slot].size() / dim;
	}

	void Reset() {
		initialized = false;
		has_dim = false;
		dim = 0;
		groups.labels.clear();
		values.clear();
	}
};

struct EmbeddingBindData : public FunctionData {
	int kind = EMB_DRIFT;
	string name;
	bool boolean_group = false;
	EmbeddingDriftMapOptions drift;
	MultivariateTestMapOptions test;
	string options_text; // for Equals

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<EmbeddingBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<EmbeddingBindData>();
		return kind == other.kind && boolean_group == other.boolean_group && options_text == other.options_text;
	}
};

//===--------------------------------------------------------------------===//
// Result types
//===--------------------------------------------------------------------===//
static LogicalType EmbeddingResultType(int kind) {
	child_list_t<LogicalType> c;
	auto add = [&](const char *name, const LogicalType &type) { c.push_back(MakeChild(name, type)); };
	if (kind == EMB_DRIFT) {
		// evidently's embedding drift: no unified test fields (a score and a
		// threshold, no p-value), as column_drift_agg.
		add("method", LogicalType::VARCHAR);
		add("score", LogicalType::DOUBLE);
		add("threshold", LogicalType::DOUBLE);
		add("drifted", LogicalType::BOOLEAN);
		add("n1", LogicalType::BIGINT);
		add("n2", LogicalType::BIGINT);
		add("dim", LogicalType::BIGINT);
		// Kernel bandwidth of method 'mmd' (drift::embedding_mmd_bandwidth); NULL otherwise.
		add("bandwidth", LogicalType::DOUBLE);
		return LogicalType::STRUCT(std::move(c));
	}
	add("statistic", LogicalType::DOUBLE);
	add("p_value", LogicalType::DOUBLE);
	add("n1", LogicalType::BIGINT);
	add("n2", LogicalType::BIGINT);
	add("dim", LogicalType::BIGINT);
	add("n_permutations", LogicalType::BIGINT);
	AppendTestResultFields(c);
	if (kind == EMB_MMD) {
		// Bandwidth of the gaussian / laplacian kernel used (the median heuristic's by default
		// for the gaussian); NULL for the linear and polynomial kernels.
		add("bandwidth", LogicalType::DOUBLE);
	}
	return LogicalType::STRUCT(std::move(c));
}

//===--------------------------------------------------------------------===//
// Aggregate operations
//===--------------------------------------------------------------------===//
static void EmbeddingInitialize(data_ptr_t state_p) {
	new (state_p) EmbeddingAggregateState();
}

static void EmbeddingDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EmbeddingAggregateState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~EmbeddingAggregateState();
	}
}

static void CheckDim(EmbeddingAggregateState &state, idx_t dim, const string &name) {
	if (!state.has_dim) {
		state.has_dim = true;
		state.dim = dim;
	} else if (dim != state.dim) {
		throw InvalidInputException("%s: every embedding must have the same length, got %llu and %llu", name,
		                            static_cast<unsigned long long>(state.dim),
		                            static_cast<unsigned long long>(dim));
	}
}

static void ThirdLabel(const EmbeddingAggregateState &state, const GroupLabel &label, const string &name) {
	if (state.groups.labels.size() > 2) {
		throw InvalidInputException("%s compares exactly two groups, but found a third group label %s (already saw "
		                            "%s and %s)",
		                            name, label.ToString(), state.groups.labels[0].ToString(),
		                            state.groups.labels[1].ToString());
	}
}

static void EmbeddingUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                            Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<EmbeddingBindData>();
	UnifiedVectorFormat list_data, group_data;
	inputs[0].ToUnifiedFormat(count, list_data);
	inputs[1].ToUnifiedFormat(count, group_data);
	auto entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);
	auto &child = ListVector::GetEntry(inputs[0]);
	auto child_data = FlatDataMut<double>(child);
	auto &child_validity = FlatValidityMut(child);
	const auto group_type = inputs[1].GetType().id();

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EmbeddingAggregateState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		state.initialized = true;
		auto list_idx = list_data.sel->get_index(i);
		auto grp_idx = group_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_idx) || !group_data.validity.RowIsValid(grp_idx)) {
			continue;
		}
		auto entry = entries[list_idx];
		CheckDim(state, entry.length, bind.name);
		if (ListHasNullElement(child_validity, entry)) {
			continue;
		}
		bool has_nan = false;
		for (idx_t j = 0; j < entry.length; j++) {
			has_nan = has_nan || std::isnan(child_data[entry.offset + j]);
		}
		if (has_nan) {
			continue;
		}
		GroupLabel label;
		if (group_type == LogicalTypeId::BOOLEAN) {
			// is_reference: TRUE rows (label 0, the smaller) are the reference.
			label = GroupLabel::FromInt(UnifiedVectorFormat::GetData<bool>(group_data)[grp_idx] ? 0 : 1);
		} else {
			label = ReadGroupLabel(group_data, grp_idx, group_type == LogicalTypeId::VARCHAR);
		}
		auto &slot = state.Slot(label);
		ThirdLabel(state, label, bind.name);
		slot.insert(slot.end(), child_data + entry.offset, child_data + entry.offset + entry.length);
	}
}

static void EmbeddingCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                             idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<EmbeddingBindData>();
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (EmbeddingAggregateState **)source_data.data;
	auto targets = (EmbeddingAggregateState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.initialized) {
			continue;
		}
		target.initialized = true;
		if (source.has_dim) {
			CheckDim(target, source.dim, bind.name);
		}
		for (idx_t g = 0; g < source.groups.labels.size(); g++) {
			auto &target_values = target.Slot(source.groups.labels[g]);
			ThirdLabel(target, source.groups.labels[g], bind.name);
			if (g >= source.values.size()) {
				continue;
			}
			auto &source_values = source.values[g];
			if (target_values.empty()) {
				target_values = CombineTake(source_values, aggr_input_data);
			} else {
				target_values.insert(target_values.end(), source_values.begin(), source_values.end());
			}
		}
	}
}

//! Sort the rows of a row-major sample lexicographically, so the FFI input (and
//! a seeded permutation or bootstrap draw) is independent of the row order.
static void SortRows(vector<double> &values, idx_t dim) {
	if (dim == 0) {
		return;
	}
	idx_t n = values.size() / dim;
	vector<idx_t> order(n);
	std::iota(order.begin(), order.end(), idx_t(0));
	std::sort(order.begin(), order.end(), [&](idx_t a, idx_t b) {
		return std::lexicographical_compare(values.begin() + a * dim, values.begin() + (a + 1) * dim,
		                                    values.begin() + b * dim, values.begin() + (b + 1) * dim);
	});
	vector<double> sorted;
	sorted.reserve(values.size());
	for (auto r : order) {
		sorted.insert(sorted.end(), values.begin() + r * dim, values.begin() + (r + 1) * dim);
	}
	values = std::move(sorted);
}

//! Index of the reference (or first) group among the two labels.
static idx_t ReferenceSlot(const EmbeddingAggregateState &state, const EmbeddingBindData &bind) {
	auto &labels = state.groups.labels;
	if (bind.kind != EMB_DRIFT || !bind.drift.reference) {
		return labels[1] < labels[0] ? 1 : 0;
	}
	auto &ref = *bind.drift.reference;
	for (idx_t k = 0; k < labels.size(); k++) {
		bool match = labels[k].is_string
		                 ? ref.ToString() == *labels[k].str_value
		                 : ref.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>() == labels[k].int_value;
		if (match) {
			return k;
		}
	}
	throw InvalidInputException("%s: reference group %s not found (groups: %s and %s)", bind.name, ref.ToString(),
	                            labels[0].ToString(), labels[1].ToString());
}

//! evidently's defaults, from upstream (drift::defaults) rather than copied here.
static const AnofoxDriftDefaults &EvidentlyDefaults() {
	static const AnofoxDriftDefaults defaults = [] {
		AnofoxDriftDefaults d {};
		anofox_drift_defaults(&d);
		return d;
	}();
	return defaults;
}

static AnofoxEmbeddingDriftOptions DriftOptions(const EmbeddingBindData &bind) {
	auto &o = bind.drift;
	auto method = o.method.value_or("distance");
	auto metric = o.metric.value_or("euclidean");
	AnofoxEmbeddingDriftOptions r;
	r.component_test = o.component_test ? o.component_test->c_str() : nullptr;
	r.threshold = o.threshold.value_or(std::nan(""));
	r.quantile = o.quantile.value_or(std::nan(""));
	r.component_threshold = o.component_threshold.value_or(std::nan(""));
	r.n_bootstrap = o.n_bootstrap.value_or(EvidentlyDefaults().embedding_bootstrap_resamples);
	r.seed = o.seed.value_or(ANOFOX_DEFAULT_MC_SEED);
	r.has_seed = true;
	r.method = method == "distance" ? 0 : method == "ratio" ? 1 : 2;
	r.metric = metric == "euclidean" ? 0 : metric == "cosine" ? 1 : metric == "cityblock" ? 2 : 3;
	r.bootstrap = o.bootstrap.value_or(false);
	return r;
}

static AnofoxMultivariateTestOptions TestOptions(const EmbeddingBindData &bind) {
	auto &o = bind.test;
	auto kernel = o.kernel.value_or("gaussian");
	AnofoxMultivariateTestOptions r;
	r.bandwidth = o.bandwidth.value_or(1.0);
	r.scale = o.scale.value_or(1.0);
	r.offset = o.offset.value_or(1.0);
	r.n_permutations = o.n_permutations.value_or(1000);
	r.seed = o.seed.value_or(ANOFOX_DEFAULT_MC_SEED);
	r.has_seed = true;
	r.degree = o.degree.value_or(2);
	r.kernel = kernel == "gaussian" ? 0 : kernel == "laplacian" ? 1 : kernel == "linear" ? 2 : 3;
	// The gaussian kernel's default bandwidth is the median heuristic (upstream
	// Kernel::GaussianMedianHeuristic), as mmd_agg and evidently's MMD use.
	if (kernel == "gaussian" && (o.median_bandwidth || !o.bandwidth)) {
		r.kernel = 4;
	}
	return r;
}

static void EmbeddingFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                              idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<EmbeddingBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (EmbeddingAggregateState **)sdata.data;
	auto &entries = StructVector::GetEntries(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		idx_t row = i + offset;
		if (!state.initialized || state.groups.labels.size() < 2 || state.Rows(0) == 0 || state.Rows(1) == 0) {
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}
		for (auto &group : state.values) {
			SortRows(group, state.dim);
		}
		auto ref = ReferenceSlot(state, bind);
		auto &a = state.values[ref];
		auto &b = state.values[1 - ref];
		auto na = state.Rows(ref), nb = state.Rows(1 - ref);
		AnofoxError error;
		if (bind.kind == EMB_DRIFT) {
			AnofoxEmbeddingDriftResult r;
			if (!anofox_embedding_drift(a.data(), na, b.data(), nb, state.dim, DriftOptions(bind), &r, &error)) {
				ThrowUnlessDegenerate(bind.name.c_str(), error);
				FlatVector::SetNull(result, row, true);
				state.Reset();
				continue;
			}
			auto method = bind.drift.method.value_or("distance");
			SetResultString(StructEntry(entries[0]), row, method.c_str());
			FlatDataMut<double>(StructEntry(entries[1]))[row] = r.score;
			FlatDataMut<double>(StructEntry(entries[2]))[row] = r.threshold;
			FlatDataMut<bool>(StructEntry(entries[3]))[row] = r.drifted;
			FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = static_cast<int64_t>(r.n1);
			FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(r.n2);
			FlatDataMut<int64_t>(StructEntry(entries[6]))[row] = static_cast<int64_t>(r.dim);
			if (std::isnan(r.bandwidth)) {
				FlatVector::SetNull(StructEntry(entries[7]), row, true);
			} else {
				FlatDataMut<double>(StructEntry(entries[7]))[row] = r.bandwidth;
			}
		} else {
			AnofoxMultivariateTestResult r;
			if (!anofox_multivariate_two_sample(bind.kind == EMB_ENERGY ? 0 : 1, a.data(), na, b.data(), nb,
			                                    state.dim, TestOptions(bind), &r, &error)) {
				ThrowUnlessDegenerate(bind.name.c_str(), error);
				FlatVector::SetNull(result, row, true);
				state.Reset();
				continue;
			}
			FlatDataMut<double>(StructEntry(entries[0]))[row] = r.statistic;
			FlatDataMut<double>(StructEntry(entries[1]))[row] = r.p_value;
			FlatDataMut<int64_t>(StructEntry(entries[2]))[row] = static_cast<int64_t>(r.n1);
			FlatDataMut<int64_t>(StructEntry(entries[3]))[row] = static_cast<int64_t>(r.n2);
			FlatDataMut<int64_t>(StructEntry(entries[4]))[row] = static_cast<int64_t>(r.dim);
			FlatDataMut<int64_t>(StructEntry(entries[5]))[row] = static_cast<int64_t>(r.n_permutations);
			WriteTestResultFields(result, row, 6, r.unified);
			if (bind.kind == EMB_MMD) {
				auto &bw = StructEntry(entries[entries.size() - 1]);
				if (std::isnan(r.bandwidth)) {
					FlatVector::SetNull(bw, row, true);
				} else {
					FlatDataMut<double>(bw)[row] = r.bandwidth;
				}
			}
		}
		state.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Bind and registration
//===--------------------------------------------------------------------===//
static const char *EmbeddingName(int kind) {
	return kind == EMB_DRIFT ? "embedding_drift_agg" : kind == EMB_ENERGY ? "energy_distance_test_agg" : "mmd_test_agg";
}

template <int KIND>
static unique_ptr<FunctionData> EmbeddingBind(ClientContext &context, AggregateBindFunction &function,
                                              vector<unique_ptr<Expression>> &arguments) {
	SetReturnTypeOf(function, EmbeddingResultType(KIND));
	auto bind = make_uniq<EmbeddingBindData>();
	bind->kind = KIND;
	bind->name = EmbeddingName(KIND);
	bind->boolean_group = ReturnTypeOf(*arguments[1]).id() == LogicalTypeId::BOOLEAN;
	if (arguments.size() >= 3) {
		Value options = EvaluateConstantOptions(context, *arguments[2], bind->name);
		if (KIND == EMB_DRIFT) {
			bind->drift = EmbeddingDriftMapOptions::ParseFromValue(options, bind->name);
		} else {
			bind->test = MultivariateTestMapOptions::ParseFromValue(options, bind->name, KIND == EMB_MMD);
		}
		bind->options_text = options.ToString();
	}
	if (KIND == EMB_DRIFT) {
		if (bind->boolean_group && bind->drift.reference) {
			throw InvalidInputException("%s: option 'reference' needs a group label; with a BOOLEAN is_reference "
			                            "the TRUE rows are the reference",
			                            bind->name);
		}
		// Reject an unknown component test at bind time: the FFI parses the
		// options before it looks at the (here empty) samples.
		AnofoxEmbeddingDriftResult r;
		AnofoxError error;
		if (!anofox_embedding_drift(nullptr, 0, nullptr, 0, 0, DriftOptions(*bind), &r, &error) &&
		    ClassifyFfiError(error.code) != FfiErrorKind::DEGENERATE) {
			ThrowFromFfiError(bind->name.c_str(), error);
		}
	}
	PostHogTelemetry::Instance().RecordFunctionCall(bind->name);
	return std::move(bind);
}

struct EmbeddingSpec {
	const char *name;
	const char *description;
	const char *example;
	const char *example_no_options;
	const char *value_name;
};

template <int KIND>
static void RegisterEmbedding(ExtensionLoader &loader, const EmbeddingSpec &spec) {
	AggregateFunctionSet set(spec.name);
	auto list_type = LogicalType::LIST(LogicalType::DOUBLE);
	vector<LogicalType> group_types = {LogicalType::BIGINT, LogicalType::VARCHAR};
	if (KIND == EMB_DRIFT) {
		group_types.push_back(LogicalType::BOOLEAN);
	}
	for (auto &group_type : group_types) {
		for (bool with_options : {true, false}) {
			vector<LogicalType> args = {list_type, group_type};
			if (with_options) {
				args.push_back(LogicalType::ANY);
			}
			AggregateFunction f(spec.name, args, LogicalType::ANY, AggregateFunction::StateSize<EmbeddingAggregateState>,
			                    ANOFOX_AGG_INIT(EmbeddingInitialize),
			                    ANOFOX_GUARDED_UPDATE(EmbeddingUpdate, EmbeddingDestroy, EmbeddingInitialize),
			                    EmbeddingCombine,
			                    ANOFOX_GUARDED_FINALIZE(EmbeddingFinalize, EmbeddingDestroy, EmbeddingInitialize),
			                    nullptr, ANOFOX_AGG_BIND(EmbeddingBind<KIND>), EmbeddingDestroy);
			set.AddFunction(f);
		}
	}
	CreateAggregateFunctionInfo info(std::move(set));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	vector<string> categories = {"distribution-comparison"};
	if (KIND == EMB_DRIFT) {
		categories.push_back("drift");
	}
	FunctionDescription d1;
	d1.description = spec.description;
	d1.examples = {spec.example};
	d1.categories = categories;
	d1.parameter_names = {spec.value_name, "group_id", "options"};
	d1.parameter_types = {list_type, LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = string(spec.description) + " Default options.";
	d2.examples = {spec.example_no_options};
	d2.categories = categories;
	d2.parameter_names = {spec.value_name, "group_id"};
	d2.parameter_types = {list_type, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(std::move(info));
}

void RegisterEmbeddingAggregateFunctions(ExtensionLoader &loader) {
	RegisterEmbedding<EMB_DRIFT>(
	    loader, {"embedding_drift_agg",
	             "evidently's embedding drift of the current group against the reference: 'distance' between the mean "
	             "embeddings (euclidean, cosine, cityblock, chebyshev; threshold 0.2), 'ratio' (share of drifted "
	             "components by a per-component drift test; threshold 0.2) or 'mmd' (squared maximum mean discrepancy "
	             "with the median bandwidth; threshold 0.015), with a fixed or bootstrapped threshold. group_id is a "
	             "label (the reference is the smaller one unless options.reference names it) or a BOOLEAN "
	             "is_reference.",
	             "embedding_drift_agg(embedding, is_reference, {'method': 'distance', 'metric': 'cosine'})",
	             "embedding_drift_agg(embedding, is_reference)", "embedding"});
	RegisterEmbedding<EMB_ENERGY>(
	    loader, {"energy_distance_test_agg",
	             "Multivariate energy distance two-sample test (Szekely and Rizzo) of two groups of equal-length "
	             "vectors, with a seeded permutation p-value.",
	             "energy_distance_test_agg(x, group_id, {'n_permutations': 999, 'seed': 42})",
	             "energy_distance_test_agg(x, group_id)", "x"});
	RegisterEmbedding<EMB_MMD>(
	    loader, {"mmd_test_agg",
	             "Multivariate maximum mean discrepancy two-sample test (Gretton et al., unbiased MMD^2) of two groups "
	             "of equal-length vectors with a gaussian, laplacian, linear or polynomial kernel, with a seeded "
	             "permutation p-value. The gaussian kernel's bandwidth defaults to the median heuristic (median "
	             "pairwise distance of the pooled rows, as mmd_agg; 'bandwidth': 'median'); the result's bandwidth "
	             "field reports the one used.",
	             "mmd_test_agg(x, group_id, {'kernel': 'gaussian', 'bandwidth': 2.0})", "mmd_test_agg(x, group_id)",
	             "x"});
}

} // namespace duckdb
