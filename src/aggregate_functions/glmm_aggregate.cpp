#include <type_traits>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/map_options_parser.hpp"
#include "../include/model_struct.hpp"
#include "telemetry.hpp"
#include "aggregate_finalize_guard.hpp"
#include "../include/canonical_order.hpp"

namespace duckdb {

namespace {
/// The index type `AnofoxGlmmOptions::random_slopes` points at, taken from the header
/// rather than written out. See the note on the fields that use it.
using RandomSlopeIndex = std::remove_const<std::remove_pointer<decltype(AnofoxGlmmOptions::random_slopes)>::type>::type;
} // namespace


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
	//! Group key -> dense index, in first-seen order. Held behind a pointer because
	//! DuckDB relocates aggregate states with a raw memory copy, and libstdc++'s
	//! unordered_map keeps pointers into its own object.
	unique_ptr<unordered_map<string, int32_t>> group_index;
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
	// The element type is *derived from* the FFI declaration rather than restated, so
	// the two cannot drift.
	//
	// Restating it is what broke every non-Linux build: this held `idx_t` while
	// `AnofoxGlmmOptions::random_slopes` is `const size_t *`. On Linux those are the
	// same *type* -- both `unsigned long` -- so `.data()` assigned cleanly and every
	// Linux job stayed green. On macOS and on wasm32 they are distinct types, and it
	// is a hard error. The same line failed the macOS and all three DuckDB-Wasm jobs,
	// and the deploys that depend on them.
	//
	// A `static_assert` here would not have helped, and that was measured rather than
	// assumed: asserting the two types are the same passes on Linux, because on Linux
	// they are. Deriving the type is the only form of the check that means anything on
	// the platform where the mistake is invisible.
	vector<RandomSlopeIndex> random_slopes;
	//! 0-based indices into x of additional crossed grouping-factor columns.
	vector<idx_t> group_columns;
	AnofoxGlmmDdf ddf;

	GlmmAggregateState()
	    : n_features(0), initialized(false), family(ANOFOX_GLMM_GAUSSIAN), fit_intercept(true), max_iterations(100),
	      tolerance(1e-8), compute_inference(false), confidence_level(0.95), reml(true),
	      theta(std::numeric_limits<double>::quiet_NaN()), power(1.5),
	      offset_column(0), ddf(ANOFOX_GLMM_DDF_DEFAULT) {
	}

	int32_t Intern(const string &key) {
		if (!group_index) {
			group_index = make_uniq<unordered_map<string, int32_t>>();
		}
		auto it = group_index->find(key);
		if (it != group_index->end()) {
			return it->second;
		}
		auto id = (int32_t)group_labels.size();
		group_index->emplace(key, id);
		group_labels.push_back(key);
		return id;
	}

	void Reset() {
		y_values.clear();
		x_columns.clear();
		group_ids.clear();
		group_index.reset();
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
	//! NaN = estimate the negative-binomial size (lme4::glmer.nb).
	double theta = std::numeric_limits<double>::quiet_NaN();
	double power = 1.5;
	idx_t offset_column = 0;
	// The element type is *derived from* the FFI declaration rather than restated, so
	// the two cannot drift.
	//
	// Restating it is what broke every non-Linux build: this held `idx_t` while
	// `AnofoxGlmmOptions::random_slopes` is `const size_t *`. On Linux those are the
	// same *type* -- both `unsigned long` -- so `.data()` assigned cleanly and every
	// Linux job stayed green. On macOS and on wasm32 they are distinct types, and it
	// is a hard error. The same line failed the macOS and all three DuckDB-Wasm jobs,
	// and the deploys that depend on them.
	//
	// A `static_assert` here would not have helped, and that was measured rather than
	// assumed: asserting the two types are the same passes on Linux, because on Linux
	// they are. Deriving the type is the only form of the check that means anything on
	// the platform where the mistake is invisible.
	vector<RandomSlopeIndex> random_slopes;
	vector<idx_t> group_columns;
	AnofoxGlmmDdf ddf = ANOFOX_GLMM_DDF_DEFAULT;

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
		r->ddf = ddf;
		return std::move(r);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<GlmmAggregateBindData>();
		return family == o.family && fit_intercept == o.fit_intercept && max_iterations == o.max_iterations &&
		       tolerance == o.tolerance && compute_inference == o.compute_inference &&
		       confidence_level == o.confidence_level && reml == o.reml &&
		       // theta is NaN when estimated; NaN != NaN would make a copy unequal to itself.
		       (theta == o.theta || (std::isnan(theta) && std::isnan(o.theta))) && power == o.power &&
		       offset_column == o.offset_column && random_slopes == o.random_slopes &&
		       group_columns == o.group_columns && ddf == o.ddf;
	}
};

static LogicalType GetGlmmResultType() {
	child_list_t<LogicalType> ranef_children;
	ranef_children.push_back(make_pair("group", LogicalType::VARCHAR));
	ranef_children.push_back(make_pair("intercept", LogicalType::DOUBLE));
	ranef_children.push_back(make_pair("se", LogicalType::DOUBLE));
	ranef_children.push_back(make_pair("n", LogicalType::BIGINT));
	// The group's full random-effect vector [intercept, slope_1, ...] and its
	// conditional standard errors.
	ranef_children.push_back(make_pair("effects", LogicalType::LIST(LogicalType::DOUBLE)));
	ranef_children.push_back(make_pair("effects_se", LogicalType::LIST(LogicalType::DOUBLE)));
	// Normal interval of each effect at confidence_level (broom.mixed ran_vals).
	ranef_children.push_back(make_pair("effects_conf_low", LogicalType::LIST(LogicalType::DOUBLE)));
	ranef_children.push_back(make_pair("effects_conf_high", LogicalType::LIST(LogicalType::DOUBLE)));
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

	// Stable shape (#152): the inference fields are always present, NULL
	// without compute_inference.
	AppendCoefficientInferenceFields(children, "z_values", false);
	children.push_back(make_pair("intercept_std_error", LogicalType::DOUBLE));

	children.push_back(make_pair("ranef", LogicalType::LIST(ranef_type)));
	// Negative-binomial size theta (estimated or fixed); NULL for other families.
	children.push_back(make_pair("nb_theta", LogicalType::DOUBLE));
	// Satterthwaite degrees of freedom of the coefficients and the intercept; NULL
	// for Wald inference (ddf := 'wald', and the non-gaussian families) or without
	// compute_inference.
	children.push_back(make_pair("df", LogicalType::LIST(LogicalType::DOUBLE)));
	children.push_back(make_pair("intercept_df", LogicalType::DOUBLE));
	// Residual degrees of freedom n - k (lme4 df.residual()).
	children.push_back(make_pair("df_residual", LogicalType::DOUBLE));
	// The coefficient statistics under Satterthwaite inference (t); z_values then is
	// NULL. Under Wald inference z_values holds them and t_values is NULL.
	children.push_back(make_pair("t_values", LogicalType::LIST(LogicalType::DOUBLE)));
	AppendModelSummaryFields(children);
	return LogicalType::STRUCT(std::move(children));
}

static void GlmmAggInitialize(data_ptr_t state_p) {
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
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	// A LIST containing NULLs is not itself NULL, so the list-level validity mask
	// says nothing about the elements. Read the child mask too and pass NaN for a
	// NULL element; the Rust side drops any row that is not finite throughout.
	auto &x_child_validity = FlatValidityMut(x_child);

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
		state.ddf = bind_data.ddf;

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
			target.ddf = source.ddf;
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

//! The inputs of anofox_glmm_fit / anofox_glmm_augment from the collected x columns:
//! the design without the crossed grouping columns, those columns dictionary-encoded
//! as extra factors, and the options with the offset / random-slope indices re-indexed
//! onto the reduced design. SETTINGS is the aggregate state or its bind data (both
//! carry the same option fields).
struct GlmmCall {
	vector<AnofoxDataArray> x_arrays;
	vector<vector<int32_t>> extra_ids;
	vector<const int32_t *> extra_ptrs;
	vector<RandomSlopeIndex> slopes;
	AnofoxGlmmOptions options {};
};

template <class SETTINGS>
static void PrepareGlmmCall(const SETTINGS &settings, vector<vector<double>> &x_columns, idx_t n_features,
                            GlmmCall &call) {
	// Additional crossed grouping factors are dictionary-encoded here and removed
	// from the design (like the offset column).
	vector<bool> is_group_col(n_features, false);
	for (auto gc : settings.group_columns) {
		if (gc < n_features) {
			is_group_col[gc] = true;
		}
	}
	for (idx_t j = 0; j < n_features; j++) {
		if (!is_group_col[j]) {
			call.x_arrays.push_back(AnofoxDataArray {x_columns[j].data(), nullptr, x_columns[j].size()});
		}
	}
	for (auto gc : settings.group_columns) {
		if (gc >= n_features) {
			continue;
		}
		unordered_map<double, int32_t> levels;
		vector<int32_t> ids;
		ids.reserve(x_columns[gc].size());
		for (double v : x_columns[gc]) {
			auto it = levels.find(v);
			int32_t id = (it == levels.end()) ? (int32_t)levels.size() : it->second;
			if (it == levels.end()) {
				levels.emplace(v, id);
			}
			ids.push_back(id);
		}
		call.extra_ids.push_back(std::move(ids));
	}
	for (auto &f : call.extra_ids) {
		call.extra_ptrs.push_back(f.data());
	}

	auto &options = call.options;
	options.family = settings.family;
	options.fit_intercept = settings.fit_intercept;
	options.max_iterations = settings.max_iterations;
	options.tolerance = settings.tolerance;
	options.compute_inference = settings.compute_inference;
	options.confidence_level = settings.confidence_level;
	options.reml = settings.reml;
	options.theta = settings.theta;
	options.power = settings.power;
	// The offset / random-slope indices refer to the caller's x; re-index them onto
	// x without the crossed grouping columns (rejected at bind if they overlap).
	auto reduced_index = [&](idx_t col0) {
		idx_t r = col0;
		for (auto gc : settings.group_columns) {
			if (gc < col0) {
				r--;
			}
		}
		return r;
	};
	options.offset_column =
	    static_cast<size_t>(settings.offset_column == 0 ? 0 : reduced_index(settings.offset_column - 1) + 1);
	for (auto c0 : settings.random_slopes) {
		call.slopes.push_back((RandomSlopeIndex)reduced_index(c0));
	}
	options.random_slopes = call.slopes.empty() ? nullptr : call.slopes.data();
	options.random_slopes_len = call.slopes.size();
	options.ddf = settings.ddf;
}

static void SetDoubleListG(Vector &target, idx_t row, const double *values, idx_t len) {
	auto list_data = FlatDataMut<list_entry_t>(target);
	auto child_offset = ListVector::GetListSize(target);
	ListVector::Reserve(target, child_offset + len);
	auto child_values = FlatDataMut<double>(ListVector::GetEntry(target));
	for (idx_t j = 0; j < len; j++) {
		child_values[child_offset + j] = values[j];
	}
	ListVector::SetListSize(target, child_offset + len);
	list_data[row].offset = child_offset;
	list_data[row].length = len;
}

//! Number the groups by sorted label and sort the rows (by group, then x, then y),
//! so the fit and the order of the random effects are bitwise independent of the
//! order in which threads delivered the rows (#226).
static void CanonicalizeGlmmRows(GlmmAggregateState &state) {
	const idx_t n_groups = state.group_labels.size();
	vector<idx_t> order(n_groups);
	std::iota(order.begin(), order.end(), idx_t(0));
	std::sort(order.begin(), order.end(),
	          [&](idx_t a, idx_t b) { return state.group_labels[a] < state.group_labels[b]; });
	vector<int32_t> rank(n_groups);
	vector<string> sorted(n_groups);
	for (idx_t r = 0; r < n_groups; r++) {
		rank[order[r]] = static_cast<int32_t>(r);
		sorted[r] = state.group_labels[order[r]];
	}
	for (auto &g : state.group_ids) {
		g = rank[static_cast<idx_t>(g)];
	}
	state.group_labels = std::move(sorted);
	state.group_index = make_uniq<unordered_map<string, int32_t>>();
	for (idx_t r = 0; r < n_groups; r++) {
		state.group_index->emplace(state.group_labels[r], static_cast<int32_t>(r));
	}
	vector<double> group_key(state.group_ids.begin(), state.group_ids.end());
	CanonicalRows().Key(group_key).Keys(state.x_columns).Key(state.y_values).Carry(state.group_ids).Sort();
}

static void GlmmAggFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAggregateState **)sdata.data;
	auto &struct_entries = StructVector::GetEntries(result);
	ModelStructWriter writer(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;

		if (!state.initialized || state.y_values.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}

		CanonicalizeGlmmRows(state);
		AnofoxDataArray y_array {state.y_values.data(), nullptr, state.y_values.size()};

		GlmmCall call;
		PrepareGlmmCall(state, state.x_columns, state.n_features, call);
		auto &x_arrays = call.x_arrays;
		auto &extra_ptrs = call.extra_ptrs;
		auto &options = call.options;

		AnofoxGlmmResult res {};
		AnofoxError error;
		bool ok = anofox_glmm_fit(y_array, x_arrays.data(), x_arrays.size(), state.group_ids.data(),
		                          state.group_ids.size(), extra_ptrs.empty() ? nullptr : extra_ptrs.data(),
		                          extra_ptrs.size(), options, &res, &error);
		if (!ok) {
			ThrowUnlessDegenerate("glmm_fit_agg", error);
			FlatVector::SetNull(result, row, true);
			state.Reset();
			continue;
		}

		idx_t c = 0;
		SetDoubleListG(StructEntry(struct_entries[c++]), row, res.coefficients, res.coefficients_len);
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.intercept;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.var_group;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.var_residual;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.icc;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.log_likelihood;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.aic;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.bic;
		FlatDataMut<double>(StructEntry(struct_entries[c++]))[row] = res.deviance;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)res.n_observations;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)res.n_groups;
		FlatDataMut<int64_t>(StructEntry(struct_entries[c++]))[row] = (int64_t)res.n_features;
		FlatDataMut<int32_t>(StructEntry(struct_entries[c++]))[row] = (int32_t)res.iterations;
		FlatDataMut<bool>(StructEntry(struct_entries[c++]))[row] = res.converged;
		SetDoubleListG(StructEntry(struct_entries[c++]), row, res.random_cov, res.random_dim * res.random_dim);
		FlatDataMut<int32_t>(StructEntry(struct_entries[c++]))[row] = (int32_t)res.random_dim;

		// Per-factor variance components LIST(STRUCT(n_levels, var)).
		{
			auto &fac_vec = StructEntry(struct_entries[c++]);
			auto fac_list = FlatDataMut<list_entry_t>(fac_vec);
			auto fac_off = ListVector::GetListSize(fac_vec);
			ListVector::Reserve(fac_vec, fac_off + res.factor_len);
			auto &fac_struct = ListVector::GetEntry(fac_vec);
			auto &fac_children = StructVector::GetEntries(fac_struct);
			auto f_levels = FlatDataMut<int64_t>(StructEntry(fac_children[0]));
			auto f_var = FlatDataMut<double>(StructEntry(fac_children[1]));
			for (idx_t j = 0; j < res.factor_len; j++) {
				f_levels[fac_off + j] = (int64_t)res.factor_n_levels[j];
				f_var[fac_off + j] = res.factor_var[j];
			}
			ListVector::SetListSize(fac_vec, fac_off + res.factor_len);
			fac_list[row].offset = fac_off;
			fac_list[row].length = res.factor_len;
		}

		// Inference (NULL unless requested), CI fields and the model summary
		if (state.compute_inference) {
			// GLMM has its own result struct; view its inference in the shared layout.
			AnofoxFitResultInference view = {res.std_errors, res.z_values,   res.p_values,         res.ci_lower,
			                                  res.ci_upper,   res.inference_len, res.confidence_level, NAN,
			                                  NAN};
			writer.WriteInference(row, &view);
		} else {
			writer.WriteInference(row, nullptr);
		}
		writer.WriteSummary(row, res.summary);
		// std_errors, z_values, p_values, ci_lower, ci_upper, intercept_std_error
		c += 6;

		// The random-effects LIST(STRUCT(group, intercept, se, n)).
		auto &ranef_vec = StructEntry(struct_entries[c]);
		auto ranef_list = FlatDataMut<list_entry_t>(ranef_vec);
		auto ranef_off = ListVector::GetListSize(ranef_vec);
		ListVector::Reserve(ranef_vec, ranef_off + res.ranef_len);
		auto &ranef_struct = ListVector::GetEntry(ranef_vec);
		auto &ranef_children = StructVector::GetEntries(ranef_struct);
		auto r_group = FlatDataMut<string_t>(StructEntry(ranef_children[0]));
		auto r_value = FlatDataMut<double>(StructEntry(ranef_children[1]));
		auto r_se = FlatDataMut<double>(StructEntry(ranef_children[2]));
		auto r_n = FlatDataMut<int64_t>(StructEntry(ranef_children[3]));
		auto &r_effects = StructEntry(ranef_children[4]);
		auto &r_effects_se = StructEntry(ranef_children[5]);
		auto &r_conf_low = StructEntry(ranef_children[6]);
		auto &r_conf_high = StructEntry(ranef_children[7]);

		for (idx_t j = 0; j < res.ranef_len; j++) {
			auto id = res.ranef_group[j];
			const string &label =
			    (id >= 0 && (idx_t)id < state.group_labels.size()) ? state.group_labels[id] : string();
			r_group[ranef_off + j] = StringVector::AddString(StructEntry(ranef_children[0]), label);
			r_value[ranef_off + j] = res.ranef_value[j];
			r_se[ranef_off + j] = res.ranef_se[j];
			r_n[ranef_off + j] = res.ranef_n[j];
			SetDoubleListG(r_effects, ranef_off + j, res.ranef_effects + j * res.ranef_dim, res.ranef_dim);
			SetDoubleListG(r_effects_se, ranef_off + j, res.ranef_effects_se + j * res.ranef_dim, res.ranef_dim);
			SetDoubleListG(r_conf_low, ranef_off + j, res.ranef_effects_conf_low + j * res.ranef_dim, res.ranef_dim);
			SetDoubleListG(r_conf_high, ranef_off + j, res.ranef_effects_conf_high + j * res.ranef_dim,
			               res.ranef_dim);
		}
		ListVector::SetListSize(ranef_vec, ranef_off + res.ranef_len);
		ranef_list[row].offset = ranef_off;
		ranef_list[row].length = res.ranef_len;

		auto &nb_theta_vec = StructEntry(struct_entries[c + 1]);
		if (std::isnan(res.nb_theta)) {
			FlatVector::SetNull(nb_theta_vec, row, true);
		} else {
			FlatDataMut<double>(nb_theta_vec)[row] = res.nb_theta;
		}
		auto &df_vec = StructEntry(struct_entries[c + 2]);
		if (res.df == nullptr) {
			FlatVector::SetNull(df_vec, row, true);
		} else {
			SetDoubleListG(df_vec, row, res.df, res.inference_len);
		}
		auto &intercept_df_vec = StructEntry(struct_entries[c + 3]);
		if (std::isnan(res.intercept_df)) {
			FlatVector::SetNull(intercept_df_vec, row, true);
		} else {
			FlatDataMut<double>(intercept_df_vec)[row] = res.intercept_df;
		}
		FlatDataMut<double>(StructEntry(struct_entries[c + 4]))[row] = res.df_residual;
		// The writer filled both t_values and z_values with the statistics: keep
		// t_values under Satterthwaite (df set), z_values under Wald.
		if (res.df != nullptr) {
			FlatVector::SetNull(StructEntry(struct_entries[c - 5]), row, true); // z_values
		} else {
			FlatVector::SetNull(StructEntry(struct_entries[c + 5]), row, true); // t_values
		}

		anofox_free_glmm_result(&res);
		state.Reset();
	}
}

//! The options MAP / STRUCT (argument 4) of glmm_fit_agg / glmm_augment_agg.
static unique_ptr<GlmmAggregateBindData> ParseGlmmBindData(ClientContext &context,
                                                           vector<unique_ptr<Expression>> &arguments,
                                                           const char *fn_name) {
	auto result = make_uniq<GlmmAggregateBindData>();

	if (arguments.size() >= 4) {
		auto opts = RegressionMapOptions::ParseFromExpression(
            context, *arguments[3], fn_name,
            {"fit_intercept", "compute_inference", "confidence_level", "max_iterations", "tolerance", "power", "family", "reml", "offset", "random", "groups", "theta", "ddf"});
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
		if (opts.glmm_ddf.has_value()) {
			result->ddf = (AnofoxGlmmDdf)opts.glmm_ddf.value();
		}
		for (auto gc : result->group_columns) {
			if (result->offset_column != 0 && gc == result->offset_column - 1) {
				throw InvalidInputException("%s: column %llu cannot be both the 'offset' and a 'groups' "
				                            "column",
				                            fn_name, (unsigned long long)(gc + 1));
			}
			for (auto rs : result->random_slopes) {
				if ((idx_t)rs == gc) {
					throw InvalidInputException("%s: column %llu cannot be both a 'random' slope and a "
					                            "'groups' column",
					                            fn_name, (unsigned long long)(gc + 1));
				}
			}
		}
	}

	return result;
}

static unique_ptr<FunctionData> GlmmAggBind(ClientContext &context, AggregateBindFunction &function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto result = ParseGlmmBindData(context, arguments, "glmm_fit_agg");
	SetReturnTypeOf(function, GetGlmmResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("glmm_fit_agg");
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// glmm_augment_agg(y, x, group [, options]) -- issue #218
//
// A window aggregate over a whole partition that fits the same model as
// glmm_fit_agg and returns one STRUCT per input row, in frame order: the
// conditional and population-level fitted values and the response and Pearson
// residuals (anofox_glmm_augment). Every row is kept, so the list lines up with
// ROW_NUMBER() over the same frame; a row the fit could not use (a NULL or
// non-finite value, a NULL group) gets NULLs and is_training = false.
//===--------------------------------------------------------------------===//
struct GlmmAugmentState {
	vector<double> y;              // NaN for a NULL
	vector<vector<double>> x_rows; // empty for a NULL x list
	vector<int32_t> group_ids;     // -1 for a NULL group
	//! Group key -> dense index; behind a pointer for the same reason as in
	//! GlmmAggregateState.
	unique_ptr<unordered_map<string, int32_t>> group_index;
	vector<string> group_labels;
	idx_t n_features;
	bool initialized;

	GlmmAugmentState() : n_features(0), initialized(false) {
	}

	int32_t Intern(const string &key) {
		if (!group_index) {
			group_index = make_uniq<unordered_map<string, int32_t>>();
		}
		auto it = group_index->find(key);
		if (it != group_index->end()) {
			return it->second;
		}
		auto id = (int32_t)group_labels.size();
		group_index->emplace(key, id);
		group_labels.push_back(key);
		return id;
	}
};

static LogicalType GetGlmmAugmentResultType() {
	child_list_t<LogicalType> row;
	row.push_back(make_pair("fitted", LogicalType::DOUBLE));
	row.push_back(make_pair("fitted_population", LogicalType::DOUBLE));
	row.push_back(make_pair("residual", LogicalType::DOUBLE));
	row.push_back(make_pair("pearson_residual", LogicalType::DOUBLE));
	row.push_back(make_pair("is_training", LogicalType::BOOLEAN));
	return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

static void GlmmAugmentInitialize(data_ptr_t state_p) {
	new (state_p) GlmmAugmentState();
}

static void GlmmAugmentDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAugmentState **)sdata.data;
	for (idx_t i = 0; i < count; i++) {
		states[sdata.sel->get_index(i)]->~GlmmAugmentState();
	}
}

static void GlmmAugmentUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat y_data, x_data;
	inputs[0].ToUnifiedFormat(count, y_data);
	inputs[1].ToUnifiedFormat(count, x_data);
	auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
	auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
	auto &x_child = ListVector::GetEntry(inputs[1]);
	auto x_child_data = FlatDataMut<double>(x_child);
	auto &x_child_validity = FlatValidityMut(x_child);
	auto &group_vec = inputs[2];

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAugmentState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		auto y_idx = y_data.sel->get_index(i);
		state.y.push_back(y_data.validity.RowIsValid(y_idx) ? y_values[y_idx]
		                                                    : std::numeric_limits<double>::quiet_NaN());
		auto group_value = group_vec.GetValue(i);
		state.group_ids.push_back(group_value.IsNull() ? -1 : state.Intern(group_value.ToString()));

		vector<double> x_row;
		auto x_idx = x_data.sel->get_index(i);
		if (x_data.validity.RowIsValid(x_idx)) {
			auto list_entry = x_list_data[x_idx];
			if (!state.initialized) {
				state.n_features = list_entry.length;
				state.initialized = true;
			} else if (list_entry.length != state.n_features) {
				throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
				                            list_entry.length);
			}
			x_row.resize(list_entry.length);
			for (idx_t j = 0; j < list_entry.length; j++) {
				const idx_t child_idx = list_entry.offset + j;
				x_row[j] = x_child_validity.RowIsValid(child_idx) ? x_child_data[child_idx]
				                                                  : std::numeric_limits<double>::quiet_NaN();
			}
		}
		state.x_rows.push_back(std::move(x_row));
	}
}

static void GlmmAugmentCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = (GlmmAugmentState **)source_data.data;
	auto targets = (GlmmAugmentState **)target_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (source.initialized && target.initialized && source.n_features != target.n_features) {
			throw InvalidInputException("Inconsistent feature count during combine");
		}
		if (source.initialized && !target.initialized) {
			target.n_features = source.n_features;
			target.initialized = true;
		}
		// Rows are appended in order; the source's group ids are remapped into the
		// target's dictionary.
		for (idx_t r = 0; r < source.y.size(); r++) {
			target.y.push_back(source.y[r]);
			auto g = source.group_ids[r];
			target.group_ids.push_back(g < 0 ? -1 : target.Intern(source.group_labels[g]));
			target.x_rows.push_back(source.x_rows[r]);
		}
	}
}

static void GlmmAugmentFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                idx_t count, idx_t offset) {
	auto &bind_data = aggr_input_data.bind_data->Cast<GlmmAggregateBindData>();
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (GlmmAugmentState **)sdata.data;

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t row = i + offset;
		if (!state.initialized || state.y.empty()) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		// Column-major design over ALL rows; a NULL x list becomes NaN features.
		const idx_t n_rows = state.y.size();
		vector<vector<double>> x_columns(state.n_features, vector<double>(n_rows));
		for (idx_t r = 0; r < n_rows; r++) {
			const auto &x_row = state.x_rows[r];
			for (idx_t j = 0; j < state.n_features; j++) {
				x_columns[j][r] = x_row.empty() ? std::numeric_limits<double>::quiet_NaN() : x_row[j];
			}
		}
		GlmmCall call;
		PrepareGlmmCall(bind_data, x_columns, state.n_features, call);
		AnofoxDataArray y_array {state.y.data(), nullptr, static_cast<size_t>(n_rows)};

		AnofoxGlmmAugmentResult aug {};
		AnofoxError error;
		bool ok = anofox_glmm_augment(y_array, call.x_arrays.data(), call.x_arrays.size(), state.group_ids.data(),
		                              n_rows, call.extra_ptrs.empty() ? nullptr : call.extra_ptrs.data(),
		                              call.extra_ptrs.size(), call.options, &aug, &error);
		if (!ok) {
			ThrowUnlessDegenerate("glmm_augment_agg", error);
			FlatVector::SetNull(result, row, true);
			continue;
		}
		if (aug.len != n_rows) {
			anofox_free_glmm_augment_result(&aug);
			throw InternalException("glmm_augment_agg: returned %llu rows for %llu inputs", (uint64_t)aug.len,
			                        (uint64_t)n_rows);
		}

		auto list_offset = ListVector::GetListSize(result);
		ListVector::Reserve(result, list_offset + n_rows);
		ListVector::SetListSize(result, list_offset + n_rows);
		ListVector::GetData(result)[row] = {list_offset, n_rows};
		auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
		const double *columns[] = {aug.fitted, aug.fitted_population, aug.residual, aug.pearson_residual};
		for (idx_t c = 0; c < 4; c++) {
			auto &vec = StructEntry(entries[c]);
			for (idx_t r = 0; r < n_rows; r++) {
				double v = columns[c][r];
				if (std::isfinite(v)) {
					FlatDataMut<double>(vec)[list_offset + r] = v;
				} else {
					FlatVector::SetNull(vec, list_offset + r, true);
				}
			}
		}
		auto training = FlatDataMut<bool>(StructEntry(entries[4]));
		for (idx_t r = 0; r < n_rows; r++) {
			training[list_offset + r] = aug.is_training[r];
		}
		anofox_free_glmm_augment_result(&aug);
	}
}

static unique_ptr<FunctionData> GlmmAugmentBind(ClientContext &context, AggregateBindFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
	auto result = ParseGlmmBindData(context, arguments, "glmm_augment_agg");
	SetReturnTypeOf(function, GetGlmmAugmentResultType());
	PostHogTelemetry::Instance().RecordFunctionCall("glmm_augment_agg");
	return std::move(result);
}

void RegisterGlmmAggregateFunction(ExtensionLoader &loader) {
	AggregateFunctionSet func_set("glmm_fit_agg");

	auto basic = AggregateFunction(
	    "glmm_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<GlmmAggregateState>, ANOFOX_AGG_INIT(GlmmAggInitialize), ANOFOX_GUARDED_UPDATE(GlmmAggUpdate, GlmmAggDestroy, GlmmAggInitialize),
	    GlmmAggCombine, ANOFOX_GUARDED_FINALIZE(GlmmAggFinalize, GlmmAggDestroy, GlmmAggInitialize), nullptr, ANOFOX_AGG_BIND(GlmmAggBind), GlmmAggDestroy);
	func_set.AddFunction(basic);

	auto with_opts = AggregateFunction(
	    "glmm_fit_agg",
	    {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY, LogicalType::ANY},
	    LogicalType::ANY, AggregateFunction::StateSize<GlmmAggregateState>, ANOFOX_AGG_INIT(GlmmAggInitialize), ANOFOX_GUARDED_UPDATE(GlmmAggUpdate, GlmmAggDestroy, GlmmAggInitialize),
	    GlmmAggCombine, ANOFOX_GUARDED_FINALIZE(GlmmAggFinalize, GlmmAggDestroy, GlmmAggInitialize), nullptr, ANOFOX_AGG_BIND(GlmmAggBind), GlmmAggDestroy);
	func_set.AddFunction(with_opts);

	CreateAggregateFunctionInfo info(func_set);
	FunctionDescription d1;
	d1.description = "Fits a generalized linear mixed model with a random intercept per group (one model across all groups); returns fixed effects, variance components and per-group random effects.";
	d1.examples = {"glmm_fit_agg(y, [x1, x2], store_id)"};
	d1.categories = {"mixed-models", "regression"};
	d1.parameter_names = {"y", "x", "group"};
	d1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
	info.descriptions.push_back(std::move(d1));
	FunctionDescription d2;
	d2.description = "Fits a generalized linear mixed model with a MAP of options (family, reml, random slopes, crossed groups, compute_inference, ...).";
	d2.examples = {"glmm_fit_agg(y, [x1, x2], store_id, {'family': 'poisson'})"};
	d2.categories = {"mixed-models", "regression"};
	d2.parameter_names = {"y", "x", "group", "options"};
	d2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY, LogicalType::ANY};
	info.descriptions.push_back(std::move(d2));
	loader.RegisterFunction(info);

	AggregateFunctionSet augment_set("glmm_augment_agg");
	for (bool with_options : {false, true}) {
		vector<LogicalType> args = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
		if (with_options) {
			args.push_back(LogicalType::ANY);
		}
		augment_set.AddFunction(AggregateFunction(
		    "glmm_augment_agg", args, LogicalType::ANY, AggregateFunction::StateSize<GlmmAugmentState>,
		    ANOFOX_AGG_INIT(GlmmAugmentInitialize),
		    ANOFOX_GUARDED_UPDATE(GlmmAugmentUpdate, GlmmAugmentDestroy, GlmmAugmentInitialize), GlmmAugmentCombine,
		    ANOFOX_GUARDED_FINALIZE(GlmmAugmentFinalize, GlmmAugmentDestroy, GlmmAugmentInitialize), nullptr,
		    ANOFOX_AGG_BIND(GlmmAugmentBind), GlmmAugmentDestroy));
	}
	CreateAggregateFunctionInfo augment_info(augment_set);
	FunctionDescription a1;
	a1.description = "Window aggregate: fits the same mixed model as glmm_fit_agg over the frame and returns one "
	                 "STRUCT per input row (in frame order) with fitted (conditional on the BLUPs, response scale), "
	                 "fitted_population (random effects at zero), residual (y - fitted), pearson_residual and "
	                 "is_training. Use it over a whole partition, as glmm_augment_by does.";
	a1.examples = {"glmm_augment_agg(y, [x1, x2], store_id) OVER ()"};
	a1.categories = {"mixed-models", "regression", "diagnostics"};
	a1.parameter_names = {"y", "x", "group"};
	a1.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY};
	augment_info.descriptions.push_back(std::move(a1));
	FunctionDescription a2 = augment_info.descriptions[0];
	a2.examples = {"glmm_augment_agg(y, [x1, x2], store_id, {'family': 'poisson'}) OVER ()"};
	a2.parameter_names = {"y", "x", "group", "options"};
	a2.parameter_types = {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY,
	                      LogicalType::ANY};
	augment_info.descriptions.push_back(std::move(a2));
	loader.RegisterFunction(augment_info);

}

} // namespace duckdb
