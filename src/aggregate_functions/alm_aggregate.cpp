#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// ALM Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct AlmAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;

    // Options
    bool fit_intercept;
    AnofoxAlmDistribution distribution;
    AnofoxAlmLoss loss;
    uint32_t max_iterations;
    double tolerance;
    double quantile;
    double role_trim;
    bool compute_inference;
    double confidence_level;

    AlmAggregateState()
        : n_features(0), initialized(false), fit_intercept(true), distribution(ANOFOX_ALM_DIST_NORMAL),
          loss(ANOFOX_ALM_LOSS_LIKELIHOOD), max_iterations(100), tolerance(1e-8), quantile(0.5), role_trim(0.05),
          compute_inference(false), confidence_level(0.95) {}

    void Reset() {
        y_values.clear();
        x_columns.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data for options
//===--------------------------------------------------------------------===//
struct AlmAggregateBindData : public FunctionData {
    bool fit_intercept = true;
    AlmDistribution distribution = AlmDistribution::NORMAL;
    AlmLoss loss = AlmLoss::LIKELIHOOD;
    uint32_t max_iterations = 100;
    double tolerance = 1e-8;
    double quantile = 0.5;
    double role_trim = 0.05;
    bool compute_inference = false;
    double confidence_level = 0.95;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<AlmAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->distribution = distribution;
        result->loss = loss;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->quantile = quantile;
        result->role_trim = role_trim;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<AlmAggregateBindData>();
        return fit_intercept == other.fit_intercept && distribution == other.distribution && loss == other.loss &&
               max_iterations == other.max_iterations && tolerance == other.tolerance && quantile == other.quantile &&
               role_trim == other.role_trim && compute_inference == other.compute_inference &&
               confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetAlmAggResultType(bool compute_inference) {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("coefficients", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("intercept", LogicalType::DOUBLE));
    children.push_back(make_pair("log_likelihood", LogicalType::DOUBLE));
    children.push_back(make_pair("aic", LogicalType::DOUBLE));
    children.push_back(make_pair("bic", LogicalType::DOUBLE));
    children.push_back(make_pair("scale", LogicalType::DOUBLE));
    children.push_back(make_pair("n_observations", LogicalType::BIGINT));
    children.push_back(make_pair("n_features", LogicalType::BIGINT));
    children.push_back(make_pair("iterations", LogicalType::INTEGER));

    if (compute_inference) {
        children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("t_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
    }

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void AlmAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) AlmAggregateState();
}

static void AlmAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AlmAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~AlmAggregateState();
    }
}

static AnofoxAlmDistribution ConvertAlmDistribution(AlmDistribution dist) {
    return static_cast<AnofoxAlmDistribution>(static_cast<int>(dist));
}

static AnofoxAlmLoss ConvertAlmLoss(AlmLoss loss) {
    return static_cast<AnofoxAlmLoss>(static_cast<int>(loss));
}

static void AlmAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<AlmAggregateBindData>();

    UnifiedVectorFormat y_data;
    UnifiedVectorFormat x_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AlmAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        state.fit_intercept = bind_data.fit_intercept;
        state.distribution = ConvertAlmDistribution(bind_data.distribution);
        state.loss = ConvertAlmLoss(bind_data.loss);
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.quantile = bind_data.quantile;
        state.role_trim = bind_data.role_trim;
        state.compute_inference = bind_data.compute_inference;
        state.confidence_level = bind_data.confidence_level;

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

        if (!state.initialized) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
            state.initialized = true;
        }

        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        state.y_values.push_back(y_val);

        for (idx_t j = 0; j < n_features; j++) {
            double x_val = x_child_data[list_entry.offset + j];
            state.x_columns[j].push_back(x_val);
        }
    }
}

static void AlmAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (AlmAggregateState **)source_data.data;
    auto targets = (AlmAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.distribution = source.distribution;
            target.loss = source.loss;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.quantile = source.quantile;
            target.role_trim = source.role_trim;
            target.compute_inference = source.compute_inference;
            target.confidence_level = source.confidence_level;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());

        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_columns[j].insert(target.x_columns[j].end(), source.x_columns[j].begin(),
                                       source.x_columns[j].end());
        }
    }
}

static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void AlmAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AlmAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t min_obs = state.fit_intercept ? state.n_features + 1 : state.n_features;
        if (state.y_values.size() <= min_obs) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
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

        AnofoxAlmOptions options;
        options.fit_intercept = state.fit_intercept;
        options.distribution = state.distribution;
        options.loss = state.loss;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;
        options.quantile = state.quantile;
        options.role_trim = state.role_trim;
        options.compute_inference = state.compute_inference;
        options.confidence_level = state.confidence_level;

        AnofoxAlmFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_alm_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                      state.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;

        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);

        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.log_likelihood;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.aic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.bic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.scale;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_observations;
        FlatVector::GetData<int64_t>(*struct_entries[struct_idx++])[result_idx] = core_result.n_features;
        FlatVector::GetData<int32_t>(*struct_entries[struct_idx++])[result_idx] = core_result.iterations;

        if (state.compute_inference) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.std_errors,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.t_values,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.p_values,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_lower,
                            inference_result.len);
            SetListInResult(*struct_entries[struct_idx++], result_idx, inference_result.ci_upper,
                            inference_result.len);

            anofox_free_result_inference(&inference_result);
        }

        anofox_free_alm_result(&core_result);

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> AlmAggBind(ClientContext &context, AggregateFunction &function,
                                           vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<AlmAggregateBindData>();

    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.compute_inference.has_value()) {
            result->compute_inference = opts.compute_inference.value();
        }
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.distribution.has_value()) {
            result->distribution = opts.distribution.value();
        }
        if (opts.loss.has_value()) {
            result->loss = opts.loss.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.quantile.has_value()) {
            result->quantile = opts.quantile.value();
        }
        if (opts.role_trim.has_value()) {
            result->role_trim = opts.role_trim.value();
        }
    }

    function.return_type = GetAlmAggResultType(result->compute_inference);

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterAlmAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_alm_fit_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_alm_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)}, LogicalType::ANY,
        AggregateFunction::StateSize<AlmAggregateState>, AlmAggInitialize, AlmAggUpdate, AlmAggCombine, AlmAggFinalize,
        nullptr, AlmAggBind, AlmAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_alm_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY},
        LogicalType::ANY, AggregateFunction::StateSize<AlmAggregateState>, AlmAggInitialize, AlmAggUpdate,
        AlmAggCombine, AlmAggFinalize, nullptr, AlmAggBind, AlmAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("alm_fit_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
