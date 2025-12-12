#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Poisson Aggregate State - accumulates y and x values for each group
//===--------------------------------------------------------------------===//
struct PoissonAggregateState {
    vector<double> y_values;
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

    PoissonAggregateState()
        : n_features(0), initialized(false), fit_intercept(true), link(ANOFOX_POISSON_LINK_LOG),
          max_iterations(100), tolerance(1e-8), compute_inference(false), confidence_level(0.95) {}

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
struct PoissonAggregateBindData : public FunctionData {
    bool fit_intercept = true;
    PoissonLink link = PoissonLink::LOG;
    uint32_t max_iterations = 100;
    double tolerance = 1e-8;
    bool compute_inference = false;
    double confidence_level = 0.95;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<PoissonAggregateBindData>();
        result->fit_intercept = fit_intercept;
        result->link = link;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->compute_inference = compute_inference;
        result->confidence_level = confidence_level;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PoissonAggregateBindData>();
        return fit_intercept == other.fit_intercept && link == other.link &&
               max_iterations == other.max_iterations && tolerance == other.tolerance &&
               compute_inference == other.compute_inference && confidence_level == other.confidence_level;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetPoissonAggResultType(bool compute_inference) {
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

    if (compute_inference) {
        children.push_back(make_pair("std_errors", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("z_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("p_values", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_lower", LogicalType::LIST(LogicalType::DOUBLE)));
        children.push_back(make_pair("ci_upper", LogicalType::LIST(LogicalType::DOUBLE)));
    }

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PoissonAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
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

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

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

static void PoissonAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
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

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.link = source.link;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
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

static void PoissonAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PoissonAggregateState **)sdata.data;

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

        AnofoxPoissonOptions options;
        options.fit_intercept = state.fit_intercept;
        options.link = state.link;
        options.max_iterations = state.max_iterations;
        options.tolerance = state.tolerance;
        options.compute_inference = state.compute_inference;
        options.confidence_level = state.confidence_level;

        AnofoxGlmFitResultCore core_result;
        AnofoxFitResultInference inference_result;
        AnofoxError error;

        bool success = anofox_poisson_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result,
                                          state.compute_inference ? &inference_result : nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        idx_t struct_idx = 0;

        SetListInResult(*struct_entries[struct_idx++], result_idx, core_result.coefficients,
                        core_result.coefficients_len);

        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.intercept;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.deviance;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.null_deviance;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.pseudo_r_squared;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.aic;
        FlatVector::GetData<double>(*struct_entries[struct_idx++])[result_idx] = core_result.dispersion;
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

        anofox_free_glm_result(&core_result);

        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PoissonAggBind(ClientContext &context, AggregateFunction &function,
                                                vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<PoissonAggregateBindData>();

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
        if (opts.poisson_link.has_value()) {
            result->link = opts.poisson_link.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
    }

    function.return_type = GetPoissonAggResultType(result->compute_inference);

    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPoissonAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_poisson_fit_agg");

    auto basic_func = AggregateFunction(
        "anofox_stats_poisson_fit_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY, AggregateFunction::StateSize<PoissonAggregateState>, PoissonAggInitialize, PoissonAggUpdate,
        PoissonAggCombine, PoissonAggFinalize, nullptr, PoissonAggBind, PoissonAggDestroy);
    func_set.AddFunction(basic_func);

    auto map_func = AggregateFunction(
        "anofox_stats_poisson_fit_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<PoissonAggregateState>, PoissonAggInitialize, PoissonAggUpdate, PoissonAggCombine,
        PoissonAggFinalize, nullptr, PoissonAggBind, PoissonAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    AggregateFunctionSet alias_set("poisson_fit_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
