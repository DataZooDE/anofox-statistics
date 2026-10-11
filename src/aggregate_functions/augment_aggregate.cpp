#include <cmath>
#include <cstring>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/error_dispatch.hpp"
#include "../include/ffi_enum_converters.hpp"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// ols_augment_agg / wls_augment_agg (#151)
//
// Window aggregates over a whole partition that return one STRUCT per input
// row, in frame order, with the per-observation diagnostics of an OLS / WLS
// fit (the integration contract's `obs` columns). The fit and every measure
// come from anofox_regression::diagnostics::augment through
// anofox_ols_augment / anofox_wls_augment; this file only collects the rows
// and lays the result out.
//
// A row is used for fitting when y, every feature (and, for WLS, a positive
// weight) are present; other rows get the prediction as `fitted` and NULL
// diagnostics.
//===--------------------------------------------------------------------===//
struct AugmentAggState {
    vector<double> y;               // NaN = row is not a training candidate
    vector<double> weight;          // prior weight (NaN for NULL); 1.0 for OLS
    vector<vector<double>> x_rows;  // empty for a NULL x list
    idx_t n_features;
    bool initialized;

    AugmentAggState() : n_features(0), initialized(false) {}
};

struct AugmentAggBindData : public FunctionData {
    bool weighted = false;
    bool fit_intercept = true;
    SolverType solver = SolverType::SVD;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<AugmentAggBindData>();
        result->weighted = weighted;
        result->fit_intercept = fit_intercept;
        result->solver = solver;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<AugmentAggBindData>();
        return weighted == other.weighted && fit_intercept == other.fit_intercept && solver == other.solver &&
               null_policy == other.null_policy;
    }
};

//! LIST(STRUCT(...)) with the `obs` columns the aggregate owns.
static LogicalType GetAugmentAggResultType() {
    child_list_t<LogicalType> row;
    row.push_back(make_pair("fitted", LogicalType::DOUBLE));
    row.push_back(make_pair("residual", LogicalType::DOUBLE));
    row.push_back(make_pair("resid_type", LogicalType::VARCHAR));
    row.push_back(make_pair("std_residual", LogicalType::DOUBLE));
    row.push_back(make_pair("stud_residual", LogicalType::DOUBLE));
    row.push_back(make_pair("leverage", LogicalType::DOUBLE));
    row.push_back(make_pair("cooks_d", LogicalType::DOUBLE));
    row.push_back(make_pair("dffits", LogicalType::DOUBLE));
    row.push_back(make_pair("weight", LogicalType::DOUBLE));
    row.push_back(make_pair("is_training", LogicalType::BOOLEAN));
    row.push_back(make_pair("n_params", LogicalType::INTEGER));
    return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

static void AugmentAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) AugmentAggState();
}

static void AugmentAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AugmentAggState **)sdata.data;
    for (idx_t i = 0; i < count; i++) {
        states[sdata.sel->get_index(i)]->~AugmentAggState();
    }
}

// Inputs: y, x [, weight] [, options]
static void AugmentAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                             idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<AugmentAggBindData>();

    UnifiedVectorFormat y_data, x_data, w_data;
    inputs[0].ToUnifiedFormat(count, y_data);
    inputs[1].ToUnifiedFormat(count, x_data);
    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto x_list_data = ListVector::GetData(inputs[1]);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatVector::GetData<double>(x_child);
    auto &x_child_validity = FlatVector::Validity(x_child);
    const double *w_values = nullptr;
    if (bind_data.weighted) {
        inputs[2].ToUnifiedFormat(count, w_data);
        w_values = UnifiedVectorFormat::GetData<double>(w_data);
    }

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AugmentAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        auto y_idx = y_data.sel->get_index(i);
        double y_val = y_data.validity.RowIsValid(y_idx) ? y_values[y_idx] : std::nan("");

        double w_val = 1.0;
        if (bind_data.weighted) {
            auto w_idx = w_data.sel->get_index(i);
            w_val = w_data.validity.RowIsValid(w_idx) ? w_values[w_idx] : std::nan("");
        }

        vector<double> x_row;
        auto x_idx = x_data.sel->get_index(i);
        if (x_data.validity.RowIsValid(x_idx)) {
            auto list_entry = x_list_data[x_idx];
            idx_t n_features = list_entry.length;
            if (!state.initialized) {
                state.n_features = n_features;
                state.initialized = true;
            } else if (n_features != state.n_features) {
                throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                            n_features);
            }
            // A NULL element's flat-vector slot is uninitialized: read it as NaN,
            // which also keeps the row out of the fit (#95).
            x_row.resize(n_features);
            for (idx_t j = 0; j < n_features; j++) {
                idx_t child_pos = list_entry.offset + j;
                x_row[j] = x_child_validity.RowIsValid(child_pos) ? x_child_data[child_pos] : std::nan("");
            }
            if (bind_data.null_policy == NullPolicy::DROP_Y_ZERO_X) {
                for (auto v : x_row) {
                    if (v == 0.0) {
                        y_val = std::nan("");
                        break;
                    }
                }
            }
        }

        state.y.push_back(y_val);
        state.weight.push_back(w_val);
        state.x_rows.push_back(std::move(x_row));
    }
}

static void AugmentAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                              idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);
    auto sources = (AugmentAggState **)source_data.data;
    auto targets = (AugmentAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (source.initialized && target.initialized && source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }
        if (source.initialized && !target.initialized) {
            target.n_features = source.n_features;
            target.initialized = true;
        }
        if (target.y.empty()) {
            target.y = CombineTake(source.y, aggr_input_data);
            target.weight = CombineTake(source.weight, aggr_input_data);
            target.x_rows = CombineTake(source.x_rows, aggr_input_data);
            continue;
        }
        target.y.insert(target.y.end(), source.y.begin(), source.y.end());
        target.weight.insert(target.weight.end(), source.weight.begin(), source.weight.end());
        target.x_rows.insert(target.x_rows.end(), source.x_rows.begin(), source.x_rows.end());
    }
}

static inline void WriteDouble(Vector &vec, idx_t idx, double v) {
    if (std::isfinite(v)) {
        FlatVector::GetData<double>(vec)[idx] = v;
    } else {
        FlatVector::SetNull(vec, idx, true);
    }
}

static void AugmentAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                               idx_t offset) {
    auto &bind_data = aggr_input_data.bind_data->Cast<AugmentAggBindData>();
    const char *fn_name = bind_data.weighted ? "wls_augment_agg" : "ols_augment_agg";

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (AugmentAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.n_features == 0) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Column-major design over ALL rows; a NULL x list becomes NaN features.
        idx_t n_rows = state.y.size();
        vector<vector<double>> x_cols(state.n_features, vector<double>(n_rows));
        for (idx_t r = 0; r < n_rows; r++) {
            const auto &row = state.x_rows[r];
            for (idx_t j = 0; j < state.n_features; j++) {
                x_cols[j][r] = row.empty() ? std::nan("") : row[j];
            }
        }
        vector<AnofoxDataArray> x_arrays;
        for (auto &col : x_cols) {
            x_arrays.push_back({col.data(), nullptr, col.size()});
        }
        AnofoxDataArray y_array {state.y.data(), nullptr, n_rows};

        AnofoxAugmentResult aug;
        std::memset(&aug, 0, sizeof(aug));
        AnofoxError error;
        bool ok;
        if (bind_data.weighted) {
            AnofoxDataArray w_array {state.weight.data(), nullptr, n_rows};
            AnofoxWlsOptions options;
            options.fit_intercept = bind_data.fit_intercept;
            options.compute_inference = false;
            options.confidence_level = 0.95;
            options.solver = ConvertSolverType(bind_data.solver);
            options.hc_type = ANOFOX_HC_NONE;
            ok = anofox_wls_augment(y_array, x_arrays.data(), x_arrays.size(), w_array, options, &aug, &error);
        } else {
            AnofoxOlsOptions options;
            options.fit_intercept = bind_data.fit_intercept;
            options.compute_inference = false;
            options.confidence_level = 0.95;
            options.solver = ConvertSolverType(bind_data.solver);
            options.hc_type = ANOFOX_HC_NONE;
            ok = anofox_ols_augment(y_array, x_arrays.data(), x_arrays.size(), options, &aug, &error);
        }
        if (!ok) {
            ThrowUnlessDegenerate(fn_name, error);
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }
        if (aug.len != n_rows) {
            anofox_free_augment_result(&aug);
            throw InternalException("%s: augment returned %llu rows for %llu inputs", fn_name, (uint64_t)aug.len,
                                    (uint64_t)n_rows);
        }

        auto *list_data = ListVector::GetData(result);
        auto list_offset = ListVector::GetListSize(result);
        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = n_rows;
        ListVector::Reserve(result, list_offset + n_rows);
        ListVector::SetListSize(result, list_offset + n_rows);

        auto &entries = StructVector::GetEntries(ListVector::GetEntry(result));
        auto &fitted_vec = *entries[0];
        auto &residual_vec = *entries[1];
        auto &resid_type_vec = *entries[2];
        auto &std_vec = *entries[3];
        auto &stud_vec = *entries[4];
        auto &leverage_vec = *entries[5];
        auto &cooks_vec = *entries[6];
        auto &dffits_vec = *entries[7];
        auto &weight_vec = *entries[8];
        auto &training_vec = *entries[9];
        auto &n_params_vec = *entries[10];

        string resid_type(aug.resid_type, strnlen(aug.resid_type, ANOFOX_MODEL_NAME_LEN));
        auto n_params = static_cast<int32_t>(aug.n_params);

        for (idx_t r = 0; r < n_rows; r++) {
            idx_t c = list_offset + r;
            bool used = aug.used[r];
            WriteDouble(fitted_vec, c, aug.fitted[r]);
            WriteDouble(residual_vec, c, aug.residual[r]);
            WriteDouble(std_vec, c, aug.std_residual[r]);
            WriteDouble(stud_vec, c, aug.stud_residual[r]);
            WriteDouble(leverage_vec, c, aug.leverage[r]);
            WriteDouble(cooks_vec, c, aug.cooks_d[r]);
            WriteDouble(dffits_vec, c, aug.dffits[r]);
            WriteDouble(weight_vec, c, state.weight[r]);
            FlatVector::GetData<string_t>(resid_type_vec)[c] = StringVector::AddString(resid_type_vec, resid_type);
            FlatVector::GetData<bool>(training_vec)[c] = used;
            FlatVector::GetData<int32_t>(n_params_vec)[c] = n_params;
        }

        anofox_free_augment_result(&aug);
    }
}

static unique_ptr<FunctionData> AugmentAggBindImpl(ClientContext &context, AggregateFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments, bool weighted) {
    const char *fn_name = weighted ? "wls_augment_agg" : "ols_augment_agg";
    auto result = make_uniq<AugmentAggBindData>();
    result->weighted = weighted;
    idx_t options_idx = weighted ? 3 : 2;
    if (arguments.size() > options_idx) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[options_idx], fn_name,
                                                              {"fit_intercept", "null_policy", "solver"});
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
        if (opts.solver.has_value()) {
            result->solver = opts.solver.value();
        }
    }
    function.return_type = GetAugmentAggResultType();
    PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
    return std::move(result);
}

static unique_ptr<FunctionData> OlsAugmentAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    return AugmentAggBindImpl(context, function, arguments, false);
}

static unique_ptr<FunctionData> WlsAugmentAggBind(ClientContext &context, AggregateFunction &function,
                                                  vector<unique_ptr<Expression>> &arguments) {
    return AugmentAggBindImpl(context, function, arguments, true);
}

static AggregateFunction MakeAugmentAgg(const string &name, vector<LogicalType> args, bind_aggregate_function_t bind) {
    return AggregateFunction(name, std::move(args), LogicalType::ANY, AggregateFunction::StateSize<AugmentAggState>,
                             AugmentAggInitialize,
                             ANOFOX_GUARDED_UPDATE(AugmentAggUpdate, AugmentAggDestroy, AugmentAggInitialize),
                             AugmentAggCombine,
                             ANOFOX_GUARDED_FINALIZE(AugmentAggFinalize, AugmentAggDestroy, AugmentAggInitialize),
                             nullptr, bind, AugmentAggDestroy);
}

static FunctionDescription AugmentDescription(const string &description, const string &example,
                                              vector<string> names, vector<LogicalType> types) {
    FunctionDescription d;
    d.description = description;
    d.examples = {example};
    d.categories = {"regression", "diagnostics"};
    d.parameter_names = std::move(names);
    d.parameter_types = std::move(types);
    return d;
}

void RegisterAugmentAggregateFunctions(ExtensionLoader &loader) {
    auto list_double = LogicalType::LIST(LogicalType::DOUBLE);

    // ols_augment_agg(y, x [, options])
    {
        AggregateFunctionSet set("ols_augment_agg");
        set.AddFunction(MakeAugmentAgg("ols_augment_agg", {LogicalType::DOUBLE, list_double}, OlsAugmentAggBind));
        set.AddFunction(MakeAugmentAgg("ols_augment_agg", {LogicalType::DOUBLE, list_double, LogicalType::ANY},
                                       OlsAugmentAggBind));
        CreateAggregateFunctionInfo info(std::move(set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        const string desc = "Fits OLS over a partition (rows with y and every feature present) and returns one STRUCT "
                            "per row, in frame order, with fitted, residual, resid_type, std_residual, stud_residual, "
                            "leverage, cooks_d, dffits, weight, is_training and n_params (broom augment).";
        info.descriptions.push_back(AugmentDescription(desc, "ols_augment_agg(y, [x1, x2]) OVER (PARTITION BY g)",
                                                       {"y", "x"}, {LogicalType::DOUBLE, list_double}));
        info.descriptions.push_back(AugmentDescription(desc,
                                                       "ols_augment_agg(y, [x1, x2], {'fit_intercept': true}) OVER ()",
                                                       {"y", "x", "options"},
                                                       {LogicalType::DOUBLE, list_double, LogicalType::ANY}));
        loader.RegisterFunction(std::move(info));
    }

    // wls_augment_agg(y, x, weight [, options])
    {
        AggregateFunctionSet set("wls_augment_agg");
        set.AddFunction(MakeAugmentAgg("wls_augment_agg", {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE},
                                       WlsAugmentAggBind));
        set.AddFunction(MakeAugmentAgg("wls_augment_agg",
                                       {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE, LogicalType::ANY},
                                       WlsAugmentAggBind));
        CreateAggregateFunctionInfo info(std::move(set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        const string desc = "Fits WLS over a partition (rows with y, every feature and a positive weight) and returns "
                            "one STRUCT per row, in frame order, with fitted, residual, resid_type, std_residual, "
                            "stud_residual, leverage, cooks_d, dffits, weight, is_training and n_params (broom augment).";
        info.descriptions.push_back(AugmentDescription(desc, "wls_augment_agg(y, [x1, x2], w) OVER (PARTITION BY g)",
                                                       {"y", "x", "weight"},
                                                       {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE}));
        info.descriptions.push_back(AugmentDescription(
            desc, "wls_augment_agg(y, [x1, x2], w, {'fit_intercept': true}) OVER ()", {"y", "x", "weight", "options"},
            {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE, LogicalType::ANY}));
        loader.RegisterFunction(std::move(info));
    }
}

} // namespace duckdb
