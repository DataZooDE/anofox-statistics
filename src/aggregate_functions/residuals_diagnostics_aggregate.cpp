#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/common/types/data_chunk.hpp"

#include "../include/anofox_stats_ffi.h"

#include <vector>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Residuals Diagnostics Aggregate State
//===--------------------------------------------------------------------===//
struct ResidualsDiagnosticsAggregateState {
    vector<double> y_values;
    vector<double> y_hat_values;
    vector<vector<double>> x_columns;
    idx_t n_features;
    bool initialized;
    bool has_x;

    ResidualsDiagnosticsAggregateState()
        : n_features(0), initialized(false), has_x(false) {}

    void Reset() {
        y_values.clear();
        y_hat_values.clear();
        x_columns.clear();
        n_features = 0;
        initialized = false;
        has_x = false;
    }
};

//===--------------------------------------------------------------------===//
// Result type definition
//===--------------------------------------------------------------------===//
static LogicalType GetResidualsDiagnosticsAggResultType() {
    child_list_t<LogicalType> children;

    children.push_back(make_pair("raw", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("standardized", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("studentized", LogicalType::LIST(LogicalType::DOUBLE)));
    children.push_back(make_pair("leverage", LogicalType::LIST(LogicalType::DOUBLE)));

    return LogicalType::STRUCT(std::move(children));
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void ResidualsDiagnosticsAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) ResidualsDiagnosticsAggregateState();
}

static void ResidualsDiagnosticsAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ResidualsDiagnosticsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~ResidualsDiagnosticsAggregateState();
    }
}

// Basic update: just y and y_hat
static void ResidualsDiagnosticsAggUpdateBasic(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                                Vector &state_vector, idx_t count) {
    UnifiedVectorFormat y_data, y_hat_data;
    inputs[0].ToUnifiedFormat(count, y_data);      // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, y_hat_data);  // y_hat: DOUBLE

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto y_hat_values = UnifiedVectorFormat::GetData<double>(y_hat_data);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ResidualsDiagnosticsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;

        auto y_idx = y_data.sel->get_index(i);
        auto y_hat_idx = y_hat_data.sel->get_index(i);

        if (!y_data.validity.RowIsValid(y_idx) || !y_hat_data.validity.RowIsValid(y_hat_idx)) {
            continue;
        }

        double y_val = y_values[y_idx];
        double y_hat_val = y_hat_values[y_hat_idx];

        if (!std::isnan(y_val) && !std::isnan(y_hat_val)) {
            state.y_values.push_back(y_val);
            state.y_hat_values.push_back(y_hat_val);
        }
    }
}

// Full update: y, y_hat, and x
static void ResidualsDiagnosticsAggUpdateFull(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                               Vector &state_vector, idx_t count) {
    UnifiedVectorFormat y_data, y_hat_data, x_data;
    inputs[0].ToUnifiedFormat(count, y_data);      // y: DOUBLE
    inputs[1].ToUnifiedFormat(count, y_hat_data);  // y_hat: DOUBLE
    inputs[2].ToUnifiedFormat(count, x_data);      // x: LIST(DOUBLE)

    auto y_values = UnifiedVectorFormat::GetData<double>(y_data);
    auto y_hat_values = UnifiedVectorFormat::GetData<double>(y_hat_data);
    auto x_list_data = ListVector::GetData(inputs[2]);
    auto &x_child = ListVector::GetEntry(inputs[2]);
    auto x_child_data = FlatVector::GetData<double>(x_child);

    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ResidualsDiagnosticsAggregateState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.initialized = true;
        state.has_x = true;

        auto y_idx = y_data.sel->get_index(i);
        auto y_hat_idx = y_hat_data.sel->get_index(i);
        auto x_idx = x_data.sel->get_index(i);

        if (!y_data.validity.RowIsValid(y_idx) ||
            !y_hat_data.validity.RowIsValid(y_hat_idx) ||
            !x_data.validity.RowIsValid(x_idx)) {
            continue;
        }

        double y_val = y_values[y_idx];
        double y_hat_val = y_hat_values[y_hat_idx];

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize x_columns on first valid row
        if (state.n_features == 0 && n_features > 0) {
            state.n_features = n_features;
            state.x_columns.resize(n_features);
        }

        if (n_features != state.n_features) {
            throw InvalidInputException(
                "Inconsistent feature count: expected %lu, got %lu",
                state.n_features, n_features);
        }

        if (!std::isnan(y_val) && !std::isnan(y_hat_val)) {
            state.y_values.push_back(y_val);
            state.y_hat_values.push_back(y_hat_val);

            for (idx_t j = 0; j < n_features; j++) {
                double x_val = x_child_data[list_entry.offset + j];
                state.x_columns[j].push_back(x_val);
            }
        }
    }
}

static void ResidualsDiagnosticsAggCombine(Vector &source_vector, Vector &target_vector,
                                            AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (ResidualsDiagnosticsAggregateState **)source_data.data;
    auto targets = (ResidualsDiagnosticsAggregateState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_values = std::move(source.y_values);
            target.y_hat_values = std::move(source.y_hat_values);
            target.x_columns = std::move(source.x_columns);
            target.n_features = source.n_features;
            target.initialized = true;
            target.has_x = source.has_x;
            continue;
        }

        target.y_values.insert(target.y_values.end(), source.y_values.begin(), source.y_values.end());
        target.y_hat_values.insert(target.y_hat_values.end(), source.y_hat_values.begin(), source.y_hat_values.end());

        if (source.has_x && target.has_x) {
            for (idx_t j = 0; j < target.n_features; j++) {
                target.x_columns[j].insert(target.x_columns[j].end(),
                                           source.x_columns[j].begin(),
                                           source.x_columns[j].end());
            }
        }
    }
}

// Helper to set a list in STRUCT result
static void SetListInResult(Vector &list_vec, idx_t row, double *data, size_t len) {
    if (data == nullptr || len == 0) {
        FlatVector::SetNull(list_vec, row, true);
        return;
    }
    auto &child = ListVector::GetEntry(list_vec);
    auto offset = ListVector::GetListSize(list_vec);
    ListVector::SetListSize(list_vec, offset + len);
    auto vec_data = FlatVector::GetData<double>(child);
    for (size_t i = 0; i < len; i++) {
        vec_data[offset + i] = data[i];
    }
    ListVector::GetData(list_vec)[row] = {offset, (idx_t)len};
}

static void ResidualsDiagnosticsAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data,
                                             Vector &result, idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (ResidualsDiagnosticsAggregateState **)sdata.data;

    auto &struct_entries = StructVector::GetEntries(result);

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        if (!state.initialized || state.y_values.size() < 3) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Prepare FFI data
        AnofoxDataArray y_array;
        y_array.data = state.y_values.data();
        y_array.validity = nullptr;
        y_array.len = state.y_values.size();

        AnofoxDataArray y_hat_array;
        y_hat_array.data = state.y_hat_values.data();
        y_hat_array.validity = nullptr;
        y_hat_array.len = state.y_hat_values.size();

        vector<AnofoxDataArray> x_arrays;
        const AnofoxDataArray* x_ptr = nullptr;
        size_t x_count = 0;

        if (state.has_x && state.n_features > 0) {
            for (auto &col : state.x_columns) {
                AnofoxDataArray arr;
                arr.data = col.data();
                arr.validity = nullptr;
                arr.len = col.size();
                x_arrays.push_back(arr);
            }
            x_ptr = x_arrays.data();
            x_count = x_arrays.size();
        }

        AnofoxResidualsResult resid_result;
        AnofoxError error;

        bool success = anofox_compute_residuals(
            y_array,
            y_hat_array,
            x_ptr,
            x_count,
            std::nan(""),  // residual_std_error computed internally
            state.has_x,   // include_studentized only if we have x
            &resid_result,
            &error
        );

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Fill STRUCT result
        idx_t struct_idx = 0;

        SetListInResult(*struct_entries[struct_idx++], result_idx, resid_result.raw, resid_result.len);

        if (resid_result.has_standardized) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, resid_result.standardized, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], result_idx, true);
        }

        if (resid_result.has_studentized) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, resid_result.studentized, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], result_idx, true);
        }

        if (resid_result.has_leverage) {
            SetListInResult(*struct_entries[struct_idx++], result_idx, resid_result.leverage, resid_result.len);
        } else {
            FlatVector::SetNull(*struct_entries[struct_idx++], result_idx, true);
        }

        anofox_free_residuals(&resid_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> ResidualsDiagnosticsAggBind(ClientContext &context, AggregateFunction &function,
                                                             vector<unique_ptr<Expression>> &arguments) {
    function.return_type = GetResidualsDiagnosticsAggResultType();
    return nullptr;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterResidualsDiagnosticsAggregateFunction(ExtensionLoader &loader) {
    AggregateFunctionSet func_set("anofox_stats_residuals_diagnostics_agg");

    // Basic version: anofox_stats_residuals_diagnostics_agg(y, y_hat)
    auto basic_func = AggregateFunction(
        "anofox_stats_residuals_diagnostics_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE},
        LogicalType::ANY,  // Set in bind
        AggregateFunction::StateSize<ResidualsDiagnosticsAggregateState>,
        ResidualsDiagnosticsAggInitialize,
        ResidualsDiagnosticsAggUpdateBasic,
        ResidualsDiagnosticsAggCombine,
        ResidualsDiagnosticsAggFinalize,
        nullptr,  // simple_update
        ResidualsDiagnosticsAggBind,
        ResidualsDiagnosticsAggDestroy
    );
    func_set.AddFunction(basic_func);

    // Full version: anofox_stats_residuals_diagnostics_agg(y, y_hat, x)
    auto full_func = AggregateFunction(
        "anofox_stats_residuals_diagnostics_agg",
        {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
        LogicalType::ANY,
        AggregateFunction::StateSize<ResidualsDiagnosticsAggregateState>,
        ResidualsDiagnosticsAggInitialize,
        ResidualsDiagnosticsAggUpdateFull,
        ResidualsDiagnosticsAggCombine,
        ResidualsDiagnosticsAggFinalize,
        nullptr,
        ResidualsDiagnosticsAggBind,
        ResidualsDiagnosticsAggDestroy
    );
    func_set.AddFunction(full_func);

    loader.RegisterFunction(func_set);

    // Also register short aliases
    AggregateFunctionSet alias_set("residuals_diagnostics_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(full_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
