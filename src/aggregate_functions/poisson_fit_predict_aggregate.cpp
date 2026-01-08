#include <cmath>
#include <vector>

#include "duckdb.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "../include/anofox_stats_ffi.h"
#include "../include/map_options_parser.hpp"
#include "telemetry.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Poisson Fit Predict Aggregate State
// Accumulates ALL rows for output, tracks which are training vs prediction
//===--------------------------------------------------------------------===//
struct PoissonFitPredictAggState {
    // Training data (rows where y is NOT NULL)
    vector<double> y_train;
    vector<vector<double>> x_train;

    // ALL rows' data for output
    vector<double> y_all;       // Original y values (NaN for NULL)
    vector<bool> y_is_null;     // True if y was NULL in input
    vector<bool> is_training;   // True if row was used for training
    vector<vector<double>> x_all;  // x values for ALL rows

    idx_t n_features;
    bool initialized;

    // Options
    bool fit_intercept;
    AnofoxPoissonLink link;
    uint32_t max_iterations;
    double tolerance;
    double confidence_level;
    NullPolicy null_policy;

    PoissonFitPredictAggState()
        : n_features(0), initialized(false), fit_intercept(true), link(ANOFOX_POISSON_LINK_LOG),
          max_iterations(100), tolerance(1e-8), confidence_level(0.95), null_policy(NullPolicy::DROP) {}

    void Reset() {
        y_train.clear();
        x_train.clear();
        y_all.clear();
        y_is_null.clear();
        is_training.clear();
        x_all.clear();
        n_features = 0;
        initialized = false;
    }
};

//===--------------------------------------------------------------------===//
// Bind Data
//===--------------------------------------------------------------------===//
struct PoissonFitPredictAggBindData : public FunctionData {
    bool fit_intercept = true;
    PoissonLink link = PoissonLink::LOG;
    uint32_t max_iterations = 100;
    double tolerance = 1e-8;
    double confidence_level = 0.95;
    NullPolicy null_policy = NullPolicy::DROP;

    unique_ptr<FunctionData> Copy() const override {
        auto result = make_uniq<PoissonFitPredictAggBindData>();
        result->fit_intercept = fit_intercept;
        result->link = link;
        result->max_iterations = max_iterations;
        result->tolerance = tolerance;
        result->confidence_level = confidence_level;
        result->null_policy = null_policy;
        return std::move(result);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<PoissonFitPredictAggBindData>();
        return fit_intercept == other.fit_intercept && link == other.link &&
               max_iterations == other.max_iterations && tolerance == other.tolerance &&
               confidence_level == other.confidence_level && null_policy == other.null_policy;
    }
};

//===--------------------------------------------------------------------===//
// Result type: LIST(STRUCT(y, x, yhat, yhat_lower, yhat_upper, is_training))
//===--------------------------------------------------------------------===//
static LogicalType GetPoissonFitPredictAggResultType() {
    child_list_t<LogicalType> row_children;
    row_children.push_back(make_pair("y", LogicalType::DOUBLE));
    row_children.push_back(make_pair("x", LogicalType::LIST(LogicalType::DOUBLE)));
    row_children.push_back(make_pair("yhat", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    row_children.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    row_children.push_back(make_pair("is_training", LogicalType::BOOLEAN));

    auto row_struct = LogicalType::STRUCT(std::move(row_children));
    return LogicalType::LIST(row_struct);
}

//===--------------------------------------------------------------------===//
// Helper functions
//===--------------------------------------------------------------------===//
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

// Apply inverse link function to get predicted mean (mu) from linear predictor (eta)
static double ApplyInverseLink(double linear_pred, AnofoxPoissonLink link) {
    switch (link) {
    case ANOFOX_POISSON_LINK_LOG:
        return std::exp(linear_pred);  // mu = exp(eta)
    case ANOFOX_POISSON_LINK_IDENTITY:
        return linear_pred;            // mu = eta
    case ANOFOX_POISSON_LINK_SQRT:
        return linear_pred * linear_pred;  // mu = eta^2
    default:
        return std::exp(linear_pred);
    }
}

// Get t critical value for confidence interval
static double GetTCritical(double confidence_level, size_t df) {
    // Approximate t-critical value using inverse normal approximation for large df
    // For df >= 30, t ~ z
    double alpha = (1.0 - confidence_level) / 2.0;
    // Use 1.96 for 95% CI as a simple approximation
    if (std::abs(confidence_level - 0.95) < 0.001) {
        return 1.96;
    } else if (std::abs(confidence_level - 0.99) < 0.001) {
        return 2.576;
    } else if (std::abs(confidence_level - 0.90) < 0.001) {
        return 1.645;
    }
    // Fallback to approximate formula
    return 1.96; // Default to 95% CI
}

//===--------------------------------------------------------------------===//
// Aggregate function operations
//===--------------------------------------------------------------------===//

static void PoissonFitPredictAggInitialize(const AggregateFunction &, data_ptr_t state_p) {
    new (state_p) PoissonFitPredictAggState();
}

static void PoissonFitPredictAggDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PoissonFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        state.~PoissonFitPredictAggState();
    }
}

// Update: accumulate ALL rows, track which are training vs prediction
static void PoissonFitPredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                       Vector &state_vector, idx_t count) {
    auto &bind_data = aggr_input_data.bind_data->Cast<PoissonFitPredictAggBindData>();

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
    auto states = (PoissonFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];

        // Copy options from bind data
        state.fit_intercept = bind_data.fit_intercept;
        state.link = ConvertPoissonLink(bind_data.link);
        state.max_iterations = bind_data.max_iterations;
        state.tolerance = bind_data.tolerance;
        state.confidence_level = bind_data.confidence_level;
        state.null_policy = bind_data.null_policy;

        // Get x values
        auto x_idx = x_data.sel->get_index(i);
        if (!x_data.validity.RowIsValid(x_idx)) {
            continue; // Skip rows with NULL x
        }

        auto list_entry = x_list_data[x_idx];
        idx_t n_features = list_entry.length;

        // Initialize on first valid row
        if (!state.initialized) {
            state.n_features = n_features;
            state.x_train.resize(n_features);
            state.initialized = true;
        }

        // Validate consistent feature count
        if (n_features != state.n_features) {
            throw InvalidInputException("Inconsistent feature count: expected %lu, got %lu", state.n_features,
                                        n_features);
        }

        // Extract x values for this row
        vector<double> x_row(n_features);
        for (idx_t j = 0; j < n_features; j++) {
            x_row[j] = x_child_data[list_entry.offset + j];
        }

        // Check y validity
        auto y_idx = y_data.sel->get_index(i);
        bool y_valid = y_data.validity.RowIsValid(y_idx);
        double y_val = y_valid ? y_values[y_idx] : std::nan("");

        // Determine if this row is for training
        bool row_is_training = y_valid;

        // Apply null_policy for drop_y_zero_x
        if (row_is_training && state.null_policy == NullPolicy::DROP_Y_ZERO_X) {
            for (idx_t j = 0; j < n_features; j++) {
                if (x_row[j] == 0.0) {
                    row_is_training = false;
                    break;
                }
            }
        }

        // Store ALL row data for output (including training flag)
        state.y_all.push_back(y_val);
        state.y_is_null.push_back(!y_valid);
        state.is_training.push_back(row_is_training);
        state.x_all.push_back(x_row);

        // Add to training data if training row
        if (row_is_training) {
            state.y_train.push_back(y_val);
            for (idx_t j = 0; j < n_features; j++) {
                state.x_train[j].push_back(x_row[j]);
            }
        }
    }
}

// Combine: merge two states
static void PoissonFitPredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);

    auto sources = (PoissonFitPredictAggState **)source_data.data;
    auto targets = (PoissonFitPredictAggState **)target_data.data;

    for (idx_t i = 0; i < count; i++) {
        auto &source = *sources[source_data.sel->get_index(i)];
        auto &target = *targets[target_data.sel->get_index(i)];

        if (!source.initialized) {
            continue;
        }

        if (!target.initialized) {
            target.y_train = std::move(source.y_train);
            target.x_train = std::move(source.x_train);
            target.y_all = std::move(source.y_all);
            target.y_is_null = std::move(source.y_is_null);
            target.is_training = std::move(source.is_training);
            target.x_all = std::move(source.x_all);
            target.n_features = source.n_features;
            target.initialized = true;
            target.fit_intercept = source.fit_intercept;
            target.link = source.link;
            target.max_iterations = source.max_iterations;
            target.tolerance = source.tolerance;
            target.confidence_level = source.confidence_level;
            target.null_policy = source.null_policy;
            continue;
        }

        if (source.n_features != target.n_features) {
            throw InvalidInputException("Cannot combine states with different feature counts: %lu vs %lu",
                                        source.n_features, target.n_features);
        }

        // Merge training data
        target.y_train.insert(target.y_train.end(), source.y_train.begin(), source.y_train.end());
        for (idx_t j = 0; j < target.n_features; j++) {
            target.x_train[j].insert(target.x_train[j].end(), source.x_train[j].begin(), source.x_train[j].end());
        }

        // Merge all data
        target.y_all.insert(target.y_all.end(), source.y_all.begin(), source.y_all.end());
        target.y_is_null.insert(target.y_is_null.end(), source.y_is_null.begin(), source.y_is_null.end());
        target.is_training.insert(target.is_training.end(), source.is_training.begin(), source.is_training.end());
        target.x_all.insert(target.x_all.end(), source.x_all.begin(), source.x_all.end());
    }
}

// Finalize: fit model and return predictions for all rows
static void PoissonFitPredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result,
                                         idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = (PoissonFitPredictAggState **)sdata.data;

    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        idx_t result_idx = i + offset;

        // Check if we have enough training data
        if (!state.initialized || state.y_train.size() < 2) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Note: Detailed min_obs validation including zero-variance column handling is done in Rust
        // Prepare FFI data for fitting
        AnofoxDataArray y_array;
        y_array.data = state.y_train.data();
        y_array.validity = nullptr;
        y_array.len = state.y_train.size();

        vector<AnofoxDataArray> x_arrays;
        for (auto &col : state.x_train) {
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
        options.compute_inference = false;
        options.confidence_level = state.confidence_level;

        AnofoxGlmFitResultCore core_result;
        AnofoxError error;

        bool success = anofox_poisson_fit(y_array, x_arrays.data(), x_arrays.size(), options, &core_result, nullptr, &error);

        if (!success) {
            FlatVector::SetNull(result, result_idx, true);
            continue;
        }

        // Build LIST result with predictions for ALL rows
        idx_t n_rows = state.y_all.size();
        auto *list_data = ListVector::GetData(result);
        auto list_offset = ListVector::GetListSize(result);

        list_data[result_idx].offset = list_offset;
        list_data[result_idx].length = n_rows;

        ListVector::Reserve(result, list_offset + n_rows);
        ListVector::SetListSize(result, list_offset + n_rows);

        auto &child_struct = ListVector::GetEntry(result);
        auto &struct_entries = StructVector::GetEntries(child_struct);

        // struct_entries: [y, x, yhat, yhat_lower, yhat_upper, is_training]
        auto &y_vec = *struct_entries[0];
        auto &x_vec = *struct_entries[1];
        auto &yhat_vec = *struct_entries[2];
        auto &yhat_lower_vec = *struct_entries[3];
        auto &yhat_upper_vec = *struct_entries[4];
        auto &is_training_vec = *struct_entries[5];

        // Get t-critical value for prediction intervals
        size_t df = core_result.n_observations - core_result.n_features - (state.fit_intercept ? 1 : 0);
        double t_crit = GetTCritical(state.confidence_level, df);

        for (idx_t row = 0; row < n_rows; row++) {
            idx_t child_idx = list_offset + row;

            // Set y (NULL if it was NULL in input)
            if (state.y_is_null[row]) {
                FlatVector::SetNull(y_vec, child_idx, true);
            } else {
                FlatVector::GetData<double>(y_vec)[child_idx] = state.y_all[row];
            }

            // Set x as LIST
            auto *x_list_data = ListVector::GetData(x_vec);
            auto x_list_offset = ListVector::GetListSize(x_vec);
            x_list_data[child_idx].offset = x_list_offset;
            x_list_data[child_idx].length = state.n_features;

            ListVector::Reserve(x_vec, x_list_offset + state.n_features);
            ListVector::SetListSize(x_vec, x_list_offset + state.n_features);

            auto &x_child = ListVector::GetEntry(x_vec);
            auto x_child_data = FlatVector::GetData<double>(x_child);
            for (idx_t j = 0; j < state.n_features; j++) {
                x_child_data[x_list_offset + j] = state.x_all[row][j];
            }

            // Compute linear predictor: eta = intercept + sum(coef * x)
            double linear_pred = core_result.intercept;
            for (idx_t j = 0; j < state.n_features; j++) {
                linear_pred += core_result.coefficients[j] * state.x_all[row][j];
            }

            // Apply inverse link function to get predicted mean
            double yhat = ApplyInverseLink(linear_pred, state.link);

            // Compute approximate prediction intervals using delta method
            // For Poisson with log link: Var(mu) ~ mu^2 * Var(eta) on log scale
            // Use dispersion parameter to account for over/underdispersion
            double yhat_lower, yhat_upper;

            if (state.link == ANOFOX_POISSON_LINK_LOG) {
                // For log link: compute CI on log scale, then transform
                // SE on log scale ~ sqrt(dispersion / yhat)
                double se_log = std::sqrt(core_result.dispersion) / std::max(yhat, 0.001);
                double log_lower = linear_pred - t_crit * se_log;
                double log_upper = linear_pred + t_crit * se_log;
                yhat_lower = std::exp(log_lower);
                yhat_upper = std::exp(log_upper);
            } else if (state.link == ANOFOX_POISSON_LINK_IDENTITY) {
                // For identity link: use standard error directly
                double se = std::sqrt(core_result.dispersion * yhat);
                yhat_lower = yhat - t_crit * se;
                yhat_upper = yhat + t_crit * se;
            } else if (state.link == ANOFOX_POISSON_LINK_SQRT) {
                // For sqrt link: compute CI on sqrt scale, then transform
                double sqrt_mu = linear_pred;
                double se_sqrt = std::sqrt(core_result.dispersion / 4.0);
                double lower = sqrt_mu - t_crit * se_sqrt;
                double upper = sqrt_mu + t_crit * se_sqrt;
                yhat_lower = std::max(0.0, lower * lower);
                yhat_upper = upper * upper;
            } else {
                // Fallback
                yhat_lower = yhat;
                yhat_upper = yhat;
            }

            FlatVector::GetData<double>(yhat_vec)[child_idx] = yhat;
            FlatVector::GetData<double>(yhat_lower_vec)[child_idx] = std::max(0.0, yhat_lower); // Poisson predictions must be >= 0
            FlatVector::GetData<double>(yhat_upper_vec)[child_idx] = yhat_upper;

            // Set is_training flag
            FlatVector::GetData<bool>(is_training_vec)[child_idx] = state.is_training[row];
        }

        anofox_free_glm_result(&core_result);
        state.Reset();
    }
}

//===--------------------------------------------------------------------===//
// Bind function
//===--------------------------------------------------------------------===//
static unique_ptr<FunctionData> PoissonFitPredictAggBind(ClientContext &context, AggregateFunction &function,
                                                         vector<unique_ptr<Expression>> &arguments) {
    auto result = make_uniq<PoissonFitPredictAggBindData>();

    // Parse MAP options if provided as 3rd argument
    if (arguments.size() >= 3 && arguments[2]->IsFoldable()) {
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[2]);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
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
        if (opts.confidence_level.has_value()) {
            result->confidence_level = opts.confidence_level.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
    }

    function.return_type = GetPoissonFitPredictAggResultType();
    PostHogTelemetry::Instance().CaptureFunctionExecution("poisson_fit_predict_agg");
    return std::move(result);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
void RegisterPoissonFitPredictAggregateFunction(ExtensionLoader &loader) {
    // Primary name
    AggregateFunctionSet func_set("anofox_stats_poisson_fit_predict_agg");

    // Basic version: poisson_fit_predict_agg(y, x)
    auto basic_func =
        AggregateFunction("anofox_stats_poisson_fit_predict_agg", {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE)},
                          LogicalType::ANY, AggregateFunction::StateSize<PoissonFitPredictAggState>, PoissonFitPredictAggInitialize,
                          PoissonFitPredictAggUpdate, PoissonFitPredictAggCombine, PoissonFitPredictAggFinalize, nullptr, PoissonFitPredictAggBind,
                          PoissonFitPredictAggDestroy);
    func_set.AddFunction(basic_func);

    // Version with MAP options: poisson_fit_predict_agg(y, x, {'link': 'log', ...})
    auto map_func = AggregateFunction(
        "anofox_stats_poisson_fit_predict_agg",
        {LogicalType::DOUBLE, LogicalType::LIST(LogicalType::DOUBLE), LogicalType::ANY}, LogicalType::ANY,
        AggregateFunction::StateSize<PoissonFitPredictAggState>, PoissonFitPredictAggInitialize, PoissonFitPredictAggUpdate,
        PoissonFitPredictAggCombine, PoissonFitPredictAggFinalize, nullptr, PoissonFitPredictAggBind, PoissonFitPredictAggDestroy);
    func_set.AddFunction(map_func);

    loader.RegisterFunction(func_set);

    // Short alias
    AggregateFunctionSet alias_set("poisson_fit_predict_agg");
    alias_set.AddFunction(basic_func);
    alias_set.AddFunction(map_func);
    loader.RegisterFunction(alias_set);
}

} // namespace duckdb
