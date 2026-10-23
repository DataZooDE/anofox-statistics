#include <cmath>
#include <cstring>
#include <limits>
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
#include "../include/glm_prior_options.hpp"
#include "telemetry.hpp"
#include "aggregate_combine.hpp"
#include "aggregate_finalize_guard.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// ols_augment_agg / wls_augment_agg and the GLM <family>_augment_agg (#151)
//
// Window aggregates over a whole partition that return one STRUCT per input
// row, in frame order, with the per-observation diagnostics of an OLS / WLS
// fit (the integration contract's `obs` columns). The fit and every measure
// come from anofox_regression::diagnostics::augment through
// anofox_ols_augment / anofox_wls_augment, and for the GLMs from
// anofox_regression::diagnostics::augment_glm through anofox_<family>_augment;
// this file only collects the rows and lays the result out.
//
// A row is used for fitting when y, every feature (and, for WLS, a positive
// weight) are present; other rows get the prediction as `fitted` and NULL
// diagnostics. GLM residuals are deviance residuals and `fitted` is on the
// response scale; a GLM offset is one of the x columns (option 'offset').
//===--------------------------------------------------------------------===//
struct AugmentAggState {
    vector<double> y;               // NaN = row is not a training candidate
    vector<double> weight;          // prior weight (NaN for NULL); 1.0 for OLS
    vector<vector<double>> x_rows;  // empty for a NULL x list
    idx_t n_features;
    bool initialized;

    AugmentAggState() : n_features(0), initialized(false) {}
};

enum class AugmentModel : uint8_t { OLS, WLS, POISSON, BINOMIAL, LOGISTIC, NEGBINOM, GAMMA, TWEEDIE };

static const char *AugmentAggName(AugmentModel model) {
    switch (model) {
    case AugmentModel::OLS:
        return "ols_augment_agg";
    case AugmentModel::WLS:
        return "wls_augment_agg";
    case AugmentModel::POISSON:
        return "poisson_augment_agg";
    case AugmentModel::BINOMIAL:
        return "binomial_augment_agg";
    case AugmentModel::LOGISTIC:
        return "logistic_augment_agg";
    case AugmentModel::NEGBINOM:
        return "negbinom_augment_agg";
    case AugmentModel::GAMMA:
        return "gamma_augment_agg";
    case AugmentModel::TWEEDIE:
        return "tweedie_augment_agg";
    }
    return "augment_agg";
}

//! <model>_partial_residuals_agg (#158): the same fit, partial residuals instead of augment.
static string PartialResidualsAggName(AugmentModel model) {
    string name = AugmentAggName(model);
    return name.substr(0, name.size() - string("_augment_agg").size()) + "_partial_residuals_agg";
}

struct AugmentAggBindData : public FunctionData {
    AugmentModel model = AugmentModel::OLS;
    bool partial = false;
    bool weighted = false;
    bool fit_intercept = true;
    SolverType solver = SolverType::SVD;
    NullPolicy null_policy = NullPolicy::DROP;
    // GLM settings, with the defaults of the matching <family>_fit_agg.
    GlmPriorBindData prior_opts;
    PoissonLink poisson_link = PoissonLink::LOG;
    BinomialLink binomial_link = BinomialLink::LOGIT;
    double power = 1.5;                                     // Tweedie
    double theta = std::numeric_limits<double>::quiet_NaN(); // Negative Binomial; NaN = estimate
    uint32_t max_iterations = 100;
    double tolerance = 1e-8;
    double glm_lambda = 0.0;
    idx_t offset_column = 0; // 1-based index into x; 0 = none

    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<AugmentAggBindData>(*this);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<AugmentAggBindData>();
        auto same_double = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
        return model == other.model && partial == other.partial && weighted == other.weighted &&
               fit_intercept == other.fit_intercept &&
               solver == other.solver && null_policy == other.null_policy && prior_opts.Equals(other.prior_opts) &&
               poisson_link == other.poisson_link && binomial_link == other.binomial_link &&
               same_double(power, other.power) && same_double(theta, other.theta) &&
               max_iterations == other.max_iterations && same_double(tolerance, other.tolerance) &&
               same_double(glm_lambda, other.glm_lambda) && offset_column == other.offset_column;
    }
};

//! Fill the fields every GLM options struct shares.
template <typename OPTIONS>
static void FillGlmOptions(OPTIONS &options, const AugmentAggBindData &bind, const GlmPriorState &priors) {
    options.fit_intercept = bind.fit_intercept;
    options.max_iterations = bind.max_iterations;
    options.tolerance = bind.tolerance;
    options.compute_inference = false;
    options.confidence_level = 0.95;
    options.lambda = bind.glm_lambda;
    options.offset_column = static_cast<size_t>(bind.offset_column);
    priors.Apply(options);
}

static AnofoxPoissonLink ToFfiPoissonLink(PoissonLink link) {
    switch (link) {
    case PoissonLink::IDENTITY:
        return ANOFOX_POISSON_LINK_IDENTITY;
    case PoissonLink::SQRT:
        return ANOFOX_POISSON_LINK_SQRT;
    case PoissonLink::LOG:
    default:
        return ANOFOX_POISSON_LINK_LOG;
    }
}

static AnofoxBinomialLink ToFfiBinomialLink(BinomialLink link) {
    switch (link) {
    case BinomialLink::PROBIT:
        return ANOFOX_BINOMIAL_LINK_PROBIT;
    case BinomialLink::CLOGLOG:
        return ANOFOX_BINOMIAL_LINK_CLOGLOG;
    case BinomialLink::LOGIT:
    default:
        return ANOFOX_BINOMIAL_LINK_LOGIT;
    }
}

//! Run the GLM augment of `bind.model`.
static bool RunGlmAugment(const AugmentAggBindData &bind, idx_t n_features, AnofoxDataArray y_array,
                          vector<AnofoxDataArray> &x_arrays, AnofoxAugmentResult *aug, AnofoxError *error) {
    GlmPriorState priors;
    priors.Materialize(bind.prior_opts, n_features, bind.fit_intercept);
    switch (bind.model) {
    case AugmentModel::POISSON: {
        AnofoxPoissonOptions options {};
        FillGlmOptions(options, bind, priors);
        options.link = ToFfiPoissonLink(bind.poisson_link);
        return anofox_poisson_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    case AugmentModel::BINOMIAL: {
        AnofoxBinomialOptions options {};
        FillGlmOptions(options, bind, priors);
        options.link = ToFfiBinomialLink(bind.binomial_link);
        return anofox_binomial_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    case AugmentModel::LOGISTIC: {
        AnofoxLogisticOptions options {};
        FillGlmOptions(options, bind, priors);
        options.threshold = 0.5;
        return anofox_logistic_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    case AugmentModel::NEGBINOM: {
        AnofoxNegBinomialOptions options {};
        FillGlmOptions(options, bind, priors);
        options.alpha = bind.theta;
        return anofox_negbinomial_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    case AugmentModel::GAMMA: {
        AnofoxGammaOptions options {};
        FillGlmOptions(options, bind, priors);
        return anofox_gamma_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    case AugmentModel::TWEEDIE: {
        AnofoxTweedieOptions options {};
        FillGlmOptions(options, bind, priors);
        options.power = bind.power;
        return anofox_tweedie_augment(y_array, x_arrays.data(), x_arrays.size(), options, aug, error);
    }
    default:
        throw InternalException("RunGlmAugment: not a GLM");
    }
}

//! Run the GLM partial residuals of `bind.model` (#158).
static bool RunGlmPartialResiduals(const AugmentAggBindData &bind, idx_t n_features, AnofoxDataArray y_array,
                                   vector<AnofoxDataArray> &x_arrays, AnofoxPartialResidualsResult *out,
                                   AnofoxError *error) {
    GlmPriorState priors;
    priors.Materialize(bind.prior_opts, n_features, bind.fit_intercept);
    switch (bind.model) {
    case AugmentModel::POISSON: {
        AnofoxPoissonOptions options {};
        FillGlmOptions(options, bind, priors);
        options.link = ToFfiPoissonLink(bind.poisson_link);
        return anofox_poisson_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    case AugmentModel::BINOMIAL: {
        AnofoxBinomialOptions options {};
        FillGlmOptions(options, bind, priors);
        options.link = ToFfiBinomialLink(bind.binomial_link);
        return anofox_binomial_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    case AugmentModel::LOGISTIC: {
        AnofoxLogisticOptions options {};
        FillGlmOptions(options, bind, priors);
        options.threshold = 0.5;
        return anofox_logistic_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    case AugmentModel::NEGBINOM: {
        AnofoxNegBinomialOptions options {};
        FillGlmOptions(options, bind, priors);
        options.alpha = bind.theta;
        return anofox_negbinomial_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    case AugmentModel::GAMMA: {
        AnofoxGammaOptions options {};
        FillGlmOptions(options, bind, priors);
        return anofox_gamma_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    case AugmentModel::TWEEDIE: {
        AnofoxTweedieOptions options {};
        FillGlmOptions(options, bind, priors);
        options.power = bind.power;
        return anofox_tweedie_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, out, error);
    }
    default:
        throw InternalException("RunGlmPartialResiduals: not a GLM");
    }
}

//! LIST (one per input row, in frame order) of LIST (one per fitted term) of the
//! partial-residual and added-variable coordinates (#158).
static LogicalType GetPartialResidualsAggResultType() {
    child_list_t<LogicalType> term;
    term.push_back(make_pair("term_index", LogicalType::INTEGER));
    term.push_back(make_pair("x", LogicalType::DOUBLE));
    term.push_back(make_pair("partial_residual", LogicalType::DOUBLE));
    term.push_back(make_pair("component", LogicalType::DOUBLE));
    term.push_back(make_pair("x_residual", LogicalType::DOUBLE));
    term.push_back(make_pair("y_residual", LogicalType::DOUBLE));
    return LogicalType::LIST(LogicalType::LIST(LogicalType::STRUCT(std::move(term))));
}

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
    // Leave-one-out (PRESS) residual and prediction (#220); NULL for the GLMs
    // and rows with leverage 1.
    row.push_back(make_pair("loo_residual", LogicalType::DOUBLE));
    row.push_back(make_pair("loo_fitted", LogicalType::DOUBLE));
    return LogicalType::LIST(LogicalType::STRUCT(std::move(row)));
}

static void AugmentAggInitialize(data_ptr_t state_p) {
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
    auto x_list_data = UnifiedVectorFormat::GetData<list_entry_t>(x_data);
    auto &x_child = ListVector::GetEntry(inputs[1]);
    auto x_child_data = FlatDataMut<double>(x_child);
    auto &x_child_validity = FlatValidityMut(x_child);
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
        double y_val = y_data.validity.RowIsValid(y_idx) ? y_values[y_idx] : NullFill(bind_data.null_policy);

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
                x_row[j] = x_child_validity.RowIsValid(child_pos) ? x_child_data[child_pos]
                                                                    : NullFill(bind_data.null_policy);
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
        FlatDataMut<double>(vec)[idx] = v;
    } else {
        FlatVector::SetNull(vec, idx, true);
    }
}

//! Partial residuals of one partition into `result[result_idx]` (#158). Terms the
//! fit dropped (aliased columns, a GLM offset) are left out.
static void WritePartialResiduals(const AugmentAggBindData &bind_data, AugmentAggState &state,
                                  AnofoxDataArray y_array, vector<AnofoxDataArray> &x_arrays, Vector &result,
                                  idx_t result_idx) {
    auto fn_name = PartialResidualsAggName(bind_data.model);
    idx_t n_rows = state.y.size();
    AnofoxPartialResidualsResult pr;
    std::memset(&pr, 0, sizeof(pr));
    AnofoxError error;
    bool ok;
    if (bind_data.model != AugmentModel::OLS && bind_data.model != AugmentModel::WLS) {
        ok = RunGlmPartialResiduals(bind_data, state.n_features, y_array, x_arrays, &pr, &error);
    } else if (bind_data.weighted) {
        AnofoxDataArray w_array {state.weight.data(), nullptr, static_cast<size_t>(n_rows)};
        AnofoxWlsOptions options {};
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = 0.95;
        options.solver = ConvertSolverType(bind_data.solver);
        options.hc_type = ANOFOX_HC_NONE;
        ok = anofox_wls_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), w_array, options, &pr, &error);
    } else {
        AnofoxOlsOptions options {};
        options.fit_intercept = bind_data.fit_intercept;
        options.compute_inference = false;
        options.confidence_level = 0.95;
        options.solver = ConvertSolverType(bind_data.solver);
        options.hc_type = ANOFOX_HC_NONE;
        ok = anofox_ols_partial_residuals(y_array, x_arrays.data(), x_arrays.size(), options, &pr, &error);
    }
    if (!ok) {
        ThrowUnlessDegenerate(fn_name.c_str(), error);
        FlatVector::SetNull(result, result_idx, true);
        return;
    }
    if (pr.len != n_rows) {
        anofox_free_partial_residuals_result(&pr);
        throw InternalException("%s: returned %llu rows for %llu inputs", fn_name.c_str(), (uint64_t)pr.len,
                                (uint64_t)n_rows);
    }
    vector<idx_t> terms;
    for (idx_t j = 0; j < pr.n_terms; j++) {
        for (idx_t r = 0; r < n_rows; r++) {
            if (std::isfinite(pr.component[j * n_rows + r])) {
                terms.push_back(j);
                break;
            }
        }
    }

    // Outer list: one entry per input row.
    auto row_offset = ListVector::GetListSize(result);
    ListVector::Reserve(result, row_offset + n_rows);
    ListVector::SetListSize(result, row_offset + n_rows);
    ListVector::GetData(result)[result_idx] = {row_offset, n_rows};
    auto &rows = ListVector::GetEntry(result);
    // Inner lists: one entry per term.
    auto term_offset = ListVector::GetListSize(rows);
    ListVector::Reserve(rows, term_offset + n_rows * terms.size());
    ListVector::SetListSize(rows, term_offset + n_rows * terms.size());
    auto &entries = StructVector::GetEntries(ListVector::GetEntry(rows));
    for (idx_t r = 0; r < n_rows; r++) {
        ListVector::GetData(rows)[row_offset + r] = {term_offset + r * terms.size(), terms.size()};
        for (idx_t t = 0; t < terms.size(); t++) {
            idx_t j = terms[t];
            idx_t c = term_offset + r * terms.size() + t;
            idx_t k = j * n_rows + r;
            FlatDataMut<int32_t>(StructEntry(entries[0]))[c] = static_cast<int32_t>(j + 1);
            WriteDouble(StructEntry(entries[1]), c, pr.x[k]);
            WriteDouble(StructEntry(entries[2]), c, pr.partial_residual[k]);
            WriteDouble(StructEntry(entries[3]), c, pr.component[k]);
            WriteDouble(StructEntry(entries[4]), c, pr.x_residual[k]);
            WriteDouble(StructEntry(entries[5]), c, pr.y_residual[k]);
        }
    }
    anofox_free_partial_residuals_result(&pr);
}

static void AugmentAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                               idx_t offset) {
    auto &bind_data = aggr_input_data.bind_data->Cast<AugmentAggBindData>();
    const char *fn_name = AugmentAggName(bind_data.model);

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
            x_arrays.push_back({col.data(), nullptr, static_cast<size_t>(col.size())});
        }
        AnofoxDataArray y_array {state.y.data(), nullptr, static_cast<size_t>(n_rows)};

        if (bind_data.partial) {
            WritePartialResiduals(bind_data, state, y_array, x_arrays, result, result_idx);
            continue;
        }

        AnofoxAugmentResult aug;
        std::memset(&aug, 0, sizeof(aug));
        AnofoxError error;
        bool ok;
        if (bind_data.model != AugmentModel::OLS && bind_data.model != AugmentModel::WLS) {
            ok = RunGlmAugment(bind_data, state.n_features, y_array, x_arrays, &aug, &error);
        } else if (bind_data.weighted) {
            AnofoxDataArray w_array {state.weight.data(), nullptr, static_cast<size_t>(n_rows)};
            AnofoxWlsOptions options {};
            options.fit_intercept = bind_data.fit_intercept;
            options.compute_inference = false;
            options.confidence_level = 0.95;
            options.solver = ConvertSolverType(bind_data.solver);
            options.hc_type = ANOFOX_HC_NONE;
            ok = anofox_wls_augment(y_array, x_arrays.data(), x_arrays.size(), w_array, options, &aug, &error);
        } else {
            AnofoxOlsOptions options {};
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
        auto &fitted_vec = StructEntry(entries[0]);
        auto &residual_vec = StructEntry(entries[1]);
        auto &resid_type_vec = StructEntry(entries[2]);
        auto &std_vec = StructEntry(entries[3]);
        auto &stud_vec = StructEntry(entries[4]);
        auto &leverage_vec = StructEntry(entries[5]);
        auto &cooks_vec = StructEntry(entries[6]);
        auto &dffits_vec = StructEntry(entries[7]);
        auto &weight_vec = StructEntry(entries[8]);
        auto &training_vec = StructEntry(entries[9]);
        auto &n_params_vec = StructEntry(entries[10]);
        auto &loo_residual_vec = StructEntry(entries[11]);
        auto &loo_fitted_vec = StructEntry(entries[12]);

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
            WriteDouble(loo_residual_vec, c, aug.loo_residual[r]);
            WriteDouble(loo_fitted_vec, c, aug.loo_fitted[r]);
            WriteDouble(weight_vec, c, state.weight[r]);
            FlatDataMut<string_t>(resid_type_vec)[c] = StringVector::AddString(resid_type_vec, resid_type);
            FlatDataMut<bool>(training_vec)[c] = used;
            FlatDataMut<int32_t>(n_params_vec)[c] = n_params;
        }

        anofox_free_augment_result(&aug);
    }
}

static unique_ptr<FunctionData> AugmentAggBindImpl(ClientContext &context, AggregateBindFunction &function,
                                                   vector<unique_ptr<Expression>> &arguments, AugmentModel model,
                                                   bool partial) {
    const string fn_name_str = partial ? PartialResidualsAggName(model) : string(AugmentAggName(model));
    const char *fn_name = fn_name_str.c_str();
    auto result = make_uniq<AugmentAggBindData>();
    result->model = model;
    result->partial = partial;
    result->weighted = model == AugmentModel::WLS;
    idx_t options_idx = result->weighted ? 3 : 2;
    if (arguments.size() > options_idx) {
        bool linear = model == AugmentModel::OLS || model == AugmentModel::WLS;
        // The keys of the matching <family>_fit_agg that change the fit; 'glm_lambda' and 'prior' are accepted
        // so that a penalized fit is rejected with upstream's error rather than as an unknown key.
        vector<string> allowed;
        if (linear) {
            allowed = {"fit_intercept", "null_policy", "solver"};
        } else {
            allowed = {"fit_intercept", "max_iterations", "tolerance", "offset", "glm_lambda", "prior", "feature_names"};
            if (model == AugmentModel::POISSON) {
                allowed.push_back("poisson_link");
            } else if (model == AugmentModel::BINOMIAL) {
                allowed.push_back("binomial_link");
            } else if (model == AugmentModel::NEGBINOM) {
                allowed.push_back("theta");
            } else if (model == AugmentModel::TWEEDIE) {
                allowed.push_back("power");
            }
        }
        auto opts = RegressionMapOptions::ParseFromExpression(context, *arguments[options_idx], fn_name, allowed);
        if (opts.fit_intercept.has_value()) {
            result->fit_intercept = opts.fit_intercept.value();
        }
        if (opts.null_policy.has_value()) {
            result->null_policy = opts.null_policy.value();
        }
        if (opts.solver.has_value()) {
            result->solver = opts.solver.value();
        }
        if (!linear) {
            result->prior_opts.LoadFrom(opts);
        }
        if (opts.poisson_link.has_value()) {
            result->poisson_link = opts.poisson_link.value();
        }
        if (opts.binomial_link.has_value()) {
            result->binomial_link = opts.binomial_link.value();
        }
        if (opts.tweedie_power.has_value()) {
            result->power = opts.tweedie_power.value();
        }
        if (opts.nb_theta.has_value()) {
            result->theta = opts.nb_theta.value();
        }
        if (opts.max_iterations.has_value()) {
            result->max_iterations = opts.max_iterations.value();
        }
        if (opts.tolerance.has_value()) {
            result->tolerance = opts.tolerance.value();
        }
        if (opts.glm_lambda.has_value()) {
            result->glm_lambda = opts.glm_lambda.value();
        }
        if (opts.offset_column.has_value()) {
            result->offset_column = opts.offset_column.value();
        }
    }
    SetReturnTypeOf(function, partial ? GetPartialResidualsAggResultType() : GetAugmentAggResultType());
    PostHogTelemetry::Instance().RecordFunctionCall(fn_name);
    return std::move(result);
}

template <AugmentModel MODEL, bool PARTIAL = false>
static unique_ptr<FunctionData> AugmentAggBind(ClientContext &context, AggregateBindFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {
    return AugmentAggBindImpl(context, function, arguments, MODEL, PARTIAL);
}

static AggregateFunction MakeAugmentAgg(const string &name, vector<LogicalType> args, bind_aggregate_function_t bind) {
    return AggregateFunction(AsDuckName(name), std::move(args), LogicalType::ANY, AggregateFunction::StateSize<AugmentAggState>,
                             ANOFOX_AGG_INIT(AugmentAggInitialize),
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
        set.AddFunction(MakeAugmentAgg("ols_augment_agg", {LogicalType::DOUBLE, list_double}, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::OLS>)));
        set.AddFunction(MakeAugmentAgg("ols_augment_agg", {LogicalType::DOUBLE, list_double, LogicalType::ANY},
                                       ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::OLS>)));
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
                                       ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::WLS>)));
        set.AddFunction(MakeAugmentAgg("wls_augment_agg",
                                       {LogicalType::DOUBLE, list_double, LogicalType::DOUBLE, LogicalType::ANY},
                                       ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::WLS>)));
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
    // <family>_augment_agg(y, x [, options]) for the six GLMs
    struct GlmAugmentAgg {
        AugmentModel model;
        const char *label;
        const char *option_example;
    };
    const GlmAugmentAgg glms[] = {
        {AugmentModel::POISSON, "a Poisson GLM", "{'poisson_link': 'log'}"},
        {AugmentModel::BINOMIAL, "a Binomial GLM", "{'binomial_link': 'logit'}"},
        {AugmentModel::LOGISTIC, "a logistic regression", "{'fit_intercept': true}"},
        {AugmentModel::NEGBINOM, "a Negative Binomial GLM", "{'theta': 2.0}"},
        {AugmentModel::GAMMA, "a Gamma GLM (log link)", "{'fit_intercept': true}"},
        {AugmentModel::TWEEDIE, "a Tweedie GLM (log link)", "{'power': 1.5}"},
    };
    for (auto &glm : glms) {
        string name = AugmentAggName(glm.model);
        bind_aggregate_function_t bind = nullptr;
        switch (glm.model) {
        case AugmentModel::POISSON:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::POISSON>);
            break;
        case AugmentModel::BINOMIAL:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::BINOMIAL>);
            break;
        case AugmentModel::LOGISTIC:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::LOGISTIC>);
            break;
        case AugmentModel::NEGBINOM:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::NEGBINOM>);
            break;
        case AugmentModel::GAMMA:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::GAMMA>);
            break;
        default:
            bind = ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::TWEEDIE>);
            break;
        }
        AggregateFunctionSet set(AsDuckName(name));
        set.AddFunction(MakeAugmentAgg(name, {LogicalType::DOUBLE, list_double}, bind));
        set.AddFunction(MakeAugmentAgg(name, {LogicalType::DOUBLE, list_double, LogicalType::ANY}, bind));
        CreateAggregateFunctionInfo info(std::move(set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        const string desc = "Fits " + string(glm.label) +
                            " over a partition (rows with y and every feature present) and returns one STRUCT per "
                            "row, in frame order, with fitted (response scale), residual (deviance), resid_type, "
                            "std_residual, stud_residual, leverage, cooks_d, dffits, weight, is_training and n_params "
                            "(broom augment). Penalized fits (glm_lambda > 0 or a prior) are rejected.";
        info.descriptions.push_back(AugmentDescription(desc, name + "(y, [x1, x2]) OVER (PARTITION BY g)",
                                                       {"y", "x"}, {LogicalType::DOUBLE, list_double}));
        info.descriptions.push_back(AugmentDescription(desc,
                                                       name + "(y, [x1, x2], " + glm.option_example + ") OVER ()",
                                                       {"y", "x", "options"},
                                                       {LogicalType::DOUBLE, list_double, LogicalType::ANY}));
        loader.RegisterFunction(std::move(info));
    }

    // <model>_partial_residuals_agg (#158): component-plus-residual (car::crPlots) and
    // added-variable (car::avPlots, OLS / WLS only) coordinates per row and term.
    struct PartialAgg {
        AugmentModel model;
        bind_aggregate_function_t bind;
        const char *label;
    };
    const PartialAgg partials[] = {
        {AugmentModel::OLS, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::OLS, true>), "OLS"},
        {AugmentModel::WLS, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::WLS, true>), "WLS"},
        {AugmentModel::POISSON, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::POISSON, true>), "a Poisson GLM"},
        {AugmentModel::BINOMIAL, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::BINOMIAL, true>), "a Binomial GLM"},
        {AugmentModel::LOGISTIC, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::LOGISTIC, true>), "a logistic regression"},
        {AugmentModel::NEGBINOM, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::NEGBINOM, true>), "a Negative Binomial GLM"},
        {AugmentModel::GAMMA, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::GAMMA, true>), "a Gamma GLM (log link)"},
        {AugmentModel::TWEEDIE, ANOFOX_AGG_BIND(AugmentAggBind<AugmentModel::TWEEDIE, true>), "a Tweedie GLM (log link)"},
    };
    for (auto &pa : partials) {
        string name = PartialResidualsAggName(pa.model);
        bool weighted = pa.model == AugmentModel::WLS;
        bool linear = weighted || pa.model == AugmentModel::OLS;
        vector<LogicalType> args = {LogicalType::DOUBLE, list_double};
        vector<string> arg_names = {"y", "x"};
        if (weighted) {
            args.push_back(LogicalType::DOUBLE);
            arg_names.push_back("weight");
        }
        AggregateFunctionSet set(AsDuckName(name));
        set.AddFunction(MakeAugmentAgg(name, args, pa.bind));
        auto args_opt = args;
        args_opt.push_back(LogicalType::ANY);
        set.AddFunction(MakeAugmentAgg(name, args_opt, pa.bind));
        CreateAggregateFunctionInfo info(std::move(set));
        info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        const string desc =
            "Fits " + string(pa.label) +
            " over a partition and returns, per row in frame order, a LIST with one STRUCT per fitted term: "
            "term_index (1-based position in x), x, partial_residual = residual + component (R residuals(type = "
            "'partial'), car::crPlots" +
            (linear ? string("") : string("; the working residual")) +
            "), component = b_j (x_j - mean(x_j)), and the added-variable coordinates x_residual, y_residual "
            "(car::avPlots" +
            (linear ? string(")") : string("; NULL for GLMs)")) +
            ". Rows not used for fitting get NULL residuals.";
        string example = name + (weighted ? "(y, [x1, x2], w) OVER (PARTITION BY g)" : "(y, [x1, x2]) OVER (PARTITION BY g)");
        info.descriptions.push_back(AugmentDescription(desc, example, arg_names, args));
        auto names_opt = arg_names;
        names_opt.push_back("options");
        info.descriptions.push_back(AugmentDescription(desc, example, names_opt, args_opt));
        loader.RegisterFunction(std::move(info));
    }
}

} // namespace duckdb
