#include "fit_predict_base.hpp"
#include "../utils/statistical_distributions.hpp"
#include "../utils/tracing.hpp"
#include <cmath>

namespace duckdb {
namespace anofox_statistics {

PredictionOptions PredictionOptions::ParseFromOptions(const RegressionOptions &reg_options) {
    PredictionOptions pred_opts;
    // In future, could extend RegressionOptions to include these fields
    // For now, use defaults
    pred_opts.confidence_level = 0.95;
    pred_opts.interval_type = "prediction";
    return pred_opts;
}

PredictionResult ComputePredictionWithInterval(
    const vector<double> &x_new,
    double intercept,
    const Eigen::VectorXd &coefficients,
    double mse,
    const Eigen::VectorXd &x_train_means,
    const Eigen::MatrixXd &X_train,
    idx_t df_residual,
    double confidence_level,
    const string &interval_type
) {
    PredictionResult result;

    if (x_new.size() != static_cast<size_t>(coefficients.size())) {
        result.is_valid = false;
        return result;
    }

    // Compute point prediction
    result.yhat = intercept;
    for (size_t i = 0; i < x_new.size(); i++) {
        result.yhat += coefficients(i) * x_new[i];
    }

    // Compute prediction standard error
    // SE = sqrt(MSE * (1 + 1/n + (x_new - x_mean)' * (X'X)^(-1) * (x_new - x_mean)))
    // For confidence interval: use (1/n + ...) instead of (1 + 1/n + ...)

    if (df_residual == 0 || mse <= 0 || std::isnan(mse)) {
        // Cannot compute intervals without valid MSE and df
        result.yhat_lower = result.yhat;
        result.yhat_upper = result.yhat;
        result.std_error = std::numeric_limits<double>::quiet_NaN();
        result.is_valid = true;
        return result;
    }

    // Convert x_new to Eigen vector and center it
    Eigen::VectorXd x_eigen(x_new.size());
    for (size_t i = 0; i < x_new.size(); i++) {
        x_eigen(i) = x_new[i] - x_train_means(i);
    }

    // Compute (X'X)^(-1) from training data
    // X_train is already centered in most implementations
    idx_t n_train = X_train.rows();
    idx_t p = X_train.cols();

    // Center X_train
    Eigen::MatrixXd X_centered = X_train;
    for (idx_t j = 0; j < p; j++) {
        double mean = X_train.col(j).mean();
        X_centered.col(j).array() -= mean;
    }

    Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;

    // Compute leverage: h = x' * (X'X)^{-1} * x
    double leverage = 0.0;

    // Use pseudo-inverse for numerical stability
    Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));

    Eigen::VectorXd XtX_inv_x = XtX_inv * x_eigen;
    leverage = x_eigen.dot(XtX_inv_x);

    // Compute standard error based on interval type
    double variance;
    if (interval_type == "confidence") {
        // Confidence interval for E[Y|X]
        variance = mse * (1.0 / n_train + leverage);
    } else {
        // Prediction interval for new observation
        variance = mse * (1.0 + 1.0 / n_train + leverage);
    }

    result.std_error = std::sqrt(variance);

    // Compute critical value from t-distribution
    double t_crit = StatisticalDistributions::StudentTCriticalValue(df_residual, confidence_level);

    // Compute interval bounds
    double margin = t_crit * result.std_error;
    result.yhat_lower = result.yhat - margin;
    result.yhat_upper = result.yhat + margin;
    result.is_valid = true;

    return result;
}

vector<double> ExtractListAsVector(Vector &list_vector, idx_t row_idx, UnifiedVectorFormat &list_data) {
    vector<double> result;

    if (!list_data.validity.RowIsValid(row_idx)) {
        return result;
    }

    auto list_entry = UnifiedVectorFormat::GetData<list_entry_t>(list_data)[row_idx];
    auto &child_vector = ListVector::GetEntry(list_vector);

    UnifiedVectorFormat child_data;
    child_vector.ToUnifiedFormat(ListVector::GetListSize(list_vector), child_data);
    auto child_ptr = UnifiedVectorFormat::GetData<double>(child_data);

    for (idx_t i = 0; i < list_entry.length; i++) {
        auto child_idx = child_data.sel->get_index(list_entry.offset + i);
        if (child_data.validity.RowIsValid(child_idx)) {
            result.push_back(child_ptr[child_idx]);
        } else {
            // NULL in array - return empty to indicate invalid
            return vector<double>();
        }
    }

    return result;
}

LogicalType CreateFitPredictReturnType() {
    child_list_t<LogicalType> struct_fields;
    struct_fields.push_back(make_pair("yhat", LogicalType::DOUBLE));
    struct_fields.push_back(make_pair("yhat_lower", LogicalType::DOUBLE));
    struct_fields.push_back(make_pair("yhat_upper", LogicalType::DOUBLE));
    struct_fields.push_back(make_pair("std_error", LogicalType::DOUBLE));
    return LogicalType::STRUCT(struct_fields);
}

} // namespace anofox_statistics
} // namespace duckdb
