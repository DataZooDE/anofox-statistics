#include "fit_predict_base.hpp"
#include "../utils/tracing.hpp"
#include <cmath>
#include <algorithm>

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

PredictionResult ComputePredictionWithInterval(const vector<double> &x_new, double intercept,
                                               const Eigen::VectorXd &coefficients, double mse,
                                               const Eigen::VectorXd &x_train_means, const Eigen::MatrixXd &X_train,
                                               idx_t df_residual, double confidence_level,
                                               const string &interval_type) {
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
	// Simple approximation: use z-value for now (TODO: implement proper t-distribution)
	// For 95% CI: z ≈ 1.96, for 99%: z ≈ 2.576
	// For small df, t > z, so this is conservative
	double z_crit = 1.96; // Approximation for 95% CI
	if (confidence_level > 0.98) {
		z_crit = 2.576; // 99% CI
	} else if (confidence_level < 0.92) {
		z_crit = 1.645; // 90% CI
	}

	// Compute interval bounds
	double margin = z_crit * result.std_error;
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

void LoadPartitionData(const WindowPartitionInput &partition, PartitionDataCache &cache) {
	// Only load once per partition
	if (cache.initialized) {
		return;
	}

	// Read input data from the partition
	ColumnDataScanState scan_state;
	partition.inputs->InitializeScan(scan_state);

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), partition.inputs->Types());

	while (partition.inputs->Scan(scan_state, chunk)) {
		auto &y_chunk = chunk.data[0];
		auto &x_array_chunk = chunk.data[1];
		auto &options_chunk = chunk.data[2];

		UnifiedVectorFormat y_data;
		UnifiedVectorFormat x_array_data;
		UnifiedVectorFormat options_data;
		y_chunk.ToUnifiedFormat(chunk.size(), y_data);
		x_array_chunk.ToUnifiedFormat(chunk.size(), x_array_data);
		options_chunk.ToUnifiedFormat(chunk.size(), options_data);

		auto y_ptr = UnifiedVectorFormat::GetData<double>(y_data);

		for (idx_t i = 0; i < chunk.size(); i++) {
			auto y_idx = y_data.sel->get_index(i);
			auto x_array_idx = x_array_data.sel->get_index(i);
			auto options_idx = options_data.sel->get_index(i);

			// Parse options from first valid row
			if (!cache.options_initialized && options_data.validity.RowIsValid(options_idx)) {
				auto options_value = options_chunk.GetValue(options_idx);
				cache.options = RegressionOptions::ParseFromMap(options_value);
				cache.options_initialized = true;
			}

			vector<double> features;
			bool x_valid = false;

			if (x_array_data.validity.RowIsValid(x_array_idx)) {
				auto x_array_entry = UnifiedVectorFormat::GetData<list_entry_t>(x_array_data)[x_array_idx];
				auto &x_child = ListVector::GetEntry(x_array_chunk);

				UnifiedVectorFormat x_child_data;
				x_child.ToUnifiedFormat(ListVector::GetListSize(x_array_chunk), x_child_data);
				auto x_child_ptr = UnifiedVectorFormat::GetData<double>(x_child_data);

				for (idx_t j = 0; j < x_array_entry.length; j++) {
					auto child_idx = x_child_data.sel->get_index(x_array_entry.offset + j);
					if (x_child_data.validity.RowIsValid(child_idx)) {
						features.push_back(x_child_ptr[child_idx]);
					} else {
						features.clear();
						break;
					}
				}

				if (!features.empty()) {
					if (cache.n_features == 0) {
						cache.n_features = features.size();
					}
					x_valid = (features.size() == cache.n_features);
				}
			}

			// Store data for all rows
			if (x_valid) {
				cache.all_x.push_back(features);
				if (y_data.validity.RowIsValid(y_idx)) {
					cache.all_y.push_back(y_ptr[y_idx]);
				} else {
					cache.all_y.push_back(std::numeric_limits<double>::quiet_NaN());
				}
			} else {
				cache.all_x.push_back(vector<double>());
				cache.all_y.push_back(std::numeric_limits<double>::quiet_NaN());
			}
		}
	}

	cache.initialized = true;
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
