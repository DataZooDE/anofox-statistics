#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "../utils/options_parser.hpp"
#include <vector>
#include <Eigen/Dense>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Base state for fit-predict aggregate functions
 *
 * This state accumulates data across rows in a partition, then:
 * 1. Fits a model on rows where y IS NOT NULL
 * 2. Predicts for ALL rows (including where y IS NULL)
 * 3. Returns predictions with optional intervals
 */
struct FitPredictState {
	// Training data (only non-NULL y values)
	vector<double> y_train;
	vector<vector<double>> x_train;

	// All rows (for prediction - includes NULL y)
	vector<vector<double>> x_all;
	vector<bool> is_train_row; // true if y was NOT NULL

	idx_t n_features = 0;
	RegressionOptions options;
	bool options_initialized = false;

	// Model state (filled after fitting)
	bool model_fitted = false;
	Eigen::VectorXd coefficients;
	double intercept = 0.0;
	double mse = 0.0;
	Eigen::VectorXd x_train_means;
	Eigen::VectorXd coefficient_std_errors;
	double intercept_std_error = 0.0;
	idx_t df_residual = 0;
	idx_t rank = 0;

	void Reset() {
		y_train.clear();
		x_train.clear();
		x_all.clear();
		is_train_row.clear();
		n_features = 0;
		options = RegressionOptions();
		options_initialized = false;
		model_fitted = false;
	}
};

/**
 * @brief Parse prediction options from options MAP
 */
struct PredictionOptions {
	double confidence_level = 0.95;
	string interval_type = "prediction"; // "prediction" or "confidence"

	static PredictionOptions ParseFromOptions(const RegressionOptions &reg_options);
};

/**
 * @brief Result structure for per-row predictions
 */
struct PredictionResult {
	double yhat = 0.0;
	double yhat_lower = 0.0;
	double yhat_upper = 0.0;
	double std_error = 0.0;
	bool is_valid = false;
};

/**
 * @brief Compute prediction interval for a single observation
 *
 * @param x_new Feature vector for new observation
 * @param intercept Model intercept
 * @param coefficients Model coefficients
 * @param mse Mean squared error from training
 * @param x_train_means Mean of training features
 * @param X_train Training design matrix (for leverage calculation)
 * @param df_residual Degrees of freedom for residuals
 * @param confidence_level Confidence level (e.g., 0.95)
 * @param interval_type "prediction" or "confidence"
 * @return PredictionResult with yhat and interval bounds
 */
PredictionResult ComputePredictionWithInterval(const vector<double> &x_new, double intercept,
                                               const Eigen::VectorXd &coefficients, double mse,
                                               const Eigen::VectorXd &x_train_means, const Eigen::MatrixXd &X_train,
                                               idx_t df_residual, double confidence_level, const string &interval_type);

/**
 * @brief Helper to extract list data from DuckDB vector
 */
vector<double> ExtractListAsVector(Vector &list_vector, idx_t row_idx, UnifiedVectorFormat &list_data);

/**
 * @brief Cached partition data for fit-predict window functions
 *
 * This structure caches the partition data to prevent O(n²) scanning.
 * The data is loaded once per partition and reused for all rows.
 */
struct PartitionDataCache {
	vector<double> all_y;
	vector<vector<double>> all_x;
	RegressionOptions options;
	bool options_initialized = false;
	idx_t n_features = 0;
	bool initialized = false;
};

/**
 * @brief Load partition data into cache (called once per partition)
 *
 * This function scans the entire partition once and caches all data,
 * preventing O(n²) performance issues from scanning the partition for every row.
 */
void LoadPartitionData(const WindowPartitionInput &partition, PartitionDataCache &cache);

/**
 * @brief Create return type for fit-predict functions
 *
 * Returns STRUCT with:
 *   - yhat: DOUBLE (predicted value)
 *   - yhat_lower: DOUBLE (lower bound of interval)
 *   - yhat_upper: DOUBLE (upper bound of interval)
 *   - std_error: DOUBLE (standard error of prediction)
 */
LogicalType CreateFitPredictReturnType();

/**
 * @brief Shared aggregate functions for fit-predict
 * These are implemented in ols_fit_predict.cpp but can be reused by other models
 * since they all use the same FitPredictState structure
 */
void OlsFitPredictInitialize(const AggregateFunction &function, data_ptr_t state_ptr);
void OlsFitPredictUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &state_vector,
                         idx_t count);
void OlsFitPredictCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count);

} // namespace anofox_statistics
} // namespace duckdb
