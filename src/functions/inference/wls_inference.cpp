#include "wls_inference.hpp"
#include "../utils/tracing.hpp"
#include "../bridge/libanostat_wrapper.hpp"
#include "../bridge/type_converters.hpp"
#include "../utils/options_parser.hpp"
#include "../utils/validation.hpp"
#include "../utils/statistical_distributions.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

struct WlsInferenceBindData : public FunctionData {
	vector<string> variable_names;
	vector<double> estimates;
	vector<double> std_errors;
	vector<double> t_statistics;
	vector<double> p_values;
	vector<double> ci_lowers;
	vector<double> ci_uppers;
	vector<bool> significant;

	idx_t current_row = 0;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<WlsInferenceBindData>();
		result->variable_names = variable_names;
		result->estimates = estimates;
		result->std_errors = std_errors;
		result->t_statistics = t_statistics;
		result->p_values = p_values;
		result->ci_lowers = ci_lowers;
		result->ci_uppers = ci_uppers;
		result->significant = significant;
		result->current_row = current_row;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

static unique_ptr<FunctionData> WlsInferenceBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<WlsInferenceBindData>();

	// Get parameters
	auto &y_value = input.inputs[0];
	auto &x_value = input.inputs[1];
	auto &weights_value = input.inputs[2];

	// Parse options from fourth parameter if provided
	RegressionOptions options;
	if (input.inputs.size() > 3) {
		options = RegressionOptions::ParseFromMap(input.inputs[3]);
	}

	double confidence_level = options.confidence_level;
	bool add_intercept = options.intercept;

	// Extract y array
	vector<double> y_values;
	auto &y_list = ListValue::GetChildren(y_value);
	for (auto &val : y_list) {
		y_values.push_back(val.GetValue<double>());
	}

	idx_t n = y_values.size();

	// Extract X matrix
	vector<vector<double>> x_matrix;
	auto &x_outer_list = ListValue::GetChildren(x_value);

	for (auto &row_val : x_outer_list) {
		auto &row_list = ListValue::GetChildren(row_val);
		vector<double> row;
		for (auto &val : row_list) {
			row.push_back(val.GetValue<double>());
		}
		x_matrix.push_back(row);
	}

	// Extract weights
	vector<double> weights;
	auto &weights_list = ListValue::GetChildren(weights_value);
	for (auto &val : weights_list) {
		double w = val.GetValue<double>();
		if (w <= 0.0) {
			throw InvalidInputException("All weights must be positive, got %f", w);
		}
		weights.push_back(w);
	}

	if (weights.size() != n) {
		throw InvalidInputException("Length mismatch: y has %llu observations, weights has %llu", n, weights.size());
	}

	if (x_matrix.empty() || x_matrix[0].empty()) {
		throw InvalidInputException("X matrix cannot be empty");
	}

	idx_t p = x_matrix[0].size();

	if (x_matrix.size() != n) {
		throw InvalidInputException("Length mismatch: y has %llu observations, X has %llu", n, x_matrix.size());
	}

	idx_t n_params = p + (add_intercept ? 1 : 0);

	if (n <= n_params) {
		throw InvalidInputException(
		    "Insufficient observations: need at least %llu observations for %llu parameters, got %llu",
		    n_params + 1, n_params, n);
	}

	// Fit WLS with full output
	auto wls_result = bridge::LibanostatWrapper::FitWLS(y_values, x_matrix, weights, options, true);

	// Extract results
	Eigen::VectorXd coefficients = wls_result.coefficients;
	double intercept = wls_result.intercept;
	idx_t rank = wls_result.rank;
	idx_t df = n - rank;

	if (df == 0) {
		throw InvalidInputException("Not enough observations for effective parameters: n=%llu, rank=%llu", n, rank);
	}

	double mse = wls_result.mse;
	double alpha = 1.0 - confidence_level;
	double t_crit = student_t_critical(alpha / 2.0, static_cast<int>(df));

	// Extract standard errors (heteroscedasticity-consistent)
	Eigen::VectorXd std_errors_vec;
	if (wls_result.has_std_errors) {
		std_errors_vec = wls_result.std_errors;
	} else {
		std_errors_vec = Eigen::VectorXd::Constant(coefficients.size(), std::numeric_limits<double>::quiet_NaN());
	}

	// Process intercept
	if (add_intercept) {
		double intercept_se = std::numeric_limits<double>::quiet_NaN();

		if (wls_result.has_intercept && wls_result.has_std_errors && std::isfinite(wls_result.intercept_std_error)) {
			intercept_se = wls_result.intercept_std_error;
		} else if (wls_result.has_std_errors && std_errors_vec.size() > p) {
			intercept_se = std_errors_vec(p);
		} else {
			intercept_se = std::sqrt(mse / static_cast<double>(n));
		}

		double t_stat = intercept / intercept_se;
		double p_value = student_t_pvalue(t_stat, static_cast<int>(df));
		double ci_lower = intercept - t_crit * intercept_se;
		double ci_upper = intercept + t_crit * intercept_se;
		bool is_sig = p_value < alpha;

		bind_data->variable_names.push_back("intercept");
		bind_data->estimates.push_back(intercept);
		bind_data->std_errors.push_back(intercept_se);
		bind_data->t_statistics.push_back(t_stat);
		bind_data->p_values.push_back(p_value);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->significant.push_back(is_sig);

		ANOFOX_DEBUG("WLS Inference: intercept = " << intercept << " (SE=" << intercept_se << ")");
	}

	// Process coefficients
	for (idx_t j = 0; j < p; j++) {
		string var_name = "x" + std::to_string(j + 1);
		double estimate = coefficients(j);
		bool is_aliased = wls_result.is_aliased[j];

		double std_error, t_stat, p_value, ci_lower, ci_upper;
		bool is_sig;

		if (is_aliased || std::isnan(estimate)) {
			estimate = std::numeric_limits<double>::quiet_NaN();
			std_error = std::numeric_limits<double>::quiet_NaN();
			t_stat = std::numeric_limits<double>::quiet_NaN();
			p_value = std::numeric_limits<double>::quiet_NaN();
			ci_lower = std::numeric_limits<double>::quiet_NaN();
			ci_upper = std::numeric_limits<double>::quiet_NaN();
			is_sig = false;
		} else {
			std_error = std_errors_vec(j);
			t_stat = estimate / std_error;
			p_value = student_t_pvalue(t_stat, static_cast<int>(df));
			ci_lower = estimate - t_crit * std_error;
			ci_upper = estimate + t_crit * std_error;
			is_sig = p_value < alpha;
		}

		bind_data->variable_names.push_back(var_name);
		bind_data->estimates.push_back(estimate);
		bind_data->std_errors.push_back(std_error);
		bind_data->t_statistics.push_back(t_stat);
		bind_data->p_values.push_back(p_value);
		bind_data->ci_lowers.push_back(ci_lower);
		bind_data->ci_uppers.push_back(ci_upper);
		bind_data->significant.push_back(is_sig);
	}

	names = {"variable", "estimate", "std_error", "t_statistic", "p_value", "ci_lower", "ci_upper", "significant"};
	return_types = {LogicalType::VARCHAR, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::BOOLEAN};

	return std::move(bind_data);
}

static void WlsInferenceTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<WlsInferenceBindData>();

	idx_t n_results = bind_data.variable_names.size();
	idx_t rows_to_output = std::min((idx_t)STANDARD_VECTOR_SIZE, n_results - bind_data.current_row);

	if (rows_to_output == 0) {
		return;
	}

	output.SetCardinality(rows_to_output);

	auto var_data = FlatVector::GetData<string_t>(output.data[0]);
	auto est_data = FlatVector::GetData<double>(output.data[1]);
	auto se_data = FlatVector::GetData<double>(output.data[2]);
	auto t_data = FlatVector::GetData<double>(output.data[3]);
	auto p_data = FlatVector::GetData<double>(output.data[4]);
	auto ci_lower_data = FlatVector::GetData<double>(output.data[5]);
	auto ci_upper_data = FlatVector::GetData<double>(output.data[6]);
	auto sig_data = FlatVector::GetData<bool>(output.data[7]);

	auto &est_validity = FlatVector::Validity(output.data[1]);
	auto &se_validity = FlatVector::Validity(output.data[2]);
	auto &t_validity = FlatVector::Validity(output.data[3]);
	auto &p_validity = FlatVector::Validity(output.data[4]);
	auto &ci_lower_validity = FlatVector::Validity(output.data[5]);
	auto &ci_upper_validity = FlatVector::Validity(output.data[6]);

	for (idx_t i = 0; i < rows_to_output; i++) {
		idx_t idx = bind_data.current_row + i;

		var_data[i] = StringVector::AddString(output.data[0], bind_data.variable_names[idx]);

		double est = bind_data.estimates[idx];
		if (std::isnan(est)) {
			est_validity.SetInvalid(i);
			se_validity.SetInvalid(i);
			t_validity.SetInvalid(i);
			p_validity.SetInvalid(i);
			ci_lower_validity.SetInvalid(i);
			ci_upper_validity.SetInvalid(i);
			est_data[i] = 0.0;
			se_data[i] = 0.0;
			t_data[i] = 0.0;
			p_data[i] = 0.0;
			ci_lower_data[i] = 0.0;
			ci_upper_data[i] = 0.0;
		} else {
			est_data[i] = est;
			se_data[i] = bind_data.std_errors[idx];
			t_data[i] = bind_data.t_statistics[idx];
			p_data[i] = bind_data.p_values[idx];
			ci_lower_data[i] = bind_data.ci_lowers[idx];
			ci_upper_data[i] = bind_data.ci_uppers[idx];
		}

		sig_data[i] = bind_data.significant[idx];
	}

	bind_data.current_row += rows_to_output;
}

void WlsInferenceFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering WLS inference function");

	TableFunction wls_inference_func("anofox_statistics_wls_inference",
	                                  {LogicalType::LIST(LogicalType::DOUBLE),                    // y
	                                   LogicalType::LIST(LogicalType::LIST(LogicalType::DOUBLE)), // X
	                                   LogicalType::LIST(LogicalType::DOUBLE)},                   // weights
	                                  WlsInferenceTableFunc, WlsInferenceBind);

	// Add support for optional MAP parameter via varargs
	wls_inference_func.varargs = LogicalType::ANY;

	loader.RegisterFunction(wls_inference_func);

	ANOFOX_DEBUG("WLS inference function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
