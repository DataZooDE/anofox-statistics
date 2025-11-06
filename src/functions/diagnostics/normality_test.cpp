#include "normality_test.hpp"
#include "../utils/tracing.hpp"
#include "../utils/validation.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"

#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox_statistics {

/**
 * Chi-square CDF approximation for df=2
 * Used for Jarque-Bera p-value
 */
static double chi_square_cdf_df2(double x) {
	if (x <= 0)
		return 0.0;
	// For df=2: CDF = 1 - exp(-x/2)
	return 1.0 - std::exp(-x / 2.0);
}

/**
 * Bind data for normality test
 */
struct NormalityTestBindData : public FunctionData {
	idx_t n_obs;
	double skewness;
	double kurtosis;
	double jb_statistic;
	double p_value;
	bool is_normal;
	string conclusion;

	bool result_returned = false;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<NormalityTestBindData>();
		result->n_obs = n_obs;
		result->skewness = skewness;
		result->kurtosis = kurtosis;
		result->jb_statistic = jb_statistic;
		result->p_value = p_value;
		result->is_normal = is_normal;
		result->conclusion = conclusion;
		result->result_returned = result_returned;
		return std::move(result);
	}

	bool Equals(const FunctionData &other) const override {
		return false;
	}
};

/**
 * Bind function - Compute normality test
 */
static unique_ptr<FunctionData> NormalityTestBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {

	auto bind_data = make_uniq<NormalityTestBindData>();

	// Get parameters
	auto &residuals_value = input.inputs[0];

	double alpha = 0.05;
	if (input.inputs.size() > 1 && !input.inputs[1].IsNull()) {
		alpha = input.inputs[1].GetValue<double>();
	}

	// Extract residuals array
	vector<double> residuals;
	auto &resid_list = ListValue::GetChildren(residuals_value);
	for (auto &val : resid_list) {
		residuals.push_back(val.GetValue<double>());
	}

	idx_t n = residuals.size();

	if (n < 4) {
		throw InvalidInputException("Need at least 4 observations for normality test, got %llu", n);
	}

	// Compute mean
	double mean = 0.0;
	for (double val : residuals) {
		mean += val;
	}
	mean /= n;

	// Compute moments
	double m2 = 0.0; // Second moment (variance)
	double m3 = 0.0; // Third moment
	double m4 = 0.0; // Fourth moment

	for (double val : residuals) {
		double dev = val - mean;
		double dev2 = dev * dev;
		m2 += dev2;
		m3 += dev2 * dev;
		m4 += dev2 * dev2;
	}

	m2 /= n;
	m3 /= n;
	m4 /= n;

	double sigma = std::sqrt(m2);

	// Skewness: E[(X - μ)³] / σ³
	double skewness = m3 / (sigma * sigma * sigma);

	// Kurtosis: E[(X - μ)⁴] / σ⁴
	double kurtosis = m4 / (m2 * m2);

	// Jarque-Bera statistic: JB = n/6 * (S² + (K-3)²/4)
	double S2 = skewness * skewness;
	double K_minus_3 = kurtosis - 3.0;
	double jb = (n / 6.0) * (S2 + 0.25 * K_minus_3 * K_minus_3);

	// P-value from chi-square(2) distribution
	double p_value = 1.0 - chi_square_cdf_df2(jb);

	// Decision
	bool is_normal = p_value > alpha;
	string conclusion = is_normal ? "normal" : "non-normal";

	// Store results
	bind_data->n_obs = n;
	bind_data->skewness = skewness;
	bind_data->kurtosis = kurtosis;
	bind_data->jb_statistic = jb;
	bind_data->p_value = p_value;
	bind_data->is_normal = is_normal;
	bind_data->conclusion = conclusion;

	ANOFOX_DEBUG("Normality test: JB=" << jb << ", p=" << p_value << ", skew=" << skewness << ", kurt=" << kurtosis);

	// Define return types
	names = {"n_obs", "skewness", "kurtosis", "jb_statistic", "p_value", "is_normal", "conclusion"};
	return_types = {LogicalType::BIGINT, LogicalType::DOUBLE,  LogicalType::DOUBLE, LogicalType::DOUBLE,
	                LogicalType::DOUBLE, LogicalType::BOOLEAN, LogicalType::VARCHAR};

	return std::move(bind_data);
}

/**
 * Table function implementation
 */
static void NormalityTestTableFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = data_p.bind_data->CastNoConst<NormalityTestBindData>();

	if (bind_data.result_returned) {
		return;
	}

	output.SetCardinality(1);

	auto n_obs_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto skew_data = FlatVector::GetData<double>(output.data[1]);
	auto kurt_data = FlatVector::GetData<double>(output.data[2]);
	auto jb_data = FlatVector::GetData<double>(output.data[3]);
	auto p_data = FlatVector::GetData<double>(output.data[4]);
	auto normal_data = FlatVector::GetData<bool>(output.data[5]);
	auto conclusion_data = FlatVector::GetData<string_t>(output.data[6]);

	n_obs_data[0] = bind_data.n_obs;
	skew_data[0] = bind_data.skewness;
	kurt_data[0] = bind_data.kurtosis;
	jb_data[0] = bind_data.jb_statistic;
	p_data[0] = bind_data.p_value;
	normal_data[0] = bind_data.is_normal;
	conclusion_data[0] = StringVector::AddString(output.data[6], bind_data.conclusion);

	bind_data.result_returned = true;
}

void NormalityTestFunction::Register(ExtensionLoader &loader) {
	ANOFOX_DEBUG("Registering normality test function");

	TableFunction normality_test_func("anofox_statistics_normality_test",
	                                  {LogicalType::LIST(LogicalType::DOUBLE), // residuals
	                                   LogicalType::DOUBLE},                   // alpha
	                                  NormalityTestTableFunc, NormalityTestBind);

	// Set named parameters
	normality_test_func.named_parameters["residuals"] = LogicalType::LIST(LogicalType::DOUBLE);
	normality_test_func.named_parameters["alpha"] = LogicalType::DOUBLE;

	loader.RegisterFunction(normality_test_func);

	ANOFOX_DEBUG("Normality test function registered successfully");
}

} // namespace anofox_statistics
} // namespace duckdb
