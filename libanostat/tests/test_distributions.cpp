#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <libanostat/utils/distributions.hpp>
#include <cmath>

using namespace libanostat::utils;

const double TOLERANCE = 1e-6;
const double LOOSE_TOLERANCE = 1e-4;  // For approximations

TEST_CASE("Distributions: Log Gamma Function", "[distributions][gamma]") {
	// Test known values of gamma function
	// Γ(1) = 1, log(Γ(1)) = 0
	REQUIRE_THAT(log_gamma(1.0), Catch::Matchers::WithinAbs(0.0, TOLERANCE));

	// Γ(2) = 1, log(Γ(2)) = 0
	REQUIRE_THAT(log_gamma(2.0), Catch::Matchers::WithinAbs(0.0, TOLERANCE));

	// Γ(3) = 2, log(Γ(3)) = log(2) ≈ 0.693147
	REQUIRE_THAT(log_gamma(3.0), Catch::Matchers::WithinAbs(0.693147180559945, TOLERANCE));

	// Γ(4) = 6, log(Γ(4)) = log(6) ≈ 1.791759
	REQUIRE_THAT(log_gamma(4.0), Catch::Matchers::WithinAbs(1.791759469228055, TOLERANCE));

	// Γ(5) = 24, log(Γ(5)) = log(24) ≈ 3.178054
	REQUIRE_THAT(log_gamma(5.0), Catch::Matchers::WithinAbs(3.178053830347946, TOLERANCE));

	// Test Stirling's approximation for large values
	// Γ(100) is very large, but log(Γ(100)) is manageable
	double log_gamma_100 = log_gamma(100.0);
	REQUIRE(log_gamma_100 > 359.0);  // Should be around 359.13
	REQUIRE(log_gamma_100 < 360.0);
}

TEST_CASE("Distributions: Log Gamma Recurrence Relation", "[distributions][gamma]") {
	// Test recurrence: Γ(x+1) = x·Γ(x)
	// log(Γ(x+1)) = log(x) + log(Γ(x))

	for (double x = 0.5; x < 10.0; x += 0.5) {
		double log_gamma_x = log_gamma(x);
		double log_gamma_x_plus_1 = log_gamma(x + 1.0);
		double expected = std::log(x) + log_gamma_x;

		REQUIRE_THAT(log_gamma_x_plus_1,
		             Catch::Matchers::WithinAbs(expected, TOLERANCE));
	}
}

TEST_CASE("Distributions: Log Beta Function", "[distributions][beta]") {
	// Test known values: B(a,b) = Γ(a)·Γ(b)/Γ(a+b)

	// B(1,1) = Γ(1)·Γ(1)/Γ(2) = 1·1/1 = 1, log(B(1,1)) = 0
	REQUIRE_THAT(log_beta(1.0, 1.0), Catch::Matchers::WithinAbs(0.0, TOLERANCE));

	// B(2,2) = Γ(2)·Γ(2)/Γ(4) = 1·1/6 = 1/6, log(B(2,2)) = log(1/6) ≈ -1.7918
	REQUIRE_THAT(log_beta(2.0, 2.0), Catch::Matchers::WithinAbs(-1.791759469228055, TOLERANCE));

	// B(3,2) = Γ(3)·Γ(2)/Γ(5) = 2·1/24 = 1/12, log(B(3,2)) = log(1/12) ≈ -2.4849
	REQUIRE_THAT(log_beta(3.0, 2.0), Catch::Matchers::WithinAbs(-2.484906649788000, TOLERANCE));

	// Test symmetry: B(a,b) = B(b,a)
	REQUIRE_THAT(log_beta(5.0, 3.0), Catch::Matchers::WithinAbs(log_beta(3.0, 5.0), TOLERANCE));
}

TEST_CASE("Distributions: Regularized Incomplete Beta Function", "[distributions][beta]") {
	// Test boundary conditions
	REQUIRE_THAT(beta_inc_reg(0.0, 1.0, 1.0), Catch::Matchers::WithinAbs(0.0, TOLERANCE));
	REQUIRE_THAT(beta_inc_reg(1.0, 1.0, 1.0), Catch::Matchers::WithinAbs(1.0, TOLERANCE));

	// I_0.5(1,1) = 0.5 (uniform distribution)
	REQUIRE_THAT(beta_inc_reg(0.5, 1.0, 1.0), Catch::Matchers::WithinAbs(0.5, TOLERANCE));

	// Test monotonicity: I_x should increase with x
	double prev = 0.0;
	for (double x = 0.1; x <= 0.9; x += 0.1) {
		double curr = beta_inc_reg(x, 2.0, 3.0);
		REQUIRE(curr > prev);
		prev = curr;
	}

	// Test symmetry property: I_x(a,b) = 1 - I_(1-x)(b,a)
	double x = 0.3;
	double a = 2.0, b = 3.0;
	double I_x_ab = beta_inc_reg(x, a, b);
	double I_1mx_ba = beta_inc_reg(1.0 - x, b, a);
	REQUIRE_THAT(I_x_ab + I_1mx_ba, Catch::Matchers::WithinAbs(1.0, LOOSE_TOLERANCE));
}

TEST_CASE("Distributions: Student's t CDF at Zero", "[distributions][student_t]") {
	// P(T ≤ 0) = 0.5 for all df
	for (int df = 1; df <= 30; df++) {
		REQUIRE_THAT(student_t_cdf(0.0, df), Catch::Matchers::WithinAbs(0.5, TOLERANCE));
	}
}

TEST_CASE("Distributions: Student's t CDF Symmetry", "[distributions][student_t]") {
	// P(T ≤ -t) = 1 - P(T ≤ t)
	std::vector<int> dfs = {1, 5, 10, 20, 30};
	std::vector<double> t_values = {0.5, 1.0, 1.5, 2.0, 2.5};

	for (int df : dfs) {
		for (double t : t_values) {
			double cdf_pos = student_t_cdf(t, df);
			double cdf_neg = student_t_cdf(-t, df);

			REQUIRE_THAT(cdf_pos + cdf_neg, Catch::Matchers::WithinAbs(1.0, LOOSE_TOLERANCE));
		}
	}
}

TEST_CASE("Distributions: Student's t CDF Known Values", "[distributions][student_t]") {
	// Test against known values from t-distribution tables

	// df=10, t=1.812 should give approximately 0.95 (one-tailed 95th percentile)
	double cdf_10_1812 = student_t_cdf(1.812, 10);
	REQUIRE_THAT(cdf_10_1812, Catch::Matchers::WithinAbs(0.95, 0.01));

	// df=10, t=2.228 should give approximately 0.975 (one-tailed 97.5th percentile)
	double cdf_10_2228 = student_t_cdf(2.228, 10);
	REQUIRE_THAT(cdf_10_2228, Catch::Matchers::WithinAbs(0.975, 0.01));

	// df=30, t=1.697 should give approximately 0.95
	double cdf_30_1697 = student_t_cdf(1.697, 30);
	REQUIRE_THAT(cdf_30_1697, Catch::Matchers::WithinAbs(0.95, 0.01));
}

TEST_CASE("Distributions: Student's t CDF Monotonicity", "[distributions][student_t]") {
	// CDF should be monotonically increasing
	int df = 10;
	double prev = 0.0;

	for (double t = -3.0; t <= 3.0; t += 0.5) {
		double curr = student_t_cdf(t, df);
		REQUIRE(curr >= prev);
		prev = curr;
	}
}

TEST_CASE("Distributions: Student's t P-values", "[distributions][student_t]") {
	// Test two-tailed p-values

	// t=0 should give p=1.0 (not significant)
	REQUIRE_THAT(student_t_pvalue(0.0, 10), Catch::Matchers::WithinAbs(1.0, TOLERANCE));

	// Small t-values should give large p-values
	double p_small = student_t_pvalue(0.5, 10);
	REQUIRE(p_small > 0.5);

	// Large t-values should give small p-values
	double p_large = student_t_pvalue(3.0, 10);
	REQUIRE(p_large < 0.05);

	// P-value should be symmetric for ±t
	double p_pos = student_t_pvalue(2.0, 20);
	double p_neg = student_t_pvalue(-2.0, 20);
	REQUIRE_THAT(p_pos, Catch::Matchers::WithinAbs(p_neg, TOLERANCE));
}

TEST_CASE("Distributions: Student's t Critical Values", "[distributions][student_t]") {
	// Test critical value approximations

	// For large df (>30), should approximate normal distribution
	double critical_95 = student_t_critical(0.05, 100);
	REQUIRE(critical_95 > 1.95);
	REQUIRE(critical_95 < 2.05);

	// For small df, critical values should be larger
	double critical_95_small_df = student_t_critical(0.05, 5);
	double critical_95_large_df = student_t_critical(0.05, 100);
	REQUIRE(critical_95_small_df > critical_95_large_df);

	// Smaller alpha should give larger critical values
	double critical_99 = student_t_critical(0.01, 30);
	double critical_95_30 = student_t_critical(0.05, 30);
	REQUIRE(critical_99 > critical_95_30);
}

TEST_CASE("Distributions: Large Degrees of Freedom Approximation", "[distributions][student_t]") {
	// For df > 200, t-distribution should approximate normal distribution
	// Standard normal: P(Z ≤ 1.96) ≈ 0.975

	double cdf_large_df = student_t_cdf(1.96, 250);
	REQUIRE_THAT(cdf_large_df, Catch::Matchers::WithinAbs(0.975, 0.01));

	// P(Z ≤ 0) = 0.5
	REQUIRE_THAT(student_t_cdf(0.0, 250), Catch::Matchers::WithinAbs(0.5, TOLERANCE));
}

TEST_CASE("Distributions: Edge Cases", "[distributions][student_t]") {
	// Test edge cases don't crash

	// Very small df
	double cdf_df1 = student_t_cdf(1.0, 1);
	REQUIRE(cdf_df1 > 0.5);
	REQUIRE(cdf_df1 < 1.0);

	// Very large t-values
	double cdf_large_t = student_t_cdf(10.0, 10);
	REQUIRE(cdf_large_t > 0.99);

	// Very negative t-values
	double cdf_neg_t = student_t_cdf(-10.0, 10);
	REQUIRE(cdf_neg_t < 0.01);
}

TEST_CASE("Distributions: Consistency Between CDF and P-value", "[distributions][student_t]") {
	// For a given t-statistic, p-value should be consistent with CDF
	// p-value = 2 * (1 - CDF(|t|))

	std::vector<double> t_values = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
	int df = 15;

	for (double t : t_values) {
		double pvalue = student_t_pvalue(t, df);
		double cdf = student_t_cdf(std::abs(t), df);
		double expected_pvalue = 2.0 * (1.0 - cdf);

		REQUIRE_THAT(pvalue, Catch::Matchers::WithinAbs(expected_pvalue, LOOSE_TOLERANCE));
	}
}
