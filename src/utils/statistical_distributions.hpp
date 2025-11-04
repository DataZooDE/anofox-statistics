#pragma once

#include <cmath>
#include <algorithm>

namespace duckdb {
namespace anofox_statistics {

/**
 * Statistical distribution functions
 */

// Log of gamma function using Stirling's approximation + correction terms
inline double log_gamma(double x) {
	// For small values, use lookup/recurrence
	if (x < 12.0) {
		if (x < 1.0) {
			// Use recurrence: Γ(x+1) = x·Γ(x)
			return log_gamma(x + 1.0) - std::log(x);
		}
		// Lookup table for small integers
		static const double log_gamma_table[] = {
		    0.0,                    // log(Γ(1)) = log(1) = 0
		    0.0,                    // log(Γ(2)) = log(1) = 0
		    0.69314718055994530942, // log(Γ(3)) = log(2)
		    1.79175946922805500081, // log(Γ(4)) = log(6)
		    3.17805383034794561964, // log(Γ(5)) = log(24)
		    4.78749174278204599424, // log(Γ(6)) = log(120)
		    6.57925121201010099507, // log(Γ(7)) = log(720)
		    8.52516136106541430016, // log(Γ(8)) = log(5040)
		    10.6046029027452502481, // log(Γ(9)) = log(40320)
		    12.8018274800814696112, // log(Γ(10)) = log(362880)
		    15.1044125730755153447, // log(Γ(11)) = log(3628800)
		    17.5023078458738858990  // log(Γ(12)) = log(39916800)
		};
		int n = static_cast<int>(x);
		if (x == n && n >= 1 && n < 12) {
			return log_gamma_table[n];
		}
		// Use recurrence to shift to larger x
		return log_gamma(x + 1.0) - std::log(x);
	}

	// Stirling's approximation for large x
	// log(Γ(x)) ≈ (x-0.5)·log(x) - x + 0.5·log(2π) + corrections
	const double log_2pi = 1.83787706640934548356;
	double log_result = (x - 0.5) * std::log(x) - x + 0.5 * log_2pi;

	// Add correction terms for better accuracy
	double x_inv = 1.0 / x;
	double x_inv2 = x_inv * x_inv;
	log_result += x_inv * (1.0 / 12.0 - x_inv2 * (1.0 / 360.0 - x_inv2 / 1260.0));

	return log_result;
}

// Log of beta function: log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))
inline double log_beta(double a, double b) {
	return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

// Regularized incomplete beta function I_x(a, b) using continued fraction
// This is the key function for computing the t-distribution CDF
inline double beta_inc_reg(double x, double a, double b) {
	// Handle edge cases
	if (x <= 0.0)
		return 0.0;
	if (x >= 1.0)
		return 1.0;

	// For numerical stability, use symmetry when x > (a+1)/(a+b+2)
	bool use_symmetry = (x > (a + 1.0) / (a + b + 2.0));
	if (use_symmetry) {
		return 1.0 - beta_inc_reg(1.0 - x, b, a);
	}

	// Compute log of incomplete beta function prefix
	double log_prefix = a * std::log(x) + b * std::log(1.0 - x) - log_beta(a, b);
	double prefix = std::exp(log_prefix);

	// Continued fraction expansion using modified Lentz's method
	const double epsilon = 1.0e-12;
	const int max_iter = 200;

	// Initialize continued fraction
	double f = 1.0;
	double c = 1.0;
	double d = 0.0;

	// First iteration (m=0)
	int m = 0;
	double numerator = 1.0;
	d = 1.0 / (1.0 - (a + b) * x / (a + 1.0));
	c = 1.0;
	f = d;

	// Iterate continued fraction
	for (m = 1; m <= max_iter; m++) {
		double m_d = static_cast<double>(m);

		// Even term: a_2m = m(b-m)x / [(a+2m-1)(a+2m)]
		double numerator_even = m_d * (b - m_d) * x / ((a + 2.0 * m_d - 1.0) * (a + 2.0 * m_d));

		// Update d and c for even term
		d = 1.0 + numerator_even * d;
		if (std::abs(d) < epsilon)
			d = epsilon;
		d = 1.0 / d;

		c = 1.0 + numerator_even / c;
		if (std::abs(c) < epsilon)
			c = epsilon;

		f = f * d * c;

		// Odd term: a_2m+1 = -(a+m)(a+b+m)x / [(a+2m)(a+2m+1)]
		double numerator_odd = -(a + m_d) * (a + b + m_d) * x / ((a + 2.0 * m_d) * (a + 2.0 * m_d + 1.0));

		// Update d and c for odd term
		d = 1.0 + numerator_odd * d;
		if (std::abs(d) < epsilon)
			d = epsilon;
		d = 1.0 / d;

		c = 1.0 + numerator_odd / c;
		if (std::abs(c) < epsilon)
			c = epsilon;

		double delta = d * c;
		f = f * delta;

		// Check convergence
		if (std::abs(delta - 1.0) < epsilon) {
			break;
		}
	}

	// Return regularized incomplete beta: I_x(a,b) = prefix * f / a
	return prefix * f / a;
}

// Critical value for t-distribution (approximation)
inline double student_t_critical(double alpha, int df) {
	// For large df, use normal approximation
	if (df > 30) {
		// Standard normal critical value for alpha/2
		if (alpha < 0.001)
			return 3.291; // 99.9%
		if (alpha < 0.01)
			return 2.807; // 99%
		if (alpha < 0.05)
			return 2.042; // 95%
		if (alpha < 0.10)
			return 1.697; // 90%
		return 1.96;      // Default to 95%
	} else {
		// Use lookup table adjustment for small df
		double factor = 1.0 + 2.0 / df; // Adjustment for small df
		return student_t_critical(alpha, 100) * factor;
	}
}

// Student's t CDF (cumulative distribution function)
// Uses the relationship: P(T ≤ t) = 1 - 0.5·I_x(df/2, 1/2)
// where x = df/(df + t²) and I_x is the regularized incomplete beta function
inline double student_t_cdf(double t, int df) {
	if (df <= 0)
		return 0.5; // Invalid df

	// Handle special case: t = 0
	if (t == 0.0)
		return 0.5;

	// For very large df (>200), use normal approximation
	if (df > 200) {
		// Standard normal CDF using error function
		return 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)));
	}

	// Use the incomplete beta function relationship
	// P(T ≤ t | df) = 1 - 0.5 * I_x(df/2, 1/2)  for t > 0
	// where x = df/(df + t²)

	double t_abs = std::abs(t);
	double df_d = static_cast<double>(df);
	double x = df_d / (df_d + t_abs * t_abs);
	double a = df_d / 2.0;
	double b = 0.5;

	// Compute CDF using incomplete beta
	double cdf_abs_t = 1.0 - 0.5 * beta_inc_reg(x, a, b);

	// Use symmetry for negative t
	if (t < 0.0) {
		return 1.0 - cdf_abs_t;
	} else {
		return cdf_abs_t;
	}
}

// Two-tailed p-value for t-statistic
inline double student_t_pvalue(double t_stat, int df) {
	double abs_t = std::abs(t_stat);
	double p_upper = 1.0 - student_t_cdf(abs_t, df);
	return 2.0 * p_upper; // Two-tailed
}

} // namespace anofox_statistics
} // namespace duckdb
