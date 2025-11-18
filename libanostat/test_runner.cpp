#include <iostream>
#include <cmath>
#include <vector>
#include <libanostat/utils/distributions.hpp>

using namespace libanostat::utils;

int main() {
    int tests_passed = 0;
    int tests_failed = 0;

    std::cout << "Running libanostat unit tests..." << std::endl << std::endl;

    // Test 1: Log gamma function
    std::cout << "[TEST 1] Log gamma function" << std::endl;
    double lg1 = log_gamma(1.0);
    if (std::abs(lg1 - 0.0) < 1e-6) {
        std::cout << "  PASS: log_gamma(1.0) = " << lg1 << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: log_gamma(1.0) = " << lg1 << ", expected 0.0" << std::endl;
        tests_failed++;
    }

    double lg3 = log_gamma(3.0);
    if (std::abs(lg3 - 0.693147180559945) < 1e-6) {
        std::cout << "  PASS: log_gamma(3.0) = " << lg3 << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: log_gamma(3.0) = " << lg3 << ", expected 0.693147" << std::endl;
        tests_failed++;
    }

    double lg4 = log_gamma(4.0);
    if (std::abs(lg4 - 1.791759469228055) < 1e-6) {
        std::cout << "  PASS: log_gamma(4.0) = " << lg4 << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: log_gamma(4.0) = " << lg4 << ", expected 1.791759" << std::endl;
        tests_failed++;
    }

    // Test 2: Student's t CDF
    std::cout << std::endl << "[TEST 2] Student's t CDF" << std::endl;
    double t_cdf = student_t_cdf(0.0, 10);
    if (std::abs(t_cdf - 0.5) < 1e-6) {
        std::cout << "  PASS: student_t_cdf(0.0, df=10) = " << t_cdf << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: student_t_cdf(0.0, df=10) = " << t_cdf << ", expected 0.5" << std::endl;
        tests_failed++;
    }

    // Test symmetry: P(T ≤ -t) = 1 - P(T ≤ t)
    double t_cdf_pos = student_t_cdf(2.0, 10);
    double t_cdf_neg = student_t_cdf(-2.0, 10);
    if (std::abs(t_cdf_pos + t_cdf_neg - 1.0) < 1e-4) {
        std::cout << "  PASS: t-distribution symmetry check" << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: t-distribution symmetry failed" << std::endl;
        tests_failed++;
    }

    // Test 3: T-distribution p-values
    std::cout << std::endl << "[TEST 3] Student's t p-values" << std::endl;
    double p_zero = student_t_pvalue(0.0, 10);
    if (std::abs(p_zero - 1.0) < 0.01) {
        std::cout << "  PASS: student_t_pvalue(0.0, df=10) = " << p_zero << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: student_t_pvalue(0.0, df=10) = " << p_zero << ", expected ~1.0" << std::endl;
        tests_failed++;
    }

    double p_large = student_t_pvalue(3.0, 10);
    if (p_large < 0.05) {
        std::cout << "  PASS: student_t_pvalue(3.0, df=10) = " << p_large << " (significant)" << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: student_t_pvalue(3.0, df=10) = " << p_large << ", expected < 0.05" << std::endl;
        tests_failed++;
    }

    // Test 4: Beta function
    std::cout << std::endl << "[TEST 4] Log beta function" << std::endl;
    double lb11 = log_beta(1.0, 1.0);
    if (std::abs(lb11 - 0.0) < 1e-6) {
        std::cout << "  PASS: log_beta(1.0, 1.0) = " << lb11 << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: log_beta(1.0, 1.0) = " << lb11 << ", expected 0.0" << std::endl;
        tests_failed++;
    }

    // Test symmetry: B(a,b) = B(b,a)
    double lb53 = log_beta(5.0, 3.0);
    double lb35 = log_beta(3.0, 5.0);
    if (std::abs(lb53 - lb35) < 1e-6) {
        std::cout << "  PASS: log_beta symmetry check" << std::endl;
        tests_passed++;
    } else {
        std::cout << "  FAIL: log_beta symmetry failed" << std::endl;
        tests_failed++;
    }

    // Summary
    std::cout << std::endl << "===============================================" << std::endl;
    std::cout << "Test Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
    std::cout << "===============================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
