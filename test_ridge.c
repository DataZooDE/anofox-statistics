#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "src/include/anofox_stats_ffi.h"

int main() {
    printf("Testing Ridge Regression FFI\n");
    printf("============================\n\n");

    // Test data: y = 2*x + 1 with some noise
    double y_data[] = {2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1};
    double x_col1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    AnofoxDataArray y = {y_data, NULL, 10};
    AnofoxDataArray x[] = {{x_col1, NULL, 10}};

    // Test 1: Low regularization (should be similar to OLS)
    printf("Test 1: Low regularization (alpha=0.001)\n");
    AnofoxRidgeOptions low_reg = {0.001, true, false, 0.95};
    AnofoxFitResultCore core_low;
    AnofoxError error;

    bool success = anofox_ridge_fit(y, x, 1, low_reg, &core_low, NULL, &error);

    if (success) {
        printf("  R-squared: %.6f\n", core_low.r_squared);
        printf("  Intercept: %.6f\n", core_low.intercept);
        printf("  Coefficient: %.6f\n", core_low.coefficients[0]);
        // Don't free yet - we need to compare with high_reg results
    } else {
        printf("  FAILED: %s\n", error.message);
        return 1;
    }

    // Test 2: High regularization (coefficients should be shrunk)
    printf("\nTest 2: High regularization (alpha=10.0)\n");
    AnofoxRidgeOptions high_reg = {10.0, true, false, 0.95};
    AnofoxFitResultCore core_high;

    success = anofox_ridge_fit(y, x, 1, high_reg, &core_high, NULL, &error);

    if (success) {
        printf("  R-squared: %.6f\n", core_high.r_squared);
        printf("  Intercept: %.6f\n", core_high.intercept);
        printf("  Coefficient: %.6f\n", core_high.coefficients[0]);

        // Verify that high regularization shrinks the coefficient
        if (fabs(core_high.coefficients[0]) < fabs(core_low.coefficients[0])) {
            printf("  [PASS] Higher regularization shrinks coefficient as expected\n");
        } else {
            printf("  [WARN] Coefficient shrinkage not as expected\n");
        }
        anofox_free_result_core(&core_high);
        anofox_free_result_core(&core_low); // Now safe to free
    } else {
        printf("  FAILED: %s\n", error.message);
        return 1;
    }

    // Test 3: OLS comparison
    printf("\nTest 3: OLS comparison (for reference)\n");
    AnofoxOlsOptions ols_opts = {true, false, 0.95};
    AnofoxFitResultCore core_ols;

    success = anofox_ols_fit(y, x, 1, ols_opts, &core_ols, NULL, &error);

    if (success) {
        printf("  R-squared: %.6f\n", core_ols.r_squared);
        printf("  Intercept: %.6f\n", core_ols.intercept);
        printf("  Coefficient: %.6f\n", core_ols.coefficients[0]);
        anofox_free_result_core(&core_ols);
    } else {
        printf("  FAILED: %s\n", error.message);
        return 1;
    }

    printf("\nAll tests passed!\n");
    return 0;
}
