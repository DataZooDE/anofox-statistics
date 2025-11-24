#!/usr/bin/env python3
"""
Replace RankDeficientOls with libanostat OLSSolver in ols_aggregate.cpp
"""

import re

# Read the file
with open('src/functions/aggregates/ols_aggregate.cpp', 'r') as f:
    content = f.read()

# Pattern 1: Replace the finalize function's OLS computation (lines ~447-480)
old_pattern1 = r'''(\t// Handle intercept option
\tdouble intercept = 0\.0;
\tRankDeficientOlsResult ols_result;

\t// Store x_means for later use
\tEigen::VectorXd x_means;

\tif \(state\.options\.intercept\) \{
\t\t// With intercept: center data, solve, compute intercept
\t\tdouble mean_y = y\.mean\(\);
\t\tx_means = X\.colwise\(\)\.mean\(\);

\t\tEigen::VectorXd y_centered = y\.array\(\) - mean_y;
\t\tEigen::MatrixXd X_centered = X;
\t\tfor \(idx_t j = 0; j < p; j\+\+\) \{
\t\t\tX_centered\.col\(j\)\.array\(\) -= x_means\(j\);
\t\t\}

\t\tols_result = RankDeficientOls::FitWithStdErrors\(y_centered, X_centered\);

\t\t// Compute intercept \(using only non-aliased features\)
\t\tdouble beta_dot_xmean = 0\.0;
\t\tfor \(idx_t j = 0; j < p; j\+\+\) \{
\t\t\tif \(!ols_result\.is_aliased\[j\]\) \{
\t\t\t\tbeta_dot_xmean \+= ols_result\.coefficients\[j\] \* x_means\(j\);
\t\t\t\}
\t\t\}
\t\tintercept = mean_y - beta_dot_xmean;
\t\} else \{
\t\t// No intercept: solve directly on raw data
\t\tols_result = RankDeficientOls::FitWithStdErrors\(y, X\);
\t\tintercept = 0\.0;
\t\tx_means = Eigen::VectorXd::Zero\(p\);
\t\})'''

new_code1 = '''\t// Convert to DuckDB types for libanostat
\tvector<double> y_data(n);
\tvector<vector<double>> x_data(n, vector<double>(p));
\tfor (idx_t i = 0; i < n; i++) {
\t\ty_data[i] = y(i);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tx_data[i][j] = X(i, j);
\t\t}
\t}

\t// Use libanostat OLSSolver (handles intercept automatically)
\tauto lib_result = bridge::LibanostatWrapper::FitOLS(y_data, x_data, state.options, true);

\t// Extract results from libanostat
\tdouble intercept = lib_result.intercept;
\tEigen::VectorXd x_means = lib_result.x_train_means;'''

# Try the replacement
if old_pattern1 in content:
    print("Found exact match for pattern 1")
    content = content.replace(old_pattern1, new_code1)
else:
    print("Pattern 1 not found, trying line-by-line replacement")
    # Simpler approach: just replace the function calls
    content = content.replace(
        'RankDeficientOls::FitWithStdErrors(y_centered, X_centered)',
        'PLACEHOLDER_LIBANOSTAT_1'
    )
    content = content.replace(
        'RankDeficientOls::FitWithStdErrors(y, X)',
        'PLACEHOLDER_LIBANOSTAT_2'
    )

print("Modified content (first few replacements)")
print("RankDeficientOls::FitWithStdErrors found:", content.count('RankDeficientOls::FitWithStdErrors'))
print("PLACEHOLDER found:", content.count('PLACEHOLDER_LIBANOSTAT'))

# Write it back
with open('src/functions/aggregates/ols_aggregate.cpp', 'w') as f:
    f.write(content)

print("Done! Please review the changes.")
