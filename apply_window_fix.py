#!/usr/bin/env python3
import re

filepath = 'src/functions/aggregates/ols_aggregate.cpp'

with open(filepath, 'r') as f:
    content = f.read()

# Find and replace the window function's OLS computation
# The window function builds all_y and all_x vectors, then extracts window_y and window_x

# Look for the section starting with "// Handle intercept option" in OlsArrayWindow function
# and ending before "// Store coefficients in list"

# Pattern to find: from "double intercept = 0.0;" to the end of the intercept computation
old_window_pattern = '''// Handle intercept option
\tdouble intercept = 0.0;
\tRankDeficientOlsResult ols_result;

\tif (options.intercept) {
\t\t// With intercept: center data, solve, compute intercept
\t\tdouble mean_y = y.mean();
\t\tEigen::VectorXd x_means = X.colwise().mean();

\t\tEigen::VectorXd y_centered = y.array() - mean_y;
\t\tEigen::MatrixXd X_centered = X;
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tX_centered.col(j).array() -= x_means(j);
\t\t}

\t\tols_result = RankDeficientOls::Fit(y_centered, X_centered);

\t\t// Compute intercept (using only non-aliased features)
\t\tdouble beta_dot_xmean = 0.0;
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (!ols_result.is_aliased[j]) {
\t\t\t\tbeta_dot_xmean += ols_result.coefficients[j] * x_means(j);
\t\t\t}
\t\t}
\t\tintercept = mean_y - beta_dot_xmean;
\t} else {
\t\t// No intercept: solve directly on raw data
\t\tols_result = RankDeficientOls::Fit(y, X);
\t\tintercept = 0.0;
\t}

\t// Compute predictions and R²
\tEigen::VectorXd y_pred = Eigen::VectorXd::Constant(n, intercept);
\tfor (idx_t j = 0; j < p; j++) {
\t\tif (!ols_result.is_aliased[j]) {
\t\t\ty_pred += ols_result.coefficients[j] * X.col(j);
\t\t}
\t}

\tEigen::VectorXd residuals = y - y_pred;
\tdouble ss_res = residuals.squaredNorm();

\tdouble ss_tot;
\tif (options.intercept) {
\t\tdouble mean_y = y.mean();
\t\tss_tot = (y.array() - mean_y).square().sum();
\t} else {
\t\t// No intercept: total sum of squares from zero
\t\tss_tot = y.squaredNorm();
\t}

\tdouble r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

\t// Adjusted R²: RankDeficientOls.rank is feature rank only (doesn't include intercept)
\t// Total model rank = feature_rank + (intercept ? 1 : 0)
\tidx_t df_model = ols_result.rank + (options.intercept ? 1 : 0);
\tdouble adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);'''

new_window_code = '''// Convert to DuckDB vectors for libanostat
\tvector<double> y_vec(n);
\tvector<vector<double>> x_vec(n, vector<double>(p));
\tfor (idx_t i = 0; i < n; i++) {
\t\ty_vec[i] = window_y[i];
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tx_vec[i][j] = window_x[i][j];
\t\t}
\t}

\t// Use libanostat OLSSolver
\tauto lib_result = bridge::LibanostatWrapper::FitOLS(y_vec, x_vec, options, false);

\t// Extract results
\tdouble intercept = lib_result.intercept;
\tidx_t rank = lib_result.rank; // Already includes intercept!

\t// Compute predictions
\tEigen::VectorXd y_pred(n);
\tfor (idx_t i = 0; i < n; i++) {
\t\ty_pred(i) = intercept;
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (!std::isnan(lib_result.coefficients(j))) {
\t\t\t\ty_pred(i) += lib_result.coefficients(j) * window_x[i][j];
\t\t\t}
\t\t}
\t}

\t// Compute R²
\tEigen::VectorXd y_eigen(n);
\tfor (idx_t i = 0; i < n; i++) {
\t\ty_eigen(i) = window_y[i];
\t}
\tEigen::VectorXd residuals = y_eigen - y_pred;
\tdouble ss_res = residuals.squaredNorm();

\tdouble ss_tot;
\tif (options.intercept) {
\t\tdouble mean_y = 0.0;
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\tmean_y += window_y[i];
\t\t}
\t\tmean_y /= n;
\t\tss_tot = 0.0;
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\tdouble diff = window_y[i] - mean_y;
\t\t\tss_tot += diff * diff;
\t\t}
\t} else {
\t\tss_tot = 0.0;
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\tss_tot += window_y[i] * window_y[i];
\t\t}
\t}

\tdouble r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

\t// Adjusted R²: lib_result.rank already includes intercept
\tidx_t df_model = rank;
\tdouble adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);'''

# Replace
if old_window_pattern in content:
    content = content.replace(old_window_pattern, new_window_code)
    print("Window function pattern found and replaced!")

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated {filepath}")
else:
    print("Window pattern not found - trying line-by-line approach")
    # Fallback: read as lines and replace by line numbers
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the OlsArrayWindow function
    in_window_func = False
    start_idx = -1
    for i, line in enumerate(lines):
        if 'static void OlsArrayWindow' in line:
            in_window_func = True
            print(f"Found OlsArrayWindow at line {i+1}")
        if in_window_func and '\t// Handle intercept option' in line:
            start_idx = i
            print(f"Found start of replacement section at line {i+1}")
            break

    if start_idx != -1:
        # Find the end (before "// Store coefficients in list")
        end_idx = -1
        for i in range(start_idx, len(lines)):
            if '\t// Store coefficients in list' in lines[i]:
                end_idx = i - 1
                print(f"Found end of replacement section at line {i}")
                break

        if end_idx != -1:
            output = lines[:start_idx]
            output.append('\t' + new_window_code + '\n\n')
            output.extend(lines[end_idx+1:])

            with open(filepath, 'w') as f:
                f.writelines(output)
            print(f"Replaced window function code (lines {start_idx+1}-{end_idx+1})")
        else:
            print("Could not find end marker")
    else:
        print("Could not find start marker")
