#!/usr/bin/env python3
"""
Migrate WLS aggregate to use libanostat.
Similar to OLS but uses FitWLS with weights.
"""

filepath = 'src/functions/aggregates/wls_aggregate.cpp'

with open(filepath, 'r') as f:
    lines = f.readlines()

# 1. Add includes if not present
includes_added = False
for i, line in enumerate(lines):
    if '#include "../utils/options_parser.hpp"' in line:
        # Check if bridge includes are already there
        if i+1 < len(lines) and 'libanostat_wrapper' not in lines[i+1]:
            lines.insert(i+1, '#include "../bridge/libanostat_wrapper.hpp"\n')
            lines.insert(i+2, '#include "../bridge/type_converters.hpp"\n')
            includes_added = True
            print("Added bridge includes")
        break

# 2. Find and replace the WLS finalize function's computation
# Lines ~221-271: from "// Handle intercept option" to end of intercept computation

# Find the start of WLS computation in finalize
for i, line in enumerate(lines):
    if i > 190 and i < 230 and '\t\t// Handle intercept option' in line:
        print(f"Found WLS finalize computation start at line {i+1}")
        # Find the end (before "// Compute predictions")
        end_idx = -1
        for j in range(i, min(i+60, len(lines))):
            if '\t\t// Compute predictions' in lines[j]:
                end_idx = j
                print(f"Found end at line {j+1}")
                break

        if end_idx != -1:
            # Replace this entire section with libanostat call
            new_code = '''\t\t// Use libanostat WLSSolver (handles weighting and intercept automatically)
\t\tauto lib_result = bridge::LibanostatWrapper::FitWLS(state.y_values, state.x_matrix, state.weights, state.options, true);

\t\t// Extract results
\t\tdouble intercept = bridge::TypeConverters::ExtractIntercept(lib_result, state.options.intercept);
\t\tauto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);
\t\tEigen::VectorXd feature_coefs = Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());
\t\tidx_t rank = lib_result.rank; // Already includes intercept!

\t\t// Compute x_means manually for metadata
\t\t// For WLS, we need weighted means
\t\tEigen::VectorXd w_eigen(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\tw_eigen(i) = state.weights[i];
\t\t}
\t\tdouble sum_weights = w_eigen.sum();
\t\tEigen::VectorXd x_means(p);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tdouble weighted_sum = 0.0;
\t\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\t\tweighted_sum += state.weights[i] * state.x_matrix[i][j];
\t\t\t}
\t\t\tx_means(j) = weighted_sum / sum_weights;
\t\t}

'''
            # Delete old lines and insert new
            output = lines[:i]
            output.append(new_code)
            output.extend(lines[end_idx:])
            lines = output
            print(f"Replaced WLS finalize computation (lines {i+1}-{end_idx})")
        break

# 3. Similarly find and replace in window function (around lines 520-570)
for i, line in enumerate(lines):
    if i > 490 and i < 550 and '\t// Handle intercept option' in line:
        print(f"Found WLS window computation start at line {i+1}")
        # Find the end
        end_idx = -1
        for j in range(i, min(i+60, len(lines))):
            if '\t// Compute predictions' in lines[j]:
                end_idx = j
                print(f"Found window end at line {j+1}")
                break

        if end_idx != -1:
            # Replace with libanostat call
            new_code = '''\t// Convert to DuckDB vectors for libanostat
\tvector<double> y_vec(n);
\tvector<double> w_vec(n);
\tvector<vector<double>> x_vec(n, vector<double>(p));
\tfor (idx_t i = 0; i < n; i++) {
\t\ty_vec[i] = window_y[i];
\t\tw_vec[i] = window_w[i];
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tx_vec[i][j] = window_x[i][j];
\t\t}
\t}

\t// Use libanostat WLSSolver
\tauto lib_result = bridge::LibanostatWrapper::FitWLS(y_vec, x_vec, w_vec, options, false);

\t// Extract results
\tdouble intercept = bridge::TypeConverters::ExtractIntercept(lib_result, options.intercept);
\tauto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);
\tEigen::VectorXd feature_coefs = Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());
\tidx_t rank = lib_result.rank;

'''
            output = lines[:i]
            output.append(new_code)
            output.extend(lines[end_idx:])
            lines = output
            print(f"Replaced WLS window computation (lines {i+1}-{end_idx})")
        break

# Write back
with open(filepath, 'w') as f:
    f.writelines(lines)

print("WLS migration applied - needs testing")
