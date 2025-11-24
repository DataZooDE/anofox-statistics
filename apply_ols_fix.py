#!/usr/bin/env python3
import re

filepath = 'src/functions/aggregates/ols_aggregate.cpp'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find the start of the section to replace (line 438: "// Build design matrix using Eigen")
# End at line 482 (before "// Compute predictions using only non-aliased features")

# New code block
new_code = """\t\t// Use libanostat OLSSolver (handles intercept and centering automatically)
\t\tauto lib_result = bridge::LibanostatWrapper::FitOLS(state.y_values, state.x_matrix, state.options, true);

\t\t// Extract results
\t\tdouble intercept = lib_result.intercept;
\t\tEigen::VectorXd x_means = lib_result.x_train_means;
\t\tidx_t rank = lib_result.rank; // Already includes intercept if fitted!

\t\t// Compute predictions
\t\tEigen::VectorXd y_pred(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_pred(i) = intercept;
\t\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\t\tif (!std::isnan(lib_result.coefficients(j))) {
\t\t\t\t\ty_pred(i) += lib_result.coefficients(j) * state.x_matrix[i][j];
\t\t\t\t}
\t\t\t}
\t\t}

\t\t// Compute R²
\t\tEigen::VectorXd y_eigen(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_eigen(i) = state.y_values[i];
\t\t}
\t\tEigen::VectorXd residuals = y_eigen - y_pred;
\t\tdouble ss_res = residuals.squaredNorm();

\t\tdouble ss_tot;
\t\tif (state.options.intercept) {
\t\t\tdouble mean_y = 0.0;
\t\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\t\tmean_y += state.y_values[i];
\t\t\t}
\t\t\tmean_y /= n;
\t\t\tss_tot = 0.0;
\t\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\t\tdouble diff = state.y_values[i] - mean_y;
\t\t\t\tss_tot += diff * diff;
\t\t\t}
\t\t} else {
\t\t\tss_tot = 0.0;
\t\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\t\tss_tot += state.y_values[i] * state.y_values[i];
\t\t\t}
\t\t}

\t\tdouble r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;

\t\t// Adjusted R²: lib_result.rank already includes intercept
\t\tidx_t df_model = rank;
\t\tdouble adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - df_model);

"""

# Delete lines 438-510 (old computation code) and insert new code
# Line numbers are 0-indexed in Python, so line 438 is index 437
start_idx = 437  # Line 438
end_idx = 509    # Line 510 (the line with df_model calculation)

# Keep lines before start
output = lines[:start_idx]

# Add new code
output.append(new_code)

# Keep lines after end
output.extend(lines[end_idx+1:])

# Write back
with open(filepath, 'w') as f:
    f.writelines(output)

print(f"Replaced lines {start_idx+1}-{end_idx+1} in {filepath}")
print("New code uses LibanostatWrapper::FitOLS")
