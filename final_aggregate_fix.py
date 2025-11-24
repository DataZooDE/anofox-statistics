#!/usr/bin/env python3
"""Final comprehensive fix for ols_aggregate.cpp"""

filepath = 'src/functions/aggregates/ols_aggregate.cpp'

with open(filepath, 'r') as f:
    content = f.read()

# The key insight: We need to convert vector<double> to Eigen::VectorXd
# And we need to compute x_means manually

# Fix all issues in one go by doing targeted replacements

# 1. Fix ExtractFeatureCoefficients - convert to Eigen
content = content.replace(
    'Eigen::VectorXd feature_coefs = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);',
    '''auto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);
\t\tEigen::VectorXd feature_coefs = Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());'''
)

# 2. Fix Extract ExtractXTrainMeans - compute manually
content = content.replace(
    'Eigen::VectorXd x_means = bridge::TypeConverters::ExtractXTrainMeans(lib_result, state.options.intercept);',
    '''// Compute x_means manually (libanostat doesn't store this)
\t\tEigen::VectorXd x_means(p);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tdouble sum = 0.0;
\t\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\t\tsum += state.x_matrix[i][j];
\t\t\t}
\t\t\tx_means(j) = sum / n;
\t\t}'''
)

# 3. Fix ols_result references in lines that weren't replaced
content = content.replace('if (std::isnan(ols_result.coefficients[j]))', 'if (std::isnan(feature_coefs(j)))')
content = content.replace('coef_data[list_offset + j] = ols_result.coefficients[j];', 'coef_data[list_offset + j] = feature_coefs(j);')

# 4. Fix intercept SE and has_std_errors references
content = content.replace(
    'if (state.options.intercept && ols_result.has_std_errors && df_residual > 0) {',
    'if (state.options.intercept && lib_result.has_std_errors && df_residual > 0) {'
)

content = content.replace(
    'if (ols_result.has_std_errors && !std::isnan(ols_result.std_errors(j))) {',
    'if (lib_result.has_std_errors && !std::isnan(feature_std_errors(j))) {'
)

# 5. Fix window function extraction (similar issues)
content = content.replace(
    'Eigen::VectorXd feature_coefs = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);',
    '''auto feature_coefs_vec = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);
\t\tEigen::VectorXd feature_coefs = Eigen::Map<const Eigen::VectorXd>(feature_coefs_vec.data(), feature_coefs_vec.size());'''
)

# Write back
with open(filepath, 'w') as f:
    f.write(content)

print("Applied final comprehensive fixes to ols_aggregate.cpp")
