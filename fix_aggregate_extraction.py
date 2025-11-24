#!/usr/bin/env python3
"""Fix the extraction of intercept and coefficients from libanostat result"""

filepath = 'src/functions/aggregates/ols_aggregate.cpp'

with open(filepath, 'r') as f:
    content = f.read()

# Fix 1: In OlsArrayFinalize, replace the extraction code
old_extract = '''// Extract results
\t\tdouble intercept = lib_result.intercept;
\t\tEigen::VectorXd x_means = lib_result.x_train_means;
\t\tidx_t rank = lib_result.rank; // Already includes intercept if fitted!'''

new_extract = '''// Extract results using TypeConverters
\t\tdouble intercept = bridge::TypeConverters::ExtractIntercept(lib_result, state.options.intercept);
\t\tEigen::VectorXd feature_coefs = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, state.options.intercept);
\t\tEigen::VectorXd x_means = bridge::TypeConverters::ExtractXTrainMeans(lib_result, state.options.intercept);
\t\tidx_t rank = lib_result.rank; // Already includes intercept if fitted!'''

content = content.replace(old_extract, new_extract)

# Fix 2: Update the prediction code to use feature_coefs
old_pred = '''// Compute predictions
\t\tEigen::VectorXd y_pred(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_pred(i) = intercept;
\t\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\t\tif (!std::isnan(lib_result.coefficients(j))) {
\t\t\t\t\ty_pred(i) += lib_result.coefficients(j) * state.x_matrix[i][j];
\t\t\t\t}
\t\t\t}
\t\t}'''

new_pred = '''// Compute predictions
\t\tEigen::VectorXd y_pred(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_pred(i) = intercept;
\t\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\t\tif (!std::isnan(feature_coefs(j))) {
\t\t\t\t\ty_pred(i) += feature_coefs(j) * state.x_matrix[i][j];
\t\t\t\t}
\t\t\t}
\t\t}'''

content = content.replace(old_pred, new_pred)

# Fix 3: Update coefficient storage to use feature_coefs
old_coef_store = '''// Store coefficients in child vector (NaN for aliased -> will be NULL)
\t\tauto coef_data = FlatVector::GetData<double>(coef_child);
\t\tauto &coef_validity = FlatVector::Validity(coef_child);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (std::isnan(lib_result.coefficients(j))) {
\t\t\t\t// Aliased coefficient -> set as invalid (NULL)
\t\t\t\tcoef_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_data[list_offset + j] = 0.0; // Placeholder value
\t\t\t} else {
\t\t\t\tcoef_data[list_offset + j] = lib_result.coefficients(j);
\t\t\t}
\t\t}'''

new_coef_store = '''// Store coefficients in child vector (NaN for aliased -> will be NULL)
\t\tauto coef_data = FlatVector::GetData<double>(coef_child);
\t\tauto &coef_validity = FlatVector::Validity(coef_child);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (std::isnan(feature_coefs(j))) {
\t\t\t\t// Aliased coefficient -> set as invalid (NULL)
\t\t\t\tcoef_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_data[list_offset + j] = 0.0; // Placeholder value
\t\t\t} else {
\t\t\t\tcoef_data[list_offset + j] = feature_coefs(j);
\t\t\t}
\t\t}'''

content = content.replace(old_coef_store, new_coef_store)

# Fix 4: Update intercept SE extraction
old_intercept_se = '''// 2. Intercept standard error
\t\tdouble intercept_se = std::numeric_limits<double>::quiet_NaN();
\t\tif (state.options.intercept && lib_result.has_std_errors && df_residual > 0) {
\t\t\tintercept_se = lib_result.intercept_std_error;
\t\t}'''

new_intercept_se = '''// 2. Intercept standard error
\t\tdouble intercept_se = std::numeric_limits<double>::quiet_NaN();
\t\tif (state.options.intercept && lib_result.has_std_errors && df_residual > 0) {
\t\t\tintercept_se = bridge::TypeConverters::ExtractInterceptStdError(lib_result, state.options.intercept);
\t\t}'''

content = content.replace(old_intercept_se, new_intercept_se)

# Fix 5: Update coef SE extraction
old_coef_se = '''// 4. Store coefficient_std_errors in list (same offset as coefficients)
\t\tauto coef_se_data = FlatVector::GetData<double>(coef_se_child);
\t\tauto &coef_se_validity = FlatVector::Validity(coef_se_child);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (lib_result.has_std_errors && !std::isnan(lib_result.std_errors(j))) {
\t\t\t\tcoef_se_data[list_offset + j] = lib_result.std_errors(j);
\t\t\t} else {
\t\t\t\tcoef_se_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_se_data[list_offset + j] = 0.0;
\t\t\t}
\t\t}'''

new_coef_se = '''// 4. Store coefficient_std_errors in list (same offset as coefficients)
\t\tauto coef_se_data = FlatVector::GetData<double>(coef_se_child);
\t\tauto &coef_se_validity = FlatVector::Validity(coef_se_child);
\t\tEigen::VectorXd feature_std_errors = bridge::TypeConverters::ExtractFeatureStdErrors(lib_result, state.options.intercept);
\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\tif (lib_result.has_std_errors && !std::isnan(feature_std_errors(j))) {
\t\t\t\tcoef_se_data[list_offset + j] = feature_std_errors(j);
\t\t\t} else {
\t\t\t\tcoef_se_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_se_data[list_offset + j] = 0.0;
\t\t\t}
\t\t}'''

content = content.replace(old_coef_se, new_coef_se)

# Now fix the window function similarly
# Fix 6: Window function extraction
old_window_extract = '''// Extract results
\t\tdouble intercept = lib_result.intercept;
\t\tidx_t rank = lib_result.rank; // Already includes intercept!'''

new_window_extract = '''// Extract results using TypeConverters
\t\tdouble intercept = bridge::TypeConverters::ExtractIntercept(lib_result, options.intercept);
\t\tEigen::VectorXd feature_coefs = bridge::TypeConverters::ExtractFeatureCoefficients(lib_result, options.intercept);
\t\tidx_t rank = lib_result.rank; // Already includes intercept!'''

content = content.replace(old_window_extract, new_window_extract)

# Fix 7: Window prediction code
old_window_pred = '''// Compute predictions
\t\tEigen::VectorXd y_pred(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_pred(i) = intercept;
\t\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\t\tif (!std::isnan(lib_result.coefficients(j))) {
\t\t\t\t\ty_pred(i) += lib_result.coefficients(j) * window_x[i][j];
\t\t\t\t}
\t\t\t}
\t\t}'''

new_window_pred = '''// Compute predictions
\t\tEigen::VectorXd y_pred(n);
\t\tfor (idx_t i = 0; i < n; i++) {
\t\t\ty_pred(i) = intercept;
\t\t\tfor (idx_t j = 0; j < p; j++) {
\t\t\t\tif (!std::isnan(feature_coefs(j))) {
\t\t\t\t\ty_pred(i) += feature_coefs(j) * window_x[i][j];
\t\t\t\t}
\t\t\t}
\t\t}'''

content = content.replace(old_window_pred, new_window_pred)

# Fix 8: Window coefficient storage
old_window_coef = '''if (std::isnan(ols_result.coefficients[j])) {
\t\t\t\tcoef_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_data[list_offset + j] = 0.0;
\t\t\t} else {
\t\t\t\tcoef_data[list_offset + j] = ols_result.coefficients[j];
\t\t\t}'''

new_window_coef = '''if (std::isnan(feature_coefs(j))) {
\t\t\t\tcoef_validity.SetInvalid(list_offset + j);
\t\t\t\tcoef_data[list_offset + j] = 0.0;
\t\t\t} else {
\t\t\t\tcoef_data[list_offset + j] = feature_coefs(j);
\t\t\t}'''

content = content.replace(old_window_coef, new_window_coef)

# Write back
with open(filepath, 'w') as f:
    f.write(content)

print("Fixed all extraction code to use TypeConverters")
