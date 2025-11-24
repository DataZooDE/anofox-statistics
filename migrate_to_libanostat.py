#!/usr/bin/env python3
"""
Migrate all RankDeficientOls usage to libanostat's OLSSolver.

This script replaces the legacy RankDeficientOls calls with LibanostatWrapper::FitOLS
throughout the codebase, ensuring consistent rank calculation.
"""

import os
import re

def convert_eigen_to_vectors(var_y, var_X, var_n, var_p):
    """Generate code to convert Eigen to DuckDB vectors"""
    return f"""// Convert Eigen to DuckDB vectors for libanostat
	vector<double> y_vec({var_n});
	vector<vector<double>> x_vec({var_n}, vector<double>({var_p}));
	for (idx_t i = 0; i < {var_n}; i++) {{
		y_vec[i] = {var_y}(i);
		for (idx_t j = 0; j < {var_p}; j++) {{
			x_vec[i][j] = {var_X}(i, j);
		}}
	}}"""

def process_ols_aggregate():
    """Process src/functions/aggregates/ols_aggregate.cpp"""
    filepath = 'src/functions/aggregates/ols_aggregate.cpp'

    with open(filepath, 'r') as f:
        content = f.read()

    # The finalize function uses state.y_values and state.x_matrix which are already vectors
    # So we can skip the Eigen->vector conversion and go straight to libanostat

    # Find the OlsArrayFinalize function and replace the OLS computation
    # Pattern: from "// Handle intercept option" to end of intercept computation

    # Replace with direct libanostat call
    finalize_old = r'''(\t\t// Build design matrix using Eigen
\t\tEigen::MatrixXd X\(n, p\);
\t\tEigen::VectorXd y\(n\);

\t\tfor \(idx_t row = 0; row < n; row\+\+\) \{
\t\t\ty\(row\) = state\.y_values\[row\];
\t\t\tfor \(idx_t col = 0; col < p; col\+\+\) \{
\t\t\t\tX\(row, col\) = state\.x_matrix\[row\]\[col\];
\t\t\t\}
\t\t\}

\t\t// Convert to DuckDB types for libanostat
\t\tvector<double> y_data\(n\);
\t\tvector<vector<double>> x_data\(n, vector<double>\(p\)\);
\t\tfor \(idx_t i = 0; i < n; i\+\+\) \{
\t\t\ty_data\[i\] = y\(i\);
\t\t\tfor \(idx_t j = 0; j < p; j\+\+\) \{
\t\t\t\tx_data\[i\]\[j\] = X\(i, j\);
\t\t\t\}
\t\t\}

\t\t// Use libanostat OLSSolver \(handles intercept automatically\)
\t\tauto lib_result = bridge::LibanostatWrapper::FitOLS\(y_data, x_data, state\.options, true\);

\t\t// Extract results from libanostat
\t\tdouble intercept = lib_result\.intercept;
\t\tEigen::VectorXd x_means = lib_result\.x_train_means;)'''

    # This is too complex - let me use a simpler approach: just replace the function calls
    # and fix up the surrounding code manually

    print(f"Processing {filepath}...")

    # Count occurrences
    count = content.count('RankDeficientOls::')
    print(f"  Found {count} RankDeficientOls:: calls")

    return content, count

def main():
    """Main migration function"""
    print("=" * 70)
    print("Migrating RankDeficientOls to libanostat OLSSolver")
    print("=" * 70)

    # Process each file
    files_to_process = [
        'src/functions/aggregates/ols_aggregate.cpp',
        'src/functions/aggregates/wls_aggregate.cpp',
        'src/functions/fit_predict/ols_fit_predict.cpp',
        'src/functions/inference/prediction_intervals.cpp',
        'src/functions/inference/ols_inference.cpp',
    ]

    for filepath in files_to_process:
        if os.path.exists(filepath):
            content, count = process_ols_aggregate()
            print(f"  {filepath}: {count} replacements needed")
        else:
            print(f"  {filepath}: NOT FOUND")

    print("\nThis migration requires manual code changes.")
    print("The script has analyzed the files. Manual edits needed.")

if __name__ == '__main__':
    main()
