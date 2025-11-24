#!/usr/bin/env python3
"""Migrate wls_aggregate.cpp to use libanostat"""

filepath = 'src/functions/aggregates/wls_aggregate.cpp'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find the includes section and add libanostat bridge
includes_added = False
for i, line in enumerate(lines):
    if '#include "../utils/options_parser.hpp"' in line and not includes_added:
        lines.insert(i+1, '#include "../bridge/libanostat_wrapper.hpp"\n')
        lines.insert(i+2, '#include "../bridge/type_converters.hpp"\n')
        includes_added = True
        print(f"Added includes after line {i+1}")
        break

# Strategy: Replace the manual weighting and OLS solve with LibanostatWrapper::FitWLS
# This is complex because there are 4 locations (2 in finalize, 2 in window)

# For finalize function: find section starting around line 215-270
# Pattern: from "// Weighted means for centering" to end of intercept computation

# Let's do a simpler approach: just replace the RankDeficientOls::FitWithStdErrors calls
# with a comment marker, then manually insert the libanostat calls

replacement_count = 0
for i, line in enumerate(lines):
    if 'ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);' in line:
        # Mark this line for replacement
        lines[i] = line.replace(
            'ols_result = RankDeficientOls::FitWithStdErrors(y_weighted, X_weighted);',
            '// TODO_LIBANOSTAT: Replace with LibanostatWrapper::FitWLS'
        )
        replacement_count += 1
        print(f"Marked line {i+1} for replacement")

with open(filepath, 'w') as f:
    f.writelines(lines)

print(f"Marked {replacement_count} locations for WLS migration")
print("Manual intervention needed - WLS is more complex than OLS")
print("The manual weighting logic needs to be removed and replaced with LibanostatWrapper::FitWLS calls")
