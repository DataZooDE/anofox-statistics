# Anofox Statistics - Technical Guide

This guide covers the architecture, implementation details, and performance characteristics of the Anofox Statistics DuckDB extension.

## Architecture Overview

### Hybrid Rust/C++ Design

The extension uses a hybrid architecture:

```
┌─────────────────────────────────────────────────────┐
│                   DuckDB                            │
├─────────────────────────────────────────────────────┤
│              C++ Extension Layer                    │
│  (Function registration, type conversion, DuckDB   │
│   API integration)                                  │
├─────────────────────────────────────────────────────┤
│                 FFI Boundary                        │
│  (anofox-stats-ffi crate)                          │
├─────────────────────────────────────────────────────┤
│              Rust Core Library                      │
│  (anofox-stats-core: nalgebra, regress)            │
└─────────────────────────────────────────────────────┘
```

### Crate Structure

```
crates/
├── anofox-stats-core/     # Pure Rust statistical algorithms
│   ├── models/            # OLS, Ridge, Elastic Net, WLS, RLS
│   ├── diagnostics/       # VIF, AIC/BIC, Jarque-Bera, Residuals
│   └── errors.rs          # Error types
└── anofox-stats-ffi/      # C FFI boundary
    ├── lib.rs             # FFI function exports
    └── types.rs           # FFI-safe type definitions
```

---

## Implementation Details

### Scalar Functions

Scalar functions process vectorized data through array inputs:

```cpp
// C++ side: Extract DuckDB LIST to std::vector
vector<double> y_data = ExtractDoubleList(y_vec, row_idx);

// Prepare FFI arrays
AnofoxDataArray y_array;
y_array.data = y_data.data();
y_array.len = y_data.size();

// Call Rust FFI
bool success = anofox_ols_fit(y_array, x_arrays, x_count, options, &result, &error);
```

```rust
// Rust side: Convert to nalgebra and compute
pub unsafe extern "C" fn anofox_ols_fit(...) -> bool {
    let y_vec = y.to_vec();
    let x_vecs: Vec<Vec<f64>> = x_arrays.iter().map(|arr| arr.to_vec()).collect();

    match fit_ols(&y_vec, &x_vecs, &opts) {
        Ok(result) => { /* copy to output */ true }
        Err(e) => { /* set error */ false }
    }
}
```

### Aggregate Functions

Aggregate functions maintain state across rows:

```cpp
struct OlsAggregateState {
    vector<double> y_values;
    vector<vector<double>> x_columns;
    bool initialized;
    // Options stored at bind time
};

// Update: accumulate one row
static void OlsAggUpdate(Vector inputs[], ..., Vector &state_vector, idx_t count) {
    for (idx_t i = 0; i < count; i++) {
        state.y_values.push_back(y_val);
        for (idx_t j = 0; j < n_features; j++) {
            state.x_columns[j].push_back(x_vals[j]);
        }
    }
}

// Finalize: call Rust with accumulated data
static void OlsAggFinalize(...) {
    AnofoxDataArray y_array = { state.y_values.data(), nullptr, state.y_values.size() };
    anofox_ols_fit(y_array, x_arrays.data(), x_count, options, &result, &error);
}
```

### Window Function Support

Aggregate functions automatically support window operations via DuckDB's `OVER` clause:

```sql
-- DuckDB handles windowing; aggregate sees accumulated rows per frame
SELECT (anofox_stats_ols_fit_agg(y, [x]) OVER (
    ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
)).coefficients[1] as rolling_beta
FROM data;
```

---

## Linear Algebra Implementation

### OLS: QR Decomposition

OLS uses QR decomposition via the `regress` crate for numerical stability:

```rust
// Solve: β = (X'X)⁻¹X'y via QR
let model = OrdinaryLeastSquares::from_data(&data, &regressors)?;
let params = model.params();
```

### Ridge: Modified Normal Equations

Ridge regression adds λI to X'X:

```rust
// Solve: β = (X'X + λI)⁻¹X'y
let xtx = x_centered.transpose() * &x_centered;
let xty = x_centered.transpose() * &y_centered;
let ridge_term = DMatrix::identity(n_features, n_features) * alpha;
let coefficients = (xtx + ridge_term).lu().solve(&xty)?;
```

### Elastic Net: Coordinate Descent

Elastic Net uses coordinate descent with soft thresholding:

```rust
for _ in 0..max_iterations {
    for j in 0..n_features {
        // Compute partial residual
        let r_j = compute_partial_residual(&y, &x, &coefficients, j);

        // Soft threshold for L1
        let z = x_col_j.dot(&r_j) / n;
        coefficients[j] = soft_threshold(z, alpha * l1_ratio) /
                          (1.0 + alpha * (1.0 - l1_ratio));
    }
    if converged() { break; }
}
```

### RLS: Recursive Update

RLS uses Sherman-Morrison-Woodbury for efficient updates:

```rust
pub fn update(&mut self, x: &DVector<f64>, y: f64) -> StatsResult<()> {
    // Kalman gain: k = Px / (λ + x'Px)
    let px = &self.p_matrix * x;
    let denom = self.forgetting_factor + x.dot(&px);
    let k = &px / denom;

    // Prediction error
    let y_pred = x.dot(&self.coefficients);
    let error = y - y_pred;

    // Update coefficients: β = β + k * error
    self.coefficients += &k * error;

    // Update P matrix: P = (P - k*x'P) / λ
    self.p_matrix = (&self.p_matrix - &k * px.transpose()) / self.forgetting_factor;

    Ok(())
}
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| OLS fit | O(np² + p³) | O(np + p²) |
| Ridge fit | O(np² + p³) | O(np + p²) |
| Elastic Net | O(iterations × np) | O(np + p) |
| RLS update | O(p²) per observation | O(p²) |
| VIF | O(p × (np² + p³)) | O(np + p²) |

Where n = observations, p = features.

### Benchmarks (Intel i7, 32GB RAM)

| Operation | n | p | Time | Memory |
|-----------|---|---|------|--------|
| OLS fit | 1M | 10 | ~260ms | ~80MB |
| Ridge fit | 1M | 10 | ~280ms | ~80MB |
| OLS aggregate (100 groups) | 1M | 10 | ~4.5s | ~120MB |
| RLS (streaming) | 1M | 10 | ~2.1s | ~1KB/row |

### Optimization Tips

1. **Batch over streaming**: Scalar functions are faster than aggregates when data fits in memory
2. **Disable inference**: Set `compute_inference=false` for ~30% speedup
3. **Partition large data**: Use `GROUP BY` to parallelize aggregate computations
4. **Column-major storage**: Store feature columns contiguously for cache efficiency

---

## Numerical Stability

### Condition Number Checks

The extension monitors condition numbers to detect ill-conditioned systems:

```rust
fn check_condition_number(xtx: &DMatrix<f64>) -> StatsResult<()> {
    let svd = xtx.svd(true, true);
    let condition = svd.singular_values[0] / svd.singular_values.last().unwrap_or(&1.0);

    if condition > 1e12 {
        return Err(StatsError::SingularMatrix);
    }
    Ok(())
}
```

### Handling Edge Cases

- **Perfect collinearity**: Returns `SingularMatrix` error
- **Near-zero variance**: Uses epsilon threshold (1e-10)
- **NaN/Inf values**: Filtered during accumulation
- **Empty groups**: Returns NULL in aggregate finalize

---

## Error Handling

### Error Types

```rust
pub enum StatsError {
    InvalidAlpha(f64),           // Alpha < 0
    InvalidL1Ratio(f64),         // l1_ratio not in [0,1]
    InsufficientData { need, got },
    NoValidData,
    DimensionMismatch { expected, got },
    SingularMatrix,
    ConvergenceFailure { iterations, tolerance },
    AllocationFailure,
}
```

### FFI Error Codes

```c
typedef enum {
    ANOFOX_ERROR_SUCCESS = 0,
    ANOFOX_ERROR_INVALID_INPUT = 1,
    ANOFOX_ERROR_SINGULAR_MATRIX = 2,
    ANOFOX_ERROR_CONVERGENCE_FAILURE = 3,
    // ...
} AnofoxErrorCode;
```

---

## Memory Management

### FFI Memory Protocol

```rust
// Rust allocates via libc::malloc
let coef_ptr = libc::malloc(n * size_of::<f64>()) as *mut f64;
std::ptr::copy_nonoverlapping(coefficients.as_ptr(), coef_ptr, n);

// C++ must call free function
anofox_free_result_core(&result);
```

### Aggregate State Lifecycle

```
Initialize -> Update* -> Combine? -> Finalize -> Destroy
     |           |          |           |           |
   alloc      append      merge      compute      free
```

---

## Testing

### Rust Unit Tests

```bash
cd crates/anofox-stats-core
cargo test
```

### SQL Tests

```bash
# Run DuckDB test suite
./build/release/test/unittest --test-dir=test/sql
```

### Manual Testing

```sql
LOAD 'build/release/extension/anofox_stats/anofox_stats.duckdb_extension';

-- Verify known values
SELECT ROUND((ols_fit([3.0, 5.0, 7.0, 9.0, 11.0], [[1.0, 2.0, 3.0, 4.0, 5.0]])).coefficients[1], 2);
-- Expected: 2.0
```

---

## Build System

### Prerequisites

- Rust 1.70+
- CMake 3.20+
- C++17 compiler
- DuckDB source (submodule)

### Build Commands

```bash
# Debug build
make debug

# Release build
make release

# Clean rebuild
rm -rf build && make release
```

### CMake Configuration

```cmake
set(EXTENSION_SOURCES
    src/anofox_stats_extension.cpp
    src/table_functions/ols_fit.cpp
    src/aggregate_functions/ols_aggregate.cpp
    # ...
)

# Link pre-built Rust library
target_link_libraries(${TARGET_NAME}_loadable_extension ${RUST_LIB_PATH})
```

---

## Extending the Extension

### Adding a New Function

1. **Implement in Rust** (`crates/anofox-stats-core/src/`)
2. **Add FFI bindings** (`crates/anofox-stats-ffi/src/lib.rs`)
3. **Update C header** (`src/include/anofox_stats_ffi.h`)
4. **Create C++ wrapper** (`src/table_functions/` or `src/aggregate_functions/`)
5. **Register function** (`src/anofox_stats_extension.cpp`)
6. **Add to CMakeLists.txt**
7. **Write tests** (`test/sql/`)

### Example: Adding a New Diagnostic

```rust
// 1. Core implementation
pub fn compute_new_diagnostic(data: &[f64]) -> StatsResult<f64> {
    // Implementation
}

// 2. FFI binding
#[no_mangle]
pub unsafe extern "C" fn anofox_new_diagnostic(
    data: DataArray,
    out_result: *mut f64,
    out_error: *mut AnofoxError,
) -> bool {
    // Convert and call
}
```

```cpp
// 3. C++ registration
void RegisterNewDiagnosticFunction(ExtensionLoader &loader) {
    ScalarFunctionSet func_set("anofox_stats_new_diagnostic");
    // ...
    loader.RegisterFunction(func_set);
}
```

---

## Debugging

### Enable Rust Logging

```bash
RUST_LOG=debug ./build/release/duckdb
```

### DuckDB Debug Build

```bash
make debug
./build/debug/duckdb
```

### Memory Profiling

```bash
valgrind --leak-check=full ./build/debug/duckdb -c "LOAD ...; SELECT ..."
```
