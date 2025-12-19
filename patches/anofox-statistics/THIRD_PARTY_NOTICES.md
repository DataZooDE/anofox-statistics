# Third Party Notices

This library contains independent Rust implementations inspired by algorithms and methodologies from several open-source projects. We gratefully acknowledge their contributions.

**Note**: This library is licensed under MIT. The attributions below are for methodological inspiration and academic courtesy. No source code was copied from these projects; all implementations are original Rust code.

## statrs (MIT)

**Repository**: https://github.com/statrs-dev/statrs

Statistical distributions and functions used for:
- Probability distributions (Normal, T, F, Chi-squared, etc.)
- Statistical functions (CDF, PDF, quantile)
- Distribution parameters and calculations

```
Copyright (c) statrs developers
License: MIT
```

## rand (MIT/Apache-2.0)

**Repository**: https://github.com/rust-random/rand

Random number generation used for:
- Permutation tests
- Bootstrap resampling methods
- Monte Carlo simulations

```
Copyright (c) rand developers
License: MIT OR Apache-2.0
```

## rand_chacha (MIT/Apache-2.0)

**Repository**: https://github.com/rust-random/rand

ChaCha random number generator for:
- Reproducible random number generation with seeds
- Deterministic test results

```
Copyright (c) rand developers
License: MIT OR Apache-2.0
```

## thiserror (MIT/Apache-2.0)

**Repository**: https://github.com/dtolnay/thiserror

Used for:
- Ergonomic error type definitions
- Error handling infrastructure

```
Copyright (c) David Tolnay
License: MIT OR Apache-2.0
```

## R Statistical Computing

All algorithms are validated against and inspired by R's statistical functions:
- `stats::t.test()` - T-test methodology
- `stats::wilcox.test()` - Mann-Whitney U and Wilcoxon signed-rank tests
- `stats::kruskal.test()` - Kruskal-Wallis test
- `stats::shapiro.test()` - Shapiro-Wilk normality test
- `WRS2::yuen()` - Yuen's robust t-test
- `car::leveneTest()` - Brown-Forsythe test
- `lawstat::brunner.munzel.test()` - Brunner-Munzel test
- `moments::agostino.test()` - D'Agostino's K-squared test
- `e1071::skewness()`, `e1071::kurtosis()` - Skewness and kurtosis
- `forecast::dm.test()` - Diebold-Mariano test

---

All licenses permit use in this MIT-licensed library with proper attribution.
