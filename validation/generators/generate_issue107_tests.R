#!/usr/bin/env Rscript
#
# Reference values for issue #107: explicit priors, AFT survival regression,
# empirical-Bayes shrinkage and mixed-effects GLMs.
#
# NOTE ON PROVENANCE
# ------------------
# R was not available in the environment where #107 was implemented, so this
# script has NOT been executed. The reference constants currently pinned in the
# Rust and sqllogictest suites were produced instead by an independent NumPy
# implementation of the same estimators (see
# crates/anofox-stats-core/src/models/glm_engine/parity.rs, which documents this).
#
# Running this script is therefore a genuine cross-check against R, not a
# reproduction of what is already asserted. Where a value disagrees, R wins:
# update the corresponding constant and say so in the commit.
#
# Requires: arm, survival, metafor, lme4, jsonlite
#
# Usage:  Rscript validation/generators/generate_issue107_tests.R

library(jsonlite)

set.seed(42)  # For reproducibility

out_root <- "test/data/issue107"
emit <- function(name, obj) {
  dir.create(file.path(out_root, "expected"), recursive = TRUE, showWarnings = FALSE)
  write_json(obj, file.path(out_root, "expected", paste0(name, ".json")),
             auto_unbox = TRUE, pretty = TRUE, digits = 15)
  cat("wrote", name, "\n")
}
emit_input <- function(name, df) {
  dir.create(file.path(out_root, "input"), recursive = TRUE, showWarnings = FALSE)
  write.csv(df, file.path(out_root, "input", paste0(name, ".csv")), row.names = FALSE)
}

# ---------------------------------------------------------------------------
# 1. Explicit priors: MAP with normal priors + curvature standard errors.
#
# arm::bayesglm computes exactly this -- a penalized IRLS whose covariance is the
# curvature at the mode -- so it is the right reference for `prior` + `vcov`.
# ---------------------------------------------------------------------------
prior_fixture <- local({
  n <- 60
  i <- 0:(n - 1)
  x1 <- (i %% 10) / 3
  x2 <- ((i * 7) %% 5) - 2
  y <- round(exp(0.6 + 0.25 * x1 - 0.15 * x2) + ((i * 13) %% 4) * 0.3)
  data.frame(x1 = x1, x2 = x2, y = y)
})
emit_input("prior_fixture", prior_fixture)

# Unpenalized baseline. The Rust suite pins these to 1e-7.
m_flat <- glm(y ~ x1 + x2, family = poisson(link = "log"), data = prior_fixture)
emit("prior_unpenalized", list(
  coefficients = as.numeric(coef(m_flat)),
  coefficient_names = names(coef(m_flat)),
  std_errors = as.numeric(summary(m_flat)$coefficients[, 2]),
  deviance = m_flat$deviance,
  null_deviance = m_flat$null.deviance,
  aic = m_flat$aic,
  logLik = as.numeric(logLik(m_flat))
))

if (requireNamespace("arm", quietly = TRUE)) {
  # A tight N(0, 0.02) prior on x1 only. prior.df = Inf makes it exactly normal
  # rather than arm's default t.
  m_prior <- arm::bayesglm(
    y ~ x1 + x2, family = poisson(link = "log"), data = prior_fixture,
    prior.mean = c(0, 0), prior.scale = c(0.02, Inf), prior.df = Inf,
    prior.mean.for.intercept = 0, prior.scale.for.intercept = Inf
  )
  emit("prior_normal_x1", list(
    coefficients = as.numeric(coef(m_prior)),
    coefficient_names = names(coef(m_prior)),
    std_errors = as.numeric(summary(m_prior)$coefficients[, 2])
  ))
} else {
  cat("SKIP prior_normal_x1: package 'arm' not installed\n")
}

# ---------------------------------------------------------------------------
# 2. AFT survival regression against survival::survreg.
# ---------------------------------------------------------------------------
if (requireNamespace("survival", quietly = TRUE)) {
  aft_fixture <- local({
    n <- 300
    i <- 0:(n - 1)
    x <- (i %% 10) / 3
    p <- (i + 0.5) / n
    # Invert the Weibull AFT quantile: log T = eta + sigma * log(-log(1 - p)).
    t_true <- exp(2.0 + 0.3 * x + 0.5 * log(-log(1 - p)))
    cens <- 9.0 + (i %% 7) * 0.9
    data.frame(
      x = x,
      days = pmin(t_true, cens),
      delivered = as.integer(t_true <= cens)
    )
  })
  emit_input("aft_fixture", aft_fixture)

  for (d in c("weibull", "lognormal", "loglogistic", "exponential")) {
    m <- survival::survreg(
      survival::Surv(days, delivered) ~ x, data = aft_fixture, dist = d
    )
    emit(paste0("aft_", d), list(
      dist = d,
      coefficients = as.numeric(coef(m)),
      coefficient_names = names(coef(m)),
      scale = m$scale,
      logLik = as.numeric(logLik(m)),
      std_errors = sqrt(diag(vcov(m))),
      median_at_x0 = as.numeric(predict(m, newdata = data.frame(x = 0), type = "quantile", p = 0.5))
    ))
  }
} else {
  cat("SKIP aft_*: package 'survival' not installed\n")
}

# ---------------------------------------------------------------------------
# 3. Empirical-Bayes shrinkage against metafor::rma(method = "DL").
# ---------------------------------------------------------------------------
if (requireNamespace("metafor", quietly = TRUE)) {
  eb_fixture <- data.frame(
    est = c(0.10, 0.30, 0.35, 0.65, 1.00),
    se  = c(0.30, 0.10, 0.50, 0.20, 0.40)
  )
  emit_input("eb_fixture", eb_fixture)

  m <- metafor::rma(yi = eb_fixture$est, sei = eb_fixture$se, method = "DL")
  b <- metafor::blup(m)
  emit("eb_shrink", list(
    mu = as.numeric(m$beta),
    mu_se = m$se,
    tau_squared = m$tau2,
    i_squared = m$I2 / 100,
    q = m$QE,
    shrunken = as.numeric(b$pred),
    shrunken_se = as.numeric(b$se)
  ))
} else {
  cat("SKIP eb_shrink: package 'metafor' not installed\n")
}

# ---------------------------------------------------------------------------
# 4. Mixed-effects models against lme4.
# ---------------------------------------------------------------------------
if (requireNamespace("lme4", quietly = TRUE)) {
  glmm_fixture <- local({
    n <- 300
    i <- 0:(n - 1)
    g <- i %/% 15
    x <- (i %% 15) %% 5
    y <- 1.0 + 0.5 * x + 0.8 * (((g + 0.5) / 20) * 2 - 1) * 1.732 +
      0.3 * (((g * 7 + (i %% 15) * 3) %% 11) / 5 - 1)
    data.frame(g = factor(g), x = x, y = y)
  })
  emit_input("glmm_fixture", glmm_fixture)

  for (use_reml in c(TRUE, FALSE)) {
    m <- lme4::lmer(y ~ x + (1 | g), data = glmm_fixture, REML = use_reml)
    vc <- as.data.frame(lme4::VarCorr(m))
    emit(paste0("glmm_gaussian_", if (use_reml) "reml" else "ml"), list(
      reml = use_reml,
      fixed = as.numeric(lme4::fixef(m)),
      fixed_names = names(lme4::fixef(m)),
      fixed_se = sqrt(diag(as.matrix(vcov(m)))),
      var_group = vc$vcov[vc$grp == "g"],
      var_residual = vc$vcov[vc$grp == "Residual"],
      logLik = as.numeric(logLik(m)),
      ranef = as.numeric(lme4::ranef(m)$g[, 1])
    ))
  }

  counts_fixture <- local({
    n <- 300
    i <- 0:(n - 1)
    g <- i %/% 20
    x <- (i %% 20) %% 4
    y <- round(exp(0.5 + 0.3 * x + 0.6 * (((g + 0.5) / 15) * 2 - 1)))
    data.frame(g = factor(g), x = x, y = y)
  })
  emit_input("glmm_counts_fixture", counts_fixture)

  mp <- lme4::glmer(y ~ x + (1 | g), data = counts_fixture, family = poisson)
  vcp <- as.data.frame(lme4::VarCorr(mp))
  emit("glmm_poisson", list(
    fixed = as.numeric(lme4::fixef(mp)),
    fixed_names = names(lme4::fixef(mp)),
    fixed_se = sqrt(diag(as.matrix(vcov(mp)))),
    var_group = vcp$vcov[vcp$grp == "g"],
    logLik = as.numeric(logLik(mp)),
    ranef = as.numeric(lme4::ranef(mp)$g[, 1])
  ))
} else {
  cat("SKIP glmm_*: package 'lme4' not installed\n")
}

# Provenance, per the convention in validation/generators/README.md.
writeLines(
  toJSON(list(
    generated = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    r_version = R.version.string,
    seed = 42,
    issue = 107
  ), auto_unbox = TRUE, pretty = TRUE),
  file.path(out_root, "metadata.json")
)
cat("done\n")
