# Reference values for the data-drift aggregates (#245-#255), held by
# test/sql/reference/drift_reference.test.
#
#   Rscript validation/generators/make_reference_drift.R
#
# Packages: kSamples, transport, philentropy, goftest, MASS.
# Shared data of the issues (group 0 = reference x, group 1 = current y):
#   CREATE TABLE drift AS
#   SELECT 0 AS grp, 2*sin(0.37*i) AS v, (i % 4)::INTEGER AS c FROM range(50) r(i)
#   UNION ALL
#   SELECT 1, 2*sin(0.53*i) + 0.5, ((i*3) % 5)::INTEGER FROM range(40) r(i);
# plus group 2 (k = 3 for Anderson-Darling): 2*sin(0.29*i) - 0.3, i < 30.
suppressMessages({
  library(kSamples); library(transport); library(philentropy); library(goftest)
})
p <- function(label, value) cat(sprintf("%-40s %.15g\n", label, value))

x  <- 2 * sin(0.37 * (0:49)); y  <- 2 * sin(0.53 * (0:39)) + 0.5
xc <- (0:49) %% 4;            yc <- ((0:39) * 3) %% 5
z3 <- 2 * sin(0.29 * (0:29)) - 0.3
# Ties: the values rounded to one decimal.
xt <- round(x, 1); yt <- round(y, 1)

# --- ks_test_agg (#245) -----------------------------------------------------
k <- suppressWarnings(ks.test(x, y)); p("ks exact D", k$statistic); p("ks exact p", k$p.value)
k <- ks.test(x, y, exact = FALSE); p("ks asymptotic p", k$p.value)
k <- ks.test(x, y, alternative = "less"); p("ks less D", k$statistic); p("ks less p", k$p.value)
k <- ks.test(x, y, alternative = "greater"); p("ks greater D", k$statistic); p("ks greater p", k$p.value)
k <- suppressWarnings(ks.test(xt, yt)); p("ks ties D", k$statistic); p("ks ties p", k$p.value)

# --- ad_test_agg (#246) -----------------------------------------------------
# ad.test rounds its table, so the unrounded statistics come from its C routine
# and the p-values from kSamples::ad.pval (as upstream's generate_drift_refs.R).
ad_k <- function(samples) {
  k <- length(samples); ns <- sapply(samples, length); z <- unlist(samples)
  n <- length(z); Z.star <- sort(unique(z))
  a <- .C("adkTestStat0", ans = double(2), k = as.integer(k), x = as.double(z),
          ns = as.integer(ns), Z.star = as.double(Z.star), L = as.integer(length(Z.star)),
          PACKAGE = "kSamples")$ans
  H <- sum(1 / ns); h <- sum(1 / (1:(n - 1))); g <- 0
  for (i in 1:(n - 2)) g <- g + (1 / (n - i)) * sum(1 / ((i + 1):(n - 1)))
  ca <- (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
  cb <- (2 * g - 4) * k^2 + 8 * h * k + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
  cc <- (6 * h + 2 * g - 2) * k^2 + (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
  cd <- (2 * h + 6) * k^2 - 4 * h * k
  sig <- sqrt((ca * n^3 + cb * n^2 + cc * n + cd) / ((n - 1) * (n - 2) * (n - 3)))
  t <- (a - (k - 1)) / sig
  list(ad = a, t = t, p = c(ad.pval(t[1], k - 1, 1), ad.pval(t[2], k - 1, 2)))
}
a <- ad_k(list(x, y))
p("ad v2 AD", a$ad[2]); p("ad v2 T", a$t[2]); p("ad v2 p", a$p[2])
p("ad v1 AD", a$ad[1]); p("ad v1 T", a$t[1]); p("ad v1 p", a$p[1])
a <- ad_k(list(x, y, z3))
p("ad k3 T", a$t[2]); p("ad k3 p", a$p[2])

# --- cvm_test_agg (#247) ----------------------------------------------------
cvmT <- function(x, y) {
  m <- length(x); n <- length(y); N <- m + n
  rk <- rank(c(x, y)); rx <- sort(rk[1:m]); ry <- sort(rk[(m + 1):N])
  U <- m * sum((rx - 1:m)^2) + n * sum((ry - 1:n)^2)
  U / (m * n * N) - (4 * m * n - 1) / (6 * N)
}
cvm2 <- function(x, y) {
  m <- length(x); n <- length(y); N <- m + n
  T <- cvmT(x, y)
  ET <- (1 + 1 / N) / 6
  VT <- (N + 1) * (4 * m * n * N - 3 * (m^2 + n^2) - 2 * m * n) / (45 * N^2 * 4 * m * n)
  c(T, 1 - pCvM(1 / 6 + (T - ET) / sqrt(45 * VT)))
}
r <- cvm2(x, y); p("cvm asymptotic T", r[1]); p("cvm asymptotic p", r[2])
# Exact: full enumeration of the splits of the pooled small sample.
xs <- x[1:8]; ys <- y[1:7]; zs <- c(xs, ys); T0 <- cvmT(xs, ys)
Ts <- apply(combn(15, 8), 2, function(i) cvmT(zs[i], zs[-i]))
p("cvm exact T", T0); p("cvm exact p", mean(Ts >= T0 - 1e-12))

# --- epps_singleton_agg (#248) ----------------------------------------------
es2 <- function(x, y, t = c(0.4, 0.8)) {
  nx <- length(x); ny <- length(y); n <- nx + ny
  ts <- t / (IQR(c(x, y)) / 2)
  g <- function(v) cbind(cos(outer(v, ts)), sin(outer(v, ts)))
  gx <- g(x); gy <- g(y)
  S <- (n / nx) * cov(gx) * (nx - 1) / nx + (n / ny) * cov(gy) * (ny - 1) / ny
  d <- colMeans(gx) - colMeans(gy)
  W <- n * drop(t(d) %*% MASS::ginv(S) %*% d)
  if (max(nx, ny) < 25) W <- W / (1 + n^-0.45 + 10.1 * (nx^-1.7 + ny^-1.7))
  r <- qr(S)$rank
  c(W, pchisq(W, df = r, lower.tail = FALSE), r)
}
r <- es2(x, y); p("es W", r[1]); p("es p", r[2]); p("es df", r[3])
r <- es2(x[1:20], y[1:15]); p("es small W", r[1]); p("es small p", r[2])
r <- es2(x, y, c(0.5, 1, 1.5)); p("es t3 W", r[1]); p("es t3 df", r[3])

# --- wasserstein_agg (#249) -------------------------------------------------
sdpop <- function(v) sqrt(mean((v - mean(v))^2))
w <- wasserstein1d(x, y, p = 1); p("wasserstein W1", w); p("wasserstein normed", w / max(sdpop(x), 0.001))
w2 <- wasserstein1d(x, y, p = 2); p("wasserstein W2", w2); p("wasserstein W2 normed", w2 / max(sdpop(x), 0.001))
w <- wasserstein1d(y, x, p = 1); p("wasserstein normed (ref = y)", w / max(sdpop(y), 0.001))

# --- binning shared by psi / kl / js ------------------------------------------
z <- c(x, y); br <- seq(min(z), max(z), length.out = nclass.Sturges(z) + 1)
pn <- tabulate(findInterval(x, br, rightmost.closed = TRUE), length(br) - 1) / length(x)
qn <- tabulate(findInterval(y, br, rightmost.closed = TRUE), length(br) - 1) / length(y)
pc <- tabulate(xc + 1, 5) / 50; qc <- tabulate(yc + 1, 5) / 40
fill <- function(a) { m <- min(a[a > 0]); a[a == 0] <- if (m <= 1e-4) m / 1e6 else 1e-4; a }
# Reference-decile bins with open outer edges.
qb <- unique(quantile(x, 0:10 / 10, type = 7)); qb[1] <- -Inf; qb[length(qb)] <- Inf
pq <- tabulate(findInterval(x, qb, left.open = FALSE), length(qb) - 1) / length(x)
qq <- tabulate(findInterval(y, qb, left.open = FALSE), length(qb) - 1) / length(y)

# --- psi_agg (#250) ---------------------------------------------------------
psi <- function(p, q) { P <- fill(p); Q <- fill(q); sum((P - Q) * log(P / Q)) }
p("psi numeric", psi(pn, qn)); p("psi categorical", psi(pc, qc)); p("psi quantile", psi(pq, qq))
cfill <- function(a, e) { a[a == 0] <- e; a }   # constant zero fill ('epsilon')
P <- cfill(pc, 1e-3); Q <- cfill(qc, 1e-3); p("psi categorical eps 1e-3", sum((P - Q) * log(P / Q)))
p("kl categorical eps 1e-3", sum((P / sum(P)) * log((P / sum(P)) / (Q / sum(Q)))))

# --- kl_divergence_agg (#251) -----------------------------------------------
kl <- function(p, q, unit = "log") {
  P <- fill(p); P <- P / sum(P); Q <- fill(q); Q <- Q / sum(Q)
  suppressMessages(KL(rbind(P, Q), unit = unit))
}
p("kl numeric", kl(pn, qn)); p("kl categorical", kl(pc, qc)); p("kl categorical log2", kl(pc, qc, "log2"))

# --- jensen_shannon_agg (#252) ----------------------------------------------
js <- function(p, q, unit = "log") suppressMessages(JSD(rbind(p, q), unit = unit))
p("js numeric distance", sqrt(js(pn, qn))); p("js numeric divergence", js(pn, qn))
p("js categorical distance", sqrt(js(pc, qc))); p("js categorical log2 distance", sqrt(js(pc, qc, "log2")))

# --- hellinger_agg (#253) ---------------------------------------------------
k <- unique(c(x, y)); bh <- seq(min(k), max(k), length.out = nclass.Sturges(k) + 1)
ph <- tabulate(findInterval(x, bh, rightmost.closed = TRUE), length(bh) - 1) / length(x)
qh <- tabulate(findInterval(y, bh, rightmost.closed = TRUE), length(bh) - 1) / length(y)
p("hellinger numeric", sqrt(1 - min(1, sum(sqrt(ph * qh)))))
p("hellinger categorical", sqrt(1 - sum(sqrt(pc * qc))))

# --- tvd_agg (#254) ---------------------------------------------------------
p("tvd statistic", 0.5 * sum(abs(pc - qc)))
