# Reference values for durbin_watson_agg(resid, x ...) (#177): R lmtest::dwtest.
# Rscript validation/generators/make_reference_dwtest.R
suppressMessages(library(lmtest))
i  <- 0:39
t  <- i
x1 <- sin(i / 3) + (i %% 5) / 10
y  <- 1 + 0.5 * x1 + 0.02 * t + cumsum(((i * 7) %% 11 - 5) / 20)   # autocorrelated errors
y2 <- 1 + 0.5 * x1 + ((i * 7) %% 11 - 5) / 20                          # white-ish errors
fmt <- function(v) formatC(v, digits = 15, format = "g")
show <- function(label, d) cat(label, fmt(d$statistic), fmt(d$p.value), "\n")
show("y~x1+t greater  ", dwtest(y ~ x1 + t))
show("y~x1+t less     ", dwtest(y ~ x1 + t, alternative = "less"))
show("y~x1+t two.sided", dwtest(y ~ x1 + t, alternative = "two.sided"))
show("y2~x1 greater   ", dwtest(y2 ~ x1))
show("y~1 greater     ", dwtest(y ~ 1))
show("y~x1-1 greater  ", dwtest(y ~ x1 - 1))
# n >= 100: normal approximation
j  <- 0:149
xa <- cos(j / 7)
ya <- 2 + xa + cumsum(((j * 5) %% 13 - 6) / 30)
show("n150 greater    ", dwtest(ya ~ xa))
