# Replication of the SARIMA backtest in base R (no extra packages needed).
#
# Same design as the Python study: SARIMA(0,1,1)(0,1,1)[12] on log CPI, re-estimated on an
# expanding window; year-on-year inflation is derived from the forecast price path and
# compared with a random-walk (no-change) benchmark for targets Jan 2021 - Jan 2025.
#
# Run from the repository root:   Rscript R/sarima_backtest.R

panel <- read.csv("data/processed/cpi_all_india_combined_monthly.csv", stringsAsFactors = FALSE)
panel$date <- as.Date(panel$date)
panel <- panel[order(panel$date), ]

cpi <- ts(panel$cpi_general, start = c(as.integer(format(panel$date[1], "%Y")),
                                       as.integer(format(panel$date[1], "%m"))), frequency = 12)
yoy <- 100 * (panel$cpi_general / c(rep(NA, 12), head(panel$cpi_general, -12)) - 1)

horizons <- c(1, 3, 6)
test_idx <- which(panel$date >= as.Date("2021-01-01"))

# Diebold-Mariano test with Harvey et al. (1997) correction, squared-error loss.
dm_pvalue <- function(e1, e2, h) {
  d <- e1^2 - e2^2
  n <- length(d)
  d_bar <- mean(d)
  gamma <- sapply(0:(h - 1), function(k) mean((d[(k + 1):n] - d_bar) * (d[1:(n - k)] - d_bar)))
  v <- (gamma[1] + 2 * sum(gamma[-1])) / n
  stat <- d_bar / sqrt(v) * sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
  2 * pt(-abs(stat), df = n - 1)
}

results <- list()
forecasts <- data.frame()
for (h in horizons) {
  fc_sarima <- fc_rw <- actual <- numeric(0)
  for (t in test_idx) {
    origin <- t - h
    y_train <- window(log(cpi), end = time(cpi)[origin])
    fit <- arima(y_train, order = c(0, 1, 1),
                 seasonal = list(order = c(0, 1, 1), period = 12), method = "ML")
    path <- exp(predict(fit, n.ahead = h)$pred)
    base <- panel$cpi_general[t - 12]          # known at the origin because h <= 12
    fc_sarima <- c(fc_sarima, 100 * (path[h] / base - 1))
    fc_rw <- c(fc_rw, yoy[origin])
    actual <- c(actual, yoy[t])
  }
  e_s <- fc_sarima - actual
  e_rw <- fc_rw - actual
  rmse_s <- sqrt(mean(e_s^2))
  rmse_rw <- sqrt(mean(e_rw^2))
  results[[length(results) + 1]] <- data.frame(
    horizon = h, n = length(actual),
    rmse_random_walk = rmse_rw, rmse_sarima = rmse_s,
    relative_rmse = rmse_s / rmse_rw,
    dm_pvalue = dm_pvalue(e_s, e_rw, h))
  forecasts <- rbind(forecasts, data.frame(horizon = h, target = panel$date[test_idx],
                                           actual = actual, random_walk = fc_rw,
                                           sarima_r = fc_sarima))
}

summary_table <- do.call(rbind, results)
dir.create("results", showWarnings = FALSE)
write.csv(summary_table, "results/r_sarima_metrics.csv", row.names = FALSE)
write.csv(forecasts, "results/r_sarima_forecasts.csv", row.names = FALSE)

cat("SARIMA(0,1,1)(0,1,1)[12] on log CPI, expanding window, targets Jan 2021 - Jan 2025\n\n")
print(format(summary_table, digits = 3), row.names = FALSE)

# Cross-check against the Python (statsmodels) results, if present.
py_file <- "results/metrics.csv"
if (file.exists(py_file)) {
  py <- read.csv(py_file, check.names = FALSE)
  py <- py[py$model == "SARIMA", c("horizon", "RMSE")]
  cmp <- merge(summary_table[, c("horizon", "rmse_sarima")], py, by = "horizon")
  names(cmp) <- c("horizon", "rmse_R", "rmse_Python")
  cat("\nR vs Python SARIMA RMSE:\n")
  print(format(cmp, digits = 3), row.names = FALSE)
}
