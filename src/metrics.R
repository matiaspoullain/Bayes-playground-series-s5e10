rmse <- function(actual, predicted) {
  sqrt(mean((predicted - actual)^2))
}

squeeze_unit_interval <- function(y, n = length(y)) {
  (y * (n - 1) + 0.5) / n
}
