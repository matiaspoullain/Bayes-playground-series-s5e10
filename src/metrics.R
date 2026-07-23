rmse <- function(actual, predicted) {
  sqrt(mean((predicted - actual)^2))
}

clip_mean <- function(mu, sigma) {
  a <- (0 - mu) / sigma
  b <- (1 - mu) / sigma
  Phi_a <- pnorm(a)
  Phi_b <- pnorm(b)
  phi_a <- dnorm(a)
  phi_b <- dnorm(b)

  mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + (1 - Phi_b)
}

squeeze_unit_interval <- function(y, n = length(y)) {
  (y * (n - 1) + 0.5) / n
}
