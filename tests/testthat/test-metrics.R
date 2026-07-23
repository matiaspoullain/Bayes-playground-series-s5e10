source(file.path("..", "..", "src", "metrics.R"))

test_that("rmse is zero for perfect predictions", {
  expect_equal(rmse(c(1, 2, 3), c(1, 2, 3)), 0)
})

test_that("rmse computes root mean squared error", {
  actual <- c(0, 1)
  predicted <- c(0.3, 0.7)
  expect_equal(rmse(actual, predicted), sqrt(mean(c(0.09, 0.09))))
})

test_that("clip_mean matches direct Monte Carlo simulation of clip(N(mu,sigma),0,1)", {
  set.seed(1)
  mu <- 0.5
  sigma <- 0.05
  draws <- rnorm(2e6, mu, sigma)
  clipped <- pmin(pmax(draws, 0), 1)
  expect_equal(clip_mean(mu, sigma), mean(clipped), tolerance = 1e-3)
})

test_that("clip_mean matches direct Monte Carlo simulation near a boundary", {
  set.seed(2)
  mu <- 0.02
  sigma <- 0.05
  draws <- rnorm(2e6, mu, sigma)
  clipped <- pmin(pmax(draws, 0), 1)
  expect_equal(clip_mean(mu, sigma), mean(clipped), tolerance = 1e-3)
})

test_that("clip_mean is vectorized over mu", {
  result <- clip_mean(c(0.1, 0.5, 0.9), sigma = 0.05)
  expect_equal(length(result), 3)
})

test_that("squeeze_unit_interval maps 0 and 1 into the open interval", {
  y <- c(0, 0.5, 1)
  result <- squeeze_unit_interval(y, n = 100)

  expect_true(all(result > 0 & result < 1))
  expect_equal(result[2], 0.5, tolerance = 1e-8)
})
