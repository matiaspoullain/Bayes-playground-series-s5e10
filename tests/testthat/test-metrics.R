source(file.path("..", "..", "src", "metrics.R"))

test_that("rmse is zero for perfect predictions", {
  expect_equal(rmse(c(1, 2, 3), c(1, 2, 3)), 0)
})

test_that("rmse computes root mean squared error", {
  actual <- c(0, 1)
  predicted <- c(0.3, 0.7)
  expect_equal(rmse(actual, predicted), sqrt(mean(c(0.09, 0.09))))
})

test_that("squeeze_unit_interval maps 0 and 1 into the open interval", {
  y <- c(0, 0.5, 1)
  result <- squeeze_unit_interval(y, n = 100)

  expect_true(all(result > 0 & result < 1))
  expect_equal(result[2], 0.5, tolerance = 1e-8)
})
