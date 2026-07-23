source(file.path("..", "..", "src", "features.R"))

test_that("engineer_predictors computes the 4 binary indicators correctly", {
  df <- data.frame(
    lighting = c("night", "dim", "daylight"),
    weather = c("clear", "rainy", "foggy"),
    speed_limit = c(70, 45, 60),
    num_reported_accidents = c(1, 3, 2)
  )
  result <- engineer_predictors(df)

  expect_equal(result$lighting_night, c(1, 0, 0))
  expect_equal(result$weather_not_clear, c(0, 1, 1))
  expect_equal(result$speed_fast, c(1, 0, 1))
  expect_equal(result$accidents_many, c(0, 1, 0))
})

test_that("stratified_subsample returns exactly n rows", {
  set.seed(1)
  df <- data.frame(id = 1:10000, y = runif(10000))
  sub <- stratified_subsample(df, target_col = "y", n = 2000, seed = 42)

  expect_equal(nrow(sub), 2000)
})

test_that("stratified_subsample is deterministic given the same seed", {
  df <- data.frame(id = 1:10000, y = runif(10000))
  sub1 <- stratified_subsample(df, target_col = "y", n = 500, seed = 7)
  sub2 <- stratified_subsample(df, target_col = "y", n = 500, seed = 7)

  expect_equal(sub1$id, sub2$id)
})

test_that("stratified_subsample guarantees full inclusion of boundary_values rows", {
  set.seed(1)
  df <- data.frame(id = 1:10000, y = runif(10000))
  df$y[1:3] <- 0
  df$y[4] <- 1
  sub <- stratified_subsample(df, target_col = "y", n = 2000, seed = 42, boundary_values = c(0, 1))

  expect_equal(sum(sub$y == 0), 3)
  expect_equal(sum(sub$y == 1), 1)
  expect_equal(nrow(sub), 2000)
})

test_that("stratified_subsample represents all deciles of the target", {
  set.seed(1)
  df <- data.frame(id = 1:20000, y = runif(20000))
  sub <- stratified_subsample(df, target_col = "y", n = 4000, n_strata = 10, seed = 42)
  deciles <- cut(df$y, breaks = quantile(df$y, probs = seq(0, 1, 0.1)), include.lowest = TRUE)
  sub_deciles <- cut(sub$y, breaks = quantile(df$y, probs = seq(0, 1, 0.1)), include.lowest = TRUE)

  expect_equal(nlevels(droplevels(sub_deciles)), 10)
  expect_true(all(table(sub_deciles) >= 300))
})
