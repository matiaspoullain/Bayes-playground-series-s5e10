source("src/features.R")
source("src/metrics.R")
library(brms)
library(data.table)

winner <- trimws(readLines("models/winning_family.txt"))
fit <- readRDS(sprintf("models/fit_%s.rds", winner))

train <- fread("data/raw/train.csv")
train <- as.data.frame(train)
train <- engineer_predictors(train)

complement_ids <- read.csv("data/processed/model_subsample_complement_ids.csv")$id
holdout <- train[train$id %in% complement_ids, ]

draw_ids <- 1:500
pred_draws <- posterior_epred(fit, newdata = holdout, allow_new_levels = TRUE, draw_ids = draw_ids)
pred <- colMeans(pred_draws)

full_data_rmse <- rmse(holdout$accident_risk, pred)

cv_summary <- read.csv("tables_figures/cv/cv_summary.csv")
cv_rmse_col <- sprintf("%s_rmse", winner)
subsample_cv_rmse <- mean(cv_summary[[cv_rmse_col]])

cat(sprintf("Winning family: %s\n", winner))
cat(sprintf("Full-data (%d held-out rows) RMSE: %.5f\n", nrow(holdout), full_data_rmse))
cat(sprintf("Subsample CV RMSE: %.5f\n", subsample_cv_rmse))
cat(sprintf("Difference: %.5f\n", full_data_rmse - subsample_cv_rmse))

dir.create("tables_figures/validation", showWarnings = FALSE, recursive = TRUE)
write.csv(
  data.frame(
    winning_family = winner, n_holdout_rows = nrow(holdout),
    full_data_rmse = full_data_rmse, subsample_cv_rmse = subsample_cv_rmse,
    difference = full_data_rmse - subsample_cv_rmse
  ),
  "tables_figures/validation/full_data_honesty_check.csv", row.names = FALSE
)
