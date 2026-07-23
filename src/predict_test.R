source("src/features.R")
source("src/metrics.R")
library(brms)
library(data.table)

winner <- trimws(readLines("models/winning_family.txt"))
fit <- readRDS(sprintf("models/fit_%s.rds", winner))

test <- fread("data/raw/test.csv")
test <- as.data.frame(test)
test <- engineer_predictors(test)

draw_ids <- 1:500
pred_draws <- posterior_epred(fit, newdata = test, allow_new_levels = TRUE, draw_ids = draw_ids)
pred <- colMeans(pred_draws)

pred <- pmin(pmax(pred, 0), 1)

submission <- data.frame(id = test$id, accident_risk = pred)
write.csv(submission, "data/processed/submission.csv", row.names = FALSE)

cat(sprintf("Submission written: %d rows. Winning family: %s\n", nrow(submission), winner))
cat(sprintf("Prediction range: [%.4f, %.4f], median: %.4f\n", min(pred), max(pred), median(pred)))
