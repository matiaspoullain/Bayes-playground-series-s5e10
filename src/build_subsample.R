source("src/features.R")
library(data.table)

train <- fread("data/raw/train.csv")
train <- as.data.frame(train)
train <- engineer_predictors(train)

subsample <- stratified_subsample(
  train, target_col = "accident_risk", n = 20000, n_strata = 10, seed = 42,
  boundary_values = c(0, 1)
)

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)
write.csv(subsample, "data/processed/model_subsample.csv", row.names = FALSE)

complement_ids <- data.frame(id = setdiff(train$id, subsample$id))
write.csv(complement_ids, "data/processed/model_subsample_complement_ids.csv", row.names = FALSE)

cat(sprintf("Subsample: %d rows. Complement: %d rows.\n", nrow(subsample), nrow(complement_ids)))
cat(sprintf(
  "Subsample boundary counts: exact-0 = %d, exact-1 = %d\n",
  sum(subsample$accident_risk == 0), sum(subsample$accident_risk == 1)
))
