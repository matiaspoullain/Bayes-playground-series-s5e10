source("src/features.R")
source("src/metrics.R")
source("src/model_spec.R")
library(brms)

sub <- read.csv("data/processed/model_subsample.csv")

set.seed(42)
n_folds <- 5
sub$fold <- sample(rep(1:n_folds, length.out = nrow(sub)))

dir.create("tables_figures/cv", showWarnings = FALSE, recursive = TRUE)

fold_summaries <- list()
oof <- data.frame(
  id = sub$id, accident_risk = sub$accident_risk,
  beta_pred = NA_real_
)

for (k in 1:n_folds) {
  train_fold <- sub[sub$fold != k, ]
  holdout_fold <- sub[sub$fold == k, ]
  train_fold$accident_risk_squeezed <- squeeze_unit_interval(train_fold$accident_risk, n = nrow(train_fold))

  fit_beta <- brm(
    formula = beta_formula(), data = train_fold, family = beta_family(), prior = beta_priors(),
    backend = "rstan", chains = 4, cores = 4, iter = 2000, warmup = 1000,
    control = list(adapt_delta = 0.95), seed = 1
  )

  beta_pred_draws <- posterior_epred(fit_beta, newdata = holdout_fold, allow_new_levels = TRUE)
  beta_pred <- colMeans(beta_pred_draws)

  oof$beta_pred[sub$fold == k] <- beta_pred

  beta_summ <- summary(fit_beta)

  fold_summaries[[k]] <- data.frame(
    fold = k,
    beta_rmse = rmse(holdout_fold$accident_risk, beta_pred),
    beta_max_rhat = max(beta_summ$fixed[, "Rhat"]),
    beta_divergences = sum(subset(nuts_params(fit_beta), Parameter == "divergent__")$Value)
  )

  cat(sprintf(
    "Fold %d: Beta RMSE=%.5f (Rhat=%.4f, div=%d)\n",
    k, fold_summaries[[k]]$beta_rmse, fold_summaries[[k]]$beta_max_rhat, fold_summaries[[k]]$beta_divergences
  ))
}

cv_summary <- do.call(rbind, fold_summaries)
write.csv(cv_summary, "tables_figures/cv/cv_summary.csv", row.names = FALSE)
write.csv(oof, "tables_figures/cv/oof_predictions.csv", row.names = FALSE)

cat(sprintf(
  "\nMean Beta RMSE: %.5f (sd %.5f)\n",
  mean(cv_summary$beta_rmse), sd(cv_summary$beta_rmse)
))
