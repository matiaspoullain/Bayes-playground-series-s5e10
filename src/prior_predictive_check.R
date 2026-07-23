source("src/features.R")
source("src/metrics.R")
source("src/model_spec.R")
library(brms)

sub <- read.csv("data/processed/model_subsample.csv")
sub$accident_risk_squeezed <- squeeze_unit_interval(sub$accident_risk, n = nrow(sub))

fit_beta_prior <- brm(
  formula = beta_formula(), data = sub, family = beta_family(), prior = beta_priors(),
  sample_prior = "only", backend = "rstan",
  chains = 2, iter = 500, warmup = 250, seed = 1
)

draw_ids <- 1:200
beta_probs <- posterior_epred(fit_beta_prior, draw_ids = draw_ids)

cat("Beta prior-predictive probability range: [", min(beta_probs), ",", max(beta_probs), "]\n")
cat("Beta prior-predictive probability median:", median(beta_probs), "\n")
cat("Observed accident_risk median (subsample):", median(sub$accident_risk), "\n")
