source("src/features.R")
source("src/metrics.R")
source("src/model_spec.R")
library(brms)

sub <- read.csv("data/processed/model_subsample.csv")
sub$cens_indicator <- censoring_indicator(sub$accident_risk)
sub$accident_risk_squeezed <- squeeze_unit_interval(sub$accident_risk, n = nrow(sub))

fit_tobit_prior <- brm(
  formula = tobit_formula(), data = sub, family = tobit_family(), prior = tobit_priors(),
  sample_prior = "only", backend = "rstan",
  chains = 2, iter = 500, warmup = 250, seed = 1
)

fit_beta_prior <- brm(
  formula = beta_formula(), data = sub, family = beta_family(), prior = beta_priors(),
  sample_prior = "only", backend = "rstan",
  chains = 2, iter = 500, warmup = 250, seed = 1
)

draw_ids <- 1:200
tobit_mu_draws <- posterior_epred(fit_tobit_prior, draw_ids = draw_ids)
sigma_draws <- as.data.frame(fit_tobit_prior, variable = "sigma")$sigma[draw_ids]
tobit_clipped <- sapply(seq_along(draw_ids), function(i) clip_mean(tobit_mu_draws[i, ], sigma_draws[i]))

beta_probs <- posterior_epred(fit_beta_prior, draw_ids = draw_ids)

cat("Tobit prior-predictive clipped-mean range: [", min(tobit_clipped), ",", max(tobit_clipped), "]\n")
cat("Tobit prior-predictive clipped-mean median:", median(tobit_clipped), "\n")
cat("Beta prior-predictive probability range: [", min(beta_probs), ",", max(beta_probs), "]\n")
cat("Beta prior-predictive probability median:", median(beta_probs), "\n")
cat("Observed accident_risk median (subsample):", median(sub$accident_risk), "\n")
