source("src/features.R")
source("src/metrics.R")
source("src/model_spec.R")
library(brms)

# Beta only -- see tables_figures/compare/tobit_non_convergence_investigation.md

sub <- read.csv("data/processed/model_subsample.csv")
sub$accident_risk_squeezed <- squeeze_unit_interval(sub$accident_risk, n = nrow(sub))

fit_beta <- brm(
  formula = beta_formula(), data = sub, family = beta_family(), prior = beta_priors(),
  backend = "rstan", chains = 4, cores = 4, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.95), seed = 1, save_pars = save_pars(all = TRUE)
)

print(summary(fit_beta))

dir.create("models", showWarnings = FALSE)
saveRDS(fit_beta, "models/fit_beta.rds")
