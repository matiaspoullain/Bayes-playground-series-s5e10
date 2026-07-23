library(brms)

beta_formula <- function() {
  bf(accident_risk_squeezed ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many)
}

beta_family <- function() {
  Beta(link = "logit")
}

beta_priors <- function() {
  c(
    set_prior("normal(-2, 2)", class = "Intercept"),
    set_prior("normal(0, 2)", class = "b", coef = "curvature"),
    set_prior("normal(0, 2)", class = "b", coef = "lighting_night"),
    set_prior("normal(0, 2)", class = "b", coef = "weather_not_clear"),
    set_prior("normal(0, 2)", class = "b", coef = "speed_fast"),
    set_prior("normal(0, 2)", class = "b", coef = "accidents_many")
  )
}
