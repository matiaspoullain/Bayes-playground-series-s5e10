library(brms)

tobit_formula <- function() {
  bf(accident_risk | cens(cens_indicator) ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many)
}

tobit_family <- function() {
  gaussian()
}

tobit_priors <- function() {
  c(
    set_prior("normal(0.02, 0.05)", class = "Intercept"),
    set_prior("normal(0.3, 0.15)", class = "b", coef = "curvature"),
    set_prior("normal(0.2, 0.15)", class = "b", coef = "lighting_night"),
    set_prior("normal(0.1, 0.15)", class = "b", coef = "weather_not_clear"),
    set_prior("normal(0.2, 0.15)", class = "b", coef = "speed_fast"),
    set_prior("normal(0.1, 0.15)", class = "b", coef = "accidents_many"),
    set_prior("normal(0.05, 0.02)", class = "sigma", lb = 0)
  )
}

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
