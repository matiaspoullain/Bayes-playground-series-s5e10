library(brms)
library(ggplot2)

fit <- readRDS("models/fit_beta.rds")
sub <- read.csv("data/processed/model_subsample.csv")

dir.create("tables_figures/report", showWarnings = FALSE, recursive = TRUE)

# --- Raw logit-scale coefficient table (for completeness / diagnostics) ---

logit_fixef <- as.data.frame(fixef(fit))
logit_fixef$term <- rownames(logit_fixef)
write.csv(logit_fixef, "tables_figures/report/logit_scale_coefficients.csv", row.names = FALSE)
print(logit_fixef)

# --- Average marginal effects (AME) on the natural accident_risk [0,1] scale ---
# Beta's coefficients live on the logit scale and are not directly comparable to
# the published formula's linear-probability-scale coefficients (0.3/0.2/0.1/0.2/0.1).
# AMEs restore that comparability: for each predictor, the average change in
# predicted accident_risk across the subsample when that predictor is moved across
# its full observed range, holding every other predictor at each row's own value,
# with the full posterior propagated (not just a point estimate).

draw_ids <- 1:500
binary_predictors <- c("lighting_night", "weather_not_clear", "speed_fast", "accidents_many")

compute_ame <- function(predictor, low_val, high_val) {
  data_low <- sub
  data_low[[predictor]] <- low_val
  data_high <- sub
  data_high[[predictor]] <- high_val

  pred_low <- posterior_epred(fit, newdata = data_low, draw_ids = draw_ids)
  pred_high <- posterior_epred(fit, newdata = data_high, draw_ids = draw_ids)

  # per-draw average effect across all rows -- the full posterior distribution
  # of the AME, not just its summary
  rowMeans(pred_high - pred_low)
}

terms_order <- c("curvature", binary_predictors)
published <- c(
  curvature = 0.3, lighting_night = 0.2, weather_not_clear = 0.1,
  speed_fast = 0.2, accidents_many = 0.1
)

ame_draws <- lapply(terms_order, function(p) compute_ame(p, 0, 1))
names(ame_draws) <- terms_order

ame_results <- do.call(rbind, lapply(terms_order, function(p) {
  draws <- ame_draws[[p]]
  data.frame(
    term = p, ame = mean(draws),
    ame_q2_5 = unname(quantile(draws, 0.025)),
    ame_q97_5 = unname(quantile(draws, 0.975)),
    published_original_data = published[p]
  )
}))
rownames(ame_results) <- NULL

write.csv(ame_results, "tables_figures/report/average_marginal_effects.csv", row.names = FALSE)
print(ame_results)

phi_summary <- summary(fit)$spec_pars["phi", ]
cat(sprintf(
  "\nPosterior phi (Beta precision): %.3f [%.3f, %.3f] -- no published analogue (Beta has no direct sigma mapping)\n",
  phi_summary["Estimate"], phi_summary["l-95% CI"], phi_summary["u-95% CI"]
))

ame_draws_long <- do.call(rbind, lapply(terms_order, function(p) {
  data.frame(term = p, ame_draw = ame_draws[[p]])
}))
ame_draws_long$term <- factor(ame_draws_long$term, levels = terms_order)
published_df <- data.frame(term = factor(terms_order, levels = terms_order), published = published[terms_order])

ggplot(ame_draws_long, aes(x = ame_draw)) +
  geom_density(fill = "#1B6CA8", alpha = 0.4, color = "#1B6CA8") +
  geom_vline(
    data = published_df, aes(xintercept = published),
    linetype = "dashed", color = "firebrick", linewidth = 0.7
  ) +
  facet_wrap(~ term, scales = "free") +
  labs(
    x = "Average marginal effect on accident_risk (probability-scale)", y = "Posterior density",
    title = "Posterior distribution of each predictor's effect vs. the published formula",
    subtitle = "Dashed red line = published (original data) coefficient"
  ) +
  theme_minimal()
ggsave("tables_figures/report/coefficient_forest_plot.png", width = 10, height = 7)
