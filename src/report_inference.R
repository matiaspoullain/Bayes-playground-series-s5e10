library(brms)
library(ggplot2)

winner <- trimws(readLines("models/winning_family.txt"))
fit <- readRDS(sprintf("models/fit_%s.rds", winner))
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

  # per-draw average effect across all rows, then summarize over draws
  per_draw_ame <- rowMeans(pred_high - pred_low)
  c(
    estimate = mean(per_draw_ame),
    q2_5 = unname(quantile(per_draw_ame, 0.025)),
    q97_5 = unname(quantile(per_draw_ame, 0.975))
  )
}

ame_results <- data.frame(
  term = character(), ame = numeric(), ame_q2_5 = numeric(), ame_q97_5 = numeric(),
  published_original_data = numeric(), stringsAsFactors = FALSE
)

published <- c(
  curvature = 0.3, lighting_night = 0.2, weather_not_clear = 0.1,
  speed_fast = 0.2, accidents_many = 0.1
)

ame_curv <- compute_ame("curvature", 0, 1)
ame_results <- rbind(ame_results, data.frame(
  term = "curvature", ame = ame_curv["estimate"], ame_q2_5 = ame_curv["q2_5"],
  ame_q97_5 = ame_curv["q97_5"], published_original_data = published["curvature"]
))

for (p in binary_predictors) {
  ame_p <- compute_ame(p, 0, 1)
  ame_results <- rbind(ame_results, data.frame(
    term = p, ame = ame_p["estimate"], ame_q2_5 = ame_p["q2_5"],
    ame_q97_5 = ame_p["q97_5"], published_original_data = published[p]
  ))
}
rownames(ame_results) <- NULL

write.csv(ame_results, "tables_figures/report/average_marginal_effects.csv", row.names = FALSE)
print(ame_results)

phi_summary <- summary(fit)$spec_pars["phi", ]
cat(sprintf(
  "\nPosterior phi (Beta precision): %.3f [%.3f, %.3f] -- no published analogue (Beta has no direct sigma mapping)\n",
  phi_summary["Estimate"], phi_summary["l-95% CI"], phi_summary["u-95% CI"]
))

ggplot(ame_results, aes(y = term)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(x = published_original_data, color = "Published (original data)"), size = 3, shape = 17) +
  geom_pointrange(aes(x = ame, xmin = ame_q2_5, xmax = ame_q97_5, color = "Posterior AME (this competition)")) +
  labs(
    x = "Effect on accident_risk (probability-scale)", y = NULL, color = NULL,
    title = "Average marginal effects (Beta model) vs. published formula coefficients"
  ) +
  theme_minimal()
ggsave("tables_figures/report/coefficient_forest_plot.png", width = 9, height = 6)
