source("src/metrics.R")
library(ggplot2)

winner <- trimws(readLines("models/winning_family.txt"))
oof <- read.csv("tables_figures/cv/oof_predictions.csv")
pred_col <- sprintf("%s_pred", winner)
oof$pred <- oof[[pred_col]]

dir.create("tables_figures/calibration", showWarnings = FALSE, recursive = TRUE)

oof$decile <- cut(oof$pred, breaks = quantile(oof$pred, probs = seq(0, 1, 0.1)),
                   include.lowest = TRUE, labels = FALSE)

reliability <- aggregate(
  cbind(observed_mean = accident_risk, predicted_mean = pred) ~ decile,
  data = oof, FUN = mean
)

overall_rmse <- rmse(oof$accident_risk, oof$pred)
cat(sprintf("Pooled out-of-fold RMSE (%s): %.5f\n", winner, overall_rmse))

write.csv(
  data.frame(metric = "rmse", value = overall_rmse, family = winner),
  "tables_figures/calibration/calibration_summary.csv", row.names = FALSE
)

ggplot(reliability, aes(x = predicted_mean, y = observed_mean)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(size = 3) +
  geom_line() +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    x = "Mean predicted accident_risk (per decile bin)", y = "Mean observed accident_risk",
    title = sprintf("Reliability diagram (%s, out-of-fold predictions, 10 bins)", winner)
  ) +
  theme_minimal()
ggsave("tables_figures/calibration/reliability_diagram.png", width = 7, height = 7)

# Boundary-mass check: does predicted mass concentrate near 0/1 in proportion
# to how often accident_risk is exactly 0/1 in the observed data?
boundary_check <- data.frame(
  boundary = c("exact_0", "exact_1"),
  observed_count = c(sum(oof$accident_risk == 0), sum(oof$accident_risk == 1)),
  observed_proportion = c(mean(oof$accident_risk == 0), mean(oof$accident_risk == 1)),
  mean_predicted_at_boundary = c(
    mean(oof$pred[oof$accident_risk == 0]),
    mean(oof$pred[oof$accident_risk == 1])
  )
)
write.csv(boundary_check, "tables_figures/calibration/boundary_mass_check.csv", row.names = FALSE)
print(boundary_check)
