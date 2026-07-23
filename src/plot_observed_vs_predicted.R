library(ggplot2)
library(ggExtra)
library(grid)
library(gridExtra)

oof <- read.csv("tables_figures/cv/oof_predictions.csv")

dir.create("tables_figures/calibration", showWarnings = FALSE, recursive = TRUE)

# Title/subtitle are added separately below (via grid.arrange's `top`), not as
# ggplot labs() -- ggMarginal() rebuilds its own gtable around the panel and
# does not reserve space for a ggplot title, so it ends up squeezed/hidden
# behind the top marginal density plot.
p <- ggplot(oof, aes(x = beta_pred, y = accident_risk)) +
  geom_point(alpha = 0.15, size = 0.8, color = "#1B6CA8") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "firebrick", linewidth = 0.7) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Predicted accident_risk", y = "Observed accident_risk") +
  theme_minimal()

p_marginal <- ggMarginal(p, type = "density", fill = "#1B6CA8", alpha = 0.4)

png("tables_figures/calibration/observed_vs_predicted.png", width = 7, height = 7.5, units = "in", res = 300, bg = "white")
grid.arrange(
  p_marginal,
  top = textGrob(
    "Observed vs. predicted accident_risk (out-of-fold, subsample)\nDashed line = perfect prediction",
    gp = gpar(fontsize = 13), just = "left", x = 0.02
  )
)
dev.off()
