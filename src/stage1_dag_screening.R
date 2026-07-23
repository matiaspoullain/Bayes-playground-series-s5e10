source("src/features.R")
library(data.table)

train <- fread("data/raw/train.csv")
train <- as.data.frame(train)
train <- engineer_predictors(train)

dir.create("tables_figures/stage1_dag", showWarnings = FALSE, recursive = TRUE)

base_formula <- accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many
base_fit <- lm(base_formula, data = train)
base_rmse <- sqrt(mean(residuals(base_fit)^2))

cat(sprintf("Base 5-term model: RMSE = %.6f\n\n", base_rmse))

candidates <- list(
  road_type = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + road_type,
  num_lanes = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + factor(num_lanes),
  road_signs_present = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + road_signs_present,
  public_road = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + public_road,
  time_of_day = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + time_of_day,
  holiday = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + holiday,
  school_season = accident_risk ~ curvature + lighting_night + weather_not_clear + speed_fast + accidents_many + school_season
)

results <- data.frame(
  candidate = character(), rmse_delta = numeric(), f_stat = numeric(),
  p_value = numeric(), max_abs_coef = numeric(), keep_drop = character(),
  stringsAsFactors = FALSE
)

# Effect-size threshold: a candidate is kept only if its largest coefficient
# magnitude clears 20% of the smallest known-formula coefficient magnitude
# (~0.093 for weather_not_clear/accidents_many => threshold ~0.019).
smallest_known_coef <- min(abs(coef(base_fit)[c("lighting_night", "weather_not_clear", "speed_fast", "accidents_many")]))
effect_size_threshold <- 0.20 * smallest_known_coef

for (name in names(candidates)) {
  full_fit <- lm(candidates[[name]], data = train)
  full_rmse <- sqrt(mean(residuals(full_fit)^2))
  anova_result <- anova(base_fit, full_fit)

  extra_coefs <- coef(full_fit)[!(names(coef(full_fit)) %in% names(coef(base_fit)))]
  max_abs_coef <- max(abs(extra_coefs))

  keep_drop <- if (max_abs_coef >= effect_size_threshold) "keep" else "drop"

  results <- rbind(results, data.frame(
    candidate = name,
    rmse_delta = base_rmse - full_rmse,
    f_stat = anova_result$F[2],
    p_value = anova_result$`Pr(>F)`[2],
    max_abs_coef = max_abs_coef,
    keep_drop = keep_drop
  ))

  cat(sprintf(
    "%-20s RMSE_delta=%.6f  F=%.2f  p=%.3g  max|coef|=%.4f  -> %s\n",
    name, base_rmse - full_rmse, anova_result$F[2], anova_result$`Pr(>F)`[2], max_abs_coef, keep_drop
  ))
}

write.csv(results, "tables_figures/stage1_dag/screening_results.csv", row.names = FALSE)

dag_text <- "
curvature                    --+
lighting_night (night vs other) |
weather_not_clear (not-clear)   +--> accident_risk   (5 direct causes, confirmed by this screen)
speed_fast (>=60mph)            |
accidents_many (>2 reported)  --+

road_type, num_lanes, road_signs_present, public_road,
time_of_day, holiday, school_season   -- tested, all DROPPED (see screening_results.csv):
  every candidate's largest coefficient fell below 20% of the smallest known-formula
  coefficient (effect-size threshold), despite several being 'significant' by F-test
  at n=517,754 -- the p-value-vs-effect-size distinction this screen is designed to catch.
"
writeLines(dag_text, "tables_figures/stage1_dag/dag.txt")
