engineer_predictors <- function(df) {
  df$lighting_night <- as.integer(df$lighting == "night")
  df$weather_not_clear <- as.integer(df$weather != "clear")
  df$speed_fast <- as.integer(df$speed_limit >= 60)
  df$accidents_many <- as.integer(df$num_reported_accidents > 2)
  df
}

stratified_subsample <- function(df, target_col, n, n_strata = 10, seed = 42, boundary_values = NULL) {
  set.seed(seed)

  # Boundary values (e.g. exact 0/1 for accident_risk) are guaranteed full
  # inclusion rather than proportional decile representation -- a rare boundary
  # value (e.g. 61 of 517,754 rows) can easily be reduced to 0-1 rows by
  # proportional sampling alone, which undermines the model's ability to learn
  # anything about that part of the outcome's range.
  boundary_idx <- integer(0)
  if (!is.null(boundary_values)) {
    boundary_idx <- which(df[[target_col]] %in% boundary_values)
  }

  remainder_df <- df[setdiff(seq_len(nrow(df)), boundary_idx), ]
  n_remainder <- n - length(boundary_idx)

  breaks <- quantile(remainder_df[[target_col]], probs = seq(0, 1, length.out = n_strata + 1))
  breaks[1] <- -Inf
  breaks[length(breaks)] <- Inf
  strata <- cut(remainder_df[[target_col]], breaks = breaks, include.lowest = TRUE, labels = FALSE)

  strata_sizes <- table(strata)
  n_per_stratum <- floor(n_remainder * strata_sizes / nrow(remainder_df))

  shortfall <- n_remainder - sum(n_per_stratum)
  if (shortfall > 0) {
    largest_stratum <- names(which.max(strata_sizes))
    n_per_stratum[largest_stratum] <- n_per_stratum[largest_stratum] + shortfall
  }

  sampled_rows <- unlist(lapply(names(n_per_stratum), function(s) {
    idx <- which(strata == as.integer(s))
    sample(idx, size = min(n_per_stratum[[s]], length(idx)))
  }))

  rbind(df[boundary_idx, ], remainder_df[sampled_rows, ])
}
