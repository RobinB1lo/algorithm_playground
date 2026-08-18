#!/usr/bin/env Rscript

# ------------------------------------------------------------------------------
# GMDH Comparison – Custom R implementation (with robust fallbacks)
# Hyperparameters: prune=8, maxlayers=15, ridge=1e-6
# Script location: r_gmdh/
# Data: ../data/
# Results: ../r_results/
# ------------------------------------------------------------------------------

# Install required packages
required_pkgs <- c("Metrics", "ggplot2", "gridExtra", "grid", "caret")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

# Load packages
library(Metrics)
library(ggplot2)
library(gridExtra)
library(grid)
library(caret)

# ------------------------------------------------------------------------------
# Custom GMDH implementation with robust checks
# ------------------------------------------------------------------------------

gmdh_fit <- function(X, y, n_keep = 8, max_layers = 15, ridge = 1e-6) {
  # Standardize
  x_mean <- colMeans(X)
  x_sd <- apply(X, 2, sd) + 1e-12
  y_mean <- mean(y)
  y_sd <- sd(y) + 1e-12
  
  X_scaled <- scale(X, center = x_mean, scale = x_sd)
  y_scaled <- (y - y_mean) / y_sd
  
  # Split into train (50%) and selection (50%)
  n <- nrow(X_scaled)
  set.seed(42)  # for reproducibility
  train_idx <- sample(n, size = floor(0.5 * n))
  Z_tr <- X_scaled[train_idx, ]
  Z_se <- X_scaled[-train_idx, ]
  y_tr <- y_scaled[train_idx]
  y_se <- y_scaled[-train_idx]
  
  layers <- list()
  best_error <- Inf
  best_layer <- 0
  
  for (layer_idx in 1:max_layers) {
    if (ncol(Z_tr) < 2) break
    
    n_features <- ncol(Z_tr)
    candidates <- list()
    
    for (i in 1:(n_features-1)) {
      for (j in (i+1):n_features) {
        a <- Z_tr[, i]
        b <- Z_tr[, j]
        # Build polynomial features
        X_poly <- cbind(1, a, b, a*b, a^2, b^2)
        
        # Skip if any column is constant or has NaN
        if (any(is.na(X_poly)) || any(is.infinite(X_poly))) next
        
        # Ridge regression with error handling
        coef <- tryCatch({
          solve(t(X_poly) %*% X_poly + ridge * diag(6),
                t(X_poly) %*% y_tr)
        }, error = function(e) NULL)
        if (is.null(coef)) next
        if (any(is.na(coef)) || any(is.infinite(coef))) next
        
        # Evaluate on selection set
        a_se <- Z_se[, i]
        b_se <- Z_se[, j]
        X_poly_se <- cbind(1, a_se, b_se, a_se*b_se, a_se^2, b_se^2)
        pred_se <- X_poly_se %*% coef
        if (any(is.na(pred_se)) || any(is.infinite(pred_se))) next
        
        error <- sqrt(mean((y_se - pred_se)^2))
        if (is.na(error) || is.infinite(error)) next
        
        candidates <- c(candidates, list(list(
          i = i, j = j,
          coef = coef,
          error = error,
          output_tr = X_poly %*% coef,
          output_se = pred_se
        )))
      }
    }
    
    # If no candidates, break (or try with larger ridge)
    if (length(candidates) == 0) {
      cat("  Layer", layer_idx, "produced no valid neurons. Breaking.\n")
      break
    }
    
    # Keep n_keep best
    errors <- sapply(candidates, function(x) x$error)
    best_idx <- order(errors)[1:min(n_keep, length(candidates))]
    kept <- candidates[best_idx]
    
    # Check for improvement
    current_best <- min(errors)
    if (current_best < best_error - 1e-12) {
      best_error <- current_best
      best_layer <- layer_idx
    }
    
    # Prepare next layer inputs
    Z_tr <- do.call(cbind, lapply(kept, function(x) x$output_tr))
    Z_se <- do.call(cbind, lapply(kept, function(x) x$output_se))
    layers <- c(layers, list(kept))
  }
  
  # If no layers were built, return a simple linear model (fallback)
  if (length(layers) == 0) {
    cat("  No layers built. Using linear regression fallback.\n")
    # Simple linear regression on original scaled features
    X_tr <- X_scaled[train_idx, ]
    y_tr <- y_scaled[train_idx]
    coef <- solve(t(X_tr) %*% X_tr + ridge * diag(ncol(X_tr)),
                  t(X_tr) %*% y_tr)
    # We'll store as a single "neuron" with i=j=1 (but we'll handle prediction)
    layers <- list(list(list(
      i = 1, j = 1,
      coef = c(0, coef),  # intercept + slopes
      error = best_error,
      output_tr = X_tr %*% coef,
      output_se = X_scaled[-train_idx, ] %*% coef
    )))
    best_layer <- 1
  }
  
  # Keep only layers up to best_layer
  if (best_layer == 0) best_layer <- 1
  layers <- layers[1:best_layer]
  
  list(
    layers = layers,
    x_mean = x_mean,
    x_sd = x_sd,
    y_mean = y_mean,
    y_sd = y_sd
  )
}

predict_gmdh <- function(model, X) {
  X_scaled <- scale(X, center = model$x_mean, scale = model$x_sd)
  Z <- X_scaled
  for (layer in model$layers) {
    Z_new <- matrix(NA, nrow = nrow(Z), ncol = length(layer))
    for (k in seq_along(layer)) {
      neuron <- layer[[k]]
      a <- Z[, neuron$i]
      b <- Z[, neuron$j]
      # For linear fallback, i and j may be 1; we handle it
      if (neuron$i == 1 && neuron$j == 1 && ncol(Z) == 1) {
        # Linear model with intercept: use only a
        X_poly <- cbind(1, a)
      } else {
        X_poly <- cbind(1, a, b, a*b, a^2, b^2)
      }
      Z_new[, k] <- X_poly %*% neuron$coef
    }
    Z <- Z_new
  }
  # Extract first neuron output and de-standardize
  if (ncol(Z) == 0) {
    y_pred_scaled <- rep(0, nrow(X))
  } else {
    y_pred_scaled <- Z[, 1]
  }
  y_pred <- y_pred_scaled * model$y_sd + model$y_mean
  return(y_pred)
}

# ------------------------------------------------------------------------------
# Wrapper to match Python interface
# ------------------------------------------------------------------------------

gmdh <- function(X, y, prune, maxlayers, ridge, verbose = FALSE) {
  if (verbose) cat("Using custom GMDH with prune=", prune, ", maxlayers=", maxlayers, ", ridge=", ridge, "\n")
  gmdh_fit(X, y, n_keep = prune, max_layers = maxlayers, ridge = ridge)
}

# ------------------------------------------------------------------------------
# Paths – script is in r_gmdh/, data in ../data/, results in ../r_results/
# ------------------------------------------------------------------------------

data_dir <- file.path("..", "data")
results_dir <- file.path("..", "r_results")

if (!dir.exists(data_dir)) {
  stop("Data directory not found: ", data_dir,
       "\nPlease ensure the 'data/' folder is at the same level as the 'r_gmdh/' folder.")
}

datasets <- c("wildfire", "weather", "ecological", "air_quality")
missing_files <- datasets[!file.exists(file.path(data_dir, paste0(datasets, ".csv")))]
if (length(missing_files) > 0) {
  stop("Missing dataset(s): ", paste(missing_files, collapse=", "),
       "\nRun 'python data_generation.py' from the project root.")
}

if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

# ------------------------------------------------------------------------------
# Helper: standardise features and target
# ------------------------------------------------------------------------------

standardise <- function(X_train, y_train, X_test, y_test) {
  x_mean <- colMeans(X_train)
  x_sd   <- apply(X_train, 2, sd)
  x_sd[x_sd == 0] <- 1e-12
  
  y_mean <- mean(y_train)
  y_sd   <- sd(y_train)
  y_sd   <- ifelse(y_sd == 0, 1e-12, y_sd)
  
  X_train_scaled <- scale(X_train, center = x_mean, scale = x_sd)
  X_test_scaled  <- scale(X_test,  center = x_mean, scale = x_sd)
  y_train_scaled <- (y_train - y_mean) / y_sd
  
  list(
    X_train = X_train_scaled,
    X_test  = X_test_scaled,
    y_train = y_train_scaled,
    y_test  = y_test,
    y_mean  = y_mean,
    y_sd    = y_sd
  )
}

# ------------------------------------------------------------------------------
# Evaluate GMDH on a single dataset – with NaN/Inf checks
# ------------------------------------------------------------------------------

evaluate_gmdh <- function(X_train, y_train, X_test, y_test) {
  scaler <- standardise(X_train, y_train, X_test, y_test)
  
  start_time <- Sys.time()
  
  fit <- tryCatch({
    gmdh(scaler$X_train, scaler$y_train,
         prune = 8,
         maxlayers = 15,
         ridge = 1e-6,
         verbose = TRUE)
  }, error = function(e) {
    cat("ERROR in gmdh():", e$message, "\n")
    return(NULL)
  })
  
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  if (is.null(fit)) {
    return(list(status = "failed", error = "gmdh() returned NULL"))
  }
  
  # Predict on test set
  y_pred_scaled <- tryCatch({
    predict_gmdh(fit, scaler$X_test)
  }, error = function(e) {
    cat("ERROR in predict():", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(y_pred_scaled)) {
    return(list(status = "failed", error = "predict() failed"))
  }
  
  # Check for NaN/Inf
  if (any(is.na(y_pred_scaled)) || any(is.infinite(y_pred_scaled))) {
    return(list(status = "failed", error = "predictions contain NaN/Inf"))
  }
  
  y_pred <- y_pred_scaled * scaler$y_sd + scaler$y_mean
  
  # Compute metrics; check if finite
  r2   <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)
  rmse <- rmse(y_test, y_pred)
  mae  <- mae(y_test, y_pred)
  
  if (!is.finite(r2) || !is.finite(rmse) || !is.finite(mae)) {
    return(list(status = "failed", error = "metrics contain NaN/Inf"))
  }
  
  list(
    model = "GMDH",
    test_r2 = r2,
    test_rmse = rmse,
    test_mae = mae,
    time = elapsed,
    status = "success"
  )
}

# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------

set.seed(42)
results_all <- list()

for (ds in datasets) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("DATASET:", ds, "\n")
  
  file_path <- file.path(data_dir, paste0(ds, ".csv"))
  data <- read.csv(file_path)
  X <- as.matrix(data[, -ncol(data)])
  y <- data[, ncol(data)]
  
  train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[train_idx, ]
  X_test  <- X[-train_idx, ]
  y_train <- y[train_idx]
  y_test  <- y[-train_idx]
  
  cat("  Training GMDH...\n")
  res <- evaluate_gmdh(X_train, y_train, X_test, y_test)
  
  if (res$status == "success") {
    cat("    Test R² =", round(res$test_r2, 4),
        " RMSE =", round(res$test_rmse, 4),
        " MAE =", round(res$test_mae, 4),
        " time =", round(res$time, 3), "s\n")
    results_all <- c(results_all, list(c(dataset = ds, res)))
  } else {
    cat("    Failed:", res$error, "\n")
    # Still record as failed so we have an entry? We can skip or add with NA.
    # For plotting, we'll skip failed ones.
  }
}

# If all failed, stop
if (length(results_all) == 0) {
  stop("No successful GMDH evaluations. Check the error messages above.")
}

# Convert to data frame
df <- do.call(rbind, lapply(results_all, function(x) as.data.frame(t(unlist(x)))))
df$dataset <- as.character(df$dataset)
df$test_r2 <- as.numeric(df$test_r2)
df$test_rmse <- as.numeric(df$test_rmse)
df$test_mae <- as.numeric(df$test_mae)
df$time <- as.numeric(df$time)

# ------------------------------------------------------------------------------
# Plot results – save as r_results.png and r_results.csv
# ------------------------------------------------------------------------------

p_r2 <- ggplot(df, aes(x = dataset, y = test_r2)) +
  geom_col(fill = "#378ADD") +
  labs(title = "Test R² by Dataset (GMDH)", x = "Dataset", y = "R²") +
  theme_minimal() +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  ylim(0, 1)

p_time <- ggplot(df, aes(x = dataset, y = time)) +
  geom_col(fill = "#185FA5") +
  labs(title = "Computation Time (seconds)", x = "Dataset", y = "Time (s)") +
  theme_minimal() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

summary_table <- data.frame(
  Dataset = df$dataset,
  R2 = round(df$test_r2, 4),
  RMSE = round(df$test_rmse, 4),
  MAE = round(df$test_mae, 4),
  Time = round(df$time, 3)
)
table_grob <- tableGrob(summary_table, rows = NULL,
                        theme = ttheme_minimal(
                          core = list(fg_params = list(cex = 0.8)),
                          colhead = list(fg_params = list(cex = 0.9, fontface = "bold"))
                        ))

final_plot <- grid.arrange(
  p_r2, p_time, table_grob,
  ncol = 2,
  nrow = 2,
  layout_matrix = rbind(c(1, 2), c(3, 3)),
  heights = c(1, 0.6),
  top = textGrob("GMDH Performance (vanilla settings – custom R implementation)",
                 gp = gpar(fontsize = 16, fontface = "bold"))
)

ggsave(file.path(results_dir, "r_results.png"), final_plot, width = 10, height = 8, dpi = 150)
cat("\n✓ Plot saved to", file.path(results_dir, "r_results.png"), "\n")

write.csv(df, file.path(results_dir, "r_results.csv"), row.names = FALSE)
cat("✓ Raw results saved to", file.path(results_dir, "r_results.csv"), "\n")