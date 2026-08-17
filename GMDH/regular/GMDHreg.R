# Faithful GMDH Implementation in R
# Based on Ivakhnenko (1970, 1971)
# Pure R, no external GMDH packages - implements the algorithm directly

#' Fit a GMDH Network
#'
#' Builds a multilayer GMDH network using pairwise second-degree polynomials,
#' layer-by-layer survivor selection, and a single train/selection split.
#'
#' @param X_train Training features, n_train x p matrix
#' @param y_train Training target, length n_train vector
#' @param X_selection Selection features, n_selection x p matrix
#' @param y_selection Selection target, length n_selection vector
#' @param n_keep Number of best neurons to keep per layer (integer)
#' @param max_layers Maximum number of layers to build (integer)
#' @param patience Number of non-improving layers tolerated (integer)
#' @param ridge Ridge regression parameter for stability (numeric)
#'
#' @return A list containing:
#'   - layers: list of layer objects, each containing surviving neurons
#'   - best_layer_idx: index of the best-performing layer
#'   - best_error: lowest selection-set error reached
#'   - x_mean, x_std, y_mean, y_std: standardization statistics
#'
gmdh_fit <- function(X_train, y_train, X_selection, y_selection,
                     n_keep = 8, max_layers = 10, patience = 1, ridge = 1e-6) {

  # Standardize
  x_mean <- colMeans(X_train)
  x_std <- apply(X_train, 2, sd)
  x_std[x_std < 1e-12] <- 1e-12  # avoid division by zero
  y_mean <- mean(y_train)
  y_std <- sd(y_train)
  y_std <- max(y_std, 1e-12)

  X_train_std <- scale(X_train, center = x_mean, scale = x_std)
  X_selection_std <- scale(X_selection, center = x_mean, scale = x_std)
  y_train_std <- (y_train - y_mean) / y_std
  y_selection_std <- (y_selection - y_mean) / y_std

  # Initialize
  layers <- list()
  Z_train <- X_train_std
  Z_selection <- X_selection_std
  best_error <- Inf
  best_layer_idx <- 0
  no_improve_count <- 0

  # Layer loop
  for (layer_num in 1:max_layers) {
    m <- ncol(Z_train)

    # Stop if fewer than 2 columns (can't form a pair)
    if (m < 2) break

    # Generate all pairwise candidates
    candidates <- list()
    for (i in 1:(m - 1)) {
      for (j in (i + 1):m) {
        a_tr <- Z_train[, i]
        b_tr <- Z_train[, j]
        a_sel <- Z_selection[, i]
        b_sel <- Z_selection[, j]

        # Fit neuron on training data
        neuron <- gmdh_fit_neuron(a_tr, b_tr, y_train_std, ridge)

        # Score on selection data
        y_pred_sel <- gmdh_predict_neuron(neuron, a_sel, b_sel)
        err <- sqrt(mean((y_selection_std - y_pred_sel)^2))

        # Compute outputs for both sets (to pass forward to next layer)
        y_pred_tr <- gmdh_predict_neuron(neuron, a_tr, b_tr)

        candidates[[length(candidates) + 1]] <- list(
          error = err,
          neuron = neuron,
          output_train = y_pred_tr,
          output_selection = y_pred_sel,
          i = i,
          j = j
        )
      }
    }

    # Rank by error and keep top n_keep
    n_candidates <- length(candidates)
    errors <- sapply(candidates, function(c) c$error)
    order_idx <- order(errors)
    survivors_idx <- order_idx[1:min(n_keep, n_candidates)]

    # Build layer
    layer <- list(
      neurons = lapply(survivors_idx, function(idx) candidates[[idx]]$neuron),
      pairs = lapply(survivors_idx, function(idx) {
        list(i = candidates[[idx]]$i, j = candidates[[idx]]$j)
      }),
      best_error = errors[survivors_idx[1]]
    )
    layers[[layer_num]] <- layer

    # Prepare outputs for next layer
    outputs_train <- sapply(survivors_idx, function(idx) {
      candidates[[idx]]$output_train
    })
    outputs_selection <- sapply(survivors_idx, function(idx) {
      candidates[[idx]]$output_selection
    })

    Z_train <- as.matrix(outputs_train)
    Z_selection <- as.matrix(outputs_selection)

    # Check for improvement
    if (layer$best_error < best_error - 1e-12) {
      best_error <- layer$best_error
      best_layer_idx <- layer_num
      no_improve_count <- 0
    } else {
      no_improve_count <- no_improve_count + 1
    }

    # Early stopping
    if (no_improve_count > patience) {
      break
    }
  }

  # Trim to best layer
  if (best_layer_idx > 0) {
    layers <- layers[1:best_layer_idx]
  }

  list(
    layers = layers,
    best_layer_idx = best_layer_idx,
    best_error = best_error,
    x_mean = x_mean,
    x_std = x_std,
    y_mean = y_mean,
    y_std = y_std
  )
}


#' Fit a single GMDH neuron (second-degree polynomial on two inputs)
#'
#' @param a First input vector
#' @param b Second input vector
#' @param y Target vector
#' @param ridge Ridge regularization parameter
#'
#' @return Coefficient vector c(a0, a1, a2, a3, a4, a5) for:
#'   y = a0 + a1*a + a2*b + a3*a^2 + a4*b^2 + a5*a*b

gmdh_fit_neuron <- function(a, b, y, ridge = 1e-6) {
  X <- cbind(1, a, b, a^2, b^2, a * b)
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)

  # Ridge regression: solve (X'X + ridge*I) * w = X'y
  A <- XtX + ridge * diag(ncol(X))
  w <- solve(A, Xty)
  as.vector(w)
}


#' Predict using a fitted GMDH neuron
#'
#' @param neuron Coefficient vector from gmdh_fit_neuron
#' @param a First input vector
#' @param b Second input vector
#'
#' @return Predicted values

gmdh_predict_neuron <- function(neuron, a, b) {
  a0 <- neuron[1]
  a1 <- neuron[2]
  a2 <- neuron[3]
  a3 <- neuron[4]
  a4 <- neuron[5]
  a5 <- neuron[6]

  a0 + a1 * a + a2 * b + a3 * a^2 + a4 * b^2 + a5 * a * b
}


#' Predict using a fitted GMDH network
#'
#' @param gmdh_model Fitted model from gmdh_fit
#' @param X_test Test features, n_test x p matrix
#'
#' @return Predicted values on original (de-standardized) scale

gmdh_predict <- function(gmdh_model, X_test) {
  # Standardize using training statistics
  X_test_std <- scale(X_test, center = gmdh_model$x_mean, scale = gmdh_model$x_std)

  # Replay through layers
  Z <- X_test_std
  for (layer in gmdh_model$layers) {
    outputs <- matrix(nrow = nrow(Z), ncol = length(layer$neurons))
    for (k in 1:length(layer$neurons)) {
      neuron <- layer$neurons[[k]]
      pair <- layer$pairs[[k]]
      i <- pair$i
      j <- pair$j
      outputs[, k] <- gmdh_predict_neuron(neuron, Z[, i], Z[, j])
    }
    Z <- outputs
  }

  # Extract best neuron (first column of final layer)
  y_pred_std <- Z[, 1]

  # De-standardize
  y_pred <- y_pred_std * gmdh_model$y_std + gmdh_model$y_mean
  y_pred
}