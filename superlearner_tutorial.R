# ==============================================================================
# Super Learner: A Practical Tutorial - Complete Reproducible R Code
# ==============================================================================
# This script contains all R code from the Overleaf article "Super Learner:
# A Practical Tutorial", organized by code boxes with their original labels.
#
# Article: Super Learner: A Practical Tutorial
# Overleaf: https://git.overleaf.com/6abe3298dba216ce9b241bf0
# GitHub repo (code only): https://github.com/migariane/SL_Tutorial
#   This repo keeps ONLY README.md + superlearner_tutorial.R (+ LICENSE).
#   All LaTeX / PDF / image artefacts live in the Overleaf project.
#
# Synced to Overleaf commit ad15a36 (2026-10-06):
#   "Fix LaTeX errors, broken code blocks, and mismatched output boxes"
#   58 rcode blocks. Obsolete Box code:cvsl-nnloglik (standalone) removed;
#   NNloglik is now demonstrated inside the clinical application.
#
# To replicate the tutorial:
#   1. Install required packages (Box code:install)
#   2. Run sections sequentially - each box builds on previous ones
#   3. Set seeds as shown for exact reproducibility
# ==============================================================================

# ---- PACKAGES ----
# Box: code:install - Installing required packages
# install.packages("SuperLearner")
# install.packages(c(
#   "glmnet", "randomForest", "ranger", "xgboost", "earth", "gam",
#   "gbm", "rpart", "nnet", "e1071", "ipred", "ada",
#   "ggplot2", "ROCR", "pROC", "caret", "data.table", "mlbench",
#   "RhpcBLASctl", "rmutil", "vcd", "nnls", "Matrix", "tmle"
# ))

# Load required packages
library(SuperLearner)
library(glmnet)
library(randomForest)
library(ranger)
library(xgboost)
library(earth)
library(gam)
library(gbm)
library(rpart)
library(nnet)
library(e1071)
library(ipred)
library(ada)
library(ggplot2)
library(ROCR)
library(pROC)
library(caret)
library(data.table)
library(mlbench)
library(RhpcBLASctl)
library(rmutil)
library(vcd)
library(nnls)
library(Matrix)

# Box: code:load-sl - Loading the SuperLearner package
library(SuperLearner)


# ==============================================================================
# SECTION 0: THEORETICAL FOUNDATION - WEIGHTED SL PROCEDURE
# (Overleaf Section "SL Fundamentals"; standalone illustration with
#  simulated Z matrix. Variables n/Y here are local and redefined below.)
# ==============================================================================

# Box: code:weighted-procedure - Implementing the weighted SL procedure
# Simulate cross-validated predictions Z (n x K matrix)
set.seed(123)
n <- 400
K <- 3
Z <- matrix(runif(n * K), nrow = n, ncol = K)
colnames(Z) <- c("SL.glmnet", "SL.randomForest", "SL.mean")
# Outcome vector
Y <- rbinom(n, 1, 0.44)
# Meta-learner: Non-negative least squares (NNLS) for convex combination
library(nnls)
fit_nnls <- nnls(Z, Y)
alpha <- fit_nnls$x / sum(fit_nnls$x)  # normalize to sum to 1
names(alpha) <- colnames(Z)
print(round(alpha, 3))
# Predict Candidate Learners on New Data (Test Set)
n_new <- 50 # Number of new test observations to predict
# Generate individual candidate predictions for new test instances
# (In real workflows, replace these with predict(fit_model, newdata = X_new))
psi_glmnet <- runif(n_new)
psi_rf     <- runif(n_new)
psi_mean   <- rep(mean(Y), n_new)
# Combine candidate predictions into a matrix matching Z's column structure
Z_new <- cbind(
  SL.glmnet       = psi_glmnet,
  SL.randomForest = psi_rf,
  SL.mean         = psi_mean
)
# Final SL Ensemble Predictions
# Option A: Explicit Manual Indexing
# Multiplies each algorithm's prediction vector by its assigned weight by name
psi_SL_manual <- (alpha["SL.glmnet"]       * psi_glmnet) +
                 (alpha["SL.randomForest"] * psi_rf)     +
                 (alpha["SL.mean"]         * psi_mean)
# Option B: Matrix Multiplication (Vectorized / Production)
# Computes the exact same linear combination across all rows simultaneously
psi_SL_matrix <- drop(Z_new %*% alpha)
# Output Results and Verify Equivalence
# Inspect first few predictions from both methods
results_comparison <- head(data.frame(
  Manual_Indexing = psi_SL_manual,
  Matrix_Product  = psi_SL_matrix
))
cat("--- First Few SL Predictions ---\n")
print(results_comparison)
cat("\n")


# ==============================================================================
# SECTION 1: SIMULATED EPIDEMIOLOGICAL DATA
# ==============================================================================

# Box: code:sim-covariates - Generating simulated covariates
set.seed(2743)
n <- 500

# Covariates (standardised to unit scale)
W1 <- runif(n, 0.5, 1)
W2 <- runif(n, 0, 1)
W3 <- runif(n, 0.25, 0.75)
W4 <- runif(n, 0, 1)

# A derived comorbidity indicator
W5 <- rbinom(n, 1, 1 / (1 + exp(1.5 * W2 - W3)))

# Box: code:sim-outcome - Generating the binary outcome
# True data-generating mechanism: interactions + non-linearity
logit_p <- -0.2 * W5 - 2 * W1 + 4 * W5 * W1 - 1.5 * W2 + sin(W4 * pi)
Y <- rbinom(n, 1, 1 / (1 + exp(-logit_p)))

dat <- data.frame(W1, W2, W3, W4, W5, Y)

# Box: code:descriptive - Descriptive summary of the simulated data
cat("Prevalence of the outcome:", round(mean(Y), 3), "\n")
summary(dat)


# ==============================================================================
# SECTION 2: BASIC SUPER LEARNER
# ==============================================================================

# Box: code:sl-single - Fitting SL with a single algorithm
set.seed(123)

# Train / test split (80 % / 20 %)
train_idx <- sample(1:n, 400)
X_train   <- dat[train_idx, 1:5]
Y_train   <- dat[train_idx, 6]
X_test    <- dat[-train_idx, 1:5]
Y_test    <- dat[-train_idx, 6]

sl_lasso <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = "SL.glmnet"
)
sl_lasso

# Box: code:lasso-coef - Extracting LASSO coefficients
# The fitted glmnet object is stored in the fitLibrary
lasso_fit <- sl_lasso$fitLibrary$SL.glmnet_All$object
round(as.matrix(coef(lasso_fit, s = "lambda.min")), 3)


# ---- Two Algorithms ----
# Box: code:sl-two - Fitting SL with two algorithms
set.seed(123)

sl_two <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c("SL.glmnet", "SL.randomForest")
)
sl_two


# ---- Full Ensemble with Reference Model ----
# Box: code:sl-reference - Including a reference model in the ensemble
set.seed(123)
sl <- SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c("SL.mean", "SL.glmnet", "SL.randomForest")
)
sl

# Box: code:sl-time - Checking computational time
# Time it took to fit
sl$times$everything


# ==============================================================================
# SECTION 3: PREDICTIONS AND PERFORMANCE
# ==============================================================================

# ---- Generating Predictions ----
# Box: code:predict - Generating predictions on the test set
# Keep ensemble and individual learner predictions
pred <- predict(sl, newdata = X_test, onlySL = FALSE)
str(pred)

# Box: code:head-pred - Examining ensemble predictions
# Ensemble predictions (probability of disease)
head(pred$pred)

# Box: code:head-library - Examining individual learner predictions
# Predictions from each individual learner
head(pred$library.predict)


# ---- Manual Weighted Sum Reconstruction ----
# Box: code:manual-prediction - Manual weighted sum reconstruction
# Extract fitted ensemble weights from the SL object
alpha_sl <- sl$coef
names(alpha_sl) <- c("SL.mean", "SL.glmnet", "SL.randomForest")
print(round(alpha_sl, 4))

# Extract individual learner predictions for test set
lib_pred <- pred$library.predict
colnames(lib_pred) <- c("SL.mean", "SL.glmnet", "SL.randomForest")

# Manual weighted sum: psi_SL = sum(alpha_k * psi_k)
manual_sl_pred <- as.vector(lib_pred %*% alpha_sl)

# Compare with predict() output
ensemble_pred <- pred$pred

# Show first 6 predictions
data.frame(
  SL.mean = round(lib_pred[1:6, "SL.mean"], 3),
  SL.glmnet = round(lib_pred[1:6, "SL.glmnet"], 3),
  SL.randomForest = round(lib_pred[1:6, "SL.randomForest"], 3),
  Manual_SL = round(manual_sl_pred[1:6], 3),
  Auto_SL = round(ensemble_pred[1:6], 3),
  Match = round(manual_sl_pred[1:6], 6) == round(ensemble_pred[1:6], 6)
)


# ---- Area Under the ROC Curve ----
# Box: code:auc - Computing the AUC on the test set
pred_rocr <- ROCR::prediction(pred$pred, Y_test)

auc_sl <- ROCR::performance(
  pred_rocr,
  measure = "auc",
  x.measure = "cutoff"
)@y.values[[1]]

cat("Test AUC:", round(auc_sl, 4), "\n")

# Box: code:roc-curves - ROC curves for the ensemble and individual algorithms
# Column order of library.predict matches SL.library:
#  1 = SL.mean, 2 = SL.glmnet, 3 = SL.randomForest
auc_mean   <- pROC::auc(Y_test, pred$library.predict[, 1])
auc_glmnet <- pROC::auc(Y_test, pred$library.predict[, 2])
auc_rf     <- pROC::auc(Y_test, pred$library.predict[, 3])

roc_mean   <- pROC::roc(Y_test, pred$library.predict[, 1], quiet = TRUE)
roc_glmnet <- pROC::roc(Y_test, pred$library.predict[, 2], quiet = TRUE)
roc_rf     <- pROC::roc(Y_test, pred$library.predict[, 3], quiet = TRUE)
roc_sl     <- pROC::roc(Y_test, pred$pred, quiet = TRUE)

roc_df <- data.frame(
  FPR   = c(1 - roc_mean$specificities, 1 - roc_glmnet$specificities,
            1 - roc_rf$specificities, 1 - roc_sl$specificities),
  TPR   = c(roc_mean$sensitivities, roc_glmnet$sensitivities,
            roc_rf$sensitivities, roc_sl$sensitivities),
  Model = rep(c(
    paste0("Mean (AUC = ", round(auc_mean, 3), ")"),
    paste0("LASSO (AUC = ", round(auc_glmnet, 3), ")"),
    paste0("Random Forest (AUC = ", round(auc_rf, 3), ")"),
    paste0("SuperLearner (AUC = ", round(auc_sl, 3), ")")
  ), times = sapply(list(roc_mean, roc_glmnet, roc_rf, roc_sl),
                    function(x) length(x$sensitivities)))
)

# Plot ROC curves
ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              color = "gray50") +
  scale_color_manual(values = c("#3498DB", "#E74C3C", "#9B59B6", "#2ECC71")) +
  labs(x = "1 - Specificity", y = "Sensitivity") +
  xlim(0, 1) + ylim(0, 1) +
  theme_minimal(base_size = 14) +
  theme(legend.position = c(0.65, 0.25))


# ==============================================================================
# SECTION 4: CROSS-VALIDATED SUPER LEARNER
# ==============================================================================

# ---- Fitting CV.SuperLearner ----
# Box: code:cvsl - Fitting CV.SuperLearner
set.seed(123)

system.time({
  cv_sl <- CV.SuperLearner(
    Y = Y_train, X = X_train,
    family = binomial(),
    V = 5,
    SL.library = c("SL.mean", "SL.glmnet", "SL.randomForest")
  )
})

# Box: code:cvsl-summary - Summarizing CV.SuperLearner results
summary(cv_sl)


# ---- Discrete Super Learner ----
# Box: code:discrete-sl - Identifying the Discrete SL
table(simplify2array(cv_sl$whichDiscreteSL))


# ---- Distribution of Ensemble Weights ----
# Box: code:review-weights - Distribution of ensemble weights across folds
review_weights <- function(cv_sl) {
  meta_weights <- coef(cv_sl)              # matrix: folds x learners
  means <- colMeans(meta_weights)
  sds   <- apply(meta_weights, 2, sd)
  mins  <- apply(meta_weights, 2, min)
  maxs  <- apply(meta_weights, 2, max)
  stats <- cbind("mean(weight)" = means, sd = sds, min = mins, max = maxs)
  stats[order(stats[, 1], decreasing = TRUE), ]
}

print(review_weights(cv_sl), digits = 3)


# ==============================================================================
# SECTION 5: HYPERPARAMETER TUNING
# ==============================================================================

# ---- Algorithm Variants ----
# Box: code:rf-wrapper - Random Forest variant with custom wrapper
SL.rf.better <- function(...) {
  SL.randomForest(..., ntree = 3000)
}

set.seed(123)

cv_sl_rf <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.rf.better",
    "SL.randomForest"
  )
)

summary(cv_sl_rf)


# Box: code:create-learner - Generating a variant with create.Learner
learners <- create.Learner(
  "SL.randomForest",
  params = list(ntree = 3000)
)

learners$names


# ---- Hyperparameter Grids ----
# Box: code:mtry-default - Reference value of mtry
floor(sqrt(ncol(X_train)))


# Box: code:mtry-grid - Generating mtry configurations
mtry_seq <- floor(
  sqrt(ncol(X_train)) * c(0.5, 1, 2)
)

learners_mtry <- create.Learner(
  "SL.randomForest",
  tune = list(mtry = mtry_seq)
)

learners_mtry$names


# Box: code:cvsl-mtry - Evaluating mtry configurations with CV.SuperLearner
set.seed(123)

cv_sl_mtry <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    learners_mtry$names,
    "SL.randomForest"
  )
)

summary(cv_sl_mtry)


# ---- XGBoost Grid ----
# Box: code:xgb-grid - Generating XGBoost hyperparameters grid
tune_xgb <- list(
  ntrees    = c(200, 500, 1000, 2000, 5000),
  max_depth = 1:4,
  shrinkage = c(0.001, 0.01, 0.1)
)

learners_xgb <- create.Learner(
  "SL.xgboost",
  tune = tune_xgb,
  detailed_names = TRUE,
  name_prefix = "xgb"
)

cat("Number of XGBoost configurations:", length(learners_xgb$names), "\n")


# ==============================================================================
# SECTION 6: FEATURE SCREENING
# ==============================================================================

# ---- Correlation-based Screening ----
# Box: code:screen-corp - Feature screening based on correlation
set.seed(123)

cv_sl_screen <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = list(
    "SL.mean",
    "SL.glmnet",
    c("SL.glmnet", "screen.corP")
  )
)

summary(cv_sl_screen)


# ---- LASSO-based Screening ----
# Box: code:screen-glmnet - LASSO-based feature screening
set.seed(123)

cv_sl_screen2 <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  SL.library = list(
    "SL.mean",
    "SL.glmnet",
    c("SL.glmnet", "screen.glmnet")
  )
)

summary(cv_sl_screen2)


# ==============================================================================
# SECTION 7: OPTIMIZING FOR AUC
# (method.NNloglik standalone box removed in Overleaf; NNloglik is now
#  demonstrated in the clinical application with method = "method.NNloglik".)
# ==============================================================================

# Box: code:cvsl-auc - CV.SuperLearner with method.AUC
set.seed(123)
cv_sl_auc <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 5,
  method = "method.AUC",
  SL.library = list("SL.mean", "SL.glmnet", "SL.randomForest",
                    c("SL.glmnet", "screen.corP"))
)
summary(cv_sl_auc)


# ==============================================================================
# SECTION 8: PARALLELIZATION
# ==============================================================================

# ---- Multicore Execution ----
# Box: code:parallel-setup - Configuring parallel execution
num_cores <- RhpcBLASctl::get_num_cores()
cat("Cores available on this machine:", num_cores, "\n")

options(mc.cores = num_cores)
set.seed(123, "L'Ecuyer-CMRG")


# Box: code:cvsl-parallel - Parallel execution of CV.SuperLearner
system.time({
  cv_sl_par <- CV.SuperLearner(
    Y = Y_train, X = X_train,
    family = binomial(),
    V = 5,
    parallel = "multicore",
    SL.library = c(
      "SL.mean",
      "SL.glmnet",
      "SL.randomForest"
    )
  )
})


# ---- mcSuperLearner ----
# Box: code:mcsuperlearner - Parallelized SL fit
set.seed(123, "L'Ecuyer-CMRG")

sl_mc <- mcSuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.randomForest"
  )
)

sl_mc


# ---- Cluster Parallel ----
# Box: code:cluster-parallel - Parallel execution using a cluster object
cl <- parallel::makeCluster(4)

# Load required packages in each worker
parallel::clusterEvalQ(cl, {
  library(SuperLearner)
  library(glmnet)
  library(randomForest)
})

set.seed(123)

cv_sl_snow <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(),
  V = 10,
  parallel = cl,
  SL.library = c(
    "SL.mean",
    "SL.glmnet",
    "SL.randomForest"
  )
)

parallel::stopCluster(cl)


# ==============================================================================
# SECTION 9: CUSTOM WRAPPERS
# ==============================================================================

# ---- Custom Learner: GBM ----
# Box: code:gbm-wrapper - Custom wrapper for GBM
SL.gbm2 <- function(Y, X, newX, family, obsWeights,
                    gbm.trees = 5000, interaction.depth = 2,
                    shrinkage = 0.001, ...) {
  form <- as.formula(paste("Y ~", paste(colnames(X), collapse = " + ")))

  distribution <- switch(family$family,
    gaussian = "gaussian",
    binomial = "bernoulli"
  )

  fit.gbm <- gbm::gbm(formula = form, data = X,
    distribution = distribution,
    n.trees = gbm.trees,
    interaction.depth = interaction.depth,
    shrinkage = shrinkage,
    cv.folds = 5, keep.data = TRUE,
    weights = obsWeights, verbose = FALSE)

  best.iter <- gbm::gbm.perf(fit.gbm, method = "cv", plot.it = FALSE)
  pred <- predict(fit.gbm, newdata = newX, best.iter, type = "response")

  out <- list(pred = pred, fit = list(object = fit.gbm, n.trees = best.iter))
  class(out$fit) <- "SL.gbm"
  return(out)
}


# ---- Custom Learner: MARS (earth) ----
# Box: code:earth-wrapper - Custom wrapper for MARS
SL.earth2 <- function(Y, X, newX, family, obsWeights, id,
                      degree = 2, penalty = 3,
                      nk = max(21, 2 * ncol(X) + 1),
                      pmethod = "backward", ...) {
  glm_arg <- NULL
  if (family$family == "binomial" && all(Y %in% c(0, 1))) {
    glm_arg <- list(family = binomial())
  }
  fit.earth <- earth::earth(x = X, y = Y, degree = degree,
    nk = nk, penalty = penalty,
    pmethod = pmethod, glm = glm_arg)
  pred <- predict(fit.earth, newdata = newX, type = "response")
  out  <- list(pred = pred, fit = list(object = fit.earth))
  class(out$fit) <- "SL.earth"
  return(out)
}


# ---- Custom Screener ----
# Box: code:custom-screener - Custom variable screening procedure
screen.glmnet3 <- function(Y, X, family, alpha = 1, minscreen = 2,
                           nfolds = 10, nlambda = 200, ...) {
  myfamily <- if (family$family == "binomial" && !all(Y %in% c(0, 1)))
    "gaussian" else family$family

  if (!is.matrix(X)) X <- model.matrix(~ -1 + ., data = X)

  fitCV <- tryCatch(
    glmnet::cv.glmnet(x = X, y = Y, lambda = NULL,
      type.measure = "deviance", nfolds = nfolds,
      family = myfamily, alpha = alpha,
      nlambda = nlambda, keep = TRUE),
    error = function(e) NULL
  )

  whichVariable <- NULL
  if (!is.null(fitCV)) {
    si <- abs((max(fitCV$nzero) - ncol(X) / 2) - fitCV$nzero)
    w2 <- as.numeric(coef(fitCV$glmnet.fit, s = fitCV$lambda.min))[-1] != 0
    w3 <- as.numeric(glmnet::coef.glmnet(
      fitCV, s = fitCV$lambda[si == min(si)][1]))[-1] != 0
    if (sum(w2) > 1)      whichVariable <- w2
    else if (sum(w3) > 1 & sum(w3) < ncol(X) / 2) whichVariable <- w3
  }
  if (is.null(whichVariable)) whichVariable <- screen.cramersv(Y, X)
  whichVariable
}

# Cramer's V fallback screener (works with binary outcome)
screen.cramersv <- function(Y, X, nscreen = 6, cts.num = 10, ...) {
  var_cont <- apply(X, 2, function(x) length(unique(x)) > cts.num)
  cvdata   <- X
  cutf <- function(x) cut(x,
    unique(quantile(x, prob = seq(0, 1, 0.2))),
    include.lowest = TRUE)
  if (sum(var_cont) > 0) cvdata[, var_cont] <- apply(cvdata[, var_cont], 2, cutf)
  cvf <- function(xx) vcd::assocstats(table(Y, xx))$cramer
  whichVariable <- order(apply(cvdata, 2, cvf), decreasing = TRUE) <= nscreen
  whichVariable
}


# ---- Incorporating Custom Wrappers ----
# Box: code:cvsl-custom - Evaluating custom wrappers
set.seed(123)

cv_sl_custom <- CV.SuperLearner(
  Y = Y_train, X = X_train,
  family = binomial(), V = 5,
  SL.library = c("SL.mean", "SL.glmnet", "SL.gbm2", "SL.earth2")
)
summary(cv_sl_custom)


# ==============================================================================
# SECTION 10: FULL EPIDEMIOLOGICAL APPLICATION (1000-patient CVD cohort)
# ==============================================================================

# Box: code:clinical-cohort - Simulating the clinical cohort
set.seed(2024)
n_clinical <- 1000

age         <- rnorm(n_clinical, 65, 10)
bmi         <- rnorm(n_clinical, 28, 5)
sbp         <- rnorm(n_clinical, 140, 20)
dbp         <- rnorm(n_clinical, 85, 12)
cholesterol <- rnorm(n_clinical, 220, 40)
glucose     <- rnorm(n_clinical, 110, 30)
smoking     <- rbinom(n_clinical, 1, 0.25)
diabetes    <- rbinom(n_clinical, 1, plogis(-3 + 0.1 * (bmi - 28) + 0.01 * (glucose - 110)))
family_hx   <- rbinom(n_clinical, 1, 0.30)

# Standardised predictors for stable estimation
stand <- function(x) (x - mean(x)) / sd(x)

logit_cvd <- -2.2 +
  0.5 * stand(age) + 0.4 * stand(bmi) + 0.3 * stand(sbp) +
  0.2 * stand(cholesterol) + 0.25 * stand(glucose) +
  0.6 * smoking + 0.5 * diabetes + 0.4 * family_hx +
  0.7 * stand(age) * smoking + 0.3 * stand(sbp) * stand(cholesterol) -
  0.2 * stand(age)^2

Y_cvd <- rbinom(n_clinical, 1, plogis(logit_cvd))
cat("CVD prevalence:", round(mean(Y_cvd), 3), "\n")

# Box: code:clinical-data - Organizing the simulated clinical data
clinical_data <- data.frame(age, bmi, sbp, dbp, cholesterol, glucose,
  smoking, diabetes, family_hx, Y = Y_cvd)

# Box: code:clinical-library - Training-test split and candidate library specification
set.seed(42)
idx <- sample(1:n_clinical, 700)
X_clin_train <- clinical_data[idx, 1:9]
Y_clin_train <- clinical_data[idx, 10]
X_clin_test  <- clinical_data[-idx, 1:9]
Y_clin_test  <- clinical_data[-idx, 10]

# Core library
sl_library <- c(
  "SL.mean", "SL.glm", "SL.glmnet", "SL.randomForest", "SL.ranger",
  "SL.earth", "SL.xgboost", "SL.rpartPrune", "SL.nnet"
)

# Hyperparameter variants
mtry_vals    <- floor(sqrt(ncol(X_clin_train)) * c(0.5, 1, 2))
rf_learners  <- create.Learner("SL.randomForest", tune = list(mtry = mtry_vals))

xgb_tune     <- list(ntrees = c(500, 1000),
  max_depth = 3:4,
  shrinkage = c(0.01, 0.1))
xgb_learners <- create.Learner("SL.xgboost", tune = xgb_tune,
  detailed_names = TRUE, name_prefix = "xgb")

full_library <- c(sl_library, rf_learners$names, xgb_learners$names)
cat("Total number of candidate learners:", length(full_library), "\n")

# Box: code:clinical-cv - Cross-validated evaluation of the candidate library
set.seed(42, "L'Ecuyer-CMRG")
options(mc.cores = RhpcBLASctl::get_num_cores())

system.time({
  cv_sl_clinical <- CV.SuperLearner(
    Y = Y_clin_train, X = X_clin_train,
    family = binomial(),
    method = "method.NNloglik",
    parallel = "multicore",
    cvControl = list(V = 10, stratifyCV = TRUE),
    SL.library = full_library
  )
})

# Box: code:clinical-cv-summary - Summarizing cross-validated performance
summary(cv_sl_clinical)

# Box: code:clinical-weights - Examining ensemble weights across CV folds
print(review_weights(cv_sl_clinical), digits = 3)

# Box: code:clinical-evaluation - Evaluating the final SL model on the test set
set.seed(42)

sl_clinical_full <- SuperLearner(
  Y = Y_clin_train, X = X_clin_train,
  family = binomial(), method = "method.NNloglik",
  SL.library = full_library
)

pred_clinical <- predict(sl_clinical_full, X_clin_test, onlySL = TRUE)

roc_test <- pROC::roc(Y_clin_test, pred_clinical$pred, quiet = TRUE)
cat("Test AUC:", round(as.numeric(pROC::auc(roc_test)), 4), "\n")
cat("95% CI :", round(as.numeric(pROC::ci.auc(roc_test)), 4), "\n")

# Box: code:clinical-roc - Plotting the ROC curve on the independent test set
plot(roc_test,
  main = "SL on the 30% held-out test set",
  print.auc = TRUE, print.auc.y = 0.4,
  col = "#2ECC71", lwd = 3)
abline(a = 0, b = 1, lty = 2, col = "gray50")

# Box: code:clinical-calibration - Assessing calibration on the independent test set
cal_df <- data.frame(predicted = as.vector(pred_clinical$pred),
  observed  = Y_clin_test)

cal_bins <- cut(cal_df$predicted, breaks = 10)
cal_sum  <- aggregate(cbind(predicted, observed) ~ cal_bins,
  data = cal_df, FUN = mean)

ggplot(cal_sum, aes(x = predicted, y = observed)) +
  geom_point(size = 3, color = "#2ECC71") +
  geom_smooth(method = "loess", se = FALSE, color = "#E74C3C") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
    color = "gray50") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Predicted probability", y = "Observed proportion") +
  theme_minimal(base_size = 14)


# ==============================================================================
# SECTION 11: CONTINUOUS OUTCOMES
# ==============================================================================

# Box: code:continuous-data - Generating data with a continuous outcome
set.seed(99)

n_reg <- 500

X_reg <- data.frame(
  x1 = rnorm(n_reg),
  x2 = rnorm(n_reg),
  x3 = rnorm(n_reg)
)

Y_reg <- with(
  X_reg,
  3 + 2 * x1 + x1^2 - 1.5 * x2 * x3 +
    rnorm(n_reg, 0, 0.5)
)

# Box: code:continuous-sl - Fitting SL with a continuous outcome
set.seed(99)
sl_reg <- SuperLearner(
  Y = Y_reg,
  X = X_reg,
  family = gaussian(),
  SL.library = c("SL.mean", "SL.glm", "SL.glmnet", "SL.randomForest", "SL.earth"))
sl_reg

# Box: code:continuous-error - Evaluating prediction error
pred_reg <- predict(sl_reg, X_reg)
mse <- mean((Y_reg - pred_reg$pred)^2)
cat("MSE :", round(mse, 4), "\n")
cat("RMSE:", round(sqrt(mse), 4), "\n")


# ==============================================================================
# SECTION 12: DOSE-RESPONSE ESTIMATION
# ==============================================================================

# Box: code:dose-response - Data generation and true curve
set.seed(12345)
n_dose <- 1000
x_dose <- runif(n_dose, 0, 8)
# True dose-response: sqrt branch below 2, quadratic branch above 2
y_dose <- 5 + 4 * sqrt(9 * x_dose) * as.numeric(x_dose < 2) +
  as.numeric(x_dose >= 2) * (abs(x_dose - 6)^2) + rlaplace(n_dose)
D_dose <- data.frame(x = x_dose, y = y_dose)
# The true curve, to compare against
xl <- seq(0, 8, 0.1)
yl <- 5 + 4 * sqrt(9 * xl) * as.numeric(xl < 2) +
  as.numeric(xl >= 2) * (abs(xl - 6)^2)
truth_df <- data.frame(x = xl, y = yl)

# Box: code:dose-sl-fit - Super Learner fit for dose-response
set.seed(123)
sl_dose <- SuperLearner(
  Y = D_dose$y, X = data.frame(x = D_dose$x),
  family = gaussian(),
  method = "method.NNLS",
  SL.library = c("SL.mean", "SL.glm", "SL.gam", "SL.earth")
)
sl_dose

# Box: code:dose-predict - Predict on fine grid
# Predict along a fine grid of "doses".
# onlySL = FALSE so library.predict keeps the genuine curve of every
# learner (with TRUE, zero-weight learners would be returned as 0)
pred_dose <- predict(sl_dose, data.frame(x = xl), onlySL = FALSE)

# Box: code:dose-response-plot - Dose-response curve plot
# Column order matches SL.library: 1 = mean, 2 = glm, 3 = gam, 4 = earth
dose_df <- rbind(
  data.frame(x = xl, y = truth_df$y,                    Model = "Truth"),
  data.frame(x = xl, y = pred_dose$pred,                Model = "SuperLearner"),
  data.frame(x = xl, y = pred_dose$library.predict[, 2], Model = "glm"),
  data.frame(x = xl, y = pred_dose$library.predict[, 3], Model = "gam")
)

ggplot() +
  geom_point(data = D_dose, aes(x, y),
    color = "gray75", alpha = 0.3, size = 0.6) +
  geom_line(data = dose_df, aes(x, y, color = Model), size = 1.1) +
  scale_color_manual(values = c(Truth = "black", SuperLearner = "#E74C3C",
    glm = "#3498DB", gam = "#2ECC71")) +
  labs(x = "Exposure (dose)", y = "Outcome") +
  theme_minimal(base_size = 14) +
  theme(legend.position = c(0.8, 0.75))


# ==============================================================================
# SECTION 13: BEST PRACTICES (summary templates)
# ==============================================================================

# Box: code:best-practice-tune - Hyperparameter grids for key algorithms
rf_learners_bp <- create.Learner("SL.randomForest",
  tune = list(mtry = c(3, 5, 7)))
xgb_learners_bp <- create.Learner("SL.xgboost", tune = list(
  ntrees = c(500, 1000), max_depth = 3:4, shrinkage = c(0.01, 0.1)
))

# Box: code:best-practice-screen - Feature screening with learner-screener pairs
SL.library_bp <- list(
  "SL.glmnet",
  c("SL.glmnet", "screen.corP"),
  c("SL.randomForest", "screen.corRank")
)

# Box: code:best-practice-parallel - Parallel computing setup
options(mc.cores = RhpcBLASctl::get_num_cores())
set.seed(1, "L'Ecuyer-CMRG")   # reproducibility across cores


# ==============================================================================
# SECTION 14: BRIDGE TO CAUSAL INFERENCE (TMLE)
# NOTE: supply your own Y (outcome), A (treatment), W (covariates).
# install.packages("tmle") if needed.
# ==============================================================================

# Box: code:tmle-connection - TMLE with SuperLearner
library(tmle)

tmle_fit <- tmle(
  Y = Y, A = A, W = W,
  Q.SL.library = c("SL.glm", "SL.glmnet", "SL.randomForest", "SL.xgboost"),
  g.SL.library = c("SL.glm", "SL.glmnet"),
  V.Q = 5,
  V.g = 5
)


# ==============================================================================
# END OF SCRIPT
# ==============================================================================
# This script reproduces all 58 code boxes from the Overleaf article:
# "Super Learner: A Practical Tutorial"
#
# Sections: Fundamentals (NNLS) / Simulated data / Basic SL / Predictions /
# CV.SL / Tuning / Screening / AUC / Parallel / Custom wrappers /
# Clinical application (CVD, n=1000) / Continuous outcomes /
# Dose-response / Best practices / TMLE bridge.
#
# For questions or issues, refer to the original Overleaf project.
# ==============================================================================
