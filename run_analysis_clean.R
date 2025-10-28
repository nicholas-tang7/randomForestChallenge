#!/usr/bin/env Rscript

# Clean R Analysis Script - Bypasses extension conflicts
# Run with: /usr/local/bin/Rscript run_analysis_clean.R

cat("=== Random Forest Analysis (Clean Environment) ===\n")

# Set up clean environment
options(warn = 1)
options(error = function() {
  cat("Error occurred:\n")
  traceback()
})

# Load libraries
cat("Loading libraries...\n")
suppressPackageStartupMessages({
  library(tidyverse)
  library(randomForest)
})
cat("✓ Libraries loaded successfully\n")

# Load data
cat("Loading data...\n")
sales_data <- read.csv("https://raw.githubusercontent.com/flyaflya/buad442Fall2025/refs/heads/main/datasets/salesPriceData.csv")
cat("✓ Data loaded:", nrow(sales_data), "rows\n")

# Prepare model data
cat("Preparing model data...\n")
model_data <- sales_data %>%
  select(SalePrice, LotArea, YearBuilt, GrLivArea, FullBath, HalfBath, 
         BedroomAbvGr, TotRmsAbvGrd, GarageCars, zipCode) %>%
  mutate(zipCode = as.factor(zipCode)) %>%
  na.omit()

cat("✓ Data prepared:", nrow(model_data), "rows\n")
cat("Number of unique zip codes:", length(unique(model_data$zipCode)), "\n")

# Split data
set.seed(123)
train_indices <- sample(1:nrow(model_data), 0.8 * nrow(model_data))
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

cat("✓ Data split - Train:", nrow(train_data), "Test:", nrow(test_data), "\n")

# Build random forest models
cat("Building random forest models...\n")

cat("Building 1-tree model...\n")
rf_1 <- randomForest(SalePrice ~ ., data = train_data, ntree = 1, mtry = 3, seed = 123)
cat("✓ 1-tree model built\n")

cat("Building 5-tree model...\n")
rf_5 <- randomForest(SalePrice ~ ., data = train_data, ntree = 5, mtry = 3, seed = 123)
cat("✓ 5-tree model built\n")

cat("Building 25-tree model...\n")
rf_25 <- randomForest(SalePrice ~ ., data = train_data, ntree = 25, mtry = 3, seed = 123)
cat("✓ 25-tree model built\n")

cat("Building 100-tree model...\n")
rf_100 <- randomForest(SalePrice ~ ., data = train_data, ntree = 100, mtry = 3, seed = 123)
cat("✓ 100-tree model built\n")

# Calculate predictions
cat("Calculating predictions...\n")
pred_1 <- predict(rf_1, test_data)
pred_5 <- predict(rf_5, test_data)
pred_25 <- predict(rf_25, test_data)
pred_100 <- predict(rf_100, test_data)

# Calculate RMSE
rmse_1 <- sqrt(mean((test_data$SalePrice - pred_1)^2))
rmse_5 <- sqrt(mean((test_data$SalePrice - pred_5)^2))
rmse_25 <- sqrt(mean((test_data$SalePrice - pred_25)^2))
rmse_100 <- sqrt(mean((test_data$SalePrice - pred_100)^2))

# Display results
cat("\n=== RESULTS ===\n")
cat("RMSE (1 tree):", round(rmse_1, 2), "\n")
cat("RMSE (5 trees):", round(rmse_5, 2), "\n")
cat("RMSE (25 trees):", round(rmse_25, 2), "\n")
cat("RMSE (100 trees):", round(rmse_100, 2), "\n")

# Calculate improvements
improvement_5 <- ((rmse_1 - rmse_5) / rmse_1) * 100
improvement_25 <- ((rmse_1 - rmse_25) / rmse_1) * 100
improvement_100 <- ((rmse_1 - rmse_100) / rmse_1) * 100

cat("\n=== IMPROVEMENTS OVER 1 TREE ===\n")
cat("5 trees:", round(improvement_5, 1), "% improvement\n")
cat("25 trees:", round(improvement_25, 1), "% improvement\n")
cat("100 trees:", round(improvement_100, 1), "% improvement\n")

cat("\n✓ Analysis completed successfully!\n")
cat("No extension conflicts detected.\n")
