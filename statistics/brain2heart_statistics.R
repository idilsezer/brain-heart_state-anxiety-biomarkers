# Brain-to-heart statistics
# author : Idil S.
# 01 July 2024

getwd()
library("readxl")
library("psych")
library("ggplot2")
library("ggpubr")
library("psy")
library("lme4")
library(corrplot)
library("gpboost")
library("readr")
library(broom.mixed)
library(writexl)
library(glmmLasso)
library(glmmTMB)
library("lme4")
library(dplyr)
library(effsize)
#install.packages("lmerTest")

# Read the CSV files and remove the first column
coeff_ctl = read_csv('../data/coeff_all_ctl_stats.csv')
coeff_ctl <- coeff_ctl %>% select(-1)
coeff_zg = read_csv('../data/coeff_all_zg_seg_stats.csv')
coeff_zg <- coeff_zg %>% select(-1)

# Join the data frames by the ID column
merged_data <- inner_join(coeff_ctl, coeff_zg, by = "ID", suffix = c("_ctl", "_zg"))
merged_data <- merged_data %>% select(-group_zg)
merged_data <- merged_data %>% rename(group = group_ctl)

# Function to remove outliers based on IQR and print aberrant rows
remove_outliers_iqr <- function(df, value_col) {
  # Function to detect outliers using IQR
  is_outlier <- function(s) {
    Q1 <- quantile(s, 0.05, na.rm = TRUE)
    Q3 <- quantile(s, 0.95, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    return(!(s >= lower_bound & s <= upper_bound))
  }
  
  # Apply outlier detection and create a logical vector
  outlier_flags <- is_outlier(df[[value_col]])
  
  # Print aberrant data (outliers)
  if (any(outlier_flags)) {
    outliers <- df[outlier_flags, ]
    print("Aberrant data detected (outliers):")
    print(outliers)
  }
  
  # Remove outliers
  df_cleaned <- df[!outlier_flags, ]
  
  return(df_cleaned)
}

# Initialize cleaned_data as merged_data
cleaned_data <- merged_data

# Coefficients to check for outliers
coefficients <- c("B2HF", "B2LF", "HF2B", "LF2B")

# Remove outliers for each coefficient and print aberrant rows
for (coeff in coefficients) {
  cleaned_data <- remove_outliers_iqr(cleaned_data, value_col = paste0(coeff, "_ctl"))
}

# Function to perform the Wilcoxon signed-rank test and calculate effect size & confidence interval
perform_tests <- function(cleaned_data, group) {
  # Filter the data for the specific group
  group_data <- cleaned_data %>% 
    filter(group == !!group)  # Use only the renamed 'group' column
  
  # Print the number of rows for the current group
  cat("Number of rows for group", group, ":", nrow(group_data), "\n")
  
  # Initialize a list to store results for each coefficient
  results_list <- vector("list", length(coefficients))
  
  # Perform the Wilcoxon signed-rank test and calculate effect sizes for each coefficient
  for (i in seq_along(coefficients)) {
    coeff <- coefficients[i]
    coeff_ctl <- group_data[[paste0(coeff, "_ctl")]]
    coeff_zg <- group_data[[paste0(coeff, "_zg")]]
    
    # Default result values
    result <- list(
      Group = group,
      Coefficient = coeff,
      P_Value = NA,
      P_Adjusted = NA,
      Test_Statistic = NA,
      Cliffs_Delta = NA,
      Confidence_Interval = NA
    )
    
    # Perform the test if there are enough paired data points
    if (nrow(group_data) > 1) {
      test_result <- wilcox.test(coeff_ctl, coeff_zg, paired = TRUE)
      cliffs_delta_result <- cliff.delta(coeff_zg, coeff_ctl, paired = TRUE)
      
      # Update result with actual test values
      result$P_Value <- test_result$p.value
      result$Test_Statistic <- test_result$statistic
      result$Cliffs_Delta <- cliffs_delta_result$estimate
      result$Confidence_Interval <- paste0(
        "[", 
        formatC(cliffs_delta_result$conf.int[1], format = "f", digits = 3), ", ", 
        formatC(cliffs_delta_result$conf.int[2], format = "f", digits = 3), 
        "]"
      )
    }
    
    # Append to the results list
    results_list[[i]] <- result
  }
  
  # Convert results list to a data frame
  results_df <- do.call(rbind, lapply(results_list, as.data.frame))
  
  # Adjust p-values for multiple comparisons
  results_df$P_Adjusted <- formatC(p.adjust(as.numeric(results_df$P_Value), method = "fdr"), 
                                   format = "f", digits = 7)
  
  # Format remaining columns
  results_df$P_Value <- formatC(as.numeric(results_df$P_Value), format = "f", digits = 7)
  results_df$Test_Statistic <- formatC(as.numeric(results_df$Test_Statistic), format = "f", digits = 3)
  results_df$Cliffs_Delta <- formatC(as.numeric(results_df$Cliffs_Delta), format = "f", digits = 3)
  
  return(results_df)
}

# Perform tests for groups 0 and 1
results_group0 <- perform_tests(cleaned_data, 0)
results_group1 <- perform_tests(cleaned_data, 1)

# Combine the results
final_results <- bind_rows(results_group0, results_group1)

# Display the final results
print(final_results)
