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
#install.packages("lmerTest")
library("lmerTest")
library(tidyr)
library(dplyr)
library("effsize")
library(broom)
#install.packages("wesanderson")
library(wesanderson)
library(ggplot2)
library(effectsize)
library(readr)


df_physio = read_csv("../data/videos_all_metrics_for_stats.csv")


## create melted df based on df_physio:
df_melted <- df_physio %>%
  select(ID, condition, LF_avg, HF_avg, Fratio_avg, midline_alpha, midline_beta, group) %>%  # Select relevant columns, including 'group'
  pivot_longer(cols = c(LF_avg, HF_avg, Fratio_avg, midline_alpha, midline_beta), 
               names_to = "variable", 
               values_to = "value") # Add the 'group' column based on 'condition'

# View the result
head(df_melted)

means_df <- df_melted %>%
  group_by(variable, condition) %>%       # Group by variable and condition
  summarise(
    mean = mean(value, na.rm = TRUE),     # Calculate mean
    std = sd(value, na.rm = TRUE)         # Calculate standard deviation
  ) %>%
  arrange(variable, condition)            # Arrange the result for better readability

# View the final means dataframe
print(means_df)
#write.csv(df_melted, "df_physio_melted.csv", row.names = FALSE)

df_normalized = df_physio[, c("LF_avg", "HF_avg", "Fratio_avg", "midline_alpha", "midline_beta")]

df_normalized <- cbind(df_normalized, condition = df_physio$condition, ID = df_physio$ID)

df_normalized <- as.data.frame(df_normalized)
df_normalized$condition <- as.factor(df_normalized$condition)

#metrics_columns <- c("LF_avg", "HF_avg", "Fratio_avg", "tonic_mean", "phasic_mean", "midline_alpha", "midline_beta")
metrics_columns <- c("LF_avg", "HF_avg", "Fratio_avg", "midline_alpha", "midline_beta")

# Convert metrics columns to numeric
df_normalized[, metrics_columns] <- lapply(df_normalized[, metrics_columns], as.numeric)


##### CHECK NORMALITY DISTRIBUTION
# Assuming df_normalized has been loaded and scaled as described in your code

# Extracting the metrics columns
metrics_columns <- c("LF_avg", "HF_avg", "Fratio_avg", "midline_alpha", "midline_beta")

# Convert metrics columns to numeric
df_normalized[, metrics_columns] <- lapply(df_normalized[, metrics_columns], as.numeric)

# Shapiro-Wilk test for normality
shapiro_results <- lapply(df_normalized[, metrics_columns], shapiro.test)

# Q-Q plots
par(mfrow = c(2, 4))  # Setting up a 2x4 grid for Q-Q plots
for (i in 1:length(metrics_columns)) {
  qqnorm(df_normalized[, metrics_columns[i]], main = paste("Q-Q Plot for", metrics_columns[i]))
  qqline(df_normalized[, metrics_columns[i]], col = "red")
}

# Print Shapiro-Wilk test results
names(shapiro_results) <- metrics_columns
print(shapiro_results)

########### APPLY WILCOXON SIGNED RANKS BECAUSE NON NORMAL DISTRIBUTION

### With test statistic, effect size, confidence intervals

if (!requireNamespace("effsize", quietly = TRUE)) {
  install.packages("effsize")
}
library(effsize)

# Initialize a list to store Wilcoxon signed-rank test results
wilcox_results <- lapply(metrics_columns, function(metric) {
  # Convert the current metric's column to numeric and remove NAs
  condition_0 <- na.omit(as.numeric(unlist(df_physio[df_physio$condition == 0, metric])))
  condition_1 <- na.omit(as.numeric(unlist(df_physio[df_physio$condition == 1, metric])))
  
  # Calculate sample size
  n <- length(condition_0)
  
  # Perform Wilcoxon signed-rank test
  set.seed(123) # set random seed to fix permutation randomness in case of ties
  test_result <- wilcox.test(condition_0, condition_1, paired = TRUE, conf.int = TRUE)
  
  # Calculate Cliff's delta
  cliff_delta <- cliff.delta(condition_0, condition_1)
  
  # Return a list with all the calculated statistics
  list(
    test_result = test_result,
    sample_size = n,
    effect_size = cliff_delta$estimate,
    effect_size_ci = cliff_delta$conf.int
    #conf_int = test_result$conf.int
  )
})

# Name the list elements according to the metrics
names(wilcox_results) <- metrics_columns

# Extract statistics from the test results
results_df <- do.call(rbind, lapply(names(wilcox_results), function(metric) {
  result <- wilcox_results[[metric]]
  data.frame(
    metric = metric,
    sample_size = result$sample_size,
    test_statistic = result$test_result$statistic,
    p_value = result$test_result$p.value,
    effect_size = result$effect_size,
    effect_size_ci_lower = result$effect_size_ci[1],
    effect_size_ci_upper = result$effect_size_ci[2]
    #conf_int_lower = result$conf_int[1],
    #conf_int_upper = result$conf_int[2]
  )
}))

# Adjust p-values using different methods
#results_df$bonferroni <- p.adjust(results_df$p_value, method = "bonferroni")
#results_df$holm <- p.adjust(results_df$p_value, method = "holm")
results_df$fdr_corrected_pvalue <- p.adjust(results_df$p_value, method = "fdr")

# Print the results
print("Results for all participants:")
print(results_df)
              # metric sample_size test_statistic      p_value effect_size effect_size_ci_lower
# V         LF_avg          27            374 1.043081e-07  0.69547325           0.42647565
# V1        HF_avg          27            351 1.879036e-05  0.51440329           0.21617347
# V2    Fratio_avg          27            300 6.450757e-03  0.37722908           0.05840121
# V3 midline_alpha          27            219 4.846056e-01  0.02880658          -0.27892283
# V4  midline_beta          27            328 4.272312e-04  0.22359396          -0.09370524
# effect_size_ci_upper          fdr
# V             0.8514434 5.215406e-07
# V1            0.7248359 4.697591e-05
# V2            0.6262256 8.063447e-03
# V3            0.3311709 4.846056e-01
# V4            0.4996601 7.120520e-04



##### Within group comparison with test statistic, effect size, confidence intervals


# Ensure effsize package is installed and loaded
if (!requireNamespace("effsize", quietly = TRUE)) {
  install.packages("effsize")
}
library(effsize)

# Add group column to df_normalized
df_normalized <- cbind(df_normalized, group = df_physio$group)
df_normalized$group <- as.factor(df_normalized$group)

## calculate mean and SD

means_group_condition <- df_normalized %>%
  pivot_longer(
    cols = where(is.numeric),             # Select only numeric columns
    names_to = "variable", 
    values_to = "value"
  ) %>%
  group_by(variable, condition, group) %>%  # Group by variable, condition, and group
  summarise(
    mean = mean(value, na.rm = TRUE),       # Calculate mean
    std = sd(value, na.rm = TRUE)           # Calculate standard deviation
  ) %>%
  arrange(variable, condition, group)       # Arrange the results for clarity

# View the results
print(means_group_condition)

# Initialize a list to store Wilcoxon signed-rank test results within each group
wilcox_results_within_group <- lapply(unique(df_normalized$group), function(group) {
  lapply(metrics_columns, function(metric) {
    # Filter and remove NAs for the current metric within the current group
    condition_0 <- na.omit(df_normalized[df_normalized$condition == 0 & df_normalized$group == group, metric])
    condition_1 <- na.omit(df_normalized[df_normalized$condition == 1 & df_normalized$group == group, metric])
    
    # Calculate sample size
    n <- length(condition_0)
    
    # Perform Wilcoxon signed-rank test
    test_result <- wilcox.test(condition_0, condition_1, paired = TRUE, conf.int = TRUE)
    
    # Calculate Cliff's delta effect size
    cliff_delta <- cliff.delta(condition_0, condition_1)
    
    # Return a list with all the calculated statistics
    list(
      test_result = test_result,
      sample_size = n,
      effect_size = cliff_delta$estimate,
      effect_size_ci = cliff_delta$conf.int,
      conf_int = test_result$conf.int
    )
  })
})

# Name the list elements according to groups and metrics
names(wilcox_results_within_group) <- unique(df_normalized$group)
for (group in names(wilcox_results_within_group)) {
  names(wilcox_results_within_group[[group]]) <- metrics_columns
}

# Extract statistics from the test results
results_df <- do.call(rbind, lapply(names(wilcox_results_within_group), function(group) {
  do.call(rbind, lapply(names(wilcox_results_within_group[[group]]), function(metric) {
    result <- wilcox_results_within_group[[group]][[metric]]
    data.frame(
      metric = paste(metric, "Group", group),
      sample_size = result$sample_size,
      test_statistic = result$test_result$statistic,
      p_value = result$test_result$p.value,
      effect_size = result$effect_size,
      effect_size_ci_lower = result$effect_size_ci[1],
      effect_size_ci_upper = result$effect_size_ci[2]
    )
  }))
}))

# Adjust p-values using FDR method
results_df$fdr <- p.adjust(results_df$p_value, method = "fdr")

# Print the results
print("Adjusted p-values and statistics within each group:")
print(results_df)

# metric sample_size test_statistic                     p_value     effect_size effect_size_ci_lower
# V          LF_avg Group 0          16            136 3.051758e-05  0.74218750           0.38056886
# V1         HF_avg Group 0          16            130 4.272461e-04  0.57812500           0.17412810
# V2     Fratio_avg Group 0          16            126 1.312256e-03  0.57031250           0.16539575
# V3  midline_alpha Group 0          16             84 4.331970e-01  0.06250000          -0.34228063
# V4   midline_beta Group 0          16            121 4.180908e-03  0.25000000          -0.16667090
# V5         LF_avg Group 1          11             63 4.882812e-03  0.58677686           0.06090281
# V11        HF_avg Group 1          11             57 3.222656e-02  0.45454545          -0.06915376
# V21    Fratio_avg Group 1          11             40 5.771484e-01  0.07438017          -0.42996054
# V31 midline_alpha Group 1          11             34 9.658203e-01 -0.04132231          -0.50664452
# V41  midline_beta Group 1          11             53 8.300781e-02  0.20661157          -0.25377013
# effect_size_ci_upper          fdr
# V              0.9069283 0.0003051758
# V1             0.8155434 0.0021362305
# V2             0.8106913 0.0043741862
# V3             0.4477133 0.5414962769
# V4             0.5909119 0.0097656250
# V5             0.8576777 0.0097656250
# V11            0.7818427 0.0537109375
# V21            0.5433409 0.6412760417
# V31            0.4426433 0.9658203125
# V41            0.5906715 0.1185825893


#########################################################
################# STAIY scores ##########################
#########################################################

stai = read_csv2('../data/STAI-Y1-SCORES.csv') # use ';' as separator to read

# Subset data based on order
data_order_0 <- filter(stai, order == 0)
data_order_1 <- filter(stai, order == 1)

paired_data <- inner_join(data_order_0, data_order_1, by = "Participant", suffix = c(".0", ".1"))

wilcox.test(paired_data$score.0, paired_data$score.1, paired = TRUE)

# Wilcoxon signed rank test with continuity correction
# 
# data:  paired_data$score.0 and paired_data$score.1
# V = 230, p-value = 0.02291
# alternative hypothesis: true location shift is not equal to 0

mean_score_0 <- mean(paired_data$score.0)
sd_score_0 <- sd(paired_data$score.0)

mean_score_1 <- mean(paired_data$score.1)
sd_score_1 <- sd(paired_data$score.1)

mean_score_0
sd_score_0
mean_score_1
sd_score_1


# Set the seed for reproducibility
set.seed(123)

# Subset data based on order
data_order_0 <- filter(stai, order == 0)
data_order_1 <- filter(stai, order == 1)

# Join the data on "Participant"
paired_data <- inner_join(data_order_0, data_order_1, by = "Participant", suffix = c(".0", ".1"))

# Perform the Wilcoxon signed-rank test (paired test)
wilcox_result <- wilcox.test(paired_data$score.0, paired_data$score.1, paired = TRUE)

# Test statistic (W) and p-value from the Wilcoxon test
test_statistic <- wilcox_result$statistic
p_value <- wilcox_result$p.value

# Calculate the effect size (r) from the test statistic
n <- length(paired_data$score.0)  # Sample size
z_value <- qnorm(wilcox_result$p.value / 2)  # Approximate Z-value
effect_size_r <- z_value / sqrt(n)  # Effect size (r)

# To get a confidence interval for the effect size (r), we can use bootstrapping
# We'll use the `boot` package for this. Alternatively, if you'd like a simple CI, we could use the standard error approach.
if (!require(boot)) install.packages("boot")
library(boot)

# Define a function for bootstrapping the effect size
bootstrap_effect_size <- function(data, indices) {
  # Resample the data
  resampled_data <- data[indices, ]
  # Recompute Wilcoxon test on the resampled data
  resampled_wilcox <- wilcox.test(resampled_data$score.0, resampled_data$score.1, paired = TRUE)
  # Return the effect size (r) based on the resampled data
  resampled_z <- qnorm(resampled_wilcox$p.value / 2)
  return(resampled_z / sqrt(n))
}

# Run bootstrap with 1000 resamples
bootstrap_results <- boot(paired_data, bootstrap_effect_size, R = 1000)

# Get confidence intervals for effect size (r)
effect_size_ci <- boot.ci(bootstrap_results, type = "perc")$percent[4:5]  # 2.5% and 97.5% percentiles

# Print the results
cat(sprintf(
  "Wilcoxon Test (p=%.4f, W=%d, r=%.2f, 95%% CI [%.2f, %.2f])\n", 
  p_value, test_statistic, effect_size_r, effect_size_ci[1], effect_size_ci[2]
))

### STAI-Y1 SCORES IN EACH SUBGROUP #######
###################

# Subset the data by group
group_0_data <- filter(stai, group == 0)
group_1_data <- filter(stai, group == 1)

# Define a function to perform the analysis for a given group
perform_analysis <- function(data, group_label) {
  # Subset data by order
  data_order_0 <- filter(data, order == 0)
  data_order_1 <- filter(data, order == 1)
  
  # Join the data on "Participant"
  paired_data <- inner_join(data_order_0, data_order_1, by = "Participant", suffix = c(".0", ".1"))
  
  # Sample size
  n <- nrow(paired_data)
  
  # Perform the Wilcoxon signed-rank test
  wilcox_result <- wilcox.test(paired_data$score.0, paired_data$score.1, paired = TRUE)
  
  # Extract the test statistic and p-value
  test_statistic <- wilcox_result$statistic
  p_value <- wilcox_result$p.value
  
  # Calculate the effect size (r) and ensure correct direction
  # Direction is determined by mean difference between conditions
  mean_diff <- mean(paired_data$score.1 - paired_data$score.0)
  z_value <- qnorm(p_value / 2) * sign(mean_diff)  # Adjust sign based on direction of difference
  effect_size_r <- z_value / sqrt(n)  # Effect size (r)
  
  # Bootstrap effect size confidence intervals
  bootstrap_effect_size <- function(data, indices) {
    resampled_data <- data[indices, ]
    resampled_wilcox <- wilcox.test(resampled_data$score.0, resampled_data$score.1, paired = TRUE)
    resampled_mean_diff <- mean(resampled_data$score.1 - resampled_data$score.0)
    resampled_z <- qnorm(resampled_wilcox$p.value / 2) * sign(resampled_mean_diff)
    return(resampled_z / sqrt(n))
  }
  
  # Run bootstrap with 1000 resamples
  bootstrap_results <- boot(paired_data, bootstrap_effect_size, R = 1000)
  effect_size_ci <- boot.ci(bootstrap_results, type = "perc")$percent[4:5]  # 2.5% and 97.5% percentiles
  
  # Return results as a list
  return(list(
    n = n,
    test_statistic = test_statistic,
    p_value = p_value,
    effect_size_r = effect_size_r,
    effect_size_ci = effect_size_ci
  ))
}

# Perform the analysis for each group
results_group_0 <- perform_analysis(group_0_data, "Group 0")
results_group_1 <- perform_analysis(group_1_data, "Group 1")

# Print results for Group 0
cat(sprintf(
  "Group 0: Wilcoxon Test (n=%d, p=%.4f, W=%d, r=%.2f, 95%% CI [%.2f, %.2f])\n", 
  results_group_0$n, results_group_0$p_value, results_group_0$test_statistic, 
  results_group_0$effect_size_r, results_group_0$effect_size_ci[1], results_group_0$effect_size_ci[2]
))

# Print results for Group 1
cat(sprintf(
  "Group 1: Wilcoxon Test (n=%d, p=%.4f, W=%d, r=%.2f, 95%% CI [%.2f, %.2f])\n", 
  results_group_1$n, results_group_1$p_value, results_group_1$test_statistic, 
  results_group_1$effect_size_r, results_group_1$effect_size_ci[1], results_group_1$effect_size_ci[2]
))

#### compare at baseline for groups 0 and 1

mann_whitney_result <- wilcox.test(
  baseline ~ group, 
  data = stai, 
  exact = FALSE # Use asymptotic method for larger samples
)

print(mann_whitney_result)

#########################################################
################# BETA REGIONS ##########################
#########################################################

df_EEG_beta = read_csv('../data/EEG_beta_regions.csv')
######


df_normalized_EEG_beta <- scale(df_EEG_beta[, c("frontal_beta", "central_beta", "parietal_beta", "occipital_beta", "temporal_beta")])
df_normalized_EEG_beta <- cbind(df_normalized_EEG_beta, condition = df_EEG_beta$condition, ID = df_EEG_beta$ID)

df_normalized_EEG_beta <- as.data.frame(df_normalized_EEG_beta)
df_normalized_EEG_beta$condition <- as.factor(df_normalized_EEG_beta$condition)


##### CHECK NORMALITY DISTRIBUTION
# Assuming df_normalized has been loaded and scaled as described in your code

# Extracting the metrics columns
metrics_EEG_beta_columns <- c("frontal_beta", "central_beta", "parietal_beta", "occipital_beta", "temporal_beta")

# Convert metrics columns to numeric
df_normalized_EEG_beta[, metrics_EEG_beta_columns] <- lapply(df_normalized_EEG_beta[, metrics_EEG_beta_columns], as.numeric)

# Shapiro-Wilk test for normality
shapiro_results_beta <- lapply(df_normalized_EEG_beta[, metrics_EEG_beta_columns], shapiro.test)

# Q-Q plots
par(mfrow = c(2, 4))  # Setting up a 2x4 grid for Q-Q plots
for (i in 1:length(metrics_EEG_beta_columns)) {
  qqnorm(df_normalized_EEG_beta[, metrics_EEG_beta_columns[i]], main = paste("Q-Q Plot for", metrics_EEG_beta_columns[i]))
  qqline(df_normalized_EEG_beta[, metrics_EEG_beta_columns[i]], col = "red")
}

# Print Shapiro-Wilk test results
names(shapiro_results_beta) <- metrics_EEG_beta_columns
print(shapiro_results_beta)

########### APPLY WILCOXON SIGNED RANKS BECAUSE OF NON NORMAL DISTRIBUTION

# Normalize specified columns
df_normalized_EEG_beta <- scale(df_EEG_beta[, c("frontal_beta", "central_beta", "parietal_beta", "occipital_beta", "temporal_beta")])
df_normalized_EEG_beta <- data.frame(df_normalized_EEG_beta, condition = df_EEG_beta$condition, ID = df_EEG_beta$ID)

# Ensure 'condition' is numeric
df_normalized_EEG_beta$condition <- as.numeric(df_normalized_EEG_beta$condition)
df_normalized_EEG_beta$group <- as.factor(df_EEG_beta$group)

# # Invert 0 and 1 in 'condition' and 'group'
# df_normalized_EEG_beta$condition <- ifelse(df_normalized_EEG_beta$condition == 0, 1, 0)
# df_normalized_EEG_beta$group <- ifelse(df_normalized_EEG_beta$group == 0, 1, 0)

# Extracting the metrics columns
metrics_EEG_beta_columns <- c("frontal_beta", "central_beta", "parietal_beta", "occipital_beta", "temporal_beta")

# Melt the data for easier processing
melted_data_EEG_beta <- reshape2::melt(df_normalized_EEG_beta, id.vars = c("ID", "condition", "group"))
#write.csv(melted_data_EEG_beta, "/Users/idil.sezer/Desktop/PhD/DATA_Paul/melted_data_EEG_beta_new.csv", row.names= FALSE)

# Initialize lists to store results
wilcox_results_within_group <- list()
adjusted_p_values_within_group <- data.frame(metric = character(), p_value = numeric(), test_statistic = numeric(), cliffs_delta = numeric(), ci_lower = numeric(), ci_upper = numeric(), stringsAsFactors = FALSE)

# Perform Wilcoxon signed-rank tests within each group across metrics
for (group in unique(df_normalized_EEG_beta$group)) {
  for (metric in metrics_EEG_beta_columns) {
    # Get data for the specified group and metric
    data_condition_0 <- df_normalized_EEG_beta[df_normalized_EEG_beta$condition == 0 & df_normalized_EEG_beta$group == group, metric]
    data_condition_1 <- df_normalized_EEG_beta[df_normalized_EEG_beta$condition == 1 & df_normalized_EEG_beta$group == group, metric]
    
    # Perform the Wilcoxon signed-rank test
    set.seed(123)
    result_within_group <- wilcox.test(data_condition_0, data_condition_1, paired = TRUE)
    
    # Calculate Cliff's delta
    cliffs_delta_result <- cliff.delta(data_condition_0, data_condition_1, paired = TRUE)
    
    # Store the test result in the list
    wilcox_results_within_group[[paste(metric, "Group", group)]] <- result_within_group
    
    # Store the results in the data frame
    adjusted_p_values_within_group <- rbind(
      adjusted_p_values_within_group, 
      data.frame(
        metric = paste(metric, "Group", group), 
        p_value = result_within_group$p.value,
        test_statistic = result_within_group$statistic,
        cliffs_delta = cliffs_delta_result$estimate,
        ci_lower = cliffs_delta_result$conf.int[1],
        ci_upper = cliffs_delta_result$conf.int[2]
      )
    )
  }
}

# Adjust p-values for multiple comparisons within each group using FDR
adjusted_p_values_within_group$adjusted_p_value_fdr <- p.adjust(adjusted_p_values_within_group$p_value, method = "fdr")

# Print the adjusted p-values with test statistics, Cliff's delta, and confidence intervals
print("Adjusted p-values with test statistics, Cliff's delta, and confidence intervals:")
print(adjusted_p_values_within_group)

#           metric            p_value     test_statistic cliffs_delta   ci_lower  ci_upper
# V    frontal_beta Group 0 5.065918e-02            106    0.1875000 -0.2272603 0.5446818
# V1   central_beta Group 0 9.155273e-05            134    0.2812500 -0.1405490 0.6166388
# V2  parietal_beta Group 0 4.272461e-04            130    0.1718750 -0.2419293 0.5327796
# V3 occipital_beta Group 0 3.356934e-03            122    0.1796875 -0.2398453 0.5426679
# V4  temporal_beta Group 0 4.272461e-04            130    0.3046875 -0.1258217 0.6386257
# V5   frontal_beta Group 1 2.441406e-02             58    0.2561983 -0.2574606 0.6569664
# V6   central_beta Group 1 6.738281e-02             54    0.2066116 -0.2564721 0.5925500
# V7  parietal_beta Group 1 1.474609e-01             50    0.1074380 -0.3832249 0.5508096
# V8 occipital_beta Group 1 6.738281e-02             54    0.1735537 -0.3029549 0.5806394
# V9  temporal_beta Group 1 2.929687e-03             64    0.2066116 -0.3050751 0.6257282

# adjusted_p_value_fdr
# V          0.0723702567 
# V1         0.0009155273
# V2         0.0014241536
# V3         0.0067138672
# V4         0.0014241536
# V5         0.0406901042
# V6         0.0748697917
# V7         0.1474609375
# V8         0.0748697917
# V9         0.0067138672



########## high alpha statistics

df_high_alpha <- read.csv('../data/EEG_high_alpha_regions_stats.csv')

# df_high_alpha <- df_high_alpha %>%
#   filter(!ID %in% c(18, 19, 21, 22, 25, 27, 29))

# Ensure 'condition' is numeric
df_high_alpha$condition <- as.numeric(df_high_alpha$condition)
df_high_alpha$group <- as.factor(df_high_alpha$group)

metrics_high_alpha_columns <- unique(df_high_alpha$variable)

# Function to remove outlier IDs based on IQR and print aberrant rows
remove_outlier_ids_iqr <- function(df, value_col) {
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
  
  # Get the IDs corresponding to outliers
  outlier_ids <- df$ID[outlier_flags]
  
  # Print aberrant data (outliers) based on IDs
  if (length(outlier_ids) > 0) {
    outliers <- df[df$ID %in% outlier_ids, ]
    print("Aberrant data detected (outliers based on ID):")
    print(outliers)
  }
  
  # Remove all rows corresponding to outlier IDs
  df_cleaned <- df[!df$ID %in% outlier_ids, ]
  
  return(df_cleaned)
}

# Remove outliers based on IDs (before entering the loop)
df_high_alpha_cleaned <- df_high_alpha
for (metric in metrics_high_alpha_columns) {
  df_high_alpha_cleaned <- remove_outlier_ids_iqr(df_high_alpha_cleaned, "value")
}

df_high_alpha <- df_high_alpha_cleaned

# Remove outliers based on IDs (before scaling)
df_high_alpha_cleaned <- df_high_alpha
for (metric in metrics_high_alpha_columns) {
  df_high_alpha_cleaned <- remove_outlier_ids_iqr(df_high_alpha_cleaned, "value")
}

# Scale each variable's values (across both condition and group)
df_high_alpha_cleaned <- df_high_alpha_cleaned %>%
  group_by(variable) %>%
  mutate(value_scaled = scale(value)) %>%
  ungroup()


# Initialize a data frame to store results
high_alpha_results <- data.frame(
  metric = character(),
  p_value = numeric(),
  test_statistic = numeric(),
  cliffs_delta = numeric(),
  ci_lower = numeric(),
  ci_upper = numeric(),
  group = numeric(),
  adjusted_p_value_fdr = numeric(),
  stringsAsFactors = FALSE
)

# Perform paired Wilcoxon signed-rank test and calculate Cliff's delta for each metric and group
for (group in unique(df_high_alpha_cleaned$group)) {
  
  for (metric in metrics_high_alpha_columns) {
    
    # Subset data for the specific group and metric
    data_condition_0 <- df_high_alpha_cleaned[df_high_alpha_cleaned$condition == 0 & df_high_alpha_cleaned$group == group & df_high_alpha_cleaned$variable == metric, ]
    data_condition_1 <- df_high_alpha_cleaned[df_high_alpha_cleaned$condition == 1 & df_high_alpha_cleaned$group == group & df_high_alpha_cleaned$variable == metric, ]
    
    # Ensure the data is paired by ID
    data_condition_0 <- data_condition_0[order(data_condition_0$ID), ]
    data_condition_1 <- data_condition_1[order(data_condition_1$ID), ]
    
    # Perform Wilcoxon signed-rank test
    set.seed(123)
    wilcox_result <- wilcox.test(data_condition_0$value_scaled, data_condition_1$value_scaled, paired = TRUE)
    
    # Calculate Cliff's delta
    cliffs_delta_result <- cliff.delta(data_condition_0$value_scaled, data_condition_1$value_scaled, paired = TRUE)
    
    # Store results in the data frame
    high_alpha_results <- rbind(
      high_alpha_results, 
      data.frame(
        metric = metric, 
        p_value = wilcox_result$p.value,
        test_statistic = wilcox_result$statistic,
        cliffs_delta = cliffs_delta_result$estimate,
        ci_lower = cliffs_delta_result$conf.int[1],
        ci_upper = cliffs_delta_result$conf.int[2],
        group = group,
        adjusted_p_value_fdr = NA  # Placeholder for FDR adjusted p-values
      )
    )
  }
}

# Adjust p-values using FDR method within each group separately
high_alpha_results <- high_alpha_results %>%
  group_by(group) %>%
  mutate(adjusted_p_value_fdr = p.adjust(p_value, method = "fdr")) %>%
  ungroup()

# Print the results
print("Wilcoxon test results with Cliff's delta and FDR adjusted p-values (per group):")
print(high_alpha_results)

# metric               p_value test_statistic cliffs_delta ci_lower ci_upper group adjusted_p_value_fdr
# <chr>                  <dbl>          <dbl>        <dbl>    <dbl>    <dbl> <chr>                <dbl>
#   1 frontal_high_alpha    0.495              82       0.0781   -0.329    0.461 0                  0.495 
# 2 temporal_high_alpha   0.0214            112       0.109    -0.297    0.482 0                   0.0428
# 3 central_high_alpha    0.0290            110       0.156    -0.258    0.522 0                   0.0435
# 4 parietal_high_alpha   0.0214            112       0.125    -0.284    0.495 0                   0.0428
# 5 occipital_high_alpha  0.0110            116       0.156    -0.256    0.520 0                   0.0428
# 6 midline_high_alpha    0.252              91       0.0781   -0.325    0.457 0                   0.303 
# 7 frontal_high_alpha    0.322              17      -0.24     -0.667    0.305 1                   1     
# 8 temporal_high_alpha   0.922              29       0.0400   -0.459    0.519 1                   1     
# 9 central_high_alpha    1                  27       0.04     -0.467    0.527 1                   1     
# 10 parietal_high_alpha   0.922              26       0.06     -0.417    0.511 1                   1     
# 11 occipital_high_alpha  0.625              33       0.0400   -0.469    0.529 1                   1     
# 12 midline_high_alpha    0.695              23       0.02     -0.441    0.473 1                   1  




### GLMM

# Read the dataset
corrs <- read_excel("../data/biomarkers_GLMM.xlsx")

# Filter for condition = 0 and remove specific variables
df_filtered <- corrs %>%
  dplyr::filter(condition == 0) %>%
  dplyr::filter(variable != 'tonic_mean', variable != 'phasic_mean')

# Pivot data to wide format
df_wide <- df_filtered %>%
  tidyr::pivot_wider(names_from = variable, values_from = value)

# Fit a generalized linear mixed model (GLMM)
model <- lm(score ~ LF_avg + HF_avg + Fratio_avg, data = df_wide)

# Print model summary
model_summary <- summary(model)
print(model_summary)

# Extract test statistics
model_results <- broom::tidy(model)

# Calculate Cohen's d for each predictor
# Note: Cohen's d here is adapted to continuous predictors by standardizing variables.
# Standardize predictors and response
df_standardized <- df_wide %>%
  dplyr::mutate(across(c(LF_avg, HF_avg, Fratio_avg, score), scale))

# Fit the model to standardized data
model_standardized <- lm(score ~ LF_avg + HF_avg + Fratio_avg, data = df_standardized)

# Extract Cohen's d as standardized coefficients
standardized_results <- broom::tidy(model_standardized) %>%
  dplyr::mutate(
    cohen_d = estimate, # Standardized coefficients are equivalent to Cohen's d
    conf.low = confint(model_standardized)[, 1], # Confidence interval lower bound
    conf.high = confint(model_standardized)[, 2] # Confidence interval upper bound
  )

# Print the HRV results
print(standardized_results)
# Estimate Std. Error t value Pr(>|t|)   
# (Intercept)  0.17142    0.08457   2.027  0.05441 . 
# LF_avg      -0.15147    0.11081  -1.367  0.18488   
# HF_avg      -0.03292    0.08785  -0.375  0.71134   
# Fratio_avg   0.27881    0.09630   2.895  0.00816 **


library(ggplot2)
library(ggpubr)  # For stat_regline_equation
library(wesanderson)  # For the color palette

# Fit the model
model <- lm(score ~ Fratio_avg, data = df_wide)

# Extract coefficients
summary(model)

# Visualize the relationship between Fratio_avg and score
wes_colors <- wes_palette("GrandBudapest1", n = 3)

ggplot(df_wide, aes(x = Fratio_avg, y = score)) +
  geom_point(color = wes_colors[1]) +  # Add scatter plot points
  geom_smooth(method = "lm", color = wes_colors[2], fill = wes_colors[2], alpha = 0.2) +  # Add regression line
  labs(
    title = "Relationship between STAI-Y1 score difference and LF/HF",
    x = "LF/HF",
    y = "∆ STAI-Y1 score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, family = "Arial"),  # Center title
    axis.title.x = element_text(family = "Arial"),  # X-axis font
    axis.title.y = element_text(family = "Arial")   # Y-axis font
  )

