## this is to build an SSF for the sea lion data 

install.packages("patchwork")

library(dplyr)
library(amt)
library(ggplot2)
library(survival)
library(patchwork)

## set the directory to where the data is
setwd("C:/Users/nribeiro/OneDrive - University of Tasmania/IMOS Shared Docs/CAPSTAN 2025/Datasets/")

## load the data 
steps_all <- readRDS("sea_lion_steps_all.rds")

## remove zero steps again, as it looks like that it didn't work previously
steps_all <- steps_all %>% filter(sl_ > 0)

#------------------------------------------------------------
# 1. BUILD SSF (USED + AVAILABLE STEPS)
#------------------------------------------------------------

## generate available steps: for each used step it is generating 20 alternative steps
ssf_data <- steps_all %>%
  random_steps(
    n = 20, 
    method = "circular"
  )

## inspect the table: 1:20 ratio between true and false
table(ssf_data$case_)

## check grouping (each decision event should have its own id)
ssf_data %>%
  count(step_id_) %>%
  summary()

#------------------------------------------------------------
# 2. MOVEMENT COVARIATES
#------------------------------------------------------------

ssf_data <- ssf_data %>%
  mutate(
    log_sl_ = log(sl_),
    cos_ta_ = cos(ta_)
  )

#------------------------------------------------------------
# 3. BASIC SSF MODEL (movement only)
#------------------------------------------------------------

model1 <- clogit(
  case_ ~ log_sl_ + cos_ta_ + strata(step_id_),
  data = ssf_data
)

summary(model1)

#------------------------------------------------------------
# 4. CHECK USED VS AVAILABLE DISTRIBUTIONS
#------------------------------------------------------------

ssf_data %>%
  mutate(type = ifelse(case_ == 1, "Used", "Available")) %>%
  ggplot(aes(x = log_sl_, fill = type)) +
  geom_histogram(bins = 60, alpha = 0.5, position = "identity") +
  theme_minimal() +
  labs(title = "Step Length Distribution: Used vs Available")

ssf_data %>%
  mutate(type = ifelse(case_ == 1, "Used", "Available")) %>%
  ggplot(aes(x = cos_ta_, fill = type)) +
  geom_histogram(bins = 60, alpha = 0.5, position = "identity") +
  theme_minimal() +
  labs(title = "Turning Angle Distribution: Used vs Available")

#------------------------------------------------------------
# 5. MODEL INTERPRETATION
#------------------------------------------------------------

used_data <- model.frame(model1)
used_data$case_ <- model1$y[,2]
used_data$pred <- predict(model1, type = "lp")

ggplot(used_data, aes(x = log_sl_, y = pred, color = factor(case_))) +
  geom_point(alpha = 0.2) +
  geom_smooth(se = FALSE) +
  theme_minimal() +
  labs(
    title = "SSF Selection vs Step Length",
    x = "log(step length)",
    y = "Linear predictor"
  )

#------------------------------------------------------------
# 6. EFFECT SIZE SUMMARY
#------------------------------------------------------------

effects <- data.frame(
  variable = c("Step length", "Turning angle"),
  odds_ratio = exp(coef(model1))
)

ggplot(effects, aes(x = variable, y = odds_ratio)) +
  geom_col() +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "SSF Movement Effects",
    y = "Odds ratio",
    x = ""
  )

#------------------------------------------------------------
# 7. EXPLORATORY MOVEMENT PATTERNS
#------------------------------------------------------------

ssf_data <- ssf_data %>%
  mutate(
    straightness = cos_ta_,
    log_speed_proxy = log_sl_
  )

#------------------------------------------------------------
# 8. SIMPLE BEHAVIOUR CLASSIFICATION (EXPLORATORY ONLY)
#------------------------------------------------------------

ssf_data <- ssf_data %>%
  mutate(
    step_km = sl_ / 1000,
    behaviour = case_when(
      step_km > median(step_km, na.rm = TRUE) & cos_ta_ > 0.5 ~ "transit",
      step_km <= median(step_km, na.rm = TRUE) & cos_ta_ <= 0.5 ~ "foraging",
      TRUE ~ "mixed"
    )
  )

table(ssf_data$behaviour)

#------------------------------------------------------------
# 9. BEHAVIOUR VISUAL CHECKS
#------------------------------------------------------------

ssf_data %>%
  ggplot(aes(x = log_sl_, fill = behaviour)) +
  geom_histogram(bins = 60, alpha = 0.5, position = "identity") +
  theme_minimal() +
  labs(title = "Step Length by Behaviour")

ssf_data %>%
  ggplot(aes(x = cos_ta_, fill = behaviour)) +
  geom_histogram(bins = 60, alpha = 0.5, position = "identity") +
  theme_minimal() +
  labs(title = "Turning Behaviour by State")

#------------------------------------------------------------
# 10. BEHAVIOUR COMPOSITION (USED ONLY)
#------------------------------------------------------------

used_steps <- ssf_data %>%
  filter(case_ == 1)

behav_counts <- used_steps %>%
  count(behaviour) %>%
  mutate(prop = n / sum(n))

#------------------------------------------------------------
# 11. JOURNAL THEME
#------------------------------------------------------------

theme_journal <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.title = element_text(face = "bold"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    )
}

theme_set(theme_journal())

#------------------------------------------------------------
# 12. FINAL FIGURE PANELS
#------------------------------------------------------------

# A. SSF effects
effects_df <- data.frame(
  variable = c("Step length", "Turning angle"),
  odds_ratio = exp(coef(model1))
)

p1 <- ggplot(effects_df, aes(x = variable, y = odds_ratio, fill = variable)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  labs(
    title = "A. Movement Selection (SSF)",
    subtitle = "Population-level selection across movement variables",
    x = "",
    y = "Odds ratio (exp[β])"
  ) +
  scale_fill_manual(values = c(
    "Step length" = "#4C9F70",
    "Turning angle" = "#C76D3A"
  )) +
  guides(fill = "none")

# B. behaviour composition
p2 <- ssf_data %>%
  filter(case_ == 1) %>%
  count(behaviour) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = behaviour, y = prop, fill = behaviour)) +
  geom_col(width = 0.6) +
  theme_minimal(base_size = 14) +
  labs(
    title = "B. Behaviour Composition",
    subtitle = "Proportion of observed (used) steps only",
    x = "",
    y = "Proportion",
    fill = "Behaviour"
  ) +
  scale_fill_manual(values = c(
    "foraging" = "#1b9e77",
    "mixed" = "#7570b3",
    "transit" = "#d95f02"
  )) +
  guides(fill = "none")

# C. step length distribution
# FIGURE PANEL C (STEP LENGTH DISTRIBUTION - DENSITY)
p3 <- ggplot(ssf_data %>% filter(case_ == 1),
             aes(x = log_sl_, color = behaviour)) +
  geom_density(linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "C. Movement Scale",
    subtitle = "Step length distribution by behavioural state (used steps only)",
    x = "log(step length)",
    y = "Density",
    color = "Behaviour"
  ) +
  scale_color_manual(values = c(
    "foraging" = "#1b9e77",
    "mixed" = "#7570b3",
    "transit" = "#d95f02"
  ))

# D. turning angle distribution
# FIGURE PANEL D (TURNING ANGLE DISTRIBUTION - DENSITY)
p4 <- ggplot(ssf_data %>% filter(case_ == 1),
             aes(x = cos_ta_, color = behaviour)) +
  geom_density(linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(
    title = "D. Directional Persistence",
    subtitle = "Turning behaviour by behavioural state (used steps only)",
    x = expression(cos(theta)),
    y = "Density",
    color = "Behaviour"
  ) +
  scale_color_manual(values = c(
    "foraging" = "#1b9e77",
    "mixed" = "#7570b3",
    "transit" = "#d95f02"
  ))

#------------------------------------------------------------
# 13. COMBINE FIGURE
#------------------------------------------------------------

final_fig <- (p1 | p2) / (p3 | p4)

final_fig

