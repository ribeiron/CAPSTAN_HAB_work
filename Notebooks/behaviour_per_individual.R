############################################################
# BEHAVIOUR ANALYSIS FROM SSF STEP DATA
############################################################

## ---------------------------
## 1. PACKAGES
## ---------------------------

library(dplyr)
library(ggplot2)

## ---------------------------
## 2. LOAD DATA
## ---------------------------

# load SSF-ready step dataset
setwd("C:/Users/nribeiro/OneDrive - University of Tasmania/IMOS Shared Docs/CAPSTAN 2025/Datasets/")

dat <- readRDS("sea_lion_steps_all.rds")

# inspect structure (IMPORTANT CHECK)
names(dat)

## ---------------------------
## 3. BASIC CLEANING
## ---------------------------

# ensure no invalid steps
dat <- dat %>%
  filter(!is.na(sl_)) %>%
  filter(sl_ > 0)

## ---------------------------
## 3. BEHAVIOUR CLASSIFICATION
## ---------------------------

median_sl <- median(dat$sl_, na.rm = TRUE)

dat <- dat %>%
  mutate(
    step_km = sl_ / 1000,
    
    behaviour = case_when(
      step_km > median_sl/1000 & cos(ta_) > 0.5 ~ "transit",
      step_km <= median_sl/1000 & cos(ta_) <= 0.5 ~ "foraging",
      TRUE ~ "mixed"
    )
  )

table(dat$behaviour)


10## ---------------------------
## 4. BEHAVIOUR COMPOSITION PER INDIVIDUAL
## ---------------------------

# USED STEPS ONLY (BIOLOGICALLY MEANINGFUL)
behav_used <- ssf_data %>%
  filter(case_ == 1) %>%
  group_by(id, behaviour) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(id) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  mutate(dataset = "Used steps")

# USED + AVAILABLE (SSF STRUCTURE ONLY, NOT BIOLOGICAL)
behav_all <- dat %>%
  group_by(id, behaviour) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(id) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  mutate(dataset = "Used + Available")

# combine
behav_compare <- bind_rows(behav_used, behav_all)

## ---------------------------
## 5. PLOT: COMPOSITION COMPARISON
## ---------------------------

ggplot(behav_compare,
       aes(x = id, y = prop, fill = behaviour)) +
  geom_col() +
  facet_wrap(~dataset, ncol = 1) +
  theme_minimal() +
  labs(
    title = "Behaviour composition per individual",
    subtitle = "Used steps (biological signal) vs full SSF sample (model structure)",
    x = "Individual",
    y = "Proportion"
  ) +
  theme(axis.text.x = element_blank())

