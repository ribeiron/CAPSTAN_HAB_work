## I have installed the packages below
#install.packages("amt")
#install.packages("dplyr")
#install.packages("ggplot2")

## calling libraries I want to use

library(dplyr)
library(amt)
library(purrr)
library(ggplot2)


#setting working directory

setwd("C:/Users/nribeiro/OneDrive - University of Tasmania/IMOS Shared Docs/CAPSTAN 2025/Datasets/")

# load the data

dat <- read.csv("sea_lion_ssf_ready.csv")
dat$timestamp <- as.POSIXct(dat$timestamp, tz = "UTC")

# timestamp can't be a charcater, so have to run this first. also have to rename the column to prevent future problems. 

dat <- dat %>%
  
  rename(
    
    t = timestamp,
    
    id = animal_id
    
  )


# remove duplicates from dataset

dat <- dat %>%
  
  distinct(id, t, x, y, .keep_all = TRUE)

# check how many rows remain. I had 363636, which is a lot for 46 seals

nrow(dat)

# sort the data

dat <- dat %>%
  
  arrange(id, t)

# Basic movement QC. Checking how mny animals and how many GPS pings
dat %>%
  count(id) %>%
  summarise(
    n_animals = n(),
    min_fixes = min(n),
    median_fixes = median(n),
    max_fixes = max(n)
  )

# check tracks make sense
ggplot(dat, aes(x = x, y = y, colour = id)) +
  geom_path(alpha = 0.5) +
  theme_minimal() +
  guides(colour = "none")

# create steps per animal 
steps_all <- dat %>%
  group_split(id) %>%
  map_dfr(function(df) {
    
    trk <- make_track(
      df,
      x, y, t,
      crs = 3577
    )
    
    stp <- steps(trk)
    
    stp$id <- unique(df$id)
    
    stp
  })

# check steps were created properly. Should have id in it 
names(steps_all)

# examine steps lentghs
summary(steps_all$sl_)

# and hours
summary(as.numeric(steps_all$dt_) / 3600)

#remove problematic steps
# zero-time steps
steps_all <- steps_all %>%
  filter(dt_ > 0)

# very large time gaps
steps_all <- steps_all %>%
  filter(dt_ <= 7200)

# check distributions again
summary(steps_all$sl_)
summary(steps_all$dt_)


# diagnose plots
# step length distribution
ggplot(steps_all, aes(x = sl_)) +
  geom_histogram(bins = 50) +
  theme_minimal()


# diagnosis plot step lentgh vs time interval 
ggplot(steps_all, aes(x = dt_ / 3600, y = sl_ / 1000)) +
  geom_point(alpha = 0.2) +
  theme_minimal() +
  labs(
    x = "Time interval (hours)",
    y = "Step length (km)"
  )

# saving SSF-ready variable
saveRDS(steps_all, "sea_lion_steps_all.rds")
