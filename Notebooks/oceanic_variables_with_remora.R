############################################################
# REMORA + IMOS SEA LION DATA WORKFLOW (CLEAN VERSION)
############################################################

# ---------------------------
# 1. Install packages (run once)
# ---------------------------

install.packages(c(
  "tidyverse",
  "sf",
  "mapview",
  "terra",
  "scales",
  "rnaturalearth",
  "remotes"
))

# Install remora from GitHub (skip vignettes to avoid build errors)
library(remotes)

remotes::install_github(
  "IMOS-AnimalTracking/remora",
  build_vignettes = FALSE
)

# ---------------------------
# 2. Load libraries
# ---------------------------

library(tidyverse)
library(sf)
library(mapview)
library(terra)
library(scales)
library(rnaturalearth)
library(remora)

# ---------------------------
# 3. Check remora variables (optional)
# ---------------------------

imos_variables()

# ---------------------------
# 4. Set working directory and load data
# ---------------------------

setwd("C:/Users/nribeiro/OneDrive - University of Tasmania/IMOS Shared Docs/CAPSTAN 2025/Datasets/")

dat <- read.csv("sea_lion_SA.csv")

# Convert timestamp to proper datetime format
dat$timestamp <- as.POSIXct(dat$timestamp, tz = "UTC")

# ---------------------------
# 5. Clean column names for consistency
# ---------------------------

dat <- dat %>%
  rename(
    t = timestamp,   # time column
    id = device_id   # animal identifier
  )

# ---------------------------
# 6. Extract surface layer (SST proxy)
#    pressure = depth
#    keep shallowest measurement per location
# ---------------------------

dat0 <- dat %>%
  group_by(lon, lat, id, t) %>%
  slice_min(pressure, n = 1, with_ties = FALSE) %>%
  ungroup()

# ---------------------------
# 7. Interactive map (Mapview)
# ---------------------------

dat0 %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
  mapview(zcol = "temp_vals")

# ---------------------------
# 8. Load land for static plotting
# ---------------------------

land <- ne_countries(scale = "medium", returnclass = "sf")

# ---------------------------
# 9. Static track plot (ggplot2)
# ---------------------------

ggplot() +
  
  # Add land background
  geom_sf(data = land, fill = "grey80", color = NA) +
  
  # Animal tracks (lines)
  geom_path(
    data = dat0,
    aes(x = lon, y = lat, color = id, group = id),
    linewidth = 0.4,
    alpha = 0.7
  ) +
  
  # Locations (points)
  geom_point(
    data = dat0,
    aes(x = lon, y = lat, color = id),
    size = 0.5
  ) +
  
  # Map limits
  coord_sf(xlim = c(120, 145), ylim = c(-19, -8)) +
  
  # Colour by individual
  scale_color_viridis_d() +
  
  # Labels and theme
  labs(
    x = NULL,
    y = NULL,
    color = "Animal ID"
  ) +
  theme_bw()
