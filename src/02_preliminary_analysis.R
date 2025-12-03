packages <- c("tidyverse", "zoo", "lmtest", "car")

new <- packages[!(packages %in% installed.packages()[,"Package"])]
if (length(new)) install.packages(new)

lapply(packages, library, character.only = TRUE)

library(tidyverse)
library(zoo)
library(lmtest)
library(car)

# ==============================
# 1. Read and Prepare the Data
# ==============================

data <- read_csv("../data/processed/merged_2000_only.csv") %>%
  drop_na(PM2.5, CMR)

# Sort for moving average
data_sorted <- data %>% arrange(PM2.5)

data_sorted$CMR_MA100 <- rollmean(
  data_sorted$CMR,
  k    = 100,
  fill = NA,
  align = "center"
)

# Clean data for plotting MA line
data_clean <- data_sorted %>% filter(!is.na(CMR_MA100))


# ==============================
# 2. Fit the Hinge-Squared Model
# ==============================

data2 <- data %>%
  mutate(
    x    = PM2.5,
    h6_2 = pmax(x - 6, 0)^2,
    h7_5_2 = pmax(x - 7.5, 0)^2
  )

model_full <- lm(
  CMR ~ x + h6_2 + h7_5_2 +
    civil_unemploy +
    median_HH_inc +
    femaleHH_ns_pct +
    vacant_HHunit +
    owner_occ_pct +
    eduattain_HS +
    pctfam_pover +
    population,
  data = data2
)

# Prediction grid
newdat <- data.frame(
  x = seq(min(data2$x), max(data2$x), length.out = 300)
) %>%
  mutate(
    h6_2 = pmax(x - 6, 0)^2,
    h7_5_2 = pmax(x - 7.5, 0)^2
  )

# Means of covariates
cov_means <- data2 %>% summarise(
  civil_unemploy  = mean(civil_unemploy,  na.rm = TRUE),
  median_HH_inc   = mean(median_HH_inc,   na.rm = TRUE),
  femaleHH_ns_pct = mean(femaleHH_ns_pct, na.rm = TRUE),
  vacant_HHunit   = mean(vacant_HHunit,   na.rm = TRUE),
  owner_occ_pct   = mean(owner_occ_pct,   na.rm = TRUE),
  eduattain_HS    = mean(eduattain_HS,    na.rm = TRUE),
  pctfam_pover    = mean(pctfam_pover,    na.rm = TRUE),
  population      = mean(population,      na.rm = TRUE)
)

# Add means to prediction data
newdat <- newdat %>%
  mutate(
    civil_unemploy  = cov_means$civil_unemploy,
    median_HH_inc   = cov_means$median_HH_inc,
    femaleHH_ns_pct = cov_means$femaleHH_ns_pct,
    vacant_HHunit   = cov_means$vacant_HHunit,
    owner_occ_pct   = cov_means$owner_occ_pct,
    eduattain_HS    = cov_means$eduattain_HS,
    pctfam_pover    = cov_means$pctfam_pover,
    population      = cov_means$population
  )

# Predictions
newdat$CMR_pred <- predict(model_full, newdata = newdat)


# ==============================
# 3A. Figure 1: Scatter + MA Line
# ==============================

p1 <- ggplot() +
  geom_point(
    data = data,
    aes(x = PM2.5, y = CMR),
    alpha = 0.4,
    color = "black"
  ) +
  geom_line(
    data  = data_clean,
    aes(x = PM2.5, y = CMR_MA100),
    color = "blue",
    size  = 1.2,
    alpha = 0.8
  ) +
  labs(
    title = "CMR vs PM2.5 (Scatter with Moving Average)",
    x = "PM2.5 Concentration",
    y = "County Mortality Rate (CMR)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = "../results/figures/01_CMR_PM25_Scatter.pdf",
  plot     = p1,
  width    = 8,
  height   = 6
)


# ==============================
# 3B. Figure 2: Scatter + MA + Model Fit
# ==============================

p2 <- ggplot() +
  geom_point(
    data = data,
    aes(x = PM2.5, y = CMR),
    alpha = 0.4,
    color = "black"
  ) +
  geom_line(
    data  = data_clean,
    aes(x = PM2.5, y = CMR_MA100),
    color = "blue",
    size  = 1.2,
    alpha = 0.8
  ) +
  geom_line(
    data  = newdat,
    aes(x = x, y = CMR_pred),
    color = "red",
    size  = 1.3
  ) +
  labs(
    title = "CMR vs PM2.5 (Hinge-Squared Model Fit)",
    x = "PM2.5 Concentration",
    y = "County Mortality Rate (CMR)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = "../results/figures/03_CMR_PM25_Model_Fit.pdf",
  plot     = p2,
  width    = 8,
  height   = 6
)
