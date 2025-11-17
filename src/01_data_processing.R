## 1. Read in data (change file paths if needed)
df1 <- read.csv("../data/raw/County_annual_PM25_CMR.csv")
df2 <- read.csv("../data/raw/County_RAW_variables.csv")

## 2. From df1: keep only year 2000 and needed columns (FIPS, PM2.5, CMR)
df1_2000 <- subset(df1, Year == 2000, select = c("FIPS", "PM2.5", "CMR"))

## 3. From df2: keep FIPS + all columns whose names contain "2000"
cols_2000 <- grepl("2000", names(df2))
df2_2000 <- df2[, cols_2000 | names(df2) == "FIPS"]

## 4. STANDARDIZATION: subtract mean and divide by SD for all 2000 variables

# Identify which columns in df2_2000 are the 2000-variables
cols_2000_df2 <- grepl("2000", names(df2_2000))

# Step 4.1: Convert these columns to numeric
df2_2000[, cols_2000_df2] <- lapply(df2_2000[, cols_2000_df2, drop = FALSE], function(x) {
  as.numeric(as.character(x))
})

# Step 4.2: Z-score standardization: (x - mean) / sd
df2_2000[, cols_2000_df2] <- lapply(df2_2000[, cols_2000_df2, drop = FALSE], function(x) {

  mu <- mean(x, na.rm = TRUE)
  sd_x <- sd(x, na.rm = TRUE)

  # If sd = 0 or not finite (all same), set entire column to 0
  if (!is.finite(sd_x) || sd_x == 0) {
    rep(0, length(x))
  } else {
    (x - mu) / sd_x
  }
})

## 5. Remove the suffix "_2000" from column names
names(df2_2000) <- sub("_2000$", "", names(df2_2000))

## 6. Merge df2_2000 with df1_2000 by FIPS
merged_2000 <- merge(df1_2000, df2_2000, by = "FIPS", all = FALSE)

## 7. Sort the merged data by FIPS
merged_2000 <- merged_2000[order(merged_2000$FIPS), ]

## 8. Write the final merged table to a new CSV file
write.csv(merged_2000, "../data/processed/merged_2000_only.csv", row.names = FALSE)

