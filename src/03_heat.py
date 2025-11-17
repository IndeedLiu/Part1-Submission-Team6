import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the new dataset ---
file_path = "../data/processed/merged_2000_only.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")
    print("Please ensure the CSV file is in the correct directory.")
    exit()

# --- 2. Inspect and Prepare Data ---
print("File loaded successfully. Original columns:")
print(df.columns.tolist())

# MODIFICATION: Drop 'FIPS', 'PM2.5', and 'CMR' as requested
# We also drop 'Unnamed: 0' if it exists, for robustness
columns_to_drop = ['FIPS', 'PM2.5', 'CMR', 'Unnamed: 0']
df_numeric = df.drop(columns=columns_to_drop, axis=1, errors='ignore')

print("\nColumns used for heatmap:")
print(df_numeric.columns.tolist())

# Calculate the correlation matrix
corr_matrix = df_numeric.corr()

# --- 3. Draw the Heatmap ---
print("\nGenerating heatmap...")

# Set the figure size (width, height in inches)
plt.figure(figsize=(10, 8))  # Adjusted size for fewer variables

# Use seaborn's heatmap function
# cmap='coolwarm': Red for positive, Blue for negative correlation
# annot=True: Show the correlation values on the map
# fmt='.2f': Format values to TWO decimal places, as requested
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',  # <-- Format to two decimal places
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot_kws={"size": 10})  # Adjust annotation font size

# Add title and labels
plt.title("Covariates Correlation Heatmap (Excl. FIPS, PM2.5, CMR)", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)  # Keep y-axis labels horizontal
plt.tight_layout()  # Adjust layout to prevent labels from being cut off

# Save the plot
output_filename = "../results/figures/02_correlation_heatmap_no_FIPS_PM25_CMR.png"
plt.savefig(output_filename, dpi=300)

print(f"Heatmap saved as '{output_filename}'")

# Use plt.show() if you are running this locally and want to see the plot
# plt.show()
