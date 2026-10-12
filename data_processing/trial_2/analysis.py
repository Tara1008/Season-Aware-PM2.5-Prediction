import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. LOAD THE DATA
# Update this path if your filename or directory changed
file_path = "../../data/session_4/merged/Satellite_Data_Hyderabad_2024.csv"
year_str = file_path.split("_")[-1].split(".")[0]
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape}\n")

# ==========================================
# PART 1: NULL VALUE ANALYSIS
# ==========================================
print("--- NULL VALUE REPORT ---")
print(f"{year_str} Data\n")
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 180  # Percentage calculation

null_df = pd.DataFrame(
    {"Missing Values": null_counts, "Percentage (%)": null_percentages}
)
# Only show columns that actually contain missing values or sorted by highest
print(null_df.sort_values(by="Missing Values", ascending=False))
print("-" * 30 + "\n")


# ==========================================
# PART 2: DATA DISTRIBUTION ANALYSIS
# ==========================================
# Select the most important continuous numerical columns for visualization
numerical_cols = [
    "PM25",
    "AOD",
    "NDVI",
    "blh_mean",
    "humidity",
    "temp",
    "wind_speed",
]

# Filter down to columns that actually exist in your final DataFrame
cols_to_plot = [col for col in numerical_cols if col in df.columns]

# Set up the plotting grid grid dynamically based on found columns
num_plots = len(cols_to_plot)
cols_per_row = 3
rows = int(np.ceil(num_plots / cols_per_row))

# Style configuration
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4 * rows))
axes = axes.flatten()  # Flatten to easily loop over via a single index

for i, col in enumerate(cols_to_plot):
    # Plot histogram with a Kernel Density Estimate (KDE) curve
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color="teal", bins=30)
    axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Count")

# Turn off any unused grid subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.title(f"{year_str} Data")

plt.tight_layout()
plt.show()

# ==========================================
# PART 3: BOX PLOTS (To identify outliers)
# ==========================================
fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4 * rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.boxplot(data=df, x=col, ax=axes[i], color="coral")
    axes[i].set_title(f"Boxplot of {col} (Outliers)", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.title(f"{year_str} Data")
plt.tight_layout()
plt.show()