import glob
import os
import re
import pandas as pd

# 1. Path to your merged folder
# Adjust this relative path if your script folder changes
merged_folder = "../../data/session_4/merged"

# 2. Grab all Satellite_Data CSV files (e.g., 2023, 2024, 2025)
file_pattern = os.path.join(merged_folder, "Satellite_Data_*.csv")
files_to_merge = glob.glob(file_pattern)

# Sort them so they combine chronologically (2023 -> 2024 -> 2025)
files_to_merge.sort()

print(f"Found {len(files_to_merge)} files to combine:")
for f in files_to_merge:
    print(f" - {os.path.basename(f)}")

# 3. Read and append them
all_dfs = []

for file in files_to_merge:
    df = pd.read_csv(file)

    # Extract the year from the filename to create a tracking column
    # year_match = re.search(r"Satellite_Data_(\d{4})", os.path.basename(file))
    # if year_match:
    #     df["data_year"] = year_match.group(1)

    all_dfs.append(df)

# 4. Vertically stack (append) all DataFrames
# ignore_index=True resets the index counter so it goes cleanly from 0 to total_rows
master_df = pd.concat(all_dfs, ignore_index=True)

print("\n--- Merge Complete ---")
print(f"Master Dataset Shape: {master_df.shape}")
# print(f"Breakdown by Data Year:\n{master_df['data_year'].value_counts()}")

# 5. Save your master copy
output_path = os.path.join(merged_folder, "Master_Satellite_Data_All_Years.csv")
master_df.to_csv(output_path, index=False)
print(f"\nSaved master file to: {output_path}")