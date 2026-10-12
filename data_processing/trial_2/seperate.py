import os
import pandas as pd

# 1. Load your already cleaned master dataset
file_path = "../../data/session_4/merged/Cleaned_Master_Satellite_Data.csv"
df = pd.read_csv(file_path)


# 2. Define the seasonal classification function
def get_season(date_val):
    month = pd.to_datetime(date_val).month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    elif month in [10, 11]:
        return "post-monsoon"


# 3. Add the lowercase string season column
df["season"] = df["date"].apply(get_season)

# 4. Save the updated master copy that includes the 'season' column
df.to_csv(file_path, index=False)
print(f"Updated master dataset with 'season' column at: {file_path}")


# 5. Create the 4 seasonal CSVs with the 'season' column removed
output_dir = "../../data/session_4/merged"
seasons = ["winter", "summer", "monsoon", "post-monsoon"]

for target_season in seasons:
    # Filter for the specific season
    season_df = df[df["season"] == target_season].copy()

    # Drop the 'season' column exclusively for this file
    season_df = season_df.drop(columns=["season"])

    # Save out the separate file
    output_path = os.path.join(output_dir, f"Cleaned_Satellite_Data_{target_season}.csv")
    season_df.to_csv(output_path, index=False)
    print(f"Saved seasonal file: Cleaned_Satellite_Data_{target_season}.csv")