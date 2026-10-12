import json
import pandas as pd

# 1. LOAD YOUR COMBINED DATASET
file_path = "../../data/session_4/merged/Master_Satellite_Data_All_Years.csv"
df = pd.read_csv(file_path)

initial_rows = len(df)
print(f"Initial row count: {initial_rows}")


# ==========================================
# NEW STEP: PARSE GEOJSON TO LAT/LON
# ==========================================
def extract_lat_lon(geo_str):
    try:
        # Safely fix any double-quoted string representations and load as JSON
        geo_dict = json.loads(geo_str.replace("'", '"'))
        lon, lat = geo_dict["coordinates"]
        return pd.Series([lat, lon])
    except Exception:
        # Returns NaN if the string is corrupt or empty
        return pd.Series([None, None])


print("Extracting latitude and longitude from .geo column...")
# This applies the function and splits the result into two brand-new columns
df[["latitude", "longitude"]] = df[".geo"].apply(extract_lat_lon)


# ==========================================
# 2. DEFINE THE CLEANING FILTERS
# ==========================================
valid_humidity = df["humidity"].between(20, 98)
valid_u_wind = df["wind_u"].between(-10, 10)
valid_v_wind = df["wind_v"].between(-8, 8)
valid_ndvi = df["NDVI"].between(0.05, 0.75)
valid_aod = df["AOD"].between(0.02, 1.50)
valid_blh_max = df["blh_max"].between(200, 3500)
valid_blh_mean = df["blh_mean"].between(100, 2000)

# Temperature conversion check:
# If your 'temp' column from ERA5/GEE is in Kelvin (e.g., ~295K),
# uncomment the line below to convert it to Celsius before filtering:
# df['temp'] = df['temp'] - 273.15

valid_temp = df["temp"].between(12, 50)

# ==========================================
# 3. COMBINE FILTERS AND APPLY
# ==========================================
master_filter = (
    valid_humidity
    & valid_u_wind
    & valid_v_wind
    & valid_ndvi
    & valid_aod
    & valid_blh_max
    & valid_blh_mean
    & valid_temp
)

# Filter the dataframe
cleaned_df = df[master_filter].copy()

# Optional: Remove the raw .geo column now that we have extracted the values
# cleaned_df = cleaned_df.drop(columns=['.geo'])

# ==========================================
# 4. REPORT THE RESULTS
# ==========================================
final_rows = len(cleaned_df)
rows_removed = initial_rows - final_rows

print("\n--- Processing & Filtering Summary ---")
print(f"Rows Retained: {final_rows} ({(final_rows/initial_rows)*100:.1f}%)")
print(f"Extreme Rows Removed: {rows_removed}")

print("\nSample of parsed coordinates:")
print(cleaned_df[["station", "latitude", "longitude"]].head(3))

# 5. SAVE THE CLEANED MASTER COPY
output_path = "../../data/session_4/merged/Cleaned_Master_Satellite_Data.csv"
cleaned_df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved successfully to: {output_path}")