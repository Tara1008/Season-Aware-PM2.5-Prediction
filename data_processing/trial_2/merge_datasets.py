import pandas as pd
import glob
import os
import re

# 1. LOAD GEE DATA
gee = pd.read_csv("../../data/session_4/gee/Hyderabad_Daily_AOD_ERA5_BLH_NDVI_Stations_2025.csv")

gee["date"] = pd.to_datetime(
    gee["date"]
)
# 2. CPCB FILES
files = glob.glob(
    "../../data/session_4/CPCB_2025/*.csv"
)
all_stations = []
# 3. STATION NAME FIXES
station_mapping = {

    "Central_University":
        "University_of_Hyderabad",

    "Nacharam_TSIIC_IALA":
        "Nacharam",

    "Ramachandrapuram":
        "Ramachandrapuram_BHEL"

}

# 4. PROCESS EACH FILE

for file in files:

    df = pd.read_csv(file)

    filename = os.path.basename(file)

    # Extract station name

    match = re.search(
        r"site_\d+_(.*?)_Hyderabad",
        filename
    )

    if match:

        station = match.group(1)

    else:

        print(
            f"Could not parse station from {filename}"
        )

        continue

    # Apply station mapping

    station = station_mapping.get(
        station,
        station
    )

    # Keep only date + PM2.5
    pm_col = next(
        (col for col in df.columns if "PM2.5" in col),
        None
    )

    if pm_col is None:
        print(f"PM2.5 column not found in {filename}")
        print(df.columns.tolist())
        continue
    
    print(f"{station}->{pm_col}")

    temp = pd.DataFrame({

        "date": pd.to_datetime(
            df["Timestamp"]
        ),

        "station": station,

        "PM25": pd.to_numeric(
            df[pm_col],
            errors="coerce"
    )

})

    all_stations.append(temp)

# 5. COMBINE ALL STATIONS

cpcb = pd.concat(
    all_stations,
    ignore_index=True
)

print("\nCPCB Shape:")
print(cpcb.shape)

print("\nStations:")
print(sorted(
    cpcb["station"].unique()
))
# 6. MERGE

final_df = pd.merge(

    cpcb,

    gee,

    on=["date", "station"],

    how="left"

)

# 7. CLEAN

final_df = final_df.dropna()

# 8. SAVE

final_df.to_csv(

    "../../data/session_4/merged/Satellite_Data_Hyderabad_2025.csv",

    index=False

)

print("\nFinal Dataset Shape:")
print(final_df.shape)

print("\nColumns:")
print(final_df.columns.tolist())

print("\nFirst Rows:")
print(final_df.head())

print(
    "\nSaved:"
    " ../../data/session_4/merged/Satellite_Data_Hyderabad_2025.csv"
)