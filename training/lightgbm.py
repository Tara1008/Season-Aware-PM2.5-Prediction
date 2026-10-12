

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------------------------------
# STEP 1: Map season name to CSV filename
# -----------------------------------------------------
season_files = {
    "Summer":       "summer_cleaned.csv",
    "Monsoon":      "monsoon.csv",
    "Post-Monsoon": "postmonsoon.csv",
    "Winter":       "winter_cleaned.csv",
}

# -----------------------------------------------------
# STEP 2: Station coordinates for spatial features
# -----------------------------------------------------
station_coords = {
    "IITH_Kandi":                (17.596111, 78.125278),
    "Ramachandrapuram_BHEL":     (17.4954,   78.2960),
    "ICRISAT_Patancheru":        (17.511543, 78.275104),
    "Sanathnagar":               (17.45594,  78.43322),
    "Somajiguda":                (17.4149,   78.4576),
    "Kompally_Municipal_Office": (17.542881, 78.481445),
    "University_of_Hyderabad":   (17.465278, 78.326944),
    "Nacharam":                  (17.4168,   78.5639),
    "Zoo_Park":                  (17.3507,   78.4513),
    "New_Malakpet":              (17.3616,   78.4747),
    "Bollaram_Industrial_Area":  (17.612434, 78.086382),
    "ECIL_Kapra":                (17.4897,   78.5614),
    "Kokapet":                   (17.403458, 78.314681),
    "IDA_Pashamylaram":          (17.511510, 78.300784),
}

# -----------------------------------------------------
# STEP 3: Hyperparameter search space
# -----------------------------------------------------
param_grid = {
    "n_estimators":      [200, 500, 1000],
    "learning_rate":     [0.01, 0.05, 0.1],
    "max_depth":         [4, 6, 8, -1],
    "num_leaves":        [15, 31, 63],
    "min_child_samples": [5, 10, 20],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0]
}

# -----------------------------------------------------
# STEP 4: Helper — load and prepare one season CSV
# -----------------------------------------------------
def prepare_season(filepath, season_name):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    # --- Spatial features using lat/lon coordinates ---
    df["lat"] = df["station"].map(lambda s: station_coords.get(s, (np.nan, np.nan))[0])
    df["lon"] = df["station"].map(lambda s: station_coords.get(s, (np.nan, np.nan))[1])

    # NOTE: Station One-Hot encoding has been intentionally removed here 
    # to facilitate grid-based city wide out-of-station predictions.

    # --- Time features ---
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["day_of_week"]  = df["date"].dt.dayofweek

    # --- Interaction features ---
    df["AOD_x_humidity"] = df["AOD"] * df["humidity"]
    df["AOD_x_wind"]     = df["AOD"] * df["wind_speed"]
    df["AOD_x_temp"]     = df["AOD"] * df["temp"]
    df["BLH_x_AOD"]      = df["blh_mean"] * df["AOD"]
    df["BLH_x_wind"]     = df["blh_mean"] * df["wind_speed"]

    print(f"\n  {season_name} — loaded {len(df)} rows, {df['station'].nunique()} stations")
    return df

# -----------------------------------------------------
# STEP 5: Build generalized feature list
# -----------------------------------------------------
def get_feature_cols(df):
    feature_base = [
        "AOD", "NDVI", "blh_max", "blh_mean", "humidity", "temp",
        "wind_speed", "wind_u", "wind_v", "lat", "lon",
        "month", "day_of_year", "day_of_week",
        "AOD_x_humidity", "AOD_x_wind", "AOD_x_temp", "BLH_x_AOD", "BLH_x_wind"
    ]
    # Keep only columns that explicitly exist in the dataframe
    feature_cols = [c for c in feature_base if c in df.columns]
    return feature_cols

# -----------------------------------------------------
# STEP 6: Train, evaluate, and SAVE models
# -----------------------------------------------------
def run_season(season_name, filepath):
    print("\n" + "=" * 60)
    print(f"  TRAINING {season_name.upper()} MODEL (NO STATION DUMMIES)")
    print("=" * 60)

    df = prepare_season(filepath, season_name)
    feature_cols = get_feature_cols(df)

    # Drop rows missing critical features or target
    df = df.dropna(subset=feature_cols + ["PM25"]).reset_index(drop=True)
    print(f"  Rows after dropping NaNs : {len(df)}")

    X = df[feature_cols]
    y = df["PM25"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=42
    )
    
    # Scale continuous features
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols]  = scaler.transform(X_test[feature_cols])

    print(f"\n  Running hyperparameter search...")
    search = RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV R²  : {round(search.best_score_, 4)}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)

    # Metrics
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    bias = np.mean(y_pred - y_test)

    print(f"\n  Performance on Test Set")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  Bias : {bias:.4f}")

    # --- SAVE ARTIFACTS ---
    clean_name = season_name.lower().replace('-', '_')
    
    joblib.dump(model, f"{clean_name}_model.pkl")
    joblib.dump(scaler, f"{clean_name}_scaler.pkl")
    joblib.dump(feature_cols, f"{clean_name}_features.pkl")
    print(f"\n  [SAVED] Saved model, scaler, and features list for {season_name}")

    # Feature Importances
    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(f"\n  Top 10 Important Features")
    print(imp_df.head(10).to_string(index=False))

    imp_df.to_csv(f"importance_{clean_name}.csv", index=False)

    return {
        "Season": season_name,
        "Rows":   len(df),
        "Train":  len(X_train),
        "Test":   len(X_test),
        "R2":     round(r2,   4),
        "MAE":    round(mae,  4),
        "RMSE":   round(rmse, 4),
        "Bias":   round(bias, 4),
        "Best_CV_R2": round(search.best_score_, 4),
    }

# -----------------------------------------------------
# STEP 7: Run Pipeline
# -----------------------------------------------------
all_results = []

for season, filepath in season_files.items():
    result = run_season(season, filepath)
    all_results.append(result)

results_df = pd.DataFrame(all_results)

print("\n\n" + "=" * 60)
print("  FINAL SEASON-WISE RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

results_df.to_csv("season_results_generalized.csv", index=False)
