# =============================================================================
# PRODUCTION INFERENCE PIPELINE FOR HYDERABAD PM2.5 POINT PREDICTION
# =============================================================================

import os
import pickle
import numpy as no
import pandas as pd

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
MODEL_DIR = 'D:\\Final_Mini_Project\\XGBOOST\\trial_6\\pkl_res\\models'
SEASON = 'monsoon'  # Adjust to match the season of your input dataset

# Update these to your actual input and desired output filenames
INPUT_CSV_PATH = 'D:\\Final_Mini_Project\\updated_data\\step_2\\season-wise\\monsoon_cleaned.csv'
OUTPUT_CSV_PATH = 'D:\\Final_Mini_Project\\XGBOOST\\trial_6\\pkl_res\\hyderabad_predicted_pm25_output.csv'

DROP_COLS = ['system:index', '.geo']

# =============================================================================
# 1. LOAD MODEL & CONFIGURATION ARTIFACTS
# =============================================================================
print(f"Loading packaged pipeline files for [{SEASON.upper()}]...")
with open(os.path.join(MODEL_DIR, f'xgb_model_{SEASON}.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, f'scaler_{SEASON}.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, f'spatial_metadata_{SEASON}.pkl'), 'rb') as f:
    meta = pickle.load(f)

global_mean = meta['global_mean']
feature_order = meta['feature_order']

# =============================================================================
# 2. LOAD & PREPARE INPUT FILE
# =============================================================================
print(f"Reading input points from: {INPUT_CSV_PATH}")
df_raw = pd.read_csv(INPUT_CSV_PATH)

# Drop unwanted operational indices if they exist
df_clean = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns], errors='ignore').copy()

# Ensure all required meteorological features are present
required_features = [c for c in feature_order if c != 'station_anchor']
missing = [c for c in required_features if c not in df_clean.columns]
if missing:
    raise ValueError(f"Input CSV is missing critical feature columns: {missing}")

# Handle missing data row-wise cleanly so inference doesn't break
if df_clean[required_features].isna().sum().sum() > 0:
    print("  --> Warning: NaNs detected in feature columns. Applying interpolation.")
    df_clean[required_features] = df_clean[required_features].fillna(method='ffill').fillna(method='bfill')

# Inject the necessary spatial anchor column used during model training
df_clean['station_anchor'] = global_mean

# Sort array columns to precisely mirror the original training matrix schema
X_inference_raw = df_clean[feature_order].values

# =============================================================================
# 3. PIPELINE MATRIX SCALING AND XGBOOST INFERENCE
# =============================================================================
print("Scaling features and running XGBoost model predictions...")
X_inference_scaled = scaler.transform(X_inference_raw)
predicted_pm25 = model.predict(X_inference_scaled)

# =============================================================================
# 4. EXPORT GEOSPATIAL RESULT DATASET
# =============================================================================
# Create a streamlined output dataframe containing only what you requested
output_df = pd.DataFrame({
    'latitude'      : df_clean['latitude'],
    'longitude'     : df_clean['longitude'],
    'predicted_PM25': predicted_pm25
})

# Optional: If you want to keep the original variables (like AOD, temp, date) 
# alongside the predictions, uncomment the line below instead:
# output_df = df_raw.copy(); output_df['predicted_PM25'] = predicted_pm25

output_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\nExecution Complete! Output file written to: {OUTPUT_CSV_PATH}")
print(output_df.head())