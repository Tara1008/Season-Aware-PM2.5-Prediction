import os
import pickle
import pandas as pd

MODEL_DIR = "pkl_res/models"
DROP_COLS = ["system:index", ".geo"]


def predict_pm25(input_csv_path, season):
    #ensuring it uses a hyphen to match the file naming
    sanitized_season = season.lower().replace("_", "-")

    with open(os.path.join(MODEL_DIR, f"xgb_model_{sanitized_season}.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, f"scaler_{sanitized_season}.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(MODEL_DIR, f"spatial_metadata_{sanitized_season}.pkl"), "rb") as f:
        meta = pickle.load(f)

    global_mean = meta["global_mean"]
    feature_order = meta["feature_order"]

    df_raw = pd.read_csv(input_csv_path)

    df_clean = df_raw.drop(
        columns=[c for c in DROP_COLS if c in df_raw.columns],
        errors="ignore"
    )

    required_features = [c for c in feature_order if c != "station_anchor"]

    missing = [
        c for c in required_features
        if c not in df_clean.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_clean["station_anchor"] = global_mean

    X = df_clean[feature_order].values
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    output_df = pd.DataFrame({
        "latitude": df_clean["latitude"],
        "longitude": df_clean["longitude"],
        "predicted_PM25": predictions
    })

    return output_df