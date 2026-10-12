"""
Random Forest — Seasonal PM2.5 Prediction (Random 80/20 Split Only)
Seasons: Summer, Winter, Monsoon, Post-Monsoon
Outputs: R², RMSE, MAE per season + comparison figures
"""

import warnings
warnings.filterwarnings("ignore")
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── CONFIGURATION ────────────────────────────────────────────────────────────
SEASONS = {
    "Summer":       "../../data/session_5/summer_cleaned.csv",
    "Winter":       "../../data/session_5/winter_cleaned.csv",
    "Monsoon":      "../../data/session_5/monsoon_cleaned.csv",
    "Post_Monsoon": "../../data/session_5/post-monsoon_cleaned.csv",
}

FEATURES = ["AOD", "NDVI", "blh_max", "blh_mean",
            "humidity", "temp", "wind_speed", "wind_u", "wind_v",
            "latitude", "longitude"]
TARGET   = "PM25"

OUTPUT_DIR  = "rf_results"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

SEASON_COLORS = {
    "Summer":       "#E07B39",
    "Winter":       "#3B7DD8",
    "Monsoon":      "#3FA34D",
    "Post_Monsoon": "#9B59B6",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HYPERPARAMETER SEARCH SPACE ──────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 8, 12, 16, 20],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.5, 0.7],
}

# ── HELPERS ──────────────────────────────────────────────────────────────────
def load(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)
    return df

def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    return {"R2": round(r2,4), "RMSE": round(rmse,4),
            "MAE": round(mae,4), "MSE": round(mse,4), "N": len(y_true)}

def tune_rf(X, y):
    """RandomizedSearchCV using standard KFold validation."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        PARAM_GRID, n_iter=30, cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
results_random = {}   # standard 80/20 split metrics
obs_pred_random = {}
feat_importance = {}

t_start = time.time()

for season, csv_file in SEASONS.items():
    t0 = time.time()
    print(f"\n{'─'*55}\n  {season}\n{'─'*55}")

    if not os.path.exists(csv_file):
        print(f"  SKIP — {csv_file} not found"); continue

    df = load(csv_file)
    X  = df[FEATURES].values
    y  = df[TARGET].values
    print(f"  n={len(df)}")

    # ── Tune on full data ────────────────────────────────────────────────────
    print("  Tuning hyperparameters...")
    best_model, best_params = tune_rf(X, y)
    print(f"  Best: {best_params}")

    # ── Random 80/20 split ───────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    
    rf_rand = RandomForestRegressor(**best_params, random_state=RANDOM_SEED, n_jobs=-1)
    rf_rand.fit(X_tr, y_tr)
    y_pred_rand = rf_rand.predict(X_te)

    m_rand = metrics(y_te, y_pred_rand)
    results_random[season]   = m_rand
    obs_pred_random[season]  = (y_te, y_pred_rand)
    print(f"  [Random split] R²={m_rand['R2']}  RMSE={m_rand['RMSE']}  MAE={m_rand['MAE']}")

    # ── Feature importance (from model trained on the split training set) ───
    feat_importance[season] = pd.Series(rf_rand.feature_importances_, index=FEATURES)

    print(f"  Done in {time.time()-t0:.0f}s")

# ── TABLES ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}\n  RESULTS\n{'='*55}")

t_rand = pd.DataFrame(results_random).T
t_rand.index.name = "Season"
print("\n── Random 80/20 Split ──")
print(t_rand[["R2","RMSE","MAE","MSE","N"]].to_string())

t_rand.to_csv(os.path.join(OUTPUT_DIR, "Table_RandomSplit.csv"))

fi_df = pd.DataFrame(feat_importance).round(4)
fi_df.index.name = "Feature"
fi_df.to_csv(os.path.join(OUTPUT_DIR, "Table_FeatureImportance.csv"))

# ── FIGURE 1: Observed vs Predicted — 2×2 grid ───────────────────────────────
seasons_done = [s for s in SEASONS if s in obs_pred_random]
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
fig.suptitle("Random Forest — Observed vs Predicted PM2.5 by Season (Random Split)",
             fontsize=14, fontweight="bold", y=0.98)

axes = axes.flatten()

for idx, season in enumerate(seasons_done):
    ax = axes[idx]
    color = SEASON_COLORS.get(season, "#555555")
    y_obs, y_prd = obs_pred_random[season]
    
    ax.scatter(y_obs, y_prd, alpha=0.4, s=15, color=color, edgecolor="none")

    lims = [min(y_obs.min(), y_prd.min()) * 0.93,
            max(y_obs.max(), y_prd.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1 Reference")
    
    slope, intercept = np.polyfit(y_obs, y_prd, 1)
    ax.plot(lims, [slope*l+intercept for l in lims], color=color, lw=1.8,
            label=f"OLS Fit ({slope:.2f})")
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    d = results_random[season]
    ax.set_title(f"{season.replace('_','-')}\nR²={d['R2']} | RMSE={d['RMSE']:.1f} | MAE={d['MAE']:.1f}",
                 fontsize=10, fontweight="bold", color=color)
    ax.set_xlabel("Observed PM2.5", fontsize=9)
    ax.set_ylabel("Predicted PM2.5", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.15)

# If any empty subplots exist, turn them off
for i in range(len(seasons_done), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_ObsVsPred_AllSeasons.png"), dpi=180, bbox_inches="tight")
plt.close()
print(f"\nFig1 saved → {OUTPUT_DIR}/Fig1_ObsVsPred_AllSeasons.png")

# ── FIGURE 2: Bar chart — R², RMSE, MAE comparing seasons ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Random Forest — Seasonal Performance Comparison (Random Split)",
             fontsize=13, fontweight="bold")

x = np.arange(len(seasons_done))

for ax, metric, ylabel in zip(
        axes,
        ["R2", "RMSE", "MAE"],
        ["R²", "RMSE (µg/m³)", "MAE (µg/m³)"]):

    vals = [results_random[s][metric] for s in seasons_done]
    colors = [SEASON_COLORS[s] for s in seasons_done]

    bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.7)

    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01*abs(h),
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_","\n") for s in seasons_done], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_Metrics_Comparison.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Fig2 saved → rf_results/Fig2_Metrics_Comparison.png")

# ── FIGURE 3: Feature importance heatmap ─────────────────────────────────────
if feat_importance:
    fi = pd.DataFrame(feat_importance)
    fi = fi.loc[fi.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(fi.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Importance")
    ax.set_xticks(range(len(fi.columns)))
    ax.set_xticklabels([c.replace("_","\n") for c in fi.columns], fontsize=10)
    ax.set_yticks(range(len(fi.index)))
    ax.set_yticklabels(fi.index, fontsize=10)
    
    for i in range(len(fi.index)):
        for j in range(len(fi.columns)):
            ax.text(j, i, f"{fi.iloc[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if fi.iloc[i,j] < 0.15 else "white")
            
    ax.set_title("Feature Importance by Season (Random Forest)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_FeatureImportance_Heatmap.png"), dpi=180, bbox_inches="tight")
    plt.close()
    print("Fig3 saved → rf_results/Fig3_FeatureImportance_Heatmap.png")

# ── DONE ──────────────────────────────────────────────────────────────────────
print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} min")
print(f"All outputs in: {OUTPUT_DIR}/")