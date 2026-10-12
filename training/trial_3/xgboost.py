# =============================================================================
# SEASON-STRATIFIED XGBoost + SHAP FOR PM2.5 PREDICTION (FIXED STRATIFIED SPLIT)
# Hyderabad — CPCB Stations
# =============================================================================

import os
import warnings
import d:\Final_Mini_Project\XGBOOST\trial_6\code6.pynumpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150
})

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

CSV_PATHS = {
    'Summer'      : 'updated_data/step_2/season-wise/summer_cleaned.csv',
    'Monsoon'     : 'updated_data/step_2/season-wise/monsoon_cleaned.csv',
    'Post-Monsoon': 'updated_data/step_2/season-wise/post_monsoon_cleaned.csv',
    'Winter'      : 'updated_data/step_2/season-wise/winter_cleaned.csv',
}

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Math features for training
BASE_FEATURES = [
    'AOD', 'NDVI', 'blh_max', 'blh_mean', 'humidity', 'temp', 
    'wind_speed', 'wind_u', 'wind_v', 'latitude', 'longitude'
]
TARGET = 'PM25'
META_COLS = ['station', 'date']
DROP_COLS = ['system:index', '.geo']

# Standard high-performance params for environmental data
XGB_PARAMS = {
    'n_estimators'    : 700,       
    'max_depth'       : 5,         
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,       
    'colsample_bytree': 0.8,       
    'min_child_weight': 3,        
    'random_state'    : 42,
    'n_jobs'          : -1,
    'verbosity'       : 0,
}

SEASON_COLORS = {
    'Summer'      : '#E07B39',
    'Monsoon'     : '#4A90D9',
    'Post-Monsoon': '#7CB87C',
    'Winter'      : '#9B6BB5',
}

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def load_clean_and_stratify(path, season, test_size=0.20):
    """Loads data, removes outliers, and performs a clean 80/20 train/test split per station."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

    required_cols = BASE_FEATURES + [TARGET] + META_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{season}] Missing critical columns in CSV: {missing}")

    df = df[required_cols].dropna().reset_index(drop=True)

    # Remove PM2.5 outliers (> 3 SD)
    z = np.abs(stats.zscore(df[TARGET]))
    df = df[z < 3].reset_index(drop=True)

    # Split 80/20 *within* each station to maintain geographic representation
    train_slices = []
    test_slices = []
    
    for station in df['station'].unique():
        station_df = df[df['station'] == station].copy()
        
        # Shuffle records within station reproducibly
        station_df = station_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(station_df) * (1 - test_size))
        train_slices.append(station_df.iloc[:split_idx])
        test_slices.append(station_df.iloc[split_idx:])
        
    train_df = pd.concat(train_slices, ignore_index=True)
    test_df = pd.concat(test_slices, ignore_index=True)

    # Spatial Anchor engineering: Calculate historical target encoding safely using train data only
    station_means = train_df.groupby('station')[TARGET].mean().to_dict()
    
    # Map back to dataframes as a highly predictive baseline feature
    train_df['station_anchor'] = train_df['station'].map(station_means)
    test_df['station_anchor'] = test_df['station'].map(station_means)
    
    # Fill any missing values with total training mean if a station is sparse
    global_mean = train_df[TARGET].mean()
    train_df['station_anchor'].fillna(global_mean, inplace=True)
    test_df['station_anchor'].fillna(global_mean, inplace=True)

    print(f"  [{season}] Cleaned Dataset -> Train: {len(train_df)} rows | Test: {len(test_df)} rows across {df['station'].nunique()} stations")

    return train_df, test_df

def compute_metrics(y_true, y_pred):
    """Calculates evaluation metrics."""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mbe  = np.mean(y_pred - y_true)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MBE': mbe}

# =============================================================================
# 2. MAIN PIPELINE
# =============================================================================

all_train_cv_metrics = {}
all_test_metrics = {}
all_shap_vals = {}
season_test_results = {}

# Dynamic feature list using base meteorological data + coordinates + station anchor
FINAL_FEATURES = BASE_FEATURES + ['station_anchor']

print("\n" + "="*75)
print("  PM2.5 SEASON-STRATIFIED XGBOOST PERFORMANCE RECOVERY PIPELINE")
print("="*75 + "\n")

for season, path in CSV_PATHS.items():

    print(f"\n{'─'*65}")
    print(f"  Processing: {season.upper()}")
    print(f"{'─'*65}")

    # Load and stratify data splits safely
    train_df, test_df = load_clean_and_stratify(path, season, test_size=0.20)

    X_train_raw = train_df[FINAL_FEATURES].values
    y_train = train_df[TARGET].values
    X_test_raw = test_df[FINAL_FEATURES].values
    y_test = test_df[TARGET].values

    # Scale data for numerical stabilization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 10-Fold Cross Validation over training pool
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_cv_pred = np.zeros_like(y_train)
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        cv_model = XGBRegressor(**XGB_PARAMS)
        cv_model.fit(X_train[train_idx], y_train[train_idx])
        y_cv_pred[val_idx] = cv_model.predict(X_train[val_idx])

    cv_metrics = compute_metrics(y_train, y_cv_pred)
    all_train_cv_metrics[season] = cv_metrics

    # Train final production model and project on validation test split
    final_model = XGBRegressor(**XGB_PARAMS)
    final_model.fit(X_train, y_train)
    
    y_test_pred = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    all_test_metrics[season] = test_metrics
    season_test_results[season] = (y_test, y_test_pred)

    print(f"\n  10-Fold Cross-Validation (Train Pool) R²: {cv_metrics['R2']:.4f}")
    print(f"  *** Independent Split Evaluation Summary ***")
    print(f"    R²   = {test_metrics['R2']:.4f}")
    print(f"    RMSE = {test_metrics['RMSE']:.4f} µg/m³")
    print(f"    MAE  = {test_metrics['MAE']:.4f} µg/m³")
    print(f"    MBE  = {test_metrics['MBE']:.4f} µg/m³")

    # Generate Explainable SHAP Arrays
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_train)
    all_shap_vals[season] = shap_vals

    # Plot Diagnostics (Scatter + Summary Importance Bar Chart)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f'{season} Validation Test Set — XGBoost Evaluation\n'
        f'R² = {test_metrics["R2"]:.3f} | RMSE = {test_metrics["RMSE"]:.2f} µg/m³ | '
        f'MAE = {test_metrics["MAE"]:.2f} µg/m³', fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    c = SEASON_COLORS[season]
    ax1.scatter(y_test, y_test_pred, alpha=0.5, s=20, color=c, edgecolors='none')
    
    lims = [min(y_test.min(), y_test_pred.min()) * 0.9, max(y_test.max(), y_test_pred.max()) * 1.05]
    ax1.plot(lims, lims, 'k--', lw=1, label='1:1 Line')
    
    m, b, *_ = stats.linregress(y_test, y_test_pred)
    ax1.plot(np.array(lims), m * np.array(lims) + b, color=c, lw=1.5, label=f'Fit (slope={m:.2f})')
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('Observed Test PM₂.₅ (µg/m³)')
    ax1.set_ylabel('Predicted Test PM₂.₅ (µg/m³)')
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    feat_labels = [FINAL_FEATURES[i] for i in sorted_idx]
    feat_vals = mean_abs_shap[sorted_idx]

    bars = ax2.barh(feat_labels, feat_vals, color=c, alpha=0.75, edgecolor='white')
    ax2.set_xlabel('Mean |SHAP Value| (Impact on Training)')
    for bar, val in zip(bars, feat_vals):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{season.lower().replace("-","_")}_test_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 3. CONSOLIDATED SUMMARY PRINT & FILE EXPORT
# =============================================================================

print(f"\n\n{'='*75}")
print("  FINAL EVALUATION REPORT — STRATIFIED 80/20 TEST SET")
print(f"{'='*75}")

test_metrics_df = pd.DataFrame(all_test_metrics).T.round(4)
test_metrics_df.index.name = 'Season'
test_metrics_df.columns = ['R²', 'RMSE (µg/m³)', 'MAE (µg/m³)', 'MBE (µg/m³)']
print(test_metrics_df.to_string())

test_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'independent_test_metrics_summary.csv'))
print(f"\nExecution finished! Clean plots and summary table saved to ./{OUTPUT_DIR}/")
