"""
RDU Temperature Prediction

MODELS:
- Ridge Regression
- LightGBM

DATA SPLIT:
- Training: 2018 through August 31, 2025
- Validation: September 1-16, 2025
- Test: September 17-30, 2025

FEATURES:
- Temperature lags: 24h, 48h, 168h (actual historical values)
- Rolling statistics: 24h mean, 168h mean, 24h std
- One-hot encoded: hour of day (24), day of week (7)
"""

from datetime import datetime
from meteostat import Hourly
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def download_data():
    """Download complete weather data including test period"""
    if not os.path.exists("rdu_weather_full.csv"):
        start = datetime(2018, 1, 1, 0, 0)
        end = datetime(2025, 9, 30, 23, 59)
        
        print("Downloading RDU weather data...")
        data = Hourly('72306', start, end, timezone="America/New_York")
        data = data.fetch()
        
        data.to_csv("rdu_weather_full.csv", index=True)
        print("Downloaded and saved complete dataset")
        return data
    else:
        print("Loading cached weather data")
        data = pd.read_csv("rdu_weather_full.csv", index_col=0, parse_dates=True)
        return data

def convert_to_datetime(df):
    """Convert index to timezone-aware datetime in America/New_York"""
    dt_index = pd.to_datetime(df.index, utc=True, errors='raise')
    df.index = dt_index.tz_convert("America/New_York")
    return df

def create_features(df):
    """
    Create features:
    - Temperature lag features (backward-looking only)
    - Rolling statistics (shifted by 1 to prevent using current value)
    - One-hot encode hour of day (24 features)
    - One-hot encode day of week (7 features)
    """
    df = df.copy()
    
    # Temperature lag features
    df['temp_lag_24h'] = df['temp'].shift(24)    # Yesterday same hour
    df['temp_lag_48h'] = df['temp'].shift(48)    # 2 days ago
    df['temp_lag_168h'] = df['temp'].shift(168)  # 1 week ago
    
    # Rolling statistics
    df['temp_mean_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).mean()
    df['temp_mean_168h'] = df['temp'].shift(1).rolling(168, min_periods=1).mean()
    df['temp_std_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).std()
    
    # One-hot encode hour of day
    for hour in range(24):
        df[f'hour_{hour}'] = (df.index.hour == hour).astype(int)
    
    # One-hot encode day of week
    for day in range(7):
        df[f'dow_{day}'] = (df.index.dayofweek == day).astype(int)
    
    return df


def prepare_splits(all_data):
    """
    Split data into train, val, and test sets.
    """
    # Split the raw data first
    train_df = all_data[all_data.index < '2025-09-01'].copy()
    val_df = all_data[(all_data.index >= '2025-09-01') & 
                      (all_data.index < '2025-09-17')].copy()
    test_df = all_data[all_data.index >= '2025-09-17'].copy()
    
    # Create features for each split
    print("  Creating features for training set...")
    train_df = create_features(train_df)
    
    print("  Creating features for validation set...")
    val_df = create_features(val_df)
    
    print("  Creating features for test set...")
    test_df = create_features(test_df)
    
    return train_df, val_df, test_df


def train_ridge(X_train, y_train, X_val, y_val):
    """Train Ridge with hyperparameter tuning"""
    print("\nRidge Regression:")
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = -float('inf')
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                                scoring='neg_mean_absolute_error', n_jobs=-1)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"  Best alpha: {best_alpha} (CV MAE: {-best_score:.3f}°C)")
    
    # Train on full training set
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train, y_train)
    
    # Validation performance
    val_pred = ridge_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"  Validation MAE: {val_mae:.3f}°C")
    
    return ridge_model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM"""
    print("\nLightGBM:")
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    val_pred = lgb_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"  Best iteration: {lgb_model.best_iteration_}")
    print(f"  Validation MAE: {val_mae:.3f}°C")
    
    return lgb_model


def create_visualizations(test_clean, ridge_preds, lgb_preds, baseline_preds):
    """Create comprehensive visualizations"""
    actual = test_clean['temp'].values
    hours = np.arange(len(actual))
    days = hours / 24
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: First week
    ax = axes[0, 0]
    n = min(168, len(hours))
    ax.plot(hours[:n], actual[:n], 'k-', label='Actual', linewidth=2.5)
    ax.plot(hours[:n], baseline_preds[:n], label='Baseline (24h persistence)', 
            color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.plot(hours[:n], ridge_preds[:n], label='Ridge Regression', 
            color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(hours[:n], lgb_preds[:n], label='LightGBM', 
            color='#E63946', linestyle='-.', linewidth=2, alpha=0.8)
    ax.set_xlabel('Hours', fontweight='bold', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_title('First Week Predictions (168 hours)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Full 14-day period
    ax = axes[0, 1]
    ax.plot(days, actual, 'k-', label='Actual', linewidth=2.5)
    ax.plot(days, baseline_preds, label='Baseline', 
            color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.plot(days, ridge_preds, label='Ridge', 
            color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8)
    ax.plot(days, lgb_preds, label='LightGBM', 
            color='#E63946', linestyle='-.', linewidth=2, alpha=0.8)
    ax.set_xlabel('Days', fontweight='bold', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_title('14-Day Forecast (Sept 17-30, 2025)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Absolute errors over time
    ax = axes[1, 0]
    ax.plot(hours, np.abs(actual - baseline_preds), 
            label='Baseline', color='gray', linewidth=2, alpha=0.7)
    ax.plot(hours, np.abs(actual - ridge_preds), 
            label='Ridge', color='#2E86AB', linewidth=2, alpha=0.7)
    ax.plot(hours, np.abs(actual - lgb_preds), 
            label='LightGBM', color='#E63946', linewidth=2, alpha=0.7)
    ax.set_xlabel('Hours', fontweight='bold', fontsize=11)
    ax.set_ylabel('Absolute Error (°C)', fontweight='bold', fontsize=11)
    ax.set_title('Prediction Errors Over Time', fontweight='bold', fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    bins = np.linspace(-8, 8, 40)
    ax.hist(actual - baseline_preds, bins=bins, alpha=0.4, 
            label=f'Baseline (MAE={mean_absolute_error(actual, baseline_preds):.2f}°C)', 
            color='gray', edgecolor='black', linewidth=0.5)
    ax.hist(actual - ridge_preds, bins=bins, alpha=0.5, 
            label=f'Ridge (MAE={mean_absolute_error(actual, ridge_preds):.2f}°C)', 
            color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax.hist(actual - lgb_preds, bins=bins, alpha=0.5, 
            label=f'LightGBM (MAE={mean_absolute_error(actual, lgb_preds):.2f}°C)', 
            color='#E63946', edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax.set_title('Error Distribution', fontweight='bold', fontsize=12)
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: test_results.png")
    plt.close()


def main():
    print("="*70)
    print("RDU AIRPORT TEMPERATURE PREDICTION")
    print("="*70)
    
    print("\n1. Data Loading")
    print("-" * 70)
    all_data = download_data()
    # all_data = prepare_data(all_data)
    all_data = convert_to_datetime(all_data)
    print(f"Full data range: {all_data.index.min()} to {all_data.index.max()}")
    print(f"Total samples: {len(all_data)}")
    
    print("\n2. Train/Validation/Test Split")
    print("-" * 70)
    train_df, val_df, test_df = prepare_splits(all_data)
    
    print(f"Training:   {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} hours)")
    print(f"Validation: {val_df.index.min()} to {val_df.index.max()} ({len(val_df)} hours)")
    print(f"Test:       {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} hours)")
    
    continuous_features = ['temp_lag_24h', 'temp_lag_48h', 'temp_lag_168h',
                          'temp_mean_24h', 'temp_mean_168h', 'temp_std_24h']
    
    categorical_features = ([f'hour_{h}' for h in range(24)] + 
                           [f'dow_{d}' for d in range(7)])
    
    all_features = continuous_features + categorical_features
    
    print(f"\nFeature breakdown:")
    print(f"  Continuous: {len(continuous_features)} features")
    print(f"  Categorical: {len(categorical_features)} features")
    print(f"  Total: {len(all_features)} features")
    
    print("\n3. Data Preparation")
    print("-" * 70)
    
    # Drop NaNs (from lag features at start of each split)
    train_clean = train_df.dropna(subset=all_features + ['temp'])
    val_clean = val_df.dropna(subset=all_features + ['temp'])
    test_clean = test_df.dropna(subset=all_features + ['temp'])
    
    print(f"After dropping NaNs from lag features:")
    print(f"  Training: {len(train_clean)} samples (dropped {len(train_df) - len(train_clean)})")
    print(f"  Validation: {len(val_clean)} samples (dropped {len(val_df) - len(val_clean)})")
    print(f"  Test: {len(test_clean)} samples (dropped {len(test_df) - len(test_clean)})")
    
    # Prepare continuous and categorical separately
    X_train_cont = train_clean[continuous_features].values
    X_train_cat = train_clean[categorical_features].values
    y_train = train_clean['temp'].values
    
    X_val_cont = val_clean[continuous_features].values
    X_val_cat = val_clean[categorical_features].values
    y_val = val_clean['temp'].values
    
    X_test_cont = test_clean[continuous_features].values
    X_test_cat = test_clean[categorical_features].values
    y_test = test_clean['temp'].values
    
    print("\n4. Feature Scaling (Fit on training data only)")
    print("-" * 70)
    print("  Scaling continuous features only (not one-hot encoded)")
    
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    X_val_cont_scaled = scaler.transform(X_val_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)
    
    # Combine scaled continuous + unscaled categorical for Ridge
    X_train_ridge = np.hstack([X_train_cont_scaled, X_train_cat])
    X_val_ridge = np.hstack([X_val_cont_scaled, X_val_cat])
    X_test_ridge = np.hstack([X_test_cont_scaled, X_test_cat])
    
    # For LightGBM: use unscaled data (tree models don't need scaling)
    X_train_lgb = np.hstack([X_train_cont, X_train_cat])
    X_val_lgb = np.hstack([X_val_cont, X_val_cat])
    X_test_lgb = np.hstack([X_test_cont, X_test_cat])
    
    print(f"  Prepared data for both models")
    
    print("\n5. Model Training")
    print("-" * 70)
    
    ridge_model = train_ridge(X_train_ridge, y_train, X_val_ridge, y_val)
    lgb_model = train_lightgbm(X_train_lgb, y_train, X_val_lgb, y_val)
    
    print("\n6. Final Training on Combined Train+Val Data")
    print("-" * 70)
    
    # Combine train and validation
    combined_clean = pd.concat([train_clean, val_clean])
    
    X_comb_cont = combined_clean[continuous_features].values
    X_comb_cat = combined_clean[categorical_features].values
    y_comb = combined_clean['temp'].values
    
    # Re-fit scaler on combined data
    scaler_final = StandardScaler()
    X_comb_cont_scaled = scaler_final.fit_transform(X_comb_cont)
    
    X_comb_ridge = np.hstack([X_comb_cont_scaled, X_comb_cat])
    X_comb_lgb = np.hstack([X_comb_cont, X_comb_cat])
    
    print(f"  Combined size: {len(X_comb_ridge)} samples")
    
    # Retrain both models
    ridge_model.fit(X_comb_ridge, y_comb)
    lgb_model.fit(X_comb_lgb, y_comb)
    
    # Re-scale test set with final scaler
    X_test_cont_scaled_final = scaler_final.transform(X_test_cont)
    X_test_ridge_final = np.hstack([X_test_cont_scaled_final, X_test_cat])
    
    print("\n7. Test Set Predictions (Sept 17-30)")
    print("-" * 70)
    
    ridge_preds = ridge_model.predict(X_test_ridge_final)
    lgb_preds = lgb_model.predict(X_test_lgb)
    
    # Baseline: 24h persistence
    baseline_preds = test_clean['temp_lag_24h'].values
    
    print(f"  Generated {len(ridge_preds)} predictions for each model")
    
    print("\n8. Test Set Evaluation")
    print("="*70)
    
    # Baseline metrics
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    baseline_r2 = r2_score(y_test, baseline_preds)
    
    # Ridge metrics
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    ridge_r2 = r2_score(y_test, ridge_preds)
    
    # LightGBM metrics
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    lgb_r2 = r2_score(y_test, lgb_preds)
    
    print(f"\nPersistence Baseline (24h lag):")
    print(f"  MAE:  {baseline_mae:.3f}°C")
    print(f"  RMSE: {baseline_rmse:.3f}°C")
    print(f"  R²:   {baseline_r2:.3f}")
    
    print(f"\nRidge Regression (required linear model):")
    improvement = ((baseline_mae - ridge_mae) / baseline_mae) * 100
    print(f"  MAE:  {ridge_mae:.3f}°C ({improvement:+.1f}% vs baseline)")
    print(f"  RMSE: {ridge_rmse:.3f}°C")
    print(f"  R²:   {ridge_r2:.3f}")
    
    print(f"\nLightGBM (required non-linear model):")
    improvement = ((baseline_mae - lgb_mae) / baseline_mae) * 100
    print(f"  MAE:  {lgb_mae:.3f}°C ({improvement:+.1f}% vs baseline)")
    print(f"  RMSE: {lgb_rmse:.3f}°C")
    print(f"  R²:   {lgb_r2:.3f}")
    
    print(f"\nBest Model: {'LightGBM' if lgb_mae < ridge_mae else 'Ridge'}")
    if lgb_mae < ridge_mae:
        imp = ((ridge_mae - lgb_mae) / ridge_mae) * 100
        print(f"  LightGBM is {imp:.1f}% better than Ridge")
    
    print("\n9. Creating Visualizations")
    print("-" * 70)
    create_visualizations(test_clean, ridge_preds, lgb_preds, baseline_preds)
    
    results = pd.DataFrame({
        'datetime': test_clean.index,
        'actual_temp_C': y_test,
        'baseline_pred_C': baseline_preds,
        'ridge_pred_C': ridge_preds,
        'lgb_pred_C': lgb_preds,
        'baseline_error': y_test - baseline_preds,
        'ridge_error': y_test - ridge_preds,
        'lgb_error': y_test - lgb_preds
    })
    results.to_csv('predictions_sept17-30.csv', index=False)
    print("  Saved: predictions_sept17-30.csv")
    
    return ridge_model, lgb_model


if __name__ == "__main__":
    models = main()