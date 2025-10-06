"""
RDU Temperature Prediction

APPROACH:
Time series forecasting using historical temperature patterns with recursive prediction.
Models learn from lag features, rolling statistics, and cyclical time patterns.

Implements recursive 14-day forecasting where each hour uses previous predictions
as inputs, simulating real-world deployment conditions.

DATA SPLIT:
- Training: 2018 through August 31, 2025
- Validation: September 1-16, 2025 (for hyperparameter tuning)
- Test: September 17-30, 2025 (final evaluation)

FEATURES:
- Temperature lags: 24h, 48h, 168h (captures recent and weekly patterns)
- Rolling statistics: 24h mean, 168h mean, 24h std (captures trends)
- Cyclical time encoding: hour and day of year (sin/cos for periodicity)

MODELS (Project Requirements):
- Ridge Regression (required linear model)
- LightGBM (required second model - non-linear)
"""

from datetime import datetime
from meteostat import Hourly
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def download_data():
    """Download weather data from Meteostat"""
    if not os.path.exists("rdu_weather_data.csv") or not os.path.exists("rdu_weather_predict.csv"):
        X_start = datetime(2018, 1, 1)
        X_end = datetime(2025, 9, 16, 23, 59)
        y_start = datetime(2025, 9, 17)
        y_end = datetime(2025, 9, 30, 23, 59)

        print("Downloading RDU weather data...")
        X = Hourly('72306', X_start, X_end)
        X = X.fetch()
        y = Hourly('72306', y_start, y_end)
        y = y.fetch()
        
        X.to_csv("rdu_weather_data.csv", index=True)
        y.to_csv("rdu_weather_predict.csv", index=True)
        print("Downloaded and saved data")
        return X, y
    else:
        print("Loading cached weather data")
        X = pd.read_csv("rdu_weather_data.csv", index_col=0, parse_dates=True)
        y = pd.read_csv("rdu_weather_predict.csv", index_col=0, parse_dates=True)
        return X, y


def prepare_data(df):
    """Convert to Eastern Time"""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


def create_features(df):
    """Create time series features for temperature prediction"""
    df = df.copy()
    
    # Cyclical time encoding (more robust than raw hour/day values)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    # Temperature lag features
    df['temp_lag_24h'] = df['temp'].shift(24)
    df['temp_lag_48h'] = df['temp'].shift(48)
    df['temp_lag_168h'] = df['temp'].shift(168)
    
    # Rolling statistics
    df['temp_mean_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).mean()
    df['temp_mean_168h'] = df['temp'].shift(1).rolling(168, min_periods=1).mean()
    df['temp_std_24h'] = df['temp'].shift(1).rolling(24, min_periods=1).std()
    
    return df


def prepare_X_y(df, target='temp'):
    """Separate features and target"""
    feature_cols = [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'temp_lag_24h', 'temp_lag_48h', 'temp_lag_168h',
        'temp_mean_24h', 'temp_mean_168h', 'temp_std_24h'
    ]
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # Remove NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    return X, y


def train_ridge(X_train, y_train, X_val, y_val):
    """Train Ridge with cross-validation on training set"""
    print("\nTraining Ridge Regression...")
    
    from sklearn.model_selection import cross_val_score
    
    # Lower alpha values - 100 was too high
    alphas = [0.01, 0.1, 1.0, 10.0]
    best_alpha = None
    best_score = -float('inf')
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        # Use CV on training set only
        scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        mean_score = scores.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    print(f"  Best alpha: {best_alpha} (CV MAE: {-best_score:.3f}°C)")
    
    # Train on full training set
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_train, y_train)
    
    # Check validation performance
    val_pred = final_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"  Validation MAE: {val_mae:.3f}°C")
    
    return final_model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM"""
    print("\nTraining LightGBM...")
    
    model = lgb.LGBMRegressor(
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
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Validation MAE: {val_mae:.3f}°C")
    
    return model


def recursive_forecast(model, history_df, n_hours, start_time, model_type='ridge', X_train=None):
    """
    Recursive forecasting with input clipping to prevent drift
    """
    print(f"\nRecursive forecasting with {model_type.upper()}...")
    
    feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                    'temp_lag_24h', 'temp_lag_48h', 'temp_lag_168h',
                    'temp_mean_24h', 'temp_mean_168h', 'temp_std_24h']
    
    # Get feature ranges from training data for clipping
    feature_ranges = {}
    if X_train is not None:
        for col in feature_cols:
            if col not in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
                feature_ranges[col] = (X_train[col].min(), X_train[col].max())
    
    # Initialize
    temp_buffer = list(history_df['temp'].tail(168).values)
    predictions = []
    
    for hour_idx in range(n_hours):
        current_time = start_time + pd.Timedelta(hours=hour_idx)
        
        # Temporal features
        features = {
            'hour_sin': np.sin(2 * np.pi * current_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * current_time.hour / 24),
            'day_sin': np.sin(2 * np.pi * current_time.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * current_time.dayofyear / 365),
        }
        
        # Lag features
        features['temp_lag_24h'] = temp_buffer[-24] if len(temp_buffer) >= 24 else temp_buffer[0]
        features['temp_lag_48h'] = temp_buffer[-48] if len(temp_buffer) >= 48 else temp_buffer[0]
        features['temp_lag_168h'] = temp_buffer[-168] if len(temp_buffer) >= 168 else temp_buffer[0]
        
        # Rolling statistics
        if len(temp_buffer) >= 24:
            features['temp_mean_24h'] = np.mean(temp_buffer[-24:])
            features['temp_std_24h'] = np.std(temp_buffer[-24:])
        else:
            features['temp_mean_24h'] = np.mean(temp_buffer)
            features['temp_std_24h'] = np.std(temp_buffer) if len(temp_buffer) > 1 else 0
        
        if len(temp_buffer) >= 168:
            features['temp_mean_168h'] = np.mean(temp_buffer[-168:])
        else:
            features['temp_mean_168h'] = np.mean(temp_buffer)
        
        # Clip features to training range to prevent drift
        for col in feature_ranges:
            if col in features:
                features[col] = np.clip(features[col], feature_ranges[col][0], feature_ranges[col][1])
        
        # Predict
        X_current = pd.DataFrame([features])[feature_cols]
        
        if model_type == 'ridge':
            pred = model.predict(X_current)[0]
        else:
            pred = model.predict(X_current)[0]
        
        # Clip prediction to reasonable range
        pred = np.clip(pred, -20, 45)  # Reasonable temp range for RDU
        
        predictions.append(pred)
        temp_buffer.append(pred)
        
        if len(temp_buffer) > 168:
            temp_buffer.pop(0)
        
        if (hour_idx + 1) % 72 == 0:
            print(f"  Progress: {hour_idx+1}/{n_hours} hours")
    
    return np.array(predictions)


def create_visualizations(test_df, ridge_preds, lgb_preds, baseline_preds):
    """Create comprehensive visualizations"""
    print("\nCreating visualizations...")
    
    actual = test_df['temp'].values
    hours = np.arange(len(actual))
    days = hours / 24
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: First week predictions
    ax = axes[0, 0]
    n = min(168, len(hours))
    ax.plot(hours[:n], actual[:n], 'k-', label='Actual', linewidth=2)
    ax.plot(hours[:n], baseline_preds[:n], label='Baseline', color='gray', 
            linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot(hours[:n], ridge_preds[:n], label='Ridge', color='#2E86AB', 
            linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(hours[:n], lgb_preds[:n], label='LightGBM', color='#E63946',
            linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Hours', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('First Week Predictions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Full period
    ax = axes[0, 1]
    ax.plot(days, actual, 'k-', label='Actual', linewidth=2)
    ax.plot(days, baseline_preds, label='Baseline', color='gray', 
            linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot(days, ridge_preds, label='Ridge', color='#2E86AB', 
            linestyle='--', linewidth=1.5, alpha=0.7)
    ax.plot(days, lgb_preds, label='LightGBM', color='#E63946',
            linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Days', fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontweight='bold')
    ax.set_title('14-Day Forecast (Sept 17-30)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error growth over time
    ax = axes[1, 0]
    window = 24
    baseline_errors = np.abs(actual - baseline_preds)
    ridge_errors = np.abs(actual - ridge_preds)
    lgb_errors = np.abs(actual - lgb_preds)
    
    baseline_rolling = pd.Series(baseline_errors).rolling(window, min_periods=1).mean()
    ridge_rolling = pd.Series(ridge_errors).rolling(window, min_periods=1).mean()
    lgb_rolling = pd.Series(lgb_errors).rolling(window, min_periods=1).mean()
    
    ax.plot(days, baseline_rolling, label='Baseline', color='gray', linewidth=2, linestyle=':')
    ax.plot(days, ridge_rolling, label='Ridge', color='#2E86AB', linewidth=2)
    ax.plot(days, lgb_rolling, label='LightGBM', color='#E63946', linewidth=2)
    ax.set_xlabel('Days into Forecast', fontweight='bold')
    ax.set_ylabel('MAE (°C) - 24h Rolling', fontweight='bold')
    ax.set_title('Error Accumulation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    ax = axes[1, 1]
    bins = np.linspace(-10, 10, 40)
    ax.hist(actual - baseline_preds, bins=bins, alpha=0.4, label='Baseline', 
            color='gray', edgecolor='black')
    ax.hist(actual - ridge_preds, bins=bins, alpha=0.5, label='Ridge', 
            color='#2E86AB', edgecolor='black')
    ax.hist(actual - lgb_preds, bins=bins, alpha=0.5, label='LightGBM', 
            color='#E63946', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error (°C)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: test_results.png")
    plt.close()


def evaluate_model(y_true, y_pred, model_name):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  MAE:  {mae:.3f}°C")
    print(f"  RMSE: {rmse:.3f}°C")
    print(f"  R²:   {r2:.3f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
    print("="*70)
    print("RDU AIRPORT TEMPERATURE PREDICTION")
    print("="*70)
    
    # Load data
    print("\n1. Data Loading")
    print("-" * 70)
    train_data, test_data = download_data()
    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)
    
    print(f"Training: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Test: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} hours)")
    
    # Feature engineering
    print("\n2. Feature Engineering")
    print("-" * 70)
    train_data = create_features(train_data)
    print("Created time series features:")
    print("  - Cyclical time encoding (sin/cos)")
    print("  - Temperature lags (24h, 48h, 168h)")
    print("  - Rolling statistics (mean, std)")
    
    # Split train/validation
    print("\n3. Train/Validation Split")
    print("-" * 70)
    
    # Use Sept 1-16 as validation (closer to test period)
    train_df = train_data[train_data.index < '2025-09-01']
    val_df = train_data[(train_data.index >= '2025-09-01') & (train_data.index < '2025-09-17')]
    
    X_train, y_train = prepare_X_y(train_df)
    X_val, y_val = prepare_X_y(val_df)
    
    print(f"Training: {len(X_train)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Validation: {len(X_val)} samples ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Features: {list(X_train.columns)}")
    
    # Model training
    print("\n4. Model Training")
    print("-" * 70)
    
    ridge_model = train_ridge(X_train, y_train, X_val, y_val)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Retrain on train+validation for final models
    print("\nRetraining on combined train+validation data...")
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])
    
    ridge_model.fit(X_combined, y_combined)
    lgb_model.fit(X_combined, y_combined)
    print(f"  Final training size: {len(X_combined)} samples")
    
    # Test forecasting
    print("\n5. Test Period Forecasting (Sept 17-30)")
    print("-" * 70)
    print("Recursive forecasting: 336 hours (14 days)")
    
    # All data through Sept 16
    history_df = train_data[train_data.index <= '2025-09-16 23:59:59']
    history_df = history_df.dropna(subset=['temp'])
    
    test_start = pd.Timestamp('2025-09-17 00:00:00', tz='America/New_York')
    
    ridge_preds = recursive_forecast(ridge_model, history_df, len(test_data), 
                                     test_start, 'ridge', X_combined)
    lgb_preds = recursive_forecast(lgb_model, history_df, len(test_data), 
                                   test_start, 'lgb', X_combined)
    
    # Baseline: 24h persistence (uses actual past temps - standard approach)
    print("\nGenerating persistence baseline...")
    baseline_preds = []
    for idx in range(len(test_data)):
        if idx < 24:
            baseline_preds.append(history_df['temp'].iloc[-24+idx])
        else:
            baseline_preds.append(test_data['temp'].iloc[idx-24])
    baseline_preds = np.array(baseline_preds)
    
    # Evaluation
    print("\n6. Test Set Evaluation")
    print("-" * 70)
    
    actual = test_data['temp'].values
    
    baseline_results = evaluate_model(actual, baseline_preds, "Persistence Baseline")
    ridge_results = evaluate_model(actual, ridge_preds, "Ridge Regression")
    lgb_results = evaluate_model(actual, lgb_preds, "LightGBM")
    
    # Visualizations
    print("\n7. Visualizations")
    print("-" * 70)
    create_visualizations(test_data, ridge_preds, lgb_preds, baseline_preds)
    
    # Save predictions
    results = pd.DataFrame({
        'datetime': test_data.index,
        'actual_temp_C': actual,
        'baseline_pred_C': baseline_preds,
        'ridge_pred_C': ridge_preds,
        'lgb_pred_C': lgb_preds,
        'baseline_error': actual - baseline_preds,
        'ridge_error': actual - ridge_preds,
        'lgb_error': actual - lgb_preds
    })
    results.to_csv('predictions_sept17-30.csv', index=False)
    print("  Saved: predictions_sept17-30.csv")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nTest Period (Sept 17-30) - 14-Day Recursive Forecast:")
    print(f"  Baseline MAE:  {baseline_results['mae']:.3f}°C")
    print(f"  Ridge MAE:     {ridge_results['mae']:.3f}°C")
    print(f"  LightGBM MAE:  {lgb_results['mae']:.3f}°C")
    
    return ridge_model, lgb_model


if __name__ == "__main__":
    models = main()