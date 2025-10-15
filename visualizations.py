"""
Visualization utilities for model results.

Usage:
    from visualizations import generate_all_plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Create output directory
os.makedirs("visualizations", exist_ok=True)


def plot_test_performance(forecast_files):
    """Plot actual vs predicted for the test period."""
    # Ensure Linear Regression is plotted first
    lr_file = [f for f in forecast_files if 'linear_regression' in f.lower()]
    other_files = [f for f in forecast_files if 'linear_regression' not in f.lower()]
    ordered_files = lr_file + other_files
    
    fig, axes = plt.subplots(len(ordered_files), 1, figsize=(14, 4 * len(ordered_files)), sharex=True)
    if len(ordered_files) == 1:
        axes = [axes]
    
    for i, filepath in enumerate(ordered_files):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        model_name = os.path.basename(filepath).replace('forecast_', '').replace('.csv', '').replace('_', ' ').title()
        
        axes[i].plot(df.index, df['temp'], label='Actual', color='black', linewidth=2, marker='o', markersize=3)
        axes[i].plot(df.index, df['predicted_temp'], label='Predicted', color='blue', linewidth=1.5, linestyle='--')
        axes[i].fill_between(df.index, df['temp'], df['predicted_temp'], alpha=0.2)
        
        mae = mean_absolute_error(df['temp'], df['predicted_temp'])
        r2 = r2_score(df['temp'], df['predicted_temp'])
        
        axes[i].set_title(f'{model_name} - Test Period (Sept 17-30, 2025)', fontweight='bold')
        axes[i].set_ylabel('Temperature (C)')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
        axes[i].text(0.02, 0.98, f'MAE: {mae:.2f}C\nR2: {r2:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('visualizations/test_predictions.png', dpi=150)
    plt.close()
    print("Saved: visualizations/test_predictions.png")


def plot_forecast_horizon_degradation(forecast_files):
    """Show how prediction error changes as we forecast further into the future."""
    
    # Ensure Linear Regression is processed first for consistent coloring
    lr_file = [f for f in forecast_files if 'linear_regression' in f.lower()]
    other_files = [f for f in forecast_files if 'linear_regression' not in f.lower()]
    ordered_files = lr_file + other_files
    
    # Read the forecast files
    forecasts = {}
    for filepath in ordered_files:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        model_name = os.path.basename(filepath).replace('forecast_', '').replace('.csv', '').replace('_', ' ').title()
        forecasts[model_name] = df
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Define enough colors for any number of models
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    # Plot 1: Scatter of errors with trend lines
    for i, (model_name, df) in enumerate(forecasts.items()):
        actual = df['temp'].values
        predicted = df['predicted_temp'].values
        errors = np.abs(actual - predicted)
        hours_ahead = range(len(df))
        
        color = colors[i % len(colors)]  # Use modulo to cycle through colors if needed
        
        axes[0].scatter(hours_ahead, errors, alpha=0.4, s=20, label=model_name, color=color)
        
        # Add trend line
        z = np.polyfit(hours_ahead, errors, 1)
        p = np.poly1d(z)
        axes[0].plot(hours_ahead, p(hours_ahead), linestyle='--', linewidth=2, color=color, alpha=0.8)
    
    axes[0].set_title('Prediction Error vs Forecast Horizon', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Hours Ahead (0 = Sept 17 midnight, 335 = Sept 30, 11pm)')
    axes[0].set_ylabel('Absolute Error (C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add vertical lines for day boundaries
    for day in range(0, 336, 24):
        axes[0].axvline(day, color='gray', alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Plot 2: Rolling average to smooth out noise
    window = 24  # 24-hour rolling average
    for i, (model_name, df) in enumerate(forecasts.items()):
        actual = df['temp'].values
        predicted = df['predicted_temp'].values
        errors = np.abs(actual - predicted)
        
        color = colors[i % len(colors)]
        
        # Calculate rolling average
        rolling_error = pd.Series(errors).rolling(window=window, min_periods=1).mean()
        axes[1].plot(rolling_error.index, rolling_error.values, linewidth=2, label=f'{model_name} (24h avg)', color=color)
    
    axes[1].set_title('Smoothed Error Progression (24-hour rolling average)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Hours Ahead')
    axes[1].set_ylabel('Mean Absolute Error (C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/forecast_horizon_degradation.png', dpi=150)
    plt.close()
    print("Saved: visualizations/forecast_horizon_degradation.png")


def generate_all_plots():
    """Generate all essential visualizations from saved forecast files."""
    print("\nGenerating visualizations...")
    
    # Find all forecast CSV files
    forecast_files = [f for f in os.listdir('.') if f.startswith('forecast_') and f.endswith('.csv')]
    
    if not forecast_files:
        print("No forecast files found. Run the model training first.")
        return
    
    # Sort to ensure consistent ordering
    forecast_files.sort()
    
    plot_test_performance(forecast_files)
    plot_forecast_horizon_degradation(forecast_files)
    
    print("Done generating visualizations.")


if __name__ == "__main__":
    generate_all_plots()