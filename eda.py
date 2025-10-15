"""
Run after downloading data with the main script.

Generates:
1. Data overview (temperature time series + missing data)
2. Temporal patterns (hourly and monthly cycles)
3. Correlation matrix
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_style("whitegrid")
os.makedirs('visualizations', exist_ok=True)


def plot_data_overview(train: pd.DataFrame) -> None:
    """Temperature time series and missing data patterns."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Temperature time series
    axes[0].plot(train.index, train['temp'], linewidth=0.5, alpha=0.7, color='blue')
    axes[0].set_title('Temperature at RDU Airport (2018-2025)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].set_xlabel('Date')
    axes[0].grid(True, alpha=0.3)

    # Missing data summary
    missing = train.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]  # Only show variables with missing data
    if len(missing) > 0:
        colors = ['red' if x > len(train)*0.1 else 'orange' for x in missing.values]
        axes[1].barh(missing.index, missing.values, color=colors)
        axes[1].set_title('Missing Data by Variable (Before Imputation)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Missing Values')
        axes[1].axvline(len(train)*0.1, color='red', linestyle='--', label='10% threshold', alpha=0.7)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No missing data detected', ha='center', va='center', 
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/eda_1_data_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/eda_1_data_overview.png")
    plt.close()


def plot_temporal_patterns(train: pd.DataFrame) -> None:
    """Daily and seasonal temperature cycles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hourly pattern
    hourly_avg = train.groupby(train.index.hour)['temp'].mean()
    hourly_std = train.groupby(train.index.hour)['temp'].std()

    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8, color='blue')
    axes[0].fill_between(hourly_avg.index, hourly_avg - hourly_std, hourly_avg + hourly_std, alpha=0.3)
    axes[0].set_title('Daily Temperature Cycle', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].grid(True, alpha=0.3)

    peak_hour = hourly_avg.idxmax()
    trough_hour = hourly_avg.idxmin()
    axes[0].axvline(peak_hour, color='red', linestyle='--', alpha=0.5, label=f'Peak: {peak_hour}:00')
    axes[0].axvline(trough_hour, color='blue', linestyle='--', alpha=0.5, label=f'Low: {trough_hour}:00')
    axes[0].legend()

    # Monthly pattern
    monthly_avg = train.groupby(train.index.month)['temp'].mean()
    monthly_std = train.groupby(train.index.month)['temp'].std()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    axes[1].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='green')
    axes[1].fill_between(monthly_avg.index, monthly_avg - monthly_std, monthly_avg + monthly_std, alpha=0.3)
    axes[1].set_title('Seasonal Temperature Cycle', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Temperature (C)')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_names, rotation=45)
    axes[1].grid(True, alpha=0.3)

    peak_month = monthly_avg.idxmax()
    trough_month = monthly_avg.idxmin()
    axes[1].axvline(peak_month, color='red', linestyle='--', alpha=0.5, label=f'{month_names[peak_month-1]}')
    axes[1].axvline(trough_month, color='blue', linestyle='--', alpha=0.5, label=f'{month_names[trough_month-1]}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('visualizations/eda_2_temporal_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/eda_2_temporal_patterns.png")
    plt.close()


def plot_correlation_matrix(train: pd.DataFrame) -> None:
    """Correlation heatmap for meteorological variables."""
    numeric_cols = ['temp', 'rhum', 'wspd', 'wpgt', 'prcp', 'pres']
    available_cols = [col for col in numeric_cols if col in train.columns]
    corr_data = train[available_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('visualizations/eda_3_correlations.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/eda_3_correlations.png")
    plt.close()


def plot_seasonality_decomposition(train: pd.DataFrame) -> None:
    """Time series decomposition showing trend, seasonal, and residual components."""
    # Resample to daily to make decomposition cleaner
    daily_temp = train['temp'].resample('D').mean().dropna()
    
    if len(daily_temp) < 14:
        print("Skipping decomposition: insufficient data")
        return

    # Use 365-day period to show ANNUAL seasonality
    decomposition = seasonal_decompose(daily_temp, model='additive', period=365)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Trend component
    axes[0].plot(decomposition.trend.index, decomposition.trend, linewidth=2, color='red')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].set_title('Trend Component (Long-term pattern)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Seasonal component - zoom to last 2 years for clarity
    recent_start = decomposition.seasonal.index.max() - pd.Timedelta(days=365*2)
    recent_seasonal = decomposition.seasonal[decomposition.seasonal.index >= recent_start]
    axes[1].plot(recent_seasonal.index, recent_seasonal, linewidth=1.5, color='green')
    axes[1].set_ylabel('Temperature (C)')
    axes[1].set_title('Seasonal Component - Annual Cycle (Last 2 Years)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Residual component
    axes[2].plot(decomposition.resid.index, decomposition.resid, linewidth=0.5, alpha=0.7, color='gray')
    axes[2].set_ylabel('Temperature (C)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Residual Component (Unexplained variation)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/eda_4_decomposition.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/eda_4_decomposition.png")
    plt.close()


def generate_eda_plots():
    """Generate all EDA plots from training data."""
    print("\nGenerating EDA visualizations...")
    
    if not os.path.exists("rdu_weather_data.csv"):
        print("Error: rdu_weather_data.csv not found!")
        print("Run main.py first to download data.")
        return
    
    # Load training data
    train = pd.read_csv("rdu_weather_data.csv", index_col=0, parse_dates=True)
    
    # Convert timezone if needed
    if train.index.tz is None:
        train.index = train.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        train.index = train.index.tz_convert("America/New_York")
    
    print(f"Loaded training data: {len(train)} hours from {train.index.min()} to {train.index.max()}")
    
    # Generate plots
    plot_data_overview(train)
    plot_temporal_patterns(train)
    plot_correlation_matrix(train)
    plot_seasonality_decomposition(train)
    
    print("\nEDA complete. Saved to visualizations/ directory")


if __name__ == "__main__":
    generate_eda_plots()