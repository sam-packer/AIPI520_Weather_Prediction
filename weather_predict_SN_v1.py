from datetime import datetime
import os
from networkx import display
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pytz
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

def download_data():
    '''Download data from meteostat. Using the date range split it into training and test data sets
        Output: X - traing data , y -testing data
    '''
    if not os.path.exists("rdu_weather_data.csv") or not os.path.exists("rdu_weather_predict.csv"):
        X_start = datetime(2018, 1, 1)
        X_end = datetime(2025, 9, 16, 23, 59)
        y_start = datetime(2025, 9, 17)
        y_end = datetime(2025, 9, 30, 23, 59)

        print("Downloading RDU weather data, please wait...")
        from meteostat import Hourly
        X = Hourly('72306', X_start, X_end).fetch()
        y = Hourly('72306', y_start, y_end).fetch()
        X.to_csv("rdu_weather_data.csv", index=True)
        y.to_csv("rdu_weather_predict.csv", index=True)
        print("Downloaded and saved data!")
        return X, y
    else:
        print("You already have the weather data downloaded, not redownloading.")
        X = pd.read_csv("rdu_weather_data.csv", index_col=0, parse_dates=True)
        y = pd.read_csv("rdu_weather_predict.csv", index_col=0, parse_dates=True)
        return X, y
    
def prepare_data(X, y):
    '''Convert index timezone from UTC to America/New_York (if timezone-naive, localize first)
        Output: X - traing data , y -testing data , df_concat
    '''
    for df in [X, y]:
        # If index is timezone-naive (no tz), localize then convert.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
    df_concat = pd.concat([X, y])
    return X, y, df_concat

def build_features(df_concat, make_datetime_col=False):
    """
    Build features - lag and rolling averages.
     Input: Accepts df_concat where timestamp is the index.
     Output:df_build
    """
    df_build = df_concat.copy()

    # Converting datetime column to pandas.
    if 'time' in df_build.columns:
        df_build['datetime'] = pd.to_datetime(df_build['time'])

    # The below features decreased the R2 score drastically to negative (despite the strong correlation), which is why they are commented out.
    
    # Last hour & last day temperature
    #df_build['temp_lag_1'] = df_build['temp'].shift(1) 
        
    df_build['temp_lag_24'] = df_build['temp'].shift(24)  
    df_build['1day_avg'] = df_build['temp'].rolling(window=24).mean()
    
    # Rolling averages (assumes hourly data)
    # 7 days, 3 day (72 hours), 30 day (720 hours)
    df_build['7day_avg'] = df_build['temp'].rolling(window=168, min_periods=1).mean()
    
    df_build['30day_avg'] = df_build['temp'].rolling(window=720, min_periods=1).mean()
    
    df_build['3day_avg'] = df_build['temp'].rolling(window=72, min_periods=1).mean()
    
    print(f"Data shape after feature engineering: {df_build.shape}")
    print(df_build.head())
    return df_build

#split data into train and test

def split_data(df_build):
    """
    Use the new df with engineered features to create the training and test data sets.
     Input: df_build
     Output: X_df: training data set, y_df: test data set
    """
    # Create timezone object
    ny_tz = pytz.timezone("America/New_York")

    # Make your datetime ranges timezone-aware
    X_start = ny_tz.localize(datetime(2018, 1, 1))
    X_end = ny_tz.localize(datetime(2025, 9, 16, 23, 59))
    y_start = ny_tz.localize(datetime(2025, 9, 17))
    y_end = ny_tz.localize(datetime(2025, 9, 30, 23, 59))
    
    #df = df.drop(['month','season'])
    X_df = df_build.loc[X_start:X_end]
    y_df = df_build.loc[y_start:y_end]
    print(f"X data range: {X_df.index.min()} to {X_df.index.max()}, shape: {X_df.shape}")
    print(f"y data range: {y_df.index.min()} to {y_df.index.max()}, shape: {y_df.shape}")
    print(X_df.head())
    print(y_df.head())
    return X_df, y_df

def robust_ffill_impute(X_df,y_df, sort_index=True, drop_all_nan_cols=True, verbose=True):
    """
    Robust forward-fill imputation pipeline:
      Ensure time order by sorting index .
      Forward-fill (ffill) then back-fill (bfill).
      Optionally drop columns that are 100% NaN.
    Output: X_df_imputed , y_df_imputed - imputed train & test data
    """
    X_df_imputed = X_df.copy()

    # Ensure index (datetime) is sorted 
    if sort_index:
        
            X_df_imputed.index = pd.to_datetime(X_df_imputed.index)
            X_df_imputed = X_df_imputed.sort_index()
   

    # Forward fill then back fill
    X_df_imputed = X_df_imputed.ffill().bfill()

    # drop columns that are 100% NaN 
    if drop_all_nan_cols:
        all_nan_cols = X_df_imputed.columns[X_df_imputed.isna().all()].tolist()
        if all_nan_cols:
            if verbose:
                print(f"Dropping columns that are all-NaN and cannot be imputed: {all_nan_cols}")
            X_df_imputed = X_df_imputed.drop(columns=all_nan_cols)

    # 6) Final diagnostic
    total_remaining = int(X_df_imputed.isna().sum().sum())
    if verbose:
        print(f"Imputation done. Remaining NaNs in dataframe: {total_remaining}")
        if total_remaining > 0:
            print("NaN counts per column (top 10):")
            print(X_df_imputed.isna().sum().sort_values(ascending=False).head(10))

    y_df_imputed = y_df.drop(columns=all_nan_cols)
    #print(f"y data shape after imputation: {y_df_imputed.shape}")
    return X_df_imputed , y_df_imputed

def eda(df):
    ''' Perform EDA to understand correlation between features
        Perform time series decomposition
        Prform cross-correlation
    '''
    
    df_input_viz = df['temp']
    plot_acf(df_input_viz,lags=24)
    plt.title('Correlation of temp with time of day (across 24hrs lags)')
    plt.show()
    print('We see highest correlation at Lag1(1 hour) and Lag24(24 hours).')

    plot_acf(df_input_viz,lags=168)
    plt.title('Correlation of temp across 7 day lag')
    plt.show()
    print('For a lag of 7 days, We see highest correlation at every 24 hours. This indicats a daily cycle.')

    plot_acf(df_input_viz,lags=720)
    plt.title('Correlation of temp across 30 day lag')
    plt.show()
    print('For a lag of 30 days,We see the daily cycle is still present but dips gradually.')

    # Perform time series decomposition
    df_temp = df['temp'].copy()
    df_temp = df_temp['2018-01-01':'2025-09-16']

    # Ensure it's datetime-indexed
    df_temp.index = pd.to_datetime(df_temp.index)
    result = sm.tsa.seasonal_decompose(df_temp, model='additive', period=24)

    # Plot
    plt.figure(figsize=(12, 8))
    result.plot()
    plt.suptitle('RDU Temperature Decomposition (2018–2025)', fontsize=10)
    plt.show()
    print('Trend shows an almost constant yearly trend with slight increase over years.')
    print('Seasonality is constant. Residuals appear to be stationary with mean around 0.')
    # Cross-correlation analysis

    #Windspeed
    print('Windspeed:')
    df_input_viz = df['wspd']
    plot_acf(df_input_viz,lags=24)
    plt.title('Windspeed')
    plt.show()
    print('Windspeed has the highest correlation at lag of 1 hour. Corrlation decreases over time.')

    # DewPoint
    print('DewPoint:')
    df_input_viz = df['dwpt']
    plot_acf(df_input_viz,lags=24)
    plt.title('DewPoint')
    plt.show()
    print('DewPoint has a high correlated with the highest being at lag of 5 hours.')

    #30 day lag
    print('DewPoint:')
    df_input_viz = df['dwpt']
    plot_acf(df_input_viz,lags=720)
    plt.title('DewPoint')
    plt.show()
    print('DewPoint has a high initial correlation (24hours) but decreses over time.')

    # Correlation and Feature creation based on the season
    # ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # add month
    df['month'] = df.index.month

    conditions = [
        df['month'].isin([12, 1, 2]),   # Winter
        df['month'].isin([3, 4, 5]),    # Spring
        df['month'].isin([6, 7, 8]),    # Summer
        df['month'].isin([9, 10, 11])   # Fall
    ]
    choices = ['Winter', 'Spring', 'Summer', 'Fall']

    # set default to a string (not 0)
    df['season'] = np.select(conditions, choices, default='Unknown')

    # convert to categorical for memory/performance
    df['season'] = df['season'].astype('category')

    season_dummies = pd.get_dummies(df['season'], prefix='season')

    # example: plot ACF for each season indicator
    for col in season_dummies.columns:
        plt.figure(figsize=(8,3))
        plot_acf(season_dummies[col], lags=2160)   # or smaller lags if preferred
        plt.title(f'ACF of indicator: {col}')
        plt.show()
    print('Each season lasts about 90 days (3 months), so the correlation gradually transitions from 1 (high at lag 0) to 0 (lag 60days) to negative at lag 90 days as it is the next season.')




def select_features_univariate(X_df_imputed, y_df_imputed, target_col='temp', k=8, verbose=True):
    """
    Run SelectKBest to select the top k (8) features with highest correlation.
    Use these featutes only and drop the rest, to build the training & test data sets.
    Output: final_X_train, final_y_train, final_X_test, final_y_test
    """
    df = X_df_imputed.copy()

    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in dataframe.")

    # 1) Prepare numeric X and y
    numeric_cols = X_df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found for selection after imputation.")

    X_numeric = X_df_imputed[numeric_cols]
    y_train = X_df_imputed[target_col]

    # Fit selector
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_numeric, y_train)

    # scores_ is aligned with X_numeric.columns
    feature_scores = pd.Series(selector.scores_, index=X_numeric.columns)
    support_mask = selector.get_support()
    selected_features = X_numeric.columns[support_mask].tolist()

    if verbose:
        print("\n---- Univariate Feature Scores (f_regression) ----")
        for feat, sc in feature_scores.sort_values(ascending=False).items():
            status = "SELECTED" if feat in selected_features else "rejected"
            print(f"{feat}: {sc:.4f}  {status}")
        print(f"\nSelected features ({len(selected_features)}): {selected_features}")

    final_X_train = X_df_imputed[selected_features]
    final_y_train = X_df_imputed[target_col]
    
    final_X_test = y_df_imputed[selected_features]
    final_y_test = y_df_imputed[target_col]
    
    return final_X_train, final_y_train, final_X_test, final_y_test

def scale_X_fixed(final_X_train, final_y_train, final_X_test, final_y_test):
    """
    Split training data into train & validation sets.
    Scale features. 
    Output: X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val
    """
    # Split ONLY the training data for validation
    split_idx = int(len(final_X_train) * 0.8)
    
    X_train = final_X_train.iloc[:split_idx]
    X_val = final_X_train.iloc[split_idx:]
    y_train = final_y_train.iloc[:split_idx]
    y_val = final_y_train.iloc[split_idx:]
    
    # Fit scaler on ALL training data (not just X_train)
    scaler = StandardScaler()
    scaler.fit(final_X_train) 
    
    # Transform all splits
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(final_X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val


 
def compare_model(X_train_scaled,y_train,X_val_scaled, y_val):
   ''' Compare the performance of the regrssion models using the validation set.
       Calculate error metrics.
       Output: best_model, best_r2, results 
   '''

   best_r2 = -float('inf')  # Start with worst possible score
   best_model = None
   models = {'model_ridge' : Ridge(alpha=1.0),
   'model_lasso' : Lasso(alpha=0.1) ,
   'model_linear': LinearRegression(),
   'model_rforest' : RandomForestRegressor(n_estimators=100, random_state=42) }
   

   results = {}
   for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
        train_preds = model.predict(X_train_scaled)
        val_preds = model.predict(X_val_scaled)
        
        train_mse = mean_squared_error(y_train, train_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        
        train_mae = mean_absolute_error(y_train, train_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        
        train_r2 = r2_score(y_train, train_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        # Store results
        results[name] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
        
        # Print results
        print(f"\n{name}:")
        print(f"  MSE  - Train: {train_mse:.3f}, Val: {val_mse:.3f}")
        print(f"  RMSE - Train: {train_rmse:.3f}, Val: {val_rmse:.3f}")
        print(f"  MAE  - Train: {train_mae:.3f}, Val: {val_mae:.3f}")
        print(f"  R²   - Train: {train_r2:.3f}, Val: {val_r2:.3f}")
        
        # Track best model based on validation R²
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model = (name, model)
   return best_model, best_r2, results   

def test_modelbest(model, X_test, y_test,transform=None):
    ''' Use the best model and evaluate the test set error metrics.
        Output: mse, rmse, mae, rsqr
    '''
    
    preds = model.predict(X_test)
    mse = np.mean((y_test-preds)**2)
    rmse = mse ** 0.5
    mae = np.mean(np.abs(y_test-preds))
    rsqr = 1-np.sum((y_test-preds)**2)/np.sum((y_test-np.mean(y_test))**2)
    
    print(f"Test Set Performance:")
    print(f"  MSE:  {mse:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {rsqr:.3f}")

    print("\nWhat this means:")
    print(f"On average, predictions are off by {mae:.2f}°F (MAE)")
    print(f"Root mean squared error is {rmse:.2f}°F (RMSE)")
    print(f"Model explains {rsqr*100:.1f}% of temperature variance (R²)")
    print(f"Typical prediction error: ±{rmse:.1f} to ±{mae:.1f}°F")
    print("\nPerformance Rating: GOOD - Model captures most temperature patterns")
    
    return mse, rmse, mae, rsqr



def main():
    ''' 
    Run pipeline
    '''

    X, y = download_data()
    X, y, df_concat = prepare_data(X, y)
    df_build = build_features(df_concat, make_datetime_col=False)
    X_df, y_df = split_data(df_build)
    X_df_imputed, y_df_imputed = robust_ffill_impute(X_df,y_df, verbose=True)
    #validate_data(X_df_imputed)
    final_X_train, final_y_train, final_X_test, final_y_test = select_features_univariate(X_df_imputed, y_df_imputed, target_col='temp', k=8, verbose=True)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train,y_val = scale_X_fixed(final_X_train, final_y_train,final_X_test,final_y_test)
    best_model, best_r2, results = compare_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    print()
    print(f"Best model: {best_model[0]}")
    print(f"Best validation R²: {best_r2:.3f}")
    mse, rmse, mae, rsqr = test_modelbest(best_model[1], X_test_scaled, final_y_test)
   

if __name__ == "__main__":
    main()