from datetime import datetime
from meteostat import Hourly
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def download_data():
    # Let us be good people and not hammer their API if we already have the data
    if not os.path.exists("rdu_weather_data.csv") or not os.path.exists("rdu_weather_predict.csv"):
        # TODO: Figure out how much data we should actually use.
        # Linear regression would actually probably benefit from learning from less historical data
        # A more advanced model might not
        X_start = datetime(2018, 7, 20)
        X_end = datetime(2025, 9, 16, 23, 59)
        y_start = datetime(2025, 9, 17)
        y_end = datetime(2025, 9, 30, 23, 59)

        print("Downloading RDU weather data, please wait...")
        X = Hourly('72306', X_start, X_end)
        X = X.fetch()
        y = Hourly('72306', y_start, y_end)
        y = y.fetch()
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
    print("Preparing the data...")
    for df in [X, y]:
        # Meteostat is really sneaky and actually uses UTC by default...
        # This converts it to Eastern Time. Good thing I read the documentation before doing anything else. Ha ha.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        # We'll validate the data very early on to ensure there's no impossible values
        # AI Disclosure: ChatGPT was used to help make this final code work
        rules = {'temp': (-40, 50),
                 'rhum': (0, 100),
                 'wspd': (0, None),
                 'wpgt': (0, None),
                 'wdir': (0, 360),
                 'prcp': (0, None)}
        violations = {}
        for col, (low, high) in rules.items():
            mask = pd.Series(False, index=df.index)
            if low is not None:
                mask |= df[col] < low
            if high is not None:
                mask |= df[col] > high
            if mask.any():
                violations[col] = df[mask]
        print("Looking for impossible values:", violations)

    # Make sure we don't accidentally fetch the same dates for training and prediction data
    assert X.index.max() < y.index.min(), "Training and prediction periods overlap!"
    print("Done with data preparation.")
    return X, y


def impute_data(X, y):
    print("Imputing data...")

    def _clean(df):
        df = df.copy()
        # Drop snow and tsun. They have no data.
        df = df.drop(columns=['snow', 'tsun'])
        # Turn wind gust into a flag rather than a value since it's missing most of the time
        df['gust_flag'] = df['wpgt'].notna().astype(int)
        # Fill the gusts with regular speed so it learns a baseline
        df['wpgt'] = df['wpgt'].fillna(df['wspd'])
        # Forward fill. Take the last known value and fill it in when it's missing
        df['prcp'] = df['prcp'].ffill().fillna(0)
        # Linearly interpolate pressure since less than 1% is even missing and it's continuous
        df['pres'] = df['pres'].interpolate().ffill().bfill()
        # We're only training on July 20, 2018+, so in theory the weather code should always exist
        df['coco_missing'] = df['coco'].isna().astype(int)
        # Things never go the way you want them to though...
        df['coco'] = df['coco'].fillna(-1)
        # See the Wind Direction under the README for this. The code below is doing what's stated in the README
        # AI Disclosure: ChatGPT was used to produce the imputation code below for wind speed
        df['wdir'] = np.degrees(np.angle(np.interp(df.index, df.index[~df['wdir'].isna()],
                                                   np.exp(1j * np.radians(df['wdir'][~df['wdir'].isna()]))))) % 360
        return df

    X = _clean(X)
    y = _clean(y)
    print("Done. Verification there are no missing values:")
    X_missing_summary = X.isna().mean().sort_values(ascending=False)
    y_missing_summary = y.isna().mean().sort_values(ascending=False)
    print(X_missing_summary)
    print(y_missing_summary)
    # AI Disclosure: ChatGPT helped me figure out you need sum() here twice for this test
    assert X.isna().sum().sum() == 0, "X still has missing values after imputation!"
    assert y.isna().sum().sum() == 0, "y still has missing values after imputation!"
    return X, y


def split_data(X):
    print("Creating train / test data...")
    cutoff_date = X.index.max() - pd.Timedelta(days=90)
    X_train = X[X.index <= cutoff_date]
    X_val = X[X.index > cutoff_date]

    print(f"Train:\t\t {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} rows)")
    print(f"Validation:\t {X_val.index.min()} to {X_val.index.max()} ({len(X_val)} rows)")
    return X_train, X_val


def feature_engineering(X_train, X_test, y):
    print("Engineering features...")

    def _engineer(df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        # Transform wind direction to sin/cos to handle cyclical nature (0° = 360°)
        df['wdir_sin'] = np.sin(np.radians(df['wdir']))
        df['wdir_cos'] = np.cos(np.radians(df['wdir']))
        df = df.dropna().copy()
        return df

    X_train = _engineer(X_train)
    X_test = _engineer(X_test)
    y = _engineer(y)

    numeric_features = ["rhum", "wspd", "wpgt", "prcp", "pres", "wdir_sin", "wdir_cos"]
    categorical_features = ["month", "hour", "day_of_week", "coco"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("col", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='drop'
    )
    print("Done engineering features. Hopefully they're good.")
    return X_train, X_test, y, preprocessor


def train_and_select_models(models, X_train, X_val, preprocessor):
    print("Training models...")

    results = {}

    y_train = X_train['temp']
    X_train = X_train.drop(columns="temp")
    y_val = X_val['temp']
    X_val = X_val.drop(columns="temp")

    for name, model in models.items():
        print("Training", name)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # print(pipeline.named_steps["preprocessor"].get_feature_names_out(X_train.columns))

        val_prede = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, val_prede)
        r2 = r2_score(y_val, val_prede)

        results[name] = {"val_mae": mae, "val_r2": r2, "model": model}
        print(f"{name} model performance:")
        print(f"MAE on validation set: {mae:.2f}")
        print(f"R2 on validation set: {r2:.3f}")

    return results


def evaluate_final_models(results, X_train, X_val, y, preprocessor):
    X_full = pd.concat([X_train, X_val]).sort_index()
    y_full = X_full['temp']
    X_full = X_full.drop(columns="temp")

    final_results = {}
    # Select the top two models for the assignment
    # AI Disclosure: ChatGPT helped write the code to sort the models from the dictionary
    sorted_models = sorted(results.items(), key=lambda kv: kv[1]["val_mae"])
    top_two = sorted_models[:2]

    print("\nTop 2 models based on validation performance:")
    for i, (name, info) in enumerate(top_two, 1):
        print(f"{i}. {name} (Validation MAE: {info['val_mae']:.2f})")

    y_forecast = y.copy()
    y_forecast['hour'] = y_forecast.index.hour
    y_forecast['month'] = y_forecast.index.month
    y_forecast['day_of_week'] = y_forecast.index.dayofweek
    # Transform wind direction to sin/cos
    y_forecast['wdir_sin'] = np.sin(np.radians(y_forecast['wdir']))
    y_forecast['wdir_cos'] = np.cos(np.radians(y_forecast['wdir']))

    for name, info in top_two:
        print(f"\nDoing final training on {name} using tested model and final 2 week prediction")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", info['model'])
        ])
        pipeline.fit(X_full, y_full)
        preds = []
        rolling_window = X_full.copy()

        # AI Disclosure: ChatGPT wrote the sliding window technique used on the final test set
        for timestamp, row in y_forecast.iterrows():
            features = row.drop(labels=['temp']) if "temp" in row else row
            features = features.to_frame().T

            y_pred = pipeline.predict(features)[0]
            preds.append(y_pred)

            new_row = row.copy()
            new_row["temp"] = y_pred
            rolling_window.loc[timestamp] = new_row

        y_forecast['predicted_temp'] = preds

        common_idx = y_forecast['temp'].notna()
        mae = mean_absolute_error(y_forecast.loc[common_idx, "temp"],
                                 y_forecast.loc[common_idx, "predicted_temp"])
        r2 = r2_score(y_forecast.loc[common_idx, "temp"],
                      y_forecast.loc[common_idx, "predicted_temp"])

        final_results[name] = {"test_mae": mae, "test_r2": r2}
        print(f"{name} model performance:")
        print(f"MAE average on test set: {mae:.2f}")
        print(f"R2 average on test: {r2:.3f}")

        y_forecast.to_csv(f"forecast_{name.replace(' ', '_').lower()}.csv")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Val MAE':>10} {'Test MAE':>10} {'Test R2':>10}")
    print("-"*60)
    for name in [m[0] for m in top_two]:
        val_mae = results[name]['val_mae']
        test_mae = final_results[name]['test_mae']
        test_r2 = final_results[name]['test_r2']
        print(f"{name:<25} {val_mae:>10.2f} {test_mae:>10.2f} {test_r2:>10.3f}")
    print("="*60)

    return final_results


def main():
    X, y = download_data()
    X, y = prepare_data(X, y)
    X, y = impute_data(X, y)
    X_train, X_val = split_data(X)
    X_train, X_val, y, preprocessor = feature_engineering(X_train, X_val, y)
    random_seed = 67
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=random_seed),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=random_seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=random_seed)
    }

    model_results = train_and_select_models(models, X_train, X_val, preprocessor)
    final_results = evaluate_final_models(model_results, X_train, X_val, y, preprocessor)
    
    # generate visualizations
    from visualizations import generate_all_plots
    generate_all_plots()


if __name__ == "__main__":
    main()

