from datetime import datetime
from meteostat import Hourly
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
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
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

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


def add_temporal_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month

    # Lags and rolling computed BEFORE split
    for lag in [1, 3, 6, 12, 24]:
        df[f"temp_lag_{lag}"] = df["temp"].shift(lag)
    df["temp_rolling_6h"] = df["temp"].rolling(window=6).mean()

    # Drop NaNs created by lag/rolling
    df = df.dropna().copy()
    return df


def split_data(X):
    print("Creating train / test data...")
    cutoff_date = X.index.max() - pd.Timedelta(days=367)
    X_train = X[X.index <= cutoff_date]
    X_test = X[X.index > cutoff_date]

    print(f"Train:\t {X_train.index.min()} to {X_train.index.max()} ({len(X_train)} rows)")
    print(f"Test:\t {X_test.index.min()} to {X_test.index.max()} ({len(X_test)} rows)")
    return X_train, X_test


def feature_engineering(X_train, X_test, y):
    print("Engineering features...")

    def _engineer(df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        for lag in [1, 3, 6, 12, 24]:
            df[f'temp_lag_{lag}'] = df['temp'].shift(lag)
            df[f'temp_lag_{lag}'] = df['temp'].shift(lag)

        df['temp_rolling_6h'] = df['temp'].rolling(window=6).mean()
        df = df.dropna().copy()
        return df

    X_train = _engineer(X_train)
    X_test = _engineer(X_test)
    y = _engineer(y)

    numeric_features = ["rhum", "wspd", "wpgt", "prcp", "pres", "temp_rolling_6h"]
    categorical_features = ["month", "hour", "coco"]
    for lag in [1, 3, 6, 12, 24]:
        numeric_features.append(f"temp_lag_{lag}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("col", OneHotEncoder(), categorical_features)
        ]
    )
    print("Done engineering features. Hopefully they're good.")
    return X_train, X_test, y, preprocessor


def train_model(models, X_train, X_test, y, preprocessor):
    print("Training models...")
    results = {}

    y_train = X_train['temp']
    y_test = X_test['temp']
    X_train = X_train.drop(columns="temp")
    X_test = X_test.drop(columns="temp")

    for name, model in models.items():
        print("Training", name)

        print("Columns being passed to preprocessor:", X_train.columns.tolist())

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # print(pipeline.named_steps["preprocessor"].get_feature_names_out(X_train.columns))

        test_preds = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        results[name] = {"MSE:": mse, "R2:": r2}
        print(f"{name} model performance:")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")

    return results


def main():
    X, y = download_data()
    X, y = prepare_data(X, y)
    X, y = impute_data(X, y)
    X_train, X_test = split_data(X)
    X_train, X_test, y, preprocessor = feature_engineering(X_train, X_test, y)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=20, n_jobs=-1)
    }

    results = train_model(models, X_train, X_test, y, preprocessor)


if __name__ == "__main__":
    main()
