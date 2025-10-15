from datetime import datetime
from meteostat import Point, Hourly
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# 1) Download
# ---------------------------------------------------------------------------

def download_data():
    # Avoid hammering the Meteostat API if we already cached the CSV
    if not os.path.exists("rdu_weather_data.csv"):
        start = datetime(2022, 1, 1)
        end = datetime(2025, 9, 30, 23, 59)

        print("Fetching RDU weather data...")
        rdu = Point(35.8776, -78.7875, 132)
        df = Hourly(rdu, start, end).fetch()
        if df.empty:
            raise ValueError("Meteostat returned empty data. Check the station ID or date range.")

        df.to_csv("rdu_weather_data.csv", index=True)
        print(f"Downloaded {len(df)} hourly rows.")
        return df

    print("Using cached weather data.")
    return pd.read_csv("rdu_weather_data.csv", index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# 2) Prepare
# ---------------------------------------------------------------------------

_EXPECTED_COLS = [
    "temp", "dwpt", "rhum", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun", "coco"
]

def _ensure_expected_columns(df):
    # Ensure the frame always has all the expected Meteostat columns
    df = df.copy()
    for c in _EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[_EXPECTED_COLS]

def prepare_data(df):
    print("Tidying timestamps and sorting...")
    # Meteostat gives UTC — convert to Eastern Time for RDU
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC").tz_convert("America/New_York")
    df.sort_index(inplace=True)
    df = _ensure_expected_columns(df)
    print("Data shape:", df.shape)
    return df


# ---------------------------------------------------------------------------
# 3) Impute / Clean
# ---------------------------------------------------------------------------

def impute_data(df):
    print("Cleaning and filling missing values...")
    df = df.copy()

    # Drop these columns if present — they’re mostly empty
    for col in ("snow", "tsun"):
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Wind gust handling
    if "wpgt" in df.columns:
        df["gust_flag"] = df["wpgt"].notna().astype(int)
        if "wspd" in df.columns:
            df["wpgt"] = df["wpgt"].fillna(df["wspd"])
    else:
        df["gust_flag"] = 0

    # Fill precipitation and pressure sensibly
    df["prcp"] = df["prcp"].ffill().fillna(0)
    df["pres"] = df["pres"].interpolate().ffill().bfill()

    # Handle weather condition code
    df["coco_missing"] = df["coco"].isna().astype(int)
    df["coco"] = df["coco"].fillna(-1)

    # Wind direction interpolation (circular)
    if "wdir" in df.columns:
        s = df["wdir"]
        mask = s.notna()
        if mask.any():
            t_all = df.index.view("int64")
            t_known = df.index[mask].view("int64")
            vals = np.exp(1j * np.deg2rad(s[mask].to_numpy()))
            interp_real = np.interp(t_all, t_known, np.real(vals))
            interp_imag = np.interp(t_all, t_known, np.imag(vals))
            ang = np.angle(interp_real + 1j * interp_imag)
            df["wdir"] = (np.degrees(ang) + 360.0) % 360.0

    # Fill anything left over
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    print("Imputation done.")
    return df


# ---------------------------------------------------------------------------
# 4) Lag Features
# ---------------------------------------------------------------------------

def add_lag_features(df, lags=[1, 3, 6, 12, 24]):
    # Adds time-based lag columns for temperature
    df = df.copy()
    for lag in lags:
        df[f"temp_lag_{lag}h"] = df["temp"].shift(lag)

    # Time-of-day and seasonality helpers
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek

    # One-hour-ahead temperature target
    df["temp_target"] = df["temp"].shift(-1)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# 5) Split
# ---------------------------------------------------------------------------

def split_data(df):
    print("Splitting into train/test (chronologically)...")
    cutoff = df.index.max() - pd.Timedelta(days=30)
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]

    if train.empty or test.empty:
        raise ValueError("Split failed — not enough data for both sets.")

    print(f"Train rows: {len(train)} | Test rows: {len(test)}")
    return train, test


# ---------------------------------------------------------------------------
# 6) Train & Evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(models, train, test):
    print("Training models...")

    # Build lags inside each split to avoid leakage
    train = add_lag_features(train)
    test = add_lag_features(test)

    y_train = train["temp_target"]
    X_train = train.drop(columns=["temp_target"])
    y_test = test["temp_target"]
    X_test = test.drop(columns=["temp_target"])

    numeric_features = [
        c for c in X_train.columns if any(k in c for k in ["rhum", "wspd", "wpgt", "prcp", "pres", "temp_lag"])
    ]
    categorical_features = [c for c in ["month", "hour", "day_of_week", "coco"] if c in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    results = {}
    for name, model in models.items():
        print(f"→ {name}")
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {"test_mae": mae, "test_r2": r2}
        print(f"   MAE={mae:.3f}, R²={r2:.3f}")

    return results


# ---------------------------------------------------------------------------
# 7) Main
# ---------------------------------------------------------------------------

def main():
    df = download_data()
    df = prepare_data(df)
    df = impute_data(df)

    train, test = split_data(df)

    seed = 67
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed
        ),
    }

    results = train_and_evaluate(models, train, test)
    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()