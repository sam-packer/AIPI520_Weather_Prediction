from datetime import datetime
from meteostat import Hourly
import os
import pandas as pd

def download_data():
    # Let us be good people and not hammer their API if we already have the data
    if not os.path.exists("rdu_weather_data.csv") or not os.path.exists("rdu_weather_predict.csv"):
        # TODO: Figure out how much data we should actually use.
        # Linear regression would actually probably benefit from learning from less historical data
        # A more advanced model might not
        X_start = datetime(2018, 1, 1)
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
    for df in [X, y]:
        # Meteostat is really sneaky and actually uses UTC by default...
        # This converts it to Eastern Time. Good thing I read the documentation before doing anything else. Ha ha.
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

    return X, y

def main():
    X, y = download_data()
    X, y = prepare_data(X, y)



if __name__ == "__main__":
    main()