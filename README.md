# AIPI 520 Weather Prediction Project

Weather prediction for RDU airport for AIPI 520 Project 1

## Installation

### Requirements

Python 3.9 or newer, Git, an internet connection for the initial data download

### Instructions

Create a virtual environment and install the dependencies. Then, run the `main.py` file.

```
git clone https://github.com/sam-packer/AIPI520_Weather_Prediction

cd AIPI520_Weather_Prediction

python3 -m venv .venv         # Create the virtual environment
source .venv/bin/activate     # macOS/Linux
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

## Data Structure

We are using Meteostat to fetch hourly time series data for the RDU Airport weather station.

The data structure within the CSV files is as follows:

| Column  | Description                                                                                     | Type       |
|---------|-------------------------------------------------------------------------------------------------|------------|
| station | The Meteostat ID of the weather station (only if query refers to multiple stations)             | String     |
| time    | The datetime of the observation                                                                 | Datetime64 |
| temp    | The air temperature in °C                                                                       | Float64    |
| dwpt    | The dew point in °C                                                                             | Float64    |
| rhum    | The relative humidity in percent (%)                                                            | Float64    |
| prcp    | The one hour precipitation total in mm                                                          | Float64    |
| snow    | The snow depth in mm                                                                            | Float64    |
| wdir    | The average wind direction in degrees (°)                                                       | Float64    |
| wspd    | The average wind speed in km/h                                                                  | Float64    |
| wpgt	   | The peak wind gust in km/h                                                                      | 	Float64   |
| pres	   | The average sea-level air pressure in hPa                                                       | 	Float64   |
| tsun    | The one hour sunshine total in minutes (m)                                                      | 	Float64   |
| coco    | The [weather condition](https://dev.meteostat.net/formats.html#meteorological-data-units) code	 | Float64    |

## Data Considerations

### Snowfall

The obvious first thing to look at is missing values. We see that for snow, this dataset reports none. I (Sam) am new to
RDU and not familiar with the historical weather since 2018. According to some research from The Chronicle, we find the
following:

- https://www.dukechronicle.com/article/2025/01/duke-university-winter-snow-warnings-national-weather-service-durham-central-north-carolina-duke-activated-severe-weather-emergency-conditions-polcy-health-system-bus-routes-classes-campus-services-jobs-work
- https://dukechronicle.com/article/duke-university-snow-day-scenes-chapel-gardens-sledding-building-snowmen-snow-angels-snowball-fights-first-snowfall-in-three-years-campus-gothic-20250112

It did snow in Durham on January 11, 2025. We do find out that it was also the first measurable snow in Durham in over
1,000 days. The Chronicle even nicely tells us:
> Though Friday’s blanket didn’t quite amount to the anticipated three inches, it was the first time many Blue Devils
> saw their Gothic campus dressed in white.
>
> -- Duke Chronicle, January 11, 2025

Thank you, Duke Chronicle staff, very cool! However, Meteostat reports no snow depth. My hypothesis is that since this
is a weather station at an airport, any snow on the ground would be cleared quickly, or else flights get delayed or
cancelled.

From this snippet of data, we do see that from the weather code that it reports snowfall, but there is no ground amount.

```
time,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco
2025-01-11 06:00:00,-0.6,-2.8,85.0,1.5,,10.0,5.4,,1007.2,,15.0
2025-01-11 07:00:00,0.0,-2.2,85.0,3.3,,10.0,5.4,,1005.8,,15.0
2025-01-11 08:00:00,0.0,-2.2,85.0,1.0,,170.0,5.4,,1004.5,,15.0
```

There are two likely solutions: the weather station is not capable of reporting snow depth, or any snow is cleared too
quickly because this weather station is near an airport. Since snow depth is underreported / non-existent, we cannot use
it as a feature here. This will likely be okay though, since according to The Chronicle, snow in this area is rare.

### Total Sunshine

The `tsun` feature is also blank. Contrary to popular belief, this does *not* report tsunamis. This is actually the
total amount of sunshine for the hour. The weather station here likely just doesn't capture it.

### Wind Speed and Wind Gusts

This one is interesting because it is sparse. A lot of the time, it is just NaN. However, when it does exist, it seems
to be quite higher than the regular wind. It's likely there is a sensitivity setting on the weather station that does
not report wind gusts unless it's sustained and much higher than the regular wind speed. This might actually be worth
capturing. We have a few options: filling it or turning it into a binary feature. If we fill the gust with the wind
speed, it would teach the model about steadiness. We could also simply say that if there was a gust, then the
`gust_flag` is 1, otherwise it's 0. This assigns it meaning in that the hour was more windy than usual.

### Precipitation Total

Sometimes, the precipitation total (in mm) for the hour is NaN. It also seems mixed: sometimes it is NaN when there is
no precipitation, other times it is NaN when there was precipitation reported.

Here's an example of the precipitation having missing values:

```
time,temp,dwpt,rhum,prcp,snow,wdir,wspd,wpgt,pres,tsun,coco
2019-04-05 10:00:00,13.3,8.4,72.0,0.0,,110.0,14.8,,1022.4,,4.0
2019-04-05 11:00:00,13.3,9.0,75.0,0.0,,120.0,13.0,,1022.8,,7.0
2019-04-05 12:00:00,13.9,9.3,74.0,,,130.0,5.4,,1023.2,,7.0
2019-04-05 13:00:00,12.8,11.0,89.0,1.3,,110.0,9.4,,1024.1,,7.0
2019-04-05 14:00:00,12.8,11.7,93.0,2.3,,100.0,14.8,,1023.9,,7.0
2019-04-05 15:00:00,12.8,11.7,93.0,1.0,,110.0,20.5,,1022.3,,8.0
2019-04-05 16:00:00,14.4,12.3,87.0,,,120.0,22.3,,1021.9,,8.0
2019-04-05 17:00:00,14.4,12.3,87.0,0.5,,110.0,9.4,,1021.5,,7.0
2019-04-05 18:00:00,14.4,12.8,90.0,3.0,,80.0,13.0,,1021.4,,7.0
2019-04-05 19:00:00,14.4,13.9,97.0,2.8,,100.0,5.4,,1021.0,,7.0
2019-04-05 20:00:00,14.4,13.3,93.0,0.8,,100.0,14.8,,1019.9,,8.0
2019-04-05 21:00:00,13.9,13.3,96.0,0.5,,100.0,22.3,,1018.6,,5.0
2019-04-05 22:00:00,13.9,12.8,93.0,,,80.0,11.2,,1019.0,,8.0
2019-04-05 23:00:00,13.9,13.3,96.0,0.3,,10.0,5.4,,1019.5,,8.0
2019-04-06 00:00:00,13.3,13.3,100.0,0.3,,30.0,9.4,,1019.8,,7.0
2019-04-06 01:00:00,13.3,12.8,97.0,0.5,,40.0,9.4,,1020.3,,5.0
2019-04-06 02:00:00,13.3,12.8,97.0,,,40.0,11.2,,1020.6,,3.0
2019-04-06 03:00:00,13.3,12.8,97.0,0.5,,40.0,9.4,,1020.6,,2.0
2019-04-06 04:00:00,12.8,12.8,100.0,,,20.0,9.4,,1020.7,,0.0
```

What we should likely do is use other signals (such as humidity and the weather code) to determine if we should be
filling it in with the average, if it's just missing sensor data when it's sunny, or if there's missing sensor data, but
it stopped raining. We wouldn't want to accidentally extend a rainy period longer than it really was. We also wouldn't
want a jump from "rain" to "no rain" when it really was raining.

### Weather codes

One of the most critical features is our weather code. They allow us to encode categorical events that may not fully be
able to be captured by numerical features. However, it is mostly missing prior to July 20, 2018. After July 2018, we get
a consistent weather code. This is really only a problem if we decide to look at even older historical weather records.
If we do, then we will have to choose how we handle this. We could build a simple heuristic, but this may be inaccurate
and lead to poor model performance as it's learning from inaccurate data. We also could actually try and use a model to
predict the weather code. However, this sounds kind of insane given that we'd be making a model to predict the weather
code just so we could make another model to predict the temperature. The simplest solution would be to add a flag that
is 1 if the weather code is missing, and 0 if it is not. However, these are only considerations if we need data older
than July 20, 2018.