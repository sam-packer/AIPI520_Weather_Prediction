## Run 2

- **Period**: Sep 17–30, 2025
- **Baseline** — MAE: 2.00°C, RMSE: 2.55°C, R²: 0.655
- **Ridge** — MAE: 1.96°C, RMSE: 2.42°C, R²: 0.690  
- **LightGBM** — MAE: 1.37°C, RMSE: 1.82°C, R²: 0.825
- **Best model**: LightGBM

> Fixed lag feature implementation to use training data for validation set lags, and training+validation data for test set lags. Now we can predict the full 14-day period instead of losing the first week to NaN values. However, these scores are suspiciously good for real forecasting since we're still using actual historical temperatures for all lag features rather than our own predictions.

---

## Run 1

- **Period**: Sep 17–30, 2025  
- **Baseline** — MAE: 2.01°C, RMSE: 2.45°C, R²: 0.575
- **Ridge** — MAE: 1.77°C, RMSE: 2.19°C, R²: 0.661
- **LightGBM** — MAE: 1.15°C, RMSE: 1.46°C, R²: 0.849
- **Best model**: LightGBM

> Only able to predict the second week (7 days) due to NaN issues from creating lag features separately on each split. Each split was dropping exactly 168 rows (7 days) from the start, which meant we were only predicting days 8-14 instead of the full 1-14 day period.
