## Insights from EDA

Top correlations by lag:
- 1h: 0.990
- 2h: 0.969
- 3h: 0.939
- 24h: 0.889
- 6h: 0.831
- 48h: 0.808
- 72h: 0.777
- 96h: 0.763
- 120h: 0.754
- 144h: 0.748

Our chosen lags performance:
- Daily (24h): 0.889 (rank 4)
- Two-day (48h): 0.808 (rank 6)
- Weekly (168h): 0.746 (rank 11)

Implications for Feature Engineering
- Multicollinearity risk: The extremely high correlations at 1-3 hour lags (0.990-0.939) would create severe multicollinearity in linear models like Ridge regression
- Interpretability: While 1 hour lags are strong, "temperature from one hour ago" provides limited actual insights compared to "yesterday's temperature at this same hour"


---

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
