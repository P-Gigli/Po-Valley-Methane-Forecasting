# Po Valley Methane Forecasting

## Objective

The project studies the performance of different classical machine learning models for short-term forecasting of Sentinel-5P XCH4 observations over the Po Valley, with particular attention to their performance relative to a persistence baseline.

The forecasting task consists of predicting the weekly mean XCH4 value for a given week and geographic cell using information from the previous weeks.

The target quantity, XCH4, is the total column-averaged dry-air mole fraction of methane and is not a direct measure of methane emissions from the ground.

## Dataset

The dataset is a multiyear dataset extracted from the Copernicus Data Space Ecosystem, filtered using a geographic mask and a temporal cell-coverage mask, and subsequently subjected to cleaning and feature engineering.

The geographic mask covers a manually defined region corresponding to the Po Valley area. The temporal cell-coverage mask excludes geographic cells with an insufficient number of observations. Engineered features include temporal features, such as the change in XCH4 over the previous week, and statistical features, such as the mean and standard deviation of XCH4 over recent weeks.

The selected temporal interval covers the years 2019 to 2024, with 2024 reserved for final model testing and evaluation.

## Model-ready dataset

### Predictor and metadata columns

| Column | Data type |
| ------------- | -------------- |
| Week | datetime64[ns] |
| Target_week | datetime64[ns] |
| cell_id | int64 |
| x | float64 |
| y | float64 |
| CH4 | float32 |
| ch4_prev_1w | float32 |
| ch4_prev_2w | float32 |
| ch4_change_1w | float32 |
| ch4_mean_last_3w | float64 |
| ch4_std_last_3w | float64 |
| season_sin | float64 |
| season_cos | float64 |

### Target

| Column | Data type |
| ------------- | -------------- |
| target_next_week | float32 |

## Validation strategy

Three candidate machine learning models are tested and compared with each other and with the persistence baseline over two disjoint annual temporal validation folds, 2022 and 2023. For each fold, the preceding years are used as the training period.

The best-performing model is then trained again over the 2019-2023 period and evaluated on the final 2024 test period.

## Models

- Persistence baseline
- Ridge
- Random Forest with default-like parameters
- Regularized Random Forest with large leaf sizes

## Model selection

A small preliminary experiment on a development subset is first used to select a customized Random Forest configuration with large leaf sizes among several Random Forest parameter settings. This model is then evaluated together with Ridge, the persistence baseline, and a default-like Random Forest.

The temporal validation results show that Ridge is the only candidate model that consistently improves over the persistence baseline across both validation folds. Ridge is therefore selected for the final multiyear train-test experiment.

## Final 2024 evaluation

| Model | MAE | RMSE | R2 | prediction_std | correlation |
|------|---------:|----------:|-------:|-----:|--------:|
| Persistence | 16.220643 | 21.245858 | 0.018069 | 20.769970 | 0.494532 |
| Ridge | 14.749186 | 19.673392 | 0.158041 | 14.876011 | 0.485777 |

Ridge is the best-performing model on the final 2024 test, outperforming the persistence baseline on both MAE and RMSE.

In particular, on the final 2024 test period, Ridge reduces MAE by approximately 9.1% and RMSE by approximately 7.4% relative to the persistence baseline.

## Conclusions

In the final test, Ridge errors are higher than in the validation folds, but the model retains a modest out-of-sample advantage over the persistence baseline despite substantial week-to-week variability in the XCH4 observations.

The results suggest that a regularized linear combination of recent XCH4 history, spatial information, and seasonal features contains useful predictive information beyond the current-week observation.

This predictive relationship should not be interpreted as evidence of a causal or intrinsically linear physical dependence.
