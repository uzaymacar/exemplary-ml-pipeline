# Exemplary Machine Learning Pipeline

## Introduction
This repository aims to act as an exemplary data science & machine learning pipeline to any tabular data problem. Moreover, the notebooks aim to explore two Python packages for machine learning *automation*: `featuretools` and `h2o`. Whereas `featuretools` specializes in *feature engineering*, `h2o`specializes in *modelling*. On a broader sense, here is what we cover:
* Data Insights & Visualizations
* Data Cleaning
* Data Imputation
* Manual Feature Engineering
* Automated Feature Engineering via `featuretools`
* Feature Scaling
* Feature Selection
* Feature Encoding
* Modelling (Model Selection & Analysis) via `h2o`

There are two main arguments we can make:
1. Currently, there is a huge gap between what we call *automated machine learning* and the actual machine learning workflow we have to create in order to solve a real data problem. This is a recurring theme in all notebooks as we had to try to impute missing values, apply feature selection, and much more in order to increase our prediction score.
2. The existing gap is based on implementations, rather than *theory*. In other words, there is a great literature (papers, workshops, experiments, examples, notebooks, etc.) that has evolved around the missing points in this gap. The notebooks make the appropriate references. Essentially, the hard parts are covered by packages such as `h2o` and `featuretools`, but the easier parts are not addressed in terms of *automation*. Notice the word *automation* here, otherwise `sklearn` already has somewhat complete implementations related to the missing points mentioned in this repository. 

## Getting Started
Download the data folders with prepared training and testing data files (.csv) from [here](https://drive.google.com/drive/folders/1UmCDCZPtmsXn5kkO9hkI_PFHgq-UHqWm?usp=sharing) and replace them with their name-wise match in this repository. Or alternatively, you can only download `(0)data/` (which you can also get from [here](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)) and run the Jupyter notebooks to generate rest of the data yourself.

## Results & Comparisons
All of the below models are trained and validated by `h2o`'s `H2OAutoML` module, but the operations applied to the data before the modelling process differs for each row. For fairness of comparison, all models are trained under the time limit of *10000* seconds and with similar parameters.

| Data Directory | Data & Operations Description | Num Features | Best Model | Prediction Score (Accuracy) |
| -------------- | ----------------------------- | ------------ | ---------- | --------------------------- | 
| `(0)data` | Untouched files extracted from [Kaggle](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings) | 13 | Stacked Ensemble | 0.56189 |
| `(1)data_manual_ops` | Applied data imputation, removed nonsensical (outlier-like) values from 'age' column, and included a new feature engineered column by linking `train_users.csv` and `age_gender_bkts.csv` | 14 | Stacked Ensemble | 0.62540 |
| `(2)data_automated_ops` | Applied automated feature engineering via `featuretools` and by linking `train_users.csv` with `sessions.csv` and `age_gender_bkts.csv`. | 137 | XGBoost | 0.68853 |
| `(3)data_trimmed/raw` (^)| Applied manual feature scaling based on normal distribution for numerical variables and applied a comprehensive feature selection.  | 39 | XGBoost | 0.71580 |
| `(3)data_trimmed/raw` | Same operations and data as (^), but applied undersampling to majority classes via `h2o`. | 39 | XGBoost | 0.71440 |
| `(3)data_trimmed/label_encoded` | Same operations and data as (^), but applied label encoding to all categorical variables. Hence, all variables are numeric in the end. | 39 | Stacked Ensemble | **0.72101** |

## Future Work
* Check **Driverless AI Platform**.
* Look into more parameters of `H2OAutoML` module, and particularly try increasing the value of parameter `@max_runtime_secs` for longer training duration and hopefully better prediction scores.
* Produce more self-encoded data.

Check to see if these increase prediction scores in any way.
