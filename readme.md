# Sales Prediction Forecasting Project

## Overview

## Features
- **Dynamic Configuration**: All settings are managed through a `settings.toml` file, making the project flexible and easy to configure.
- **Robust Logging**: Each script logs its progress and errors, aiding in debugging and traceability.
- **Efficient Resource Management**: Includes utilities for reducing memory usage and leveraging parallel processing.
- **Modularity**: Refactored into clean, reusable scripts for preprocessing, training, prediction, and ensembling.

## Directory Structure
```
project/
|-- data/
|   |-- raw/           # Raw input data
|   |-- processed/     # Processed data
|   |-- system/        # System files:  Globals, hyperparameters
|
|-- data/
|
|-- jupyter/           # Jupyter notebooks to work with predictions
|
|-- models/            # Trained lgbm models
|
|-- scripts/           # Main scripts for preprocessing, training, prediction, and ensembling
|   |-- gen_sales.py
|   |-- preprocessing.py
|   |-- training.py
|   |-- prediction.py
|   |-- ensemble.py
|
|-- utils/
    |-- utilities.py   # Utility functions for memory reduction and logging setup
```

## Scripts
(generally process them in this order)

### 1. `gen_sales.py`
Comprehensive data simulation.  Creates a data file called generated_sales.csv
- Time range definition
- Seasonal item configuration
- Noise generation
- Trend application
- Item turnover

### 2. `preprocessing.py`
Handles preprocessing of raw data (generated_sales.csv):
- Reduces memory usage using `reduce_mem_usage`.
- Logs progress and saves processed data to the `processed/` folder.
- Files that are output of the script:
- Historical sales and features (grid_part_1.pkl, grid_part_2.pkl, grid_part_3.pkl). 
- Lagged and rolling features (lags_df_28.pkl). 
- Mean and standard deviation encodings for categorical groups (mean_encoding_df.pkl).

### 3. `lgb_train_cluster.py`
Implements recursive and non-recursive light gradiant boost model training:
- Supports parallel training across multiple stores.
- Saves trained models to the `models/` folder.

### 4. `prophet_train_pred.py`
Implements recursive and non-recursive model training and predictions
- Uses /data/processed/pre_processed_data_training.csv
- generates /data/processed/prophet_predictions.csv

### 5. `lgbm_prediction.py`
Generates light gradiant boost predictions using trained models:
- Uses /data/processed/pre_processed_data_training.csv
- generates /data/processed/lgbm_predictions.csv

## Jupyter Notebooks to work with predictions
### 1. merge_predictions.ipynb
Merges the prophet prediction (yhat) into the lgbm prediction file

- Uses:
  - /data/processed/lgbm_predictions.csv
  - /data/processed/prophet_predictions.csv
- generates /data/processed/merged_predictions.csv

## License
This project is for educational and non-commercial use only.

python -m Orange.canvas
