import logging
import configparser
import inspect
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from colorlog import exception

####################################
# CONSTANTS
####################################
# Invoice	StockCode	Description	Quantity	InvoiceDate	Price	Customer ID	Country

SCALE_TARGET = False
DATE_COLUMN = 'date'
EPOCH_COLUMN = 'epoch_week'
TARGET_COLUMN = 'sales'
ITEM_COLUMN = 'item_id'
CLASS_COLUMN = 'cls_id'
SUBDEPT_COLUMN = 'sbd_id'
DEPT_COLUMN = 'dept_id'
DIV_COLUMN = 'div_id'

BASE_DATE = pd.Timestamp("1970-01-01") # used for epoch week calculation

PROJ_ID = 'PREDICT_SALES_v1'
ROOT_DIR = f'/home/py/data/{PROJ_ID}/'
DATA_DIR = f'{ROOT_DIR}data/'
LOG_DIR = f'{ROOT_DIR}logs/'
MODEL_DIR = f'{ROOT_DIR}models/'
SCRIPTS_DIR = f'{ROOT_DIR}scripts/'
SYSTEM_DIR = f'{ROOT_DIR}data/system/'
RAW_DIR = f'{ROOT_DIR}data/raw/'
PROCESSED_DIR = f'{ROOT_DIR}data/processed/'

config_file = f'{SYSTEM_DIR}config.ini'
LGB_PARAM_FILE = f'{SYSTEM_DIR}lgb_params.conf'
model_training_columns_file = f'{SYSTEM_DIR}lgb_feature_columns.json'

REMOVE_4_TRAINING = ['item_id', 'cls_id', 'sbd_id', 'dept_id', 'div_id',
                     'epoch_week', 'date','woy', 'moy', 'qoy', 'soy', 'year',
                     'copy_sales', 'partition'
                     #'epoch_week_sin', 'epoch_week_cos'
                     ]
CLUSTER_MODELS = ["cluster_kmeans_fs", "cluster_hac_s", "cluster_seasonal_fs", "cluster_volume_fs"]  # Cluster categories
ALL_CLUSTER_FIELDS = ["cluster_kmeans_fs", "cluster_hac_s", "cluster_seasonal_fs", "cluster_volume_fs", "cluster_gaussian_f","cluster_trend_f"]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
####################################
# ENUMS
####################################
SELL_PATTERN_ENUM = {
    "New": 1,
    "RampingUp": 2,
    "RampingDown": 3,
    "Erratic": 4,
    "Basic": 5
}

####################################
# setup the logging information
# log_name:  the name of the log file.  will be in the logs dir
####################################
def setup_logging(log_name: str):
    logger = logging.getLogger(__name__)
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    log_file = f'/home/py/data/{PROJ_ID}/logs/{log_name}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logger

####################################
# this_df:  dataframe to determine the train/test/eval periods
# is_for_train:  True/False  split for training (else is for tuning)
# returns the: first/last train,test,and validation epoch periods
# data:     ----------------------------------
# train:    {train---------------}{test------}
# tune:     {train-------}{valid-------}{test}
####################################
def get_break_epoch_periods(this_df,is_for_train):
    # Splitting data into training, testing, and validation sets based on weeks
    epoch_start = this_df['epoch_week'].min()
    epoch_end = this_df['epoch_week'].max()
    total_weeks = epoch_end - epoch_start

    if is_for_train:
        train_pct = 0.80
        train_end_pd = int(total_weeks * train_pct) + epoch_start
        valid_start_pd = 0
        valid_end_pd = 0
        test_start_pd = train_end_pd + 1
        test_end_pd = epoch_end

    else: # for tuning
        train_pct = 0.70
        valid_pct = 0.20
        train_end_pd = int(total_weeks * train_pct) + epoch_start
        valid_start_pd = train_end_pd + 1
        valid_end_pd = int(total_weeks * valid_pct) + epoch_start
        test_start_pd = valid_end_pd + 1
        test_end_pd = epoch_end
    return epoch_start,train_end_pd,valid_start_pd,valid_end_pd,test_start_pd,test_end_pd

####################################
# Convert epoch_pd to date
####################################
def convert_epoch_to_date(epoch_pd):
    return BASE_DATE + pd.to_timedelta(epoch_pd * 7, 'D')

####################################
# CONFIG SECTIONS and KEYS
####################################
CFG_SECTION_PREDICTION_DATES = 'PREDICTION_DATE_RANGES'
CFG_SECTION_TUNE_DATES = 'TUNE_DATE_RANGES'
CFG_KEY_EPOCH_TRAIN_START = 'EPOCH_TRAIN_START'
CFG_KEY_EPOCH_TRAIN_END = 'EPOCH_TRAIN_END'
CFG_KEY_EPOCH_TEST_START = 'EPOCH_TEST_START'
CFG_KEY_EPOCH_TEST_END = 'EPOCH_TEST_END'
CFG_KEY_EPOCH_VAL_START = 'EPOCH_VAL_START'
CFG_KEY_EPOCH_VAL_END = 'EPOCH_VAL_END'
CFG_KEY_EPOCH_52AGO = 'EPOCH_52AGO'

####################################
# write_cfg writes to the config file
####################################
def write_cfg(section: str, key: str, val):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Check if the section exists, if not, create it
    if section not in config:
        config[section] = {}

    config[section][key] = str(val)

    with open(config_file, 'w') as configfile:
        config.write(configfile) # type: ignore

####################################
# get_cfg gets a value from the config file
####################################
def get_cfg(section: str, key: str):
    config = configparser.ConfigParser()
    config.read(config_file)

    # Reading a value
    val = config[section][key]
    # Return value
    return  val

####################################
# Handle duplicates
####################################
def check_for_duplicates(this_logger,df,col1,col2):
    if col1 not in df.columns or col2 not in df.columns:
        this_logger.error(f"Columns '{col1}' and/or '{col2}' not found in DataFrame.")
        return False  # Return False since the validation failed

    # Identify duplicates
    duplicates = df[df.duplicated(subset=[col1, col2], keep=False)]
    line_number = inspect.currentframe().f_back.f_lineno
    function_name = inspect.currentframe().f_back.f_code.co_name
    if duplicates.empty:
        this_logger.info(f"No duplicates using {col1} and {col2} found in dataframe at {function_name}:{line_number}.")
    else:
        dup_count = len(duplicates)
        duplicates = duplicates.sort_values(by=[col1,col2])
        this_logger.warning(f"Found {dup_count} duplicate records at {function_name}:{line_number} based on '{col1}' and '{col2}':\n{duplicates.head(20)}")
        return False  # Return False if duplicates are found

    return True  # Return True if no duplicates exist

####################################
# save_training_columns
####################################
def save_training_columns(train_df):
    """
    Saves the feature column names to a JSON file for future use in prediction.

    Parameters:
    - feature_columns_list (list): List of feature column names.
    - filename (str): File path to save the column names (default: 'lgb_training_columns.json').

    Returns:
    - None
    """
    column_info = {col: str(train_df[col].dtype) for col in train_df.columns}  # Save column names and dtypes

    with open(model_training_columns_file, "w") as f:
        json.dump(column_info, f)
    print(f"Training columns saved successfully to {model_training_columns_file}")

####################################
# load_training_columns
####################################
def load_training_columns_and_types():
    """
    Loads the saved feature column names from a JSON file.

    Parameters:
    - filename (str): File path to load the column names (default: 'lgb_training_columns.json').

    Returns:
    - list: List of feature column names.
    """
    with open(model_training_columns_file, "r") as f:
        column_info = json.load(f)

    print(f"raining columns loaded from {model_training_columns_file}")
    return column_info

####################################
# apply_column_types
####################################
def apply_column_types(df, column_info):
    """
    Ensures that the DataFrame's columns match the saved data types.

    Parameters:
    - df (pd.DataFrame): The dataframe to modify.
    - column_info (dict): Dictionary where keys are column names and values are their original data types.

    Returns:
    - pd.DataFrame: The updated dataframe with correct column types.
    """
    this_col = 'unassigned'
    try:
        for col, dtype in column_info.items():
            this_col = col
            if col in df.columns:
                if "int" in dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")  # Handles missing values safely
                elif "float" in dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                elif "bool" in dtype:
                    df[col] = df[col].astype("bool")
                elif "category" in dtype:
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype("string")  # Default to string for unexpected types
    except Exception as e:
        raise SystemExit(f"An error occurred while processing column {this_col} types: {str(e)}")

    # print("Data types applied successfully.")
    return df

def identify_sales_spikes(this_df, item_field, sales_field, multiplier=1.5):
    """
    Identify high sales outliers and add a flag for holiday shopping periods.
    """

    def calculate_thresholds(inner_df, group_field, value_field, inner_multiplier=1.5):
        """
        Calculate upper thresholds for outliers based on IQR for grouped data.
        """
        stats = inner_df.groupby(group_field)[value_field].quantile([0.25, 0.75]).unstack()
        try:
            stats['IQR'] = stats[0.75] - stats[0.25]
            stats['upper_threshold'] = stats[0.75] + inner_multiplier * stats['IQR']
        except KeyError as e:
            print(f"Key error: {e} - Check that the quantile keys exist in the DataFrame")
        return stats[['upper_threshold']]

    # Calculate thresholds and flag outliers
    thresholds = calculate_thresholds(this_df, item_field, sales_field, multiplier)
    this_df = this_df.merge(thresholds, left_on=item_field, right_index=True)
    this_df['is_high_outlier'] = this_df[sales_field] > this_df['upper_threshold']
    return this_df.drop(columns=['upper_threshold'])





# Function to assign holiday flags
def assign_holiday_flags(df):
    lookback_years = 2
    spike_threshold = 1.5
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df['year'] = df[DATE_COLUMN].dt.year
    df['month'] = df[DATE_COLUMN].dt.month
    df['week'] = df[DATE_COLUMN].dt.isocalendar().week
    df['day'] = df[DATE_COLUMN].dt.day

    # Function to get the Thanksgiving week dynamically
    def get_thanksgiving(yr):
        """Returns the date of Thanksgiving (4th Thursday of November)."""
        first_day = datetime(yr, 11, 1)
        first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
        thanksgiving_1 = first_thursday + timedelta(weeks=3)  # 4th Thursday
        return thanksgiving_1

    # Define holidays with expected retail impact
    holiday_weeks = {
        'New Year': (1, 1, 2),
        'Memorial Day': (5, -1, 1),
        'Independence Day': (7, 4, 1),
        'Labor Day': (9, 1, 1),
        'Halloween': (10, 31, 2),
        'Christmas': (12, 25, 3),
    }

    for year in df['year'].unique():
        thanksgiving = get_thanksgiving(year)
        black_friday = thanksgiving + timedelta(days=1)
        cyber_monday = thanksgiving + timedelta(days=4)

        # Dynamically assign holiday weeks
        holiday_weeks_dynamic = {
            'Thanksgiving': (thanksgiving.month, thanksgiving.day, 3),
            'Black Friday': (black_friday.month, black_friday.day, 3),
            'Cyber Monday': (cyber_monday.month, cyber_monday.day, 3),
        }

        # Add peak holiday shopping weeks (weeks after Thanksgiving)
        for i in range(1, 5):  # Assign 4 peak weeks
            shopping_week = thanksgiving + timedelta(weeks=i)
            holiday_weeks_dynamic[f'Holiday Shopping {i}'] = (shopping_week.month, shopping_week.day, 3)

        # Merge static and dynamic holidays
        full_holiday_weeks = {**holiday_weeks, **holiday_weeks_dynamic}

        # Assign holiday flags
        for holiday, (month, day, impact) in full_holiday_weeks.items():
            df.loc[(df['year'] == year) & (df['month'] == month) & (df['day'] == day), 'holiday_flag'] = impact

    # === Dynamic Promotion Flagging (Based on Past Year Sales Spikes) === #
    sales_median = df.groupby(['week'])[TARGET_COLUMN].median()

    for year in df['year'].unique():
        for week in df['week'].unique():
            current_sales = df[(df['year'] == year) & (df['week'] == week)][TARGET_COLUMN].median()
            past_sales = []

            # Look back at previous years
            for i in range(1, lookback_years + 1):
                past_year = year - i
                past_sale = df[(df['year'] == past_year) & (df['week'] == week)][TARGET_COLUMN].median()
                if not np.isnan(past_sale):
                    past_sales.append(past_sale)

            # Detect spike if the current median sales exceed past years' median sales by the threshold
            if past_sales and current_sales > np.median(past_sales) * spike_threshold:
                df.loc[
                    (df['year'] == year) & (df['week'] == week) & (df['holiday_flag'] < 2),
                    'holiday_flag'
                ] = 2  # Only update if current flag is < 2

    # Drop extra columns
    df = df.drop(columns=['year', 'month', 'week','day'])
    return df

def generate_holiday_flags(df, date_col='date', epoch_week_col='epoch_week', sales_col='sales', lookback_years=3,
                           spike_threshold=1.5):
    """
    Assigns a holiday impact flag (0-3) to a weekly time-series dataset.

    - **0**: No impact
    - **1**: Minor impact (Labor Day, Memorial Day, etc.)
    - **2**: Moderate impact (Back to School, Halloween, general holiday shopping, past spikes)
    - **3**: Major retail impact (Black Friday, Cyber Monday, Christmas)

    Adds a **dynamic promotion flag** based on historical sales trends.

    Parameters:
        df (pd.DataFrame): DataFrame with weekly time-series data.
        date_col (str): Column name for date.
        epoch_week_col (str): Column name for epoch week (for sorting).
        sales_col (str): Column for sales values (used to check past spikes).
        lookback_years (int): Number of past years to review for sales spikes.
        spike_threshold (float): Multiplier for detecting sales spikes.

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'holiday_flag' column.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    #df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week

    # Define holidays with expected retail impact
    holiday_weeks = {
        'New Year': (1, 1, 2),
        'Memorial Day': (5, -1, 1),
        'Independence Day': (7, 4, 1),
        'Labor Day': (9, 1, 1),
        'Halloween': (10, 31, 2),
        'Thanksgiving': (11, 4, 3),
        'Black Friday': (11, 4, 3),
        'Cyber Monday': (11, 5, 3),
        'Holiday Shopping 1': (12, 1, 3),
        'Holiday Shopping 2': (12, 2, 3),
        'Holiday Shopping 3': (12, 3, 3),
        'Holiday Shopping 4': (12, 4, 3),
        'Christmas': (12, 25, 3),
    }

    # Initialize the holiday flag column
    df['holiday_flag'] = 0

    # Flag Back to School (August & early September)
    df['holiday_flag'] = np.where((df['month'] == 8) | ((df['month'] == 9) & (df['week'] <= 2)), 2, df['holiday_flag'])

    # Flag Holiday Shopping Season (Thanksgiving to Christmas)
    df['holiday_flag'] = np.where(
        ((df['month'] == 11) & (df['week'] >= 4)) | (df['month'] == 12),
        np.maximum(df['holiday_flag'], 2),  # Ensure at least 2
        df['holiday_flag']
    )

    # Assign flags for fixed-date holidays
    for holiday, (month, week_or_day, impact) in holiday_weeks.items():
        if isinstance(week_or_day, int) and week_or_day > 0:  # Fixed date holidays
            df['holiday_flag'] = np.where(
                (df['month'] == month) & (df['date'].dt.day == week_or_day),
                impact,
                df['holiday_flag']
            )
        elif week_or_day == -1:  # Last Monday of the month
            last_monday = df[(df['month'] == month) & (df['date'].dt.weekday == 0)].groupby('year')[date_col].max()
            df['holiday_flag'] = np.where(df[date_col].isin(last_monday.values), impact, df['holiday_flag'])
        elif week_or_day > 0:  # Nth week of the month
            df['holiday_flag'] = np.where(
                (df['month'] == month) & (df['week'] == week_or_day),
                impact,
                df['holiday_flag']
            )

    # === Dynamic Promotion Flagging (Based on Past Year Sales Spikes) === #
    sales_median = df.groupby(['week'])[sales_col].median()

    # make all of december a 3
    df.loc[df['month'] == 12, 'holiday_flag'] = 3

    for year in df['year'].unique():
        for week in df['week'].unique():
            current_sales = df[(df['year'] == year) & (df['week'] == week)][sales_col].median()
            past_sales = []

            # Look back at previous years
            for i in range(1, lookback_years + 1):
                past_year = year - i
                past_sale = df[(df['year'] == past_year) & (df['week'] == week)][sales_col].median()
                if not np.isnan(past_sale):
                    past_sales.append(past_sale)

            # Detect spike if the current median sales exceed past years' median sales by the threshold
            if past_sales and current_sales > np.median(past_sales) * spike_threshold:
                df.loc[
                    (df['year'] == year) & (df['week'] == week) & (df['holiday_flag'] < 2),
                    'holiday_flag'
                ] = 2  # Only update if current flag is < 2

    # Drop extra columns
    df = df.drop(columns=['month', 'week'])

    return df

def calculate_build_ratio(data,lag_distance):
    """
    Calculates the build ratio as (numerator / denominator), handling NaNs and division errors.

    Parameters:
        data (pd.DataFrame or pd.Series): The dataset (DataFrame for batch processing, Series for single-row updates).
        lag_distance:  what is the lag (generally 4, 13, 52)

    Returns:
        If input is DataFrame: Returns the updated DataFrame with the new build ratio column.
        If input is Series (single row): Returns the calculated build ratio value.
    """
    epsilon = 1e-8
    numerator_col = 'lag_1'
    denominator_col = f'lag_{lag_distance}'
    new_col_name = f'build_{lag_distance}'

    if isinstance(data, pd.DataFrame):
        # Vectorized operation for batch processing (training)
        data[new_col_name] = (data[numerator_col] / (data[denominator_col] + epsilon)).fillna(0).round(2)
        return data
    elif isinstance(data, pd.Series):
        # Single-row operation for row-wise updates (prediction)
        return np.nan_to_num(data[numerator_col] / (data[denominator_col] + epsilon), nan=0).round(2)
    else:
        raise ValueError("Input must be a pandas DataFrame or Series.")


def fill_missing_with_mode(df, column_name):
    """
    Fills missing values in the specified column with the mode (most common value).

    Parameters:
        df (pd.DataFrame): The input dataframe.
        column_name (str): The name of the column to process.

    Returns:
        pd.DataFrame: DataFrame with missing values in the specified column filled.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    mode_value = df[column_name].mode().iloc[0]  # Get the most common value

    #df[column_name].fillna(mode_value, inplace=True)  # Fill NaN with mode
    df[column_name] = df[column_name].fillna(mode_value)
    return df  # Return the updated DataFrame

def filter_items_with_min_weeks(df, item_col, min_weeks=25):
    """
    Removes rows for items that appear in the DataFrame fewer than `min_weeks` times.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        item_col (str): The column name representing the item ID.
        min_weeks (int): The minimum number of weeks an item must have to be retained.

    Returns:
        pd.DataFrame: Filtered DataFrame with only items that meet the minimum weeks condition.
    """
    # Count occurrences of each item
    item_counts = df[item_col].value_counts()

    # Get items that have at least `min_weeks` occurrences
    valid_items = item_counts[item_counts >= min_weeks].index

    # Filter the DataFrame
    return df[df[item_col].isin(valid_items)]