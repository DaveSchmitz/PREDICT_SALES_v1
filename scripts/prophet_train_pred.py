import pandas as pd
import prophet
from prophet.make_holidays import make_holidays_df
import global_utils as gu
from datetime import date, timedelta
import logging

def generate_holiday_season(start_year, end_year):
    holidays_ = []
    for year in range(start_year, end_year + 1):
        # Get Thanksgiving (4th Thursday in November)
        nov1 = date(year, 11, 1)
        thanksgiving = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)  # 3 = Thursday, +3 weeks

        # Calculate the start and end of the season, excluding the week of Thanksgiving and Christmas
        season_start = thanksgiving + timedelta(days=7)  # Start the week after Thanksgiving
        christmas = date(year, 12, 25)
        season_end = christmas - timedelta(days=7)  # End the week before Christmas

        if season_start < season_end:  # Only add if there is at least one week between
            holidays_.append({
                'holiday': 'retail_peak',
                'ds': pd.Timestamp(season_start),
                'lower_window': 0,
                'upper_window': (season_end - season_start).days
            })

    return pd.DataFrame(holidays_)


# Setup logging
logger = gu.setup_logging('prophet_predictions')
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.INFO)

# Load data
logger.info("Starting model training...")
data_file = f'{gu.PROCESSED_DIR}pre_processed_data_training.pkl'
df = pd.read_pickle(data_file)

df['ds'] = pd.to_datetime(df[gu.DATE_COLUMN])
df['y'] = df[gu.TARGET_COLUMN]
#
# # Define holidays and special events
# holidays = pd.DataFrame({
#     'holiday': 'public_holiday',
#     'ds': pd.to_datetime(df.loc[df['holiday_flag'] == 1, 'date']),
#     'lower_window': 0,
#     'upper_window': 1,
# })

# # Additional special weeks - Adjust dates accordingly
# special_events = pd.DataFrame([
#     {'holiday': 'back_to_school',
#      'ds': pd.Timestamp(year=2021, month=8, day=1),
#      'lower_window': 0, 'upper_window': 14}
#     # Add more special events similarly
# ])


# Example: Generate for 2010-2025
min_year = df['ds'].dt.year.min()
max_year = df['ds'].dt.year.max()
unique_years = sorted(df['ds'].dt.year.unique())
holiday_xmas = generate_holiday_season(min_year, max_year)
holidays_standard = make_holidays_df(unique_years,'US')
holidays = pd.concat([holidays_standard, holiday_xmas], ignore_index=True)

# Parameters for the training and prediction periods
train_start_epoch = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TRAIN_START))
train_end_epoch = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TRAIN_END))
predict_start_epoch = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_START))
predict_end_epoch = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_END))

train_start_date = df[df['epoch_week'] == train_start_epoch]['date'].min()
train_end_date = df[df['epoch_week'] == train_end_epoch]['date'].min()
predict_start_date = df[df['epoch_week'] == predict_start_epoch]['date'].min()
predict_end_date = df[df['epoch_week'] == predict_end_epoch]['date'].min()

# Filter the data for the training period
train_df = df[(df['epoch_week'] >= train_start_epoch) & (df['epoch_week'] <= train_end_epoch)]

# Train and predict for each item
total_unique_items = len(train_df['item_id'].unique())
counter = 0
results = []
for item in train_df['item_id'].unique():
    counter += 1
    logger.info(f"Item ID: {item},{counter} of {total_unique_items}")
    item_data = train_df[train_df['item_id'] == item]

    if item_data.empty:
        logger.info(f"No data available for item_id: {item}")
        continue  # Skip this iteration if no data is available

    # Initialize Prophet model with holidays
    model = prophet.Prophet(holidays=holidays)
    model.fit(item_data[['ds', 'y']])

    # Make future dataframe for predictions
    # Instead of calculating periods, directly specify the range
    future = pd.date_range(start=predict_start_date, end=predict_end_date, freq='W-SUN').to_frame(index=False,
                                                                                                  name='ds')

    # Predict
    forecast = model.predict(future)
    forecast['item_id'] = item
    #forecast['epoch_week'] = forecast['ds'].dt.strftime('%s').astype(int)
    results.append(forecast[['item_id', 'ds', 'yhat']])

# Combine all results
final_predictions = pd.concat(results, ignore_index=True)
final_predictions['yhat'] = final_predictions['yhat'].round().astype(int)
data_file = f'{gu.PROCESSED_DIR}prophet_predictions.csv'
final_predictions.to_csv(data_file, index=False)
logger.info(f"Predictions saved to: {data_file}")


