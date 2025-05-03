import pandas as pd
import numpy as np
# from matplotlib.widgets import EllipseSelector
# from pyqtgraph.examples.colorMapsLinearized import intermediate

import global_utils as gu
import inspect
import clustering as cf


def generate_outlier_thresholds(df, item_field, sales_field, multiplier=1.5):
    """
    Compute and save outlier thresholds based on IQR for each item.

    Parameters:
    df (pd.DataFrame): Input DataFrame with sales data.
    item_field (str): Column representing item IDs.
    sales_field (str): Column representing sales values.
    multiplier (float): Multiplier for IQR thresholding.
    output_path (str): Path to save the CSV.
    """
    output_path = f'{gu.DATA_DIR}outlier_thresholds.csv'
    # Compute IQR-based thresholds
    stats = df.groupby(item_field)[sales_field].quantile([0.25, 0.75]).unstack()
    stats["IQR"] = stats[0.75] - stats[0.25]
    stats["upper_threshold"] = stats[0.75] + multiplier * stats["IQR"]

    # Keep only the necessary column and save
    outlier_thresholds = stats[["upper_threshold"]]
    outlier_thresholds.to_csv(output_path)

    logger.info(f"Outlier thresholds saved to {output_path}")


def main():

    logger.info("Starting data preprocessing...")

    # Load the raw data
    # Columns in raw data:  ITEM,CLS,SUBDEPT,DEPT,DIV,COMPANY,STARTDATE,SLS_UNT
    # source_file = f'{gu.RAW_DIR}online_retail_II.csv'  # Update this with the actual path
    source_file = f'{gu.RAW_DIR}generated_sales.csv'  # Update this with the actual path
    init_df = pd.read_csv(source_file)

    # drop COMPANY since this only has 1 unique value:
    columns_to_drop = ['Invoice','Description','Price','Customer ID','Country'] # these are for online_retail_II.csv
    columns_to_drop = ['Class','Dept','Season','woy','moy','qoy','soy','year'] # these are for department_store_sales.csv
    init_df = init_df.drop(columns=[col for col in columns_to_drop if col in init_df.columns], errors='ignore')

    # Invoice	StockCode	Description	Quantity	InvoiceDate	Price	Customer ID	Country
    # Date,Item,Sales,Class,Dept,Season,woy,moy,qoy,soy,year
    rename_dict = {
        "Item": gu.ITEM_COLUMN,
        "Date": gu.DATE_COLUMN,
        "Sales": gu.TARGET_COLUMN
    }
    init_df.rename(columns=rename_dict, inplace=True)

    # Remove any rows with no sales
    logger.info(f"Main dataframe shape before cleaning:{init_df.shape}")
    init_df = init_df[init_df[gu.TARGET_COLUMN] >= 1].dropna(subset=[gu.TARGET_COLUMN])
    logger.info(f"Main dataframe shape after cleaning:{init_df.shape}")

    # Create date features
    # init_df[gu.DATE_COLUMN] = pd.to_datetime(init_df[gu.DATE_COLUMN], format='%m/%d/%Y %H:%M').dt.normalize() # use this for online
    # 2021-05-29
    init_df[gu.DATE_COLUMN] = pd.to_datetime(init_df[gu.DATE_COLUMN], format='%Y-%m-%d')
    #  this data is by day, we will make the daily data by week instead
    # Adjust each 'Date' to the corresponding end of the week (Sunday)
    init_df['End_of_Week'] = init_df[gu.DATE_COLUMN].apply(lambda x: x + pd.DateOffset(days=(6 - x.weekday())))
    #drop the data column and rename eow to date
    init_df.drop(gu.DATE_COLUMN, axis=1, inplace=True)
    init_df.rename(columns={'End_of_Week': gu.DATE_COLUMN}, inplace=True)

    # Convert datetime to epoch weeks
    base_date = pd.Timestamp("1970-01-01")
    init_df[gu.EPOCH_COLUMN] = ((init_df[gu.DATE_COLUMN] - base_date) // pd.Timedelta("7D")).astype(int)

    # de-dup the data based on item_column and date
    logger.info(f"Initial dataframe shape before de-dup:{init_df.shape}")
    init_main_df = init_df.groupby([gu.ITEM_COLUMN, gu.DATE_COLUMN], as_index=False).agg({
        gu.TARGET_COLUMN: 'sum',
        gu.EPOCH_COLUMN: 'first'
    }).copy()
    logger.info(f"Main dataframe shape after de-dup:{init_main_df.shape}")
    del init_df

    init_main_df = gu.filter_items_with_min_weeks(init_main_df,gu.ITEM_COLUMN,25)

    # Determine the full range of weeks from the minimum to the maximum date
    date_range = pd.date_range(start=init_main_df[gu.DATE_COLUMN].min(), end=init_main_df[gu.DATE_COLUMN].max(),
                               freq='W-SUN')
    all_weeks_df = pd.DataFrame(date_range, columns=[gu.DATE_COLUMN])
    # Get all unique items
    unique_items = init_main_df[gu.ITEM_COLUMN].unique()
    items_df = pd.DataFrame(unique_items, columns=[gu.ITEM_COLUMN])

    # Cross join items with all weeks - ensures every item appears for each week
    all_weeks_df['key'] = 1
    items_df['key'] = 1
    complete_df = pd.merge(all_weeks_df, items_df, on='key').drop('key', axis=1)

    # Merge to ensure all weeks are accounted for each item
    pre_main_df = pd.merge(complete_df, init_main_df, on=[gu.ITEM_COLUMN, gu.DATE_COLUMN], how='left')
    pre_main_df[gu.TARGET_COLUMN].fillna(0, inplace=True)  # Fill missing sales data with zeros

    # Optional: fill other columns if there are any other metrics or features
    # For instance, if you had columns like 'promotional_flag', you might want to fill with a default value
    # final_df['promotional_flag'].fillna(0, inplace=True)

    # Also, if epoch_week could be missing due to the join, recalculate or adjust as necessary
    pre_main_df[gu.EPOCH_COLUMN] = ((pre_main_df[gu.DATE_COLUMN] - base_date) // pd.Timedelta("7D")).astype(int)

    #summarize data frame:
    item_count = pre_main_df[gu.ITEM_COLUMN].unique().size
    logger.info(f'Number of {gu.ITEM_COLUMN}s:{item_count}')
    time_min = pre_main_df[gu.DATE_COLUMN].min()
    time_max = pre_main_df[gu.DATE_COLUMN].max()
    logger.info(f"{gu.DATE_COLUMN} Range:{time_min} to {time_max}")
    epoch_start = pre_main_df[gu.EPOCH_COLUMN].min()
    epoch_end = pre_main_df[gu.EPOCH_COLUMN].max()
    total_weeks = epoch_end - epoch_start + 1
    logger.info(f"{gu.EPOCH_COLUMN} Range:{epoch_start} to {epoch_end} ({total_weeks} weeks)")

    if not gu.check_for_duplicates(logger,pre_main_df,gu.ITEM_COLUMN,gu.DATE_COLUMN):
        raise ValueError("Initial data has duplicate values.  Handle before continuing")

    logger.info(f"Current shape:\n{pre_main_df.shape}")
    logger.info(f"Current dtypes:\n{pre_main_df.dtypes}")

    #determine the start/end epoch weeks for the 5 different data sets
    tune_train_epoch_start = epoch_start
    tune_train_epoch_end = int(total_weeks * 0.7) + epoch_start
    tune_valid_epoch_start = tune_train_epoch_end + 1
    tune_valid_epoch_end = int(total_weeks * 0.9) + epoch_start
    tune_test_epoch_start = tune_valid_epoch_end + 1
    tune_test_epoch_end = epoch_end
    tune_ty_epoch = gu.convert_epoch_to_date(tune_test_epoch_start)
    tune_ty = gu.convert_epoch_to_date(tune_test_epoch_start).year
    tune_ly = tun_ly = tune_ty -1
    tune_52_ago  = tune_test_epoch_start - 53

    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_TRAIN_START, tune_train_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_TRAIN_END, tune_train_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_TEST_START, tune_test_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_TEST_END, tune_test_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_VAL_START, tune_valid_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_VAL_END, tune_valid_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_TUNE_DATES, gu.CFG_KEY_EPOCH_52AGO, tune_52_ago)

    predict_train_epoch_start = epoch_start
    predict_train_epoch_end = int(total_weeks * 0.8) + epoch_start
    predict_valid_epoch_start = 0
    predict_valid_epoch_end = 0
    predict_test_epoch_start = predict_train_epoch_end + 1
    predict_test_epoch_end = epoch_end
    predict_ty_epoch = gu.convert_epoch_to_date(predict_test_epoch_start)
    predict_ty = gu.convert_epoch_to_date(predict_test_epoch_start).year
    predict_ly = predict_ty - 1
    predict_52_ago = predict_test_epoch_start - 53

    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TRAIN_START, predict_train_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TRAIN_END, predict_train_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_START, predict_test_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_END, predict_test_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_VAL_START, predict_valid_epoch_start)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_VAL_END, predict_valid_epoch_end)
    gu.write_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_52AGO, predict_52_ago)

    # Log only string & numeric variables
    # Filter only string, int, and float variables
    # Filter and sort variables by name
    i = inspect.currentframe().f_locals
    local_vars = {k: v for k, v in i.items() if isinstance(v, (str, int, float,np.integer))}
    sorted_vars = sorted(local_vars.items())  # Sort by variable name
    # Format variables with a newline
    formatted_vars = "\n".join(f"{k}: {v}" for k, v in sorted_vars)
    # Log the sorted result
    logger.info(f"Local Variables (Sorted by Name):\n{formatted_vars}\n")

    # determine time hierarchy periods data features
    pre_main_df['woy'] = pre_main_df[gu.DATE_COLUMN].dt.isocalendar().week.astype('int32')
    pre_main_df['moy'] = pre_main_df[gu.DATE_COLUMN].dt.month.astype('int32')
    pre_main_df['qoy'] = pre_main_df[gu.DATE_COLUMN].dt.quarter.astype('int32')
    pre_main_df['soy'] = pre_main_df[gu.DATE_COLUMN].dt.month.map(lambda x: 1 if x <= 6 else 2).astype('int32')
    pre_main_df['year'] = pre_main_df[gu.DATE_COLUMN].dt.year.astype('int32')
    pre_main_df["epoch_week_sin"] = np.sin(2 * np.pi * pre_main_df[gu.EPOCH_COLUMN] / 52)
    pre_main_df["epoch_week_cos"] = np.cos(2 * np.pi * pre_main_df[gu.EPOCH_COLUMN] / 52)
    logger.info("Date features created")

    # create a copy of target (prediction logic drops the target column)
    pre_main_df[f"copy_{gu.TARGET_COLUMN}"] = pre_main_df[gu.TARGET_COLUMN]

    # Create a shifted dataframe for merging
    ly_df = pre_main_df[[gu.ITEM_COLUMN, gu.EPOCH_COLUMN, gu.TARGET_COLUMN]].copy()
    ly_df[gu.EPOCH_COLUMN] += 52  # Shift epoch_week forward by 52 weeks

    # Merge with the original dataframe to get LY values
    pre_main_df = pre_main_df.merge(
        ly_df,
        on=[gu.ITEM_COLUMN, gu.EPOCH_COLUMN],
        how='left',
        suffixes=('', '_ly')
    ).rename(columns={gu.TARGET_COLUMN + '_ly': 'ly'})

    logger.info("LY column created")

    if gu.SCALE_TARGET:
        pre_main_df[gu.TARGET_COLUMN] = np.log(pre_main_df[gu.TARGET_COLUMN] + 1)
        # pre_main_df[gu.TARGET_COLUMN] = np.log2(pre_main_df[gu.TARGET_COLUMN] + 1)
        # pre_main_df[gu.TARGET_COLUMN] = np.log10(pre_main_df[gu.TARGET_COLUMN] + 1)

    # if gu.SCALE_TARGET:
    #     scaler = StandardScaler()
    #     pre_main_df[gu.TARGET_COLUMN] = scaler.fit_transform(pre_main_df[[gu.TARGET_COLUMN]])
    #     # Save the fitted scaler
    #     scaler_path = f'{gu.DATA_DIR}scaler.pkl'
    #     joblib.dump(scaler, scaler_path)
    #     logger.info(f"Scaler saved to {scaler_path}")

    # the rest of the data is dependent on if it is tuning or training
    for data_mode in ("tuning", "training"):
        logger.info(f"Processing mode: {data_mode}:  Start")
        df_start_row_count = pre_main_df.shape[0]
        if data_mode == 'tuning':
            train_epoch_start = tune_train_epoch_start
            train_epoch_end = tune_train_epoch_end
            valid_epoch_start = tune_valid_epoch_start
            valid_epoch_end = tune_valid_epoch_end
            test_epoch_start = tune_test_epoch_start
            test_epoch_end = tune_test_epoch_end
            ty_epoch = tune_ty_epoch
            ty = tune_ty
            ly = tune_ly
            prev_52_ago = tune_52_ago
        else:
            train_epoch_start = predict_train_epoch_start
            train_epoch_end = predict_train_epoch_end
            valid_epoch_start = predict_valid_epoch_start
            valid_epoch_end = predict_valid_epoch_end
            test_epoch_start = predict_test_epoch_start
            test_epoch_end = predict_test_epoch_end
            ty_epoch = predict_ty_epoch
            ty = predict_ty
            ly = predict_ly
            prev_52_ago = predict_52_ago

        # classify the items selling type (Unknown, RampingUp, RampingDown, Erratic, Basic):
        pre_main_df = pre_main_df.sort_values([gu.DATE_COLUMN, gu.ITEM_COLUMN])
        # ========================================================================
        # Group by item and count zero sales for post and pre conditions
        zero_sales_post_train = pre_main_df[(pre_main_df[gu.EPOCH_COLUMN] > train_epoch_end) & (pre_main_df[gu.TARGET_COLUMN] == 0)].groupby(
            gu.ITEM_COLUMN).size()
        zero_sales_pre_train = pre_main_df[(pre_main_df[gu.EPOCH_COLUMN] <= train_epoch_end) & (pre_main_df[gu.TARGET_COLUMN] == 0)].groupby(
            gu.ITEM_COLUMN).size()

        # Create DataFrames from the series
        zero_sales_post_train = zero_sales_post_train.reset_index(name='zero_count_post_train')
        zero_sales_pre_train = zero_sales_pre_train.reset_index(name='zero_count_pre_train')

        # Get all unique items
        unique_items_df = pd.DataFrame(pre_main_df[gu.ITEM_COLUMN].unique(), columns=[gu.ITEM_COLUMN])

        # Merge the zero sales counts back to the unique items list
        unique_items_df = unique_items_df.merge(zero_sales_post_train, on=gu.ITEM_COLUMN, how='left')
        unique_items_df = unique_items_df.merge(zero_sales_pre_train, on=gu.ITEM_COLUMN, how='left')

        # Fill missing counts with 0 (assuming no zero sales were recorded as no data)
        unique_items_df.fillna(0, inplace=True)

        # Apply conditions to filter items
        # Less than 10 zero sales for epoch_week > train_epoch_end
        # Less than 20 zero sales for epoch_week <= train_epoch_end
        filtered_items = unique_items_df[
            (unique_items_df['zero_count_post_train'] < 10) |
            (unique_items_df['zero_count_pre_train'] < 20)
            ]

        # Get the list of item_ids to drop
        items_to_drop = filtered_items[gu.ITEM_COLUMN]

        # Filter these items out of the final DataFrame
        main_df = pre_main_df[~pre_main_df[gu.ITEM_COLUMN].isin(items_to_drop)].copy()
        main_df = pre_main_df.copy()
        # ========================================================================

        # Apply filter to main_df
        # main_df = pre_main_df[pre_main_df["item_id"].isin(valid_items)].copy()
        # del pre_main_df
        # use the tighter range to classify so to prevent data leaks during tuning
        # df = classify_selling_patterns(df,train_epoch_end,ly,12,0,.35)

        # set the dataset
        main_df['partition'] = 3 # default to all to test partition
        main_df.loc[main_df[gu.EPOCH_COLUMN] <= test_epoch_end, "partition"] = 2
        main_df.loc[main_df[gu.EPOCH_COLUMN] <= train_epoch_end, "partition"] = 1

        # sort data by item_id/date
        main_df = main_df.sort_values([gu.ITEM_COLUMN, gu.DATE_COLUMN])
        logger.info("Data sorted by item_id and date")
        output_file = f'{gu.PROCESSED_DIR}temp.csv'
        main_df.to_csv(output_file,index=False)

        for lag in [1, 4, 13, 52]:
            logger.info(f"Creating lag for period {lag}")

            # Create a shifted dataframe for merging
            lag_df = main_df[[gu.ITEM_COLUMN, gu.EPOCH_COLUMN, gu.TARGET_COLUMN]].copy()
            lag_df[gu.EPOCH_COLUMN] += lag  # Shift epoch_week forward by the lag amount

            # Merge with the original dataframe to get lag values
            main_df = main_df.merge(
                lag_df,
                on=[gu.ITEM_COLUMN, gu.EPOCH_COLUMN],
                how='left',
                suffixes=('', f'_lag{lag}')
            ).rename(columns={gu.TARGET_COLUMN + f'_lag{lag}': f'lag_{lag}'})

        logger.info("Lag features created")

        # # Create lag features
        # for lag in [1, 4, 13, 52]:
        #     logger.info(f"Creating lag for period {lag}")
        #     main_df[f'lag_{lag}'] = main_df.groupby(gu.ITEM_COLUMN)[gu.TARGET_COLUMN].shift(lag)
        # logger.info("Lag features created")

        # create an ly_bld feature:
        main_df= gu.calculate_build_ratio(main_df,4)
        main_df = gu.calculate_build_ratio(main_df,13)
        main_df = gu.calculate_build_ratio(main_df,52)

        # make sure the sort is still correct
        main_df = main_df.sort_values([gu.ITEM_COLUMN,gu.DATE_COLUMN])

        # Create aggregate features
        periods = [4, 13, 52]
        for period in periods:
            logger.info(f"Creating rolling features for period {period}")

            group = main_df.groupby(gu.ITEM_COLUMN)[gu.TARGET_COLUMN]

            main_df[f'roll_mean_{period}'] = group.shift(1).rolling(period, min_periods=1).mean()
            main_df[f'roll_median_{period}'] = group.shift(1).rolling(period, min_periods=1).median()

            if period > 1:  # Standard deviation makes sense only for period > 1
                main_df[f'roll_std_{period}'] = group.shift(1).rolling(period, min_periods=1).std()

            #main_df[f'roll_min_{period}'] = group.shift(1).rolling(period, min_periods=1).min()
            #main_df[f'roll_max_{period}'] = group.shift(1).rolling(period, min_periods=1).max()

        logger.info("Lag and rolling features created successfully")

        #Add sales spike feature
        main_df = gu.identify_sales_spikes(main_df,gu.ITEM_COLUMN,gu.TARGET_COLUMN,1.5)
        generate_outlier_thresholds(main_df, gu.ITEM_COLUMN, gu.TARGET_COLUMN, multiplier=1.5)

        main_df = main_df.sort_values(by=[gu.ITEM_COLUMN, gu.EPOCH_COLUMN])  # Sort by item & time

        # Rolling mean target encoding (no data leakage)
        # main_df["item_id_encoded"] = main_df.groupby(gu.ITEM_COLUMN)[gu.TARGET_COLUMN].transform(lambda x: x.expanding().mean().shift(1))
        main_df["item_id_encoded"] = main_df.groupby(gu.ITEM_COLUMN)[gu.TARGET_COLUMN].transform(lambda x: x.expanding().median().shift(1)).fillna(0).astype(int)

        # Life-to-date feature
        main_df["life_to_date"] = main_df.groupby(gu.ITEM_COLUMN)[gu.EPOCH_COLUMN].rank(method="first").astype(int)

        # Apply Holiday & Promo Flags
        main_df = gu.generate_holiday_flags(main_df)

        df_end_row_count = main_df.shape[0]
        # assert df_start_row_count==df_end_row_count, f"df changed row count after the split.  This could lead to data leaks.{df_start_row_count} before vs {df_end_row_count} after"
        gu.check_for_duplicates(logger,main_df,gu.ITEM_COLUMN,gu.EPOCH_COLUMN)
        gu.check_for_duplicates(logger, main_df, gu.ITEM_COLUMN, gu.DATE_COLUMN)

        # ---------------------------------------------------------------------------------------------------------------------------------
        # Clustering Features
        logger.info("Adding clusters - Start")
        # 1
        logger.info(f'Adding cluster_kmeans_fs...')
        main_df = cf.kmeans_clustering(main_df,this_logger=logger,item_col=gu.ITEM_COLUMN,last_train_epoch=train_epoch_end,n_clusters=5)  # Apply KMeans clustering
        # 2
        logger.info(f'Adding cluster_gaussian_f...')

        main_df = cf.gaussian_mixture_clustering(main_df,this_logger=logger, item_col=gu.ITEM_COLUMN, last_train_epoch=train_epoch_end, n_clusters=5)  # Soft clustering
        # 3
        logger.info(f'Adding cluster_hac_s...')
        main_df = cf.hierarchical_clustering(main_df,this_logger=logger,item_col=gu.ITEM_COLUMN, last_train_epoch=train_epoch_end, n_clusters=5)
        # 4
        logger.info(f'Adding cluster_seasonal_fs clusters...')
        main_df = cf.time_series_kmeans(main_df,this_logger=logger,item_col=gu.ITEM_COLUMN, sales_col=gu.TARGET_COLUMN, last_train_epoch=train_epoch_end, n_clusters=5)  # Detect demand-based clusters
        # 5
        logger.info(f'Adding cluster_trend_f...')
        main_df = cf.rolling_sales_trend_clustering(main_df,this_logger=logger, item_col=gu.ITEM_COLUMN, last_train_epoch=train_epoch_end, n_clusters=5)  # Segment items
        # 6
        logger.info(f'Adding cluster_volume_fs...')
        main_df = cf.volume_based_clustering(main_df,this_logger=logger, item_col=gu.ITEM_COLUMN,prev_52ago=prev_52_ago, last_train_epoch=train_epoch_end, n_clusters=5,use_total=False)  # Segment items

        logger.info("Adding clusters - Finished")
        # ---------------------------------------------------------------------------------------------------------------------------------

        # Define the range for the last 8 periods including train_epoch_end
        last_8_periods = main_df[
            (main_df[gu.EPOCH_COLUMN] <= train_epoch_end) &
            (main_df[gu.EPOCH_COLUMN] > train_epoch_end - 8)
            ]

        # Compute the mode (most common cluster assignment) for each item
        mode_clusters = last_8_periods.groupby(gu.ITEM_COLUMN)[gu.ALL_CLUSTER_FIELDS].agg(
            lambda x: x.mode()[0]).reset_index()

        # Merge mode clusters into future weeks
        main_df = main_df.merge(mode_clusters, on=gu.ITEM_COLUMN, how="left", suffixes=("", "_mode"))

        # Apply the mode values to future periods
        for cluster_field in gu.ALL_CLUSTER_FIELDS:
            main_df.loc[main_df[gu.EPOCH_COLUMN] > train_epoch_end, cluster_field] = \
                main_df.loc[main_df[gu.EPOCH_COLUMN] > train_epoch_end, f"{cluster_field}_mode"]

        # Compute the mode (most common cluster assignment) for each item
        mode_clusters = last_8_periods.groupby(gu.ITEM_COLUMN)[gu.ALL_CLUSTER_FIELDS].agg(lambda x: x.mode()[0])

        # Apply these mode cluster values to all future periods (after train_epoch_end)
        for col in gu.ALL_CLUSTER_FIELDS:  # Iterate over each cluster field separately
            main_df.loc[main_df[gu.EPOCH_COLUMN] > train_epoch_end, col] = \
                main_df[gu.ITEM_COLUMN].map(mode_clusters[col])

        # Drop temp columns
        main_df.drop(columns=[f"{col}_mode" for col in gu.ALL_CLUSTER_FIELDS], inplace=True)

        # log the fields in the training data:
        logger.info(f"Fields in {data_mode} files:\n{main_df.columns.tolist()}")

        for cluster_field in gu.ALL_CLUSTER_FIELDS:
            main_df = gu.fill_missing_with_mode(main_df,cluster_field)
        # Save the processed data
        output_file = f'{gu.PROCESSED_DIR}pre_processed_data_{data_mode}.pkl'
        main_df.to_pickle(output_file)
        output_file = f'{gu.PROCESSED_DIR}pre_processed_data_{data_mode}.csv'
        main_df.to_csv(output_file,index=False)
        logger.info(f"Processed data saved to {output_file}")
        logger.info(f"Final data shape:{main_df.shape}")
        logger.info(f"Final data dtypes:\n{main_df.dtypes}")
        # drop the cluster columns, the loop logic assumes they are not there and adds them
        columns_to_drop = ['cluster_kmeans_fs','cluster_gaussian_f','cluster_hac_s','cluster_seasonal_fs','cluster_trend_f','cluster_volume_fs','cluster_volume_fs']
        main_df = main_df.drop(columns=[col for col in columns_to_drop if col in main_df.columns], errors='ignore')
        logger.info(f"Processing mode: {data_mode}:  End")

logger = gu.setup_logging("pre_process")
if __name__ == '__main__':
    main()
