import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import clustering as cf

import global_utils as gu

# Set up logging
logger = gu.setup_logging('predictions_by_cluster')
pd.set_option('future.no_silent_downcasting', True)


def recalculate_clusters(df, current_week):
    """
    Iteratively updates cluster assignments based on predicted sales for the given week.
    """
    for cluster_field in gu.CLUSTER_MODELS:
        df.loc[df[gu.EPOCH_COLUMN] == current_week, cluster_field] = assign_cluster(df, cluster_field, current_week)

    # Forward-fill missing cluster values for later weeks
    df[gu.CLUSTER_MODELS] = df[gu.CLUSTER_MODELS].fillna(method="ffill")

    return df


def assign_cluster(df, cluster_field, current_week):
    """
    Dynamically assigns clusters using the appropriate method from clustering.py.
    Uses past predictions (`predict_avg_sales`) instead of actual sales.
    """
    past_data = df[df[gu.EPOCH_COLUMN] < current_week].copy()

    # Ensure enough past data exists
    if past_data["predict_avg_sales"].isna().sum() > len(past_data) * 0.5:
        return df[cluster_field]  # Keep old clusters if too many NaNs

    # Select clustering method based on field
    if cluster_field == "cluster_kmeans_fs":
        df = cf.kmeans_clustering(df, logger, gu.ITEM_COLUMN, last_train_epoch=current_week - 1)
    elif cluster_field == "cluster_gaussian_f":
        df = cf.gaussian_mixture_clustering(df, logger, gu.ITEM_COLUMN, last_train_epoch=current_week - 1)
    elif cluster_field == "cluster_hac_s":
        df = cf.hierarchical_clustering(df, logger, gu.ITEM_COLUMN, last_train_epoch=current_week - 1)
    elif cluster_field == "cluster_seasonal_fs":
        df = cf.time_series_kmeans(df, logger, gu.ITEM_COLUMN, gu.TARGET_COLUMN, last_train_epoch=current_week - 1)
    elif cluster_field == "cluster_trend_f":
        df = cf.rolling_sales_trend_clustering(df, logger, gu.ITEM_COLUMN, last_train_epoch=current_week - 1)

    return df[cluster_field]  # Return updated clusters

def main():
    """
    Main function to execute the model prediction process.
    """
    epsilon = 1e-8  # A small number to prevent division errors
    predict_columns = gu.CLUSTER_MODELS  # List of prediction fields

    logger.info("Starting model predictions...")

    # Retrieve prediction time frames
    first_prediction_week = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_START))
    last_prediction_week = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TEST_END))

    # Load all trained models
    models = {}
    models_field = {}
    models_value = {}
    models_features = {}

    for filename in os.listdir(gu.MODEL_DIR):
        if filename.startswith('lgbm') and filename.endswith('.pkl'):  # Ensure it's a valid model file
            parts = filename[:-4].split('~')  # Example: model~cluster_hac_s~2.0.pkl
            if len(parts) == 3:
                _, file_field_name, file_field_value = parts
                model_name = f"{file_field_name}_{file_field_value}"
                model_path = os.path.join(gu.MODEL_DIR, filename)

                try:
                    models[model_name] = lgb.Booster(model_file=model_path)
                    trained_features = models[model_name].feature_name()
                    models_features[model_name] = trained_features
                    models_field[model_name] = file_field_name
                    models_value[model_name] = file_field_value
                    logger.info(
                        f"Loaded model: {model_name} (Field: {file_field_name}, Value: {file_field_value}, Features: {trained_features})")
                except Exception as e:
                    logger.error(f"Failed to load model {filename}: {e}")
                    sys.exit(1)  # Exit script if model loading fails
            else:
                logger.error(f"Skipping file with unexpected format: {filename}")
                sys.exit(1)  # Exit script if file format is invalid

    # Load prediction data
    predict_data_path = f'{gu.PROCESSED_DIR}lgbm_predict_data.pkl'
    predict_data_v1 = pd.read_pickle(predict_data_path)

    debug_items_10 = ['STYCOL004381', 'STYCOL002796', 'STYCOL004288', 'STYCOL000964', 'STYCOL000672',
                      'STYCOL001161', 'STYCOL001071', 'STYCOL002029', 'STYCOL001941', 'STYCOL003910']
    debug_items_20 = ['ITEM_000459', 'ITEM_005241', 'ITEM_007672', 'ITEM_006367', 'ITEM_003193',
                      'ITEM_006294', 'ITEM_005257', 'ITEM_001092', 'ITEM_008137', 'ITEM_007234',
                      'ITEM_008379', 'ITEM_003606', 'ITEM_002372', 'ITEM_003542', 'ITEM_008380',
                      'ITEM_002378','ITEM_001724', 'ITEM_008097', 'ITEM_007510', 'ITEM_002443']
    debug_item_highvol_badpred = [
        'ITEM_001664', 'ITEM_001608', 'ITEM_001617', 'ITEM_001529', 'ITEM_001670',
        'ITEM_001533', 'ITEM_001851', 'ITEM_001853', 'ITEM_001646', 'ITEM_001609'
    ]
    debug_item_highvol_bestpred = [
        'ITEM_008379', 'ITEM_002372', 'ITEM_003606', 'ITEM_007522', 'ITEM_002248',
        'ITEM_002321', 'ITEM_007991', 'ITEM_002326', 'ITEM_007510', 'ITEM_002443'
    ]
    debug_items_1 = ['ITEM_002372']
    # debug_items_10 = predict_data_v1['item_id'].drop_duplicates().sample(n=10)
    # debug_items = debug_items_20
    # predict_data = predict_data_v1[predict_data_v1[gu.ITEM_COLUMN].isin(debug_items_10)].copy()
    predict_data = predict_data_v1.copy()


    # Ensure consistency in column types
    trained_feature_columns = gu.load_training_columns_and_types()
    predict_data = gu.apply_column_types(predict_data, trained_feature_columns)
    logger.info(f"Prediction data column types:\n{predict_data.dtypes}")

    # Define sales tracking column
    system_sales = 'predict_avg_sales'

    # Set system_sales equal to actual sales for elapsed weeks
    # predict_data.loc[predict_data[gu.EPOCH_COLUMN] < first_prediction_week, system_sales] = predict_data[
    #     gu.TARGET_COLUMN]
    predict_data.loc[predict_data[gu.EPOCH_COLUMN] < first_prediction_week, system_sales] = \
        predict_data.loc[predict_data[gu.EPOCH_COLUMN] < first_prediction_week, gu.TARGET_COLUMN]

    # Remove data in columns that contain actual/real sales values in prediction weeks
    # these columns are expected to get populated as predictions progress through the weeks
    columns_to_nan = [
        'lag_1','lag_4','lag_13','lag_52',
        'build_4','build_13','build_52',
        'roll_mean_4','roll_median_4','roll_std_4','roll_mean_13','roll_median_13','roll_std_13','roll_mean_52','roll_median_52','roll_std_52',
        'sales',
        'predict_all_sales', 'predict_cluster_kmeans_fs_sales', 'predict_cluster_hac_s_sales',
        'predict_cluster_seasonal_fs_sales', 'predict_avg_sales'
    ]
    existing_columns = [col for col in columns_to_nan if col in predict_data.columns]  # Only keep existing columns
    predict_data.loc[predict_data[gu.EPOCH_COLUMN] >= first_prediction_week, existing_columns] = np.nan

    # Initialize processing variables
    prev_processed_week = -1
    skip = True

    # Iterate over each row for predictions and feature updates
    for index, row in predict_data.iterrows():
        if skip and row[gu.EPOCH_COLUMN] < first_prediction_week:
            continue  # Skip rows before the first prediction week
        else:
            skip = False  # Start processing

        current_week = row[gu.EPOCH_COLUMN]
        if current_week != prev_processed_week:
            logger.info(f"Processing week: {current_week} (Last week will be {last_prediction_week})")
            prev_processed_week = current_week

        past_sales = predict_data[predict_data[gu.ITEM_COLUMN] == row[gu.ITEM_COLUMN]]
        # Dynamically update lag and rolling statistics
        if current_week >= first_prediction_week:

            # Update lag features
            for lag in [1, 4, 13, 52]:
                field_id = f'lag_{lag}'
                prev_week_sales = past_sales.loc[past_sales[gu.EPOCH_COLUMN] == current_week - lag, system_sales]
                if not prev_week_sales.empty:
                    predict_data.at[index, field_id] = prev_week_sales.values[0]

            # Update rolling statistics
            for window in [4, 13, 52]:
                rolling_window = past_sales.loc[past_sales[gu.EPOCH_COLUMN] < current_week].nlargest(window, gu.EPOCH_COLUMN)[
                    system_sales]
                if not rolling_window.empty:
                    predict_data.at[index, f'roll_mean_{window}'] = rolling_window.mean()
                    predict_data.at[index, f'roll_median_{window}'] = rolling_window.median()
                    predict_data.at[index, f'roll_std_{window}'] = rolling_window.std()
                    predict_data.at[index, f'roll_max_{window}'] = rolling_window.max()
                    predict_data.at[index, f'roll_min_{window}'] = rolling_window.min()

        # === Calculate Features Before Prediction ===
        # predict_data.at[index, "item_id_encoded"] = past_sales[gu.TARGET_COLUMN].expanding().mean().shift(1).iloc[-1]
        predict_data.at[index, "life_to_date"] = current_week - past_sales[gu.EPOCH_COLUMN].min()
        predict_data.at[index, "build_4"] = gu.calculate_build_ratio(predict_data.loc[index], 4)
        predict_data.at[index, "build_13"] = gu.calculate_build_ratio(predict_data.loc[index], 13)
        predict_data.at[index, "build_52"] = gu.calculate_build_ratio(predict_data.loc[index], 52)
        # predict_data.at[index, "build_4"] = int(
        #     np.nan_to_num(predict_data.at[index, "lag_1"] / (predict_data.at[index, "lag_4"] + epsilon), nan=0).round()
        # )
        # predict_data.at[index, "build_13"] = int(
        #     np.nan_to_num(predict_data.at[index, "lag_1"] / (predict_data.at[index, "lag_13"] + epsilon), nan=0).round()
        # )
        # predict_data.at[index, "build_52"] = int(
        #     np.nan_to_num(predict_data.at[index, "lag_1"] / (predict_data.at[index, "lag_52"] + epsilon), nan=0).round()
        # )


        # at this point all updates to the dataframe should be done so its safe to get the udpates from the underlying dataframe
        updated_row = predict_data.loc[index]

        # Apply relevant models based on clustering
        for model_key, model in models.items():
            feature_columns = models_features[model_key]
            field_name = models_field[model_key]
            field_value = models_value[model_key]
            df_column = f"predict_{field_name}_sales"
            if str(row.get(field_name)) == field_value:
                X_row = updated_row[feature_columns].to_frame().T.fillna(0).infer_objects(copy=False).values.reshape(1, -1)
                try:
                    predict_data.at[index, df_column] = model.predict(X_row)[0]
                except Exception as e:
                    logger.error(f'Error during prediction: {e}')
                    continue

        # Apply fallback model
        if "main_all" in models:
            feature_columns = models_features["main_all"]
            X_row = updated_row[feature_columns].to_frame().T.fillna(0).infer_objects(copy=False).values.reshape(1, -1)
            predict_data.at[index, "predict_all_sales"] = models["main_all"].predict(X_row)[0]
            # pred_results["predict_all_sales"] = models["main_all"].predict(X_row)[0]

        # Get the values directly from predict_data at the given index
        valid_predictions = [
            predict_data.at[index, col] for col in [
                "predict_all_sales",
                "predict_cluster_kmeans_fs_sales",
                "predict_cluster_hac_s_sales",
                "predict_cluster_seasonal_fs_sales"
            ] if not np.isnan(predict_data.at[index, col])
        ]

        # Compute and store the average in predict_data
        predict_data.at[index, "predict_avg_sales"] = np.mean(valid_predictions) if valid_predictions else np.nan
        last_processed_week = predict_data.at[index, gu.EPOCH_COLUMN]
        last_processed_item = predict_data.at[index, gu.ITEM_COLUMN]

        # # Update the dataframe with predictions
        # for predict_column in predict_columns:
        #     tmp_field = f'predict_{predict_column}_sales'
        #     if tmp_field in pred_results:
        #         predict_data.at[index, tmp_field] = pred_results[tmp_field]
        # predict_data.at[index, 'predict_all_sales'] = pred_results['predict_all_sales']
        # predict_data.at[index, 'predict_avg_sales'] = pred_results['predict_avg_sales']


    if gu.SCALE_TARGET:
        # Apply exponential function to transform predictions back to original sales scale and subtract 1
        prediction_columns = [
            "predict_all_sales", "predict_cluster_kmeans_fs_sales",
            "predict_cluster_hac_s_sales", "predict_cluster_seasonal_fs_sales"
        ]
        transformed_predictions = []

        for col in prediction_columns:
            if col in predict_data.columns:
                # Transform and update each prediction column
                predict_data[col] = np.exp(predict_data[col]) - 1
                # Collect the transformed predictions for averaging
                transformed_predictions.append(predict_data[col])

        # Calculate the average of the transformed predictions, ignoring NaN values
        if transformed_predictions:
            predict_data["predict_avg_sales"] = pd.concat(transformed_predictions, axis=1).mean(axis=1, skipna=True)

        # Optional: Logging the update or checking the results
        logger.info("Updated predict_avg_sales with the average of transformed prediction columns.")


    logger.info("Creating LY (Last Year Sales) column")

    # # Create a shifted dataframe for merging
    # ly_df = predict_data[[gu.ITEM_COLUMN, gu.EPOCH_COLUMN, gu.TARGET_COLUMN]].copy()
    # ly_df[gu.EPOCH_COLUMN] += 52  # Shift epoch_week forward by 52 weeks
    #
    # # Merge with the original dataframe to get LY values
    # predict_data = predict_data.merge(
    #     ly_df,
    #     on=[gu.ITEM_COLUMN, gu.EPOCH_COLUMN],
    #     how='left',
    #     suffixes=('', '_ly')
    # ).rename(columns={gu.TARGET_COLUMN + '_ly': 'ly'})
    #
    # logger.info("LY column created")

    # Save final predictions to CSV
    predict_data['sales'] = predict_data['copy_sales']
    output_file = f'{gu.PROCESSED_DIR}lgbm_predictions.csv'
    predict_data.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Predictions vs Actual Sales
    df = predict_data[(predict_data['epoch_week'] >= first_prediction_week) & (predict_data['epoch_week'] <= last_prediction_week)]

    mae_sales = mean_absolute_error(df['sales'], df['predict_avg_sales'])
    mse_sales = mean_squared_error(df['sales'], df['predict_avg_sales'])
    rmse_sales = mse_sales ** 0.5
    r2_sales = r2_score(df['sales'], df['predict_avg_sales'])

    # Predictions vs Last Year's Sales
    mae_ly = mean_absolute_error(df['ly'], df['predict_avg_sales'])
    mse_ly = mean_squared_error(df['ly'], df['predict_avg_sales'])
    rmse_ly = mse_ly ** 0.5
    r2_ly = r2_score(df['ly'], df['predict_avg_sales'])

    logger.info("Error Metrics for Actual Sales:")
    logger.info(f"MAE: {mae_sales}, MSE: {mse_sales}, RMSE: {rmse_sales}, R-squared: {r2_sales}")

    logger.info("\nError Metrics for Last Year's Sales:")
    logger.info(f"MAE: {mae_ly}, MSE: {mse_ly}, RMSE: {rmse_ly}, R-squared: {r2_ly}")

    summary_stats = df[['sales', 'ly', 'predict_avg_sales']].describe()
    logger.info(f'Summary stats:\n{summary_stats}')
    # import matplotlib.pyplot as plt
    # # Time Series Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(df['epoch_week'], df['sales'], label='Actual Sales')
    # plt.plot(df['epoch_week'], df['ly'], label='Last Year Sales')
    # plt.plot(df['epoch_week'], df['predict_avg_sales'], label='Predicted Avg Sales', alpha=0.7)
    # plt.legend()
    # plt.title('Comparison of Sales: Actual vs Predicted vs Last Year')
    # plt.xlabel('Epoch Week')
    # plt.ylabel('Sales')
    # plt.grid(True)
    # plt.show()
    #
    # # Scatter Plot
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df['sales'], df['predict_avg_sales'], alpha=0.5, label='Actual vs Predicted')
    # plt.scatter(df['ly'], df['predict_avg_sales'], alpha=0.5, label='Last Year vs Predicted')
    # plt.xlabel('Actual / Last Year Sales')
    # plt.ylabel('Predicted Avg Sales')
    # plt.title('Scatter Plot: Actual and Last Year Sales vs Predicted Sales')
    # plt.legend()
    # plt.grid(True)


if __name__ == '__main__':
    main()
