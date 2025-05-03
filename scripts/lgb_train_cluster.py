import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import shap
import global_utils as gu
from model_features import MAIN_KEEP_FEATURES, HAC_FULL_FEATURES, KMEANS_KEEP_FEATURES, SEASONAL_KEEP_FEATURES, VOLUME_KEEP_FEATURES

# from lightgbm.callback import early_stopping

LOG_SHAP = False
LOG_MODEL_IMPORTANCE = True

num_round = 3000

logger = gu.setup_logging("training")

#gu.setup_lightgbm_logger(logger)
####################################
# setup_lightgbm_logger
####################################
def setup_lightgbm_logger(outer_logger):
    """Set up LightGBM to use the provided logger."""
    class LGBMLogger:
        def __init__(self, logger_instance):
            self._logger = logger_instance

        def write(self, msg):
            msg = msg.strip()
            if msg:
                self._logger.info(msg)

        def flush(self):
            pass

        def info(self, msg):
            """Required by LightGBM."""
            self._logger.info(msg)

        def warning(self, msg):
            """Required by LightGBM."""
            self._logger.warning(msg)

    lgb.register_logger(LGBMLogger(outer_logger))

# , , , ,
def cluster_keep_fields(this_cluster):
    if this_cluster == "cluster_kmeans_fs":
        return KMEANS_KEEP_FEATURES
    elif this_cluster == "cluster_hac_s":
        return HAC_FULL_FEATURES
    elif this_cluster == "cluster_seasonal_fs":
        return SEASONAL_KEEP_FEATURES
    elif this_cluster == "cluster_volume_fs":
        return VOLUME_KEEP_FEATURES
    elif this_cluster == "all":
        return MAIN_KEEP_FEATURES
    else:
        logger.error(f"Unknown clustering method:{this_cluster}")
        raise ValueError(f"Unknown clustering method:{this_cluster}")

def train_model(train_data, model_name,cluster_field):
    """
    Trains and saves an LGBM model.
    """
    model_specific_features = cluster_keep_fields(cluster_field)
    X_train = train_data.drop(gu.TARGET_COLUMN, axis=1)
    intersection_columns = list(set(X_train.columns).intersection(model_specific_features))
    X_train = X_train[intersection_columns]
    y_train = train_data[gu.TARGET_COLUMN]

    base_very_important_features = ['holiday_flag']
    base_important_features = ['item_id_encoded','is_high_outlier', "life_to_date"]
    base_less_important_features = ["roll_mean_13", "roll_median_13", "roll_std_13",'cluster_hac_s','lag_4']

    # Filter features based on importance and presence in model_specific_features
    very_important_features = [f for f in base_very_important_features if f in model_specific_features]
    important_features = [f for f in base_important_features if f in model_specific_features]
    less_important_features = [f for f in base_less_important_features if f in model_specific_features]

    feature_weights = pd.DataFrame(1, index=X_train.index, columns=X_train.columns)
    feature_weights[very_important_features] = 2  # Assign high weight
    feature_weights[important_features] = 1  # Assign medium weight
    feature_weights[less_important_features] = 0.5  # Assign low weight

    cat_fields = gu.ALL_CLUSTER_FIELDS
    if cluster_field in cat_fields:
        cat_fields.remove(cluster_field)

    intersection_cat_fields = list(set(cat_fields).intersection(model_specific_features))
    train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=intersection_cat_fields,
                                weight=feature_weights.mean(axis=1))
    logger.info(f'{model_name} train datatypes:\n{X_train.dtypes}')
    with open(gu.LGB_PARAM_FILE, "r") as f:
        params = json.load(f)

    bst = lgb.train(
        params,
        train_dataset,
        num_boost_round=num_round,
    )

    model_path = f'{gu.MODEL_DIR}{model_name}.pkl'
    bst.save_model(model_path)
    logger.info(f"Model saved: {model_path}")

    if LOG_MODEL_IMPORTANCE:
        # Log LightGBM feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': bst.feature_importance(importance_type='gain')
        }).sort_values(by='Importance', ascending=False)

        logger.info(f"LightGBM Feature Importance for {model_name}:\n{feature_importance.head(100)}")

    if LOG_SHAP:
        # Compute SHAP feature importance
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_train)

        # Calculate mean absolute SHAP importance per feature
        shap_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'SHAP Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='SHAP Importance', ascending=False)

        logger.info(f"SHAP Feature Importance for {model_name}:\n{shap_importance.head(100)}")


def main():
    logger.info("Starting model training...")

    data_file = f'{gu.PROCESSED_DIR}pre_processed_data_training.pkl'
    df = pd.read_pickle(data_file)
    df[gu.DATE_COLUMN] = pd.to_datetime(df[gu.DATE_COLUMN])
    df.sort_values([gu.DATE_COLUMN, gu.ITEM_COLUMN], inplace=True)
    df[gu.TARGET_COLUMN] = df[gu.TARGET_COLUMN].astype(float)
    df[gu.ALL_CLUSTER_FIELDS] = df[gu.ALL_CLUSTER_FIELDS].astype("category")


    train_end_epoch = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_TRAIN_END))
    train_data = df[df[gu.EPOCH_COLUMN] <= train_end_epoch].copy()
    train_data = train_data.drop(columns=gu.REMOVE_4_TRAINING,errors='ignore')
    gu.save_training_columns(train_data[train_data.columns])
    logger.info(f"Training data size: {train_data.shape}")

    # ======= 1. Train full dataset model ======= #
    train_model(train_data, "lgbm~main~all",'all')

    # ======= 2-4. Train models per cluster category ======= #
    for cluster_field in gu.CLUSTER_MODELS:
        unique_clusters = train_data[cluster_field].dropna().unique()
        for cluster_id in unique_clusters:
            logger.info(f"Training model for {cluster_field} = {cluster_id}")
            cluster_train_data = train_data[train_data[cluster_field] == cluster_id].drop(columns=[cluster_field])
            if not cluster_train_data.empty:
                model_name = f"lgbm~{cluster_field}~{cluster_id}"
                train_model(cluster_train_data, model_name,cluster_field)
            else:
                logger.warning(f"Skipping model training for {cluster_field}={cluster_id} (No data)")
    # Remove features that are not to be part of training.  keep them in the files though
    prev_52 = int(gu.get_cfg(gu.CFG_SECTION_PREDICTION_DATES, gu.CFG_KEY_EPOCH_52AGO))
    prediction_data = df[(df[gu.EPOCH_COLUMN] >= prev_52)].copy()
    prediction_data = prediction_data.sort_values([gu.EPOCH_COLUMN, gu.ITEM_COLUMN])

    # prediction_data.loc[prediction_data[gu.EPOCH_COLUMN] > train_end_epoch, 'sales'] = np.nan
    output_file = f'{gu.PROCESSED_DIR}lgbm_predict_data.pkl'
    prediction_data.to_pickle(output_file)
    output_file = f'{gu.PROCESSED_DIR}lgbm_predict_data.csv'
    prediction_data.to_csv(output_file, index=False)

    del prediction_data


    logger.info("=== Model Training Complete ===")

setup_lightgbm_logger(logger)
if __name__ == '__main__':
    main()
