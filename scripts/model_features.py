# Explicitly defined feature lists for each model based on log analysis

# Main Model Feature Lists
MAIN_FULL_FEATURES = [
    "lag_1", "is_high_outlier", "roll_mean_4", "cluster_trend_f", "cluster_volume_fs",
    "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos", "cluster_hac_s", "epoch_week_sin",
    "life_to_date", "lag_4", "cluster_seasonal_fs", "lag_13", "cluster_gaussian_f",
    "item_id_encoded", "holiday_flag", "build_4", "lag_52", "roll_median_13",
    "roll_std_4", "build_52", "roll_std_52", "roll_median_52", "roll_mean_52",
    "build_13", "roll_mean_13", "roll_std_13"
]
MAIN_KEEP_FEATURES = [
    "lag_1", "lag_52", "is_high_outlier", "roll_mean_4", "cluster_trend_f", "cluster_volume_fs",
    "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos", "cluster_hac_s", "epoch_week_sin",
    "life_to_date", "lag_4", "cluster_seasonal_fs", "lag_13", "cluster_gaussian_f", "build_4"
]
MAIN_DROP_FEATURES = list(set(MAIN_FULL_FEATURES) - set(MAIN_KEEP_FEATURES))

# HAC Model Feature Lists
HAC_FULL_FEATURES = [
    "lag_1", "is_high_outlier", "cluster_trend_f", "roll_mean_4", "cluster_hac_s",
    "cluster_volume_fs", "roll_median_4", "epoch_week_cos", "cluster_kmeans_fs",
    "epoch_week_sin", "life_to_date", "lag_4", "cluster_seasonal_fs", "lag_13",
    "cluster_gaussian_f", "item_id_encoded", "holiday_flag", "build_4", "lag_52",
    "roll_median_13", "roll_std_4", "build_52", "roll_std_52", "roll_median_52",
    "roll_mean_52", "build_13", "roll_mean_13", "roll_std_13"
]
HAC_KEEP_FEATURES = [
    "lag_1",  "lag_52","is_high_outlier", "cluster_trend_f", "roll_mean_4", "cluster_hac_s",
    "cluster_volume_fs", "roll_median_4", "epoch_week_cos", "cluster_kmeans_fs",
    "epoch_week_sin", "life_to_date", "lag_4", "cluster_seasonal_fs", "build_4"
]
HAC_DROP_FEATURES = list(set(HAC_FULL_FEATURES) - set(HAC_KEEP_FEATURES))

# K-Means Model Feature Lists
KMEANS_FULL_FEATURES = MAIN_FULL_FEATURES  # Assume same importance order
KMEANS_KEEP_FEATURES = MAIN_KEEP_FEATURES
KMEANS_DROP_FEATURES = list(set(KMEANS_FULL_FEATURES) - set(KMEANS_KEEP_FEATURES))

# Seasonal Model Feature Lists
SEASONAL_FULL_FEATURES = [
    "lag_1", "roll_mean_4", "cluster_seasonal_fs", "is_high_outlier", "cluster_trend_f",
    "cluster_volume_fs", "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos",
    "cluster_hac_s", "epoch_week_sin", "life_to_date", "lag_4", "lag_13",
    "cluster_gaussian_f", "item_id_encoded", "holiday_flag", "build_4", "lag_52",
    "roll_median_13", "roll_std_4", "build_52", "roll_std_52", "roll_median_52",
    "roll_mean_52", "build_13", "roll_mean_13", "roll_std_13"
]
SEASONAL_KEEP_FEATURES = [
    "lag_1",  "lag_52","roll_mean_4", "cluster_seasonal_fs", "is_high_outlier", "cluster_trend_f",
    "cluster_volume_fs", "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos",
    "cluster_hac_s", "epoch_week_sin", "life_to_date", "lag_4", "lag_13", "build_4"
]
SEASONAL_DROP_FEATURES = list(set(SEASONAL_FULL_FEATURES) - set(SEASONAL_KEEP_FEATURES))

# Volume-Based Model Feature Lists
VOLUME_FULL_FEATURES = [
    "lag_1", "cluster_volume_fs", "roll_mean_4", "is_high_outlier", "cluster_trend_f",
    "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos", "cluster_hac_s",
    "epoch_week_sin", "life_to_date", "lag_4", "cluster_seasonal_fs", "lag_13",
    "cluster_gaussian_f", "item_id_encoded", "holiday_flag", "build_4", "lag_52",
    "roll_median_13", "roll_std_4", "build_52", "roll_std_52", "roll_median_52",
    "roll_mean_52", "build_13", "roll_mean_13", "roll_std_13"
]
VOLUME_KEEP_FEATURES = [
    "lag_1",  "lag_52","cluster_volume_fs", "roll_mean_4", "is_high_outlier", "cluster_trend_f",
    "roll_median_4", "cluster_kmeans_fs", "epoch_week_cos", "cluster_hac_s",
    "epoch_week_sin", "life_to_date", "lag_4", "cluster_seasonal_fs", "build_4"
]
VOLUME_DROP_FEATURES = list(set(VOLUME_FULL_FEATURES) - set(VOLUME_KEEP_FEATURES))

# Dictionary for quick lookup
MODEL_FEATURES = {
    "main": {
        "full": MAIN_FULL_FEATURES,
        "keep": MAIN_KEEP_FEATURES,
        "drop": MAIN_DROP_FEATURES,
    },
    "hac": {
        "full": HAC_FULL_FEATURES,
        "keep": HAC_KEEP_FEATURES,
        "drop": HAC_DROP_FEATURES,
    },
    "kmeans": {
        "full": KMEANS_FULL_FEATURES,
        "keep": KMEANS_KEEP_FEATURES,
        "drop": KMEANS_DROP_FEATURES,
    },
    "seasonal": {
        "full": SEASONAL_FULL_FEATURES,
        "keep": SEASONAL_KEEP_FEATURES,
        "drop": SEASONAL_DROP_FEATURES,
    },
    "volume": {
        "full": VOLUME_FULL_FEATURES,
        "keep": VOLUME_KEEP_FEATURES,
        "drop": VOLUME_DROP_FEATURES,
    },
}

