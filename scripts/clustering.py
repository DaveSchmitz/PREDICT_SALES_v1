from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.impute import KNNImputer
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import global_utils as gu
from sklearn.impute import SimpleImputer

####################################
# 1.  kmeans_clustering
####################################
def kmeans_clustering(df,this_logger, item_col, last_train_epoch, n_clusters=5):
    """
    Applies K-Means clustering on aggregated sales data.

    Can be used as a feature OR to segment data into multiple models.
    Groups items based on overall sales trends.
    Useful for large datasets where items have somewhat distinct behaviors.

    Parameters:
        df (pd.DataFrame): Input dataset.
        this_logger:  Logger object
        item_col (str): Column containing item IDs.
        last_train_epoch (int): Last epoch of training data.
        n_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster' column.

    """
    train_df = df[df[gu.EPOCH_COLUMN] <= last_train_epoch].copy()  # Use only training data for clustering
    this_logger.info(f'Rows before processing: {df.shape}')

    # Create binned features
    for col in ["lag_1", "lag_4", "lag_13", "lag_52", "roll_mean_4", "roll_mean_13", "roll_mean_52"]:
        train_df[f"{col}_bin"] = pd.qcut(df[col].fillna(0), q=5, labels=False, duplicates='drop')

    feature_cols = [f"{col}_bin" for col in
                    ["lag_1", "lag_4", "lag_13", "lag_52", "roll_mean_4", "roll_mean_13", "roll_mean_52"]]

    # Aggregate per item
    train_agg = train_df.groupby(item_col)[feature_cols].mean()

    # Handle missing values before clustering
    imputer = KNNImputer(n_neighbors=5)  # Fill missing values using 5 nearest neighbors
    train_agg_imputed = pd.DataFrame(imputer.fit_transform(train_agg), index=train_agg.index, columns=feature_cols)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_agg_imputed["cluster_kmeans_fs"] = kmeans.fit_predict(train_agg_imputed)

    # Merge clusters back to full dataset
    df[item_col] = df[item_col].astype(str)
    train_agg_imputed.index = train_agg_imputed.index.astype(str)
    df = df.merge(train_agg_imputed[["cluster_kmeans_fs"]], on=item_col, how="left")

    this_logger.info(f'Rows after processing: {df.shape}')
    df = gu.fill_missing_with_mode(df, "cluster_kmeans_fs")
    this_logger.info(f"Describe cluster_kmeans_fs:\n{df["cluster_kmeans_fs"].describe()}")
    return df


####################################
# 2.  gaussian_mixture_clustering
####################################
def gaussian_mixture_clustering(df,this_logger, item_col, last_train_epoch, n_clusters=5):
    """
    Applies Gaussian Mixture Model (GMM) clustering.

    Only useful as a feature (not for model segmentation).

    Uses probability-based clustering, allowing items to belong to multiple clusters (soft clustering).
    Great for capturing items with mixed sales behaviors (e.g., seasonal + steady demand).

    Parameters:
        df (pd.DataFrame): DataFrame with item sales over time.
        this_logger:  Logger object
        item_col (str): Column containing item IDs.
        last_train_epoch (int): Last epoch of training data.
        n_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster' column.

    """
    train_df = df[df[gu.EPOCH_COLUMN] <= last_train_epoch].copy()
    this_logger.info(f'Rows before filtering: {df.shape}')
    this_logger.info(f'Rows in training set: {train_df.shape}')

    # Identify items in df that are NOT in train_df
    train_items = set(train_df[item_col].unique())
    all_items = set(df[item_col].unique())
    test_only_items = all_items - train_items

    this_logger.info(f'Items only in test/validation set (not in training): {len(test_only_items)}')
    this_logger.debug(f'Test/validation-only items: {list(test_only_items)}')  # Log full list at debug level

    # Create binned features
    for col in ["lag_1", "lag_4", "lag_13", "roll_std_4", "roll_std_13", "roll_std_52"]:
        train_df[f"{col}_bin"] = pd.qcut(df[col].fillna(0), q=5, labels=False, duplicates='drop')

    feature_cols = [f"{col}_bin" for col in ["lag_1", "lag_4", "lag_13", "roll_std_4", "roll_std_13", "roll_std_52"]]

    # Aggregate per item
    train_agg = train_df.groupby(item_col)[feature_cols].mean()

    # Handle missing values before clustering
    imputer = KNNImputer(n_neighbors=5)  # Fill missing values using 5 nearest neighbors
    train_agg_imputed = pd.DataFrame(imputer.fit_transform(train_agg), index=train_agg.index, columns=feature_cols)

    this_logger.info(f'Rows after grouping: {train_agg_imputed.shape}')

    # Apply Gaussian Mixture clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    train_agg_imputed["cluster_gaussian_f"] = gmm.fit_predict(train_agg_imputed)

    # Merge clusters back to full dataset
    df[item_col] = df[item_col].astype(str)
    train_agg_imputed.index = train_agg_imputed.index.astype(str)
    df = df.merge(train_agg_imputed[["cluster_gaussian_f"]], on=item_col, how="left")

    this_logger.info(f'Final dataset shape: {df.shape}')
    df = gu.fill_missing_with_mode(df, "cluster_gaussian_f")
    this_logger.info(f"Describe cluster_gaussian_f:\n{df["cluster_gaussian_f"].describe()}")
    return df

####################################
# 3. hierarchical_clustering
####################################
def hierarchical_clustering(df,this_logger, item_col, last_train_epoch, n_clusters=5):
    """
    Applies Hierarchical Agglomerative Clustering (hac).

    Best used for segmenting items into separate models.

    Helps create very distinct clusters for training separate models.
    Works well for datasets with clear divisions (e.g., perishable vs. durable goods).

    Parameters:
        df (pd.DataFrame): DataFrame with item sales over time.
        this_logger:  Logger object
        item_col (str): Column containing item IDs.
        last_train_epoch (int): Last epoch of training data.
        n_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster' column.

    Best for:       Complex datasets
    Pros:           learns patterns automatically
    Cons:           Requires deep learning
    """

    train_df = df[df[gu.EPOCH_COLUMN] <= last_train_epoch].copy()
    this_logger.info(f'Rows before filtering: {df.shape}')
    this_logger.info(f'Rows in training set: {train_df.shape}')

    # Identify items in df that are NOT in train_df
    train_items = set(train_df[item_col].unique())
    all_items = set(df[item_col].unique())
    test_only_items = all_items - train_items

    this_logger.info(f'Items only in test/validation set (not in training): {len(test_only_items)}')
    this_logger.debug(f'Test/validation-only items: {list(test_only_items)}')  # Log full list at debug level

    # Create binned features
    for col in ["lag_1", "lag_4", "lag_13", "lag_52", "life_to_date"]:
        train_df[f"{col}_bin"] = pd.qcut(df[col].fillna(0), q=5, labels=False, duplicates='drop')

    feature_cols = ["lag_1_bin", "lag_4_bin", "lag_13_bin", "lag_52_bin", "life_to_date_bin", "epoch_week_sin", "epoch_week_cos"]

    # Aggregate per item
    train_agg = train_df.groupby(item_col)[feature_cols].mean()

    # Handle missing values before clustering
    imputer = KNNImputer(n_neighbors=5)  # Fill missing values using 5 nearest neighbors
    train_agg_imputed = pd.DataFrame(imputer.fit_transform(train_agg), index=train_agg.index, columns=feature_cols)

    this_logger.info(f'Rows after grouping: {train_agg_imputed.shape}')

    # Apply Hierarchical Clustering
    Z = linkage(train_agg_imputed.to_numpy(), method="ward")
    train_agg_imputed["cluster_hac_s"] = fcluster(Z, n_clusters, criterion="maxclust")

    # Merge clusters back to full dataset
    df[item_col] = df[item_col].astype(str)
    train_agg_imputed.index = train_agg_imputed.index.astype(str)
    df = df.merge(train_agg_imputed[["cluster_hac_s"]], on=item_col, how="left")

    this_logger.info(f'Final dataset shape: {df.shape}')
    df = gu.fill_missing_with_mode(df, "cluster_hac_s")
    this_logger.info(f"Describe cluster_hac_s:\n{df["cluster_hac_s"].describe()}")
    return df


####################################
# 4.  time_series_kmeans
####################################


def time_series_kmeans(df, this_logger, item_col, sales_col, last_train_epoch, n_clusters=5):
    """
    Applies Time-Series K-Means clustering with optimizations.
    """

    this_logger.info("Starting time-series clustering...")

    # Filter training data
    train_df = df[df[gu.EPOCH_COLUMN] <= last_train_epoch].copy()

    # Convert epoch weeks to dates and bin into months
    train_df["date"] = pd.to_datetime(train_df[gu.EPOCH_COLUMN], unit="W")
    train_df = train_df.set_index("date").groupby(item_col)[sales_col].resample("ME").sum().unstack().fillna(0)

    this_logger.info(f"Data shape after binning: {train_df.shape}")

    # Time-Series K-Means Clustering
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", n_jobs=-1, random_state=42)
    train_df["cluster_seasonal_fs"] = ts_kmeans.fit_predict(train_df)

    # Merge clusters back to full dataset
    df = df.merge(train_df[["cluster_seasonal_fs"]], on=item_col, how="left")

    df = gu.fill_missing_with_mode(df, "cluster_seasonal_fs")
    this_logger.info(f"Describe cluster_seasonal_fs:\n{df["cluster_seasonal_fs"].describe()}")
    return df


####################################
# 5.  rolling_sales_trend_clustering
####################################
def rolling_sales_trend_clustering(df,this_logger, item_col, last_train_epoch, n_clusters=5):
    """
    Clusters items based on their rolling sales trends.
    Only useful as a feature.
    - Groups items using precomputed rolling statistics.
    - Helps identify promotional vs. stable items.
    - Uses training data for clustering, then applies labels to all periods.

    Parameters:
        df (pd.DataFrame): DataFrame with time-series sales features.
        this_logger:  Logger object
        item_col (str): Column containing item IDs.
        last_train_epoch (int): Last epoch of training data.
        n_clusters (int): Number of clusters.

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster_roll_f' column.
    """
    # Select only training data
    this_logger.info("Starting rolling trend clustering...")

    # Use only training data for clustering
    train_df = df[df[gu.EPOCH_COLUMN] <= last_train_epoch].copy()

    # Select feature columns
    feature_cols = ["roll_mean_4", "roll_mean_13", "roll_mean_52", "roll_std_4", "roll_std_13", "roll_std_52"]

    # Aggregate rolling feature columns by item
    this_logger.info(f'Rows before filtering: {df.shape}')
    train_agg = train_df.groupby(item_col)[feature_cols].mean()

    # Handle missing values using imputation
    imputer = SimpleImputer(strategy="mean")
    train_agg_imputed = pd.DataFrame(imputer.fit_transform(train_agg), index=train_agg.index, columns=feature_cols)

    # Scale features
    scaler = StandardScaler()
    train_agg_scaled = scaler.fit_transform(train_agg_imputed)

    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_agg_imputed["cluster_trend_f"] = kmeans.fit_predict(train_agg_scaled)

    # Merge cluster labels back to full dataset
    df = df.merge(train_agg_imputed[["cluster_trend_f"]], on=item_col, how="left")

    this_logger.info(f"Clustering completed. {df['cluster_trend_f'].isna().sum()} items had missing clusters.")
    df = gu.fill_missing_with_mode(df,"cluster_trend_f")
    this_logger.info(f"Describe cluster_trend_f:\n{df["cluster_trend_f"].describe()}")
    return df


####################################
# 6. Volume-Based Clustering
####################################
def volume_based_clustering(df, this_logger, item_col, prev_52ago, last_train_epoch, n_clusters=4,use_total=True):
    """
    Assigns items to volume-based clusters using historical total sales within a fixed window.

    Parameters:
        df (pd.DataFrame): Input dataset.
        this_logger (Logger): Logger object.
        item_col (str): Column containing item IDs.
        prev_52ago (int): The starting epoch_week for calculating total sales.
        last_train_epoch (int): The last epoch_week for training data.
        n_clusters (int): Number of volume categories (default: 4).
        use_total:  If it should sum across time or use the mean

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster_volume_fs' column.
    """
    this_logger.info(f'Rows before processing: {df.shape}')
    this_logger.info(f'Calculating volume clusters based on sales from {prev_52ago} to {last_train_epoch}')


    if use_total:
        # Compute total sales per item within the defined window
        sales_per_item = df[(df[gu.EPOCH_COLUMN] >= prev_52ago) & (df[gu.EPOCH_COLUMN] <= last_train_epoch)] \
            .groupby(item_col)[gu.TARGET_COLUMN].sum()

    else:
        # Compute the average sales per item within the defined window
        sales_per_item = df[(df[gu.EPOCH_COLUMN] >= prev_52ago) & (df[gu.EPOCH_COLUMN] <= last_train_epoch)] \
            .groupby(item_col)[gu.TARGET_COLUMN].mean()

    # Define volume-based bins (quartiles or fixed thresholds)
    volume_clusters = pd.qcut(sales_per_item, q=n_clusters, labels=False, duplicates='drop')

    # Merge the computed clusters back to the full dataset (apply to all time periods)
    df = df.merge(volume_clusters.rename("cluster_volume_fs"), on=item_col, how="left")

    this_logger.info(f"Describe cluster_volume_fs:\n{df["cluster_volume_fs"].describe()}")
    df = gu.fill_missing_with_mode(df, "cluster_volume_fs")
    this_logger.info(f'Rows after processing: {df.shape}')
    return df