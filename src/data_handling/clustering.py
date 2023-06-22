import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift
import hdbscan
import logging
logging.basicConfig(level=logging.INFO)
from collections import defaultdict
from scipy.spatial import distance

def k_means_clustering(config):
    pca_transformed_df = pd.read_csv(config["output_directory"]+ config["embedding_algorithm"] + "_pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]

    logging.info("Perform KMeans clustering...")
    kmeans = KMeans(n_clusters=config["k_means_config"]["clusters"],
        n_init=config["k_means_config"]["n_init"], 
        max_iter=config["k_means_config"]["max_iter"], 
        random_state=config["k_means_config"]["random_state"])

    kmeans = kmeans.fit(pca_transformed_df[columns_for_clustering])
    pca_transformed_df.loc[:,"cluster"] = kmeans.labels_

    only_cluster_df = pca_transformed_df.drop(columns=columns_for_clustering)
    #only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

def dbscan_clustering(config): 
    pca_transformed_df = pd.read_csv(config["output_directory"]+ config["embedding_algorithm"] + "_pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform DBScan clustering...")

    dbscan = DBSCAN(eps=config["dbscan_config"]["eps"], 
        min_samples=config["dbscan_config"]["min_samples"], algorithm= "kd_tree").fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = dbscan.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    #only_cluster_df = only_cluster_df[only_cluster_df.cluster != -1]
    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

    #print(only_cluster_df)

def optics_clustering(config):
    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform Optics clustering...")

    dbscan = OPTICS(min_samples=config["dbscan_config"]["min_samples"]).fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = dbscan.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df = only_cluster_df[only_cluster_df.cluster != -1]
    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

def meanshift_clustering(config):


    pca_transformed_df = pd.read_csv(config["output_directory"] + "pca_transformed_data.csv")
    columns_for_clustering = [e for e in list(pca_transformed_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
    
    logging.info("Perform MeanShift clustering...")

    clustering = MeanShift().fit(pca_transformed_df[columns_for_clustering])

    pca_transformed_df.loc[:,"cluster"] = clustering.labels_

    only_cluster_df = pca_transformed_df

    logging.info("Finished Clustering")
    logging.info(only_cluster_df["cluster"].value_counts())

    only_cluster_df.to_csv(config["output_directory"] + "clustered_data.csv")

def hdbscan_clustering(config):
    
    pca_transformed_df = pd.read_csv(config["output_directory"]+ config["embedding_algorithm"] + "_pca_transformed_data.csv")
    concatenated_df = pd.DataFrame()
    cluster_centroids = defaultdict(lambda: defaultdict(list))

    for game_title in pca_transformed_df['video_id'].unique():

        game_df = pca_transformed_df.loc[pca_transformed_df["video_id"] == game_title]

        columns_for_clustering = [e for e in list(game_df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]
        clustering = hdbscan.HDBSCAN(min_cluster_size=config["hdbscan_config"]["min_cluster_size"]).fit_predict(game_df[columns_for_clustering])

        game_df.loc[:,"cluster"] = clustering

        cluster_centroids = get_centroids(cluster_centroids, game_df, game_title, columns_for_clustering)
        #only_cluster_df = game_df.drop(columns=columns_for_clustering)
        only_cluster_df = game_df
        
        concatenated_df = pd.concat([concatenated_df, only_cluster_df], ignore_index=True)

        logging.info("Finished Clustering")
        logging.info(only_cluster_df["cluster"].value_counts())

    return concatenated_df, columns_for_clustering

def get_centroids(cluster_centroids, game_df, game_title, columns_for_clustering):

    for cluster in game_df['cluster'].unique():
        game_clusters = game_df.loc[game_df["cluster"] == cluster]  
        cluster_centroids[game_title][int(cluster)] = list(game_clusters[columns_for_clustering].mean())
    
    return cluster_centroids

