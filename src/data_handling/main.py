import pandas as pd
import os
from datetime import datetime
import json

import logging
logging.basicConfig(level=logging.INFO)

import data_preprocessing
import pca
import clustering

def handle_config():
    config = {
        "source": "tagesschau",
        "data_source": "./resources/tagesschau/",
        "output_directory": "./outputs/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + "/",
        "initial_transformation": False,
        "principal_components": 30,
        "clustering": "mean_shift",
        "k_means_config": {
            "clusters": 150,
            "n_init": 3,
            "max_iter": 3000,
            "random_state": 1
        },
        "dbscan_config": {
            "eps": 0.5, # 2*dim
            "min_samples": 20 # larger dataset higher, >= dims of data (pca)
        }
    }

    os.mkdir(config["output_directory"])
    with open(config["output_directory"]+"/config.json", 'w') as convert_file:
        convert_file.write(json.dumps(config))
    
    return config

if __name__ == "__main__":
    config = handle_config()

    logging.info(str(config))

    if config["initial_transformation"]:
        logging.info("Data Transformation...")
        data_preprocessing.get_embeddings(config)
    
    transformed_data_path = "./resources/transformed_embeddings/"+ config["source"] + ".csv"
        
    logging.info("Start PCA...")
    pca.pca_main(transformed_data_path, config)

    logging.info("Start Clustering...")
    if config["clustering"] == "kmeans":
        clustering.k_means_clustering(config)
    elif config["clustering"] == "dbscan":
        clustering.dbscan_clustering(config)
    elif config["clustering"] == "mean_shift":
        clustering.meanshift_clustering(config)


    

    

