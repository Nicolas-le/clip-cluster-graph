import pandas as pd
import json
import base64
from PIL import Image
from io import BytesIO
import random
from pathlib import Path
import shutil


def show_images_cluster(cluster, cluster_video_timestamp, output_dir,  sample_size=5):

    cluster_df = cluster_video_timestamp[cluster_video_timestamp["cluster"]==cluster].sample(frac=1)

    counter = 0    
    for index, image in cluster_df.iterrows():
        #print(counter)
        thumbnails_path = "./resources/images/" + image["video_id"].replace(".hdf5","") + "/"
        image_id = str(int(image["timestamp"]))
        zeros = ""
        for i in range(len(image_id),6):
            zeros = zeros + "0"

        image_id = zeros + image_id
        image_path = thumbnails_path + image_id  + ".jpg"

        try:
            shutil.copyfile(image_path, output_dir + image["video_id"]+ "_" + str(cluster)+"_"+str(counter)+".jpg")
            counter += 1
        except FileNotFoundError:
            print("File not found. {}".format(image_path))
            continue
        
        if counter == sample_size:
            break
        


cluster_video_timestamp = pd.read_csv("./outputs/21_06_2023_12_01_27/clustered_data.csv")

for video_id in cluster_video_timestamp["video_id"].unique():
    game_clusters = cluster_video_timestamp[cluster_video_timestamp["video_id"]==video_id].sample(frac=1)

    print(video_id)
    for cluster in game_clusters['cluster'].unique():
        show_images_cluster(cluster, game_clusters, "./src/app/static/clusters/" , sample_size=10)

    print("-"*100)
#show_images_cluster(61, cluster_video_timestamp, "./outputs/clustering_test/", sample_size=500)
