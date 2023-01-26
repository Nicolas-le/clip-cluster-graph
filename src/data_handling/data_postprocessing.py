import pandas as pd
import json
import base64
from PIL import Image
from io import BytesIO

cluster_video_timestamp = pd.read_csv("./resources/clustered_embeddings/tagesschau_kmeans_25_pca20.csv")

def show_images_cluster(cluster, sample_size=10):

    cluster_df = cluster_video_timestamp[cluster_video_timestamp["cluster"]==cluster]
    sampled_df = cluster_df.sample(n=sample_size, random_state=1)

    for index, image in sampled_df.iterrows():
        thumbnails_path = "./resources/tagesschau/" + image["video_id"] + "/thumbnails.json"
        
        with open(thumbnails_path) as json_file:
            data = json.load(json_file)
            
            for time in data:
                if time["t"] == image["timestamp"]:
                    image_base64 = time["image"]

            with open("./resources/cluster_peak/cluster"+str(cluster)+"_"+ str(index) +".png", "wb") as fh:
                fh.write(base64.b64decode(image_base64))

#for i in range(1,26):
#    show_images_cluster(i)
show_images_cluster(1, sample_size= 50)
show_images_cluster(2, sample_size= 50)


