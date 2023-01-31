import pandas as pd
import json
import base64
from PIL import Image
from io import BytesIO
import random

def show_images_cluster(cluster, cluster_video_timestamp, output_dir,  sample_size=5):

    cluster_df = cluster_video_timestamp[cluster_video_timestamp["cluster"]==cluster]
    sampled_df = cluster_df.sample(n=sample_size, random_state=1)
    
    for index, image in sampled_df.iterrows():
        thumbnails_path = "./resources/tagesschau/" + image["video_id"] + "/thumbnails.json"
        
        with open(thumbnails_path) as json_file:
            data = json.load(json_file)
            
            for time in data:
                if time["t"] == image["timestamp"]:
                    image_base64 = time["image"]

            """
            with open(output_dir+"cluster"+str(cluster)+"_"+ str(random.randint(0,5000)) +".png", "wb") as fh:
                fh.write(base64.b64decode(image_base64))
            """
            with open(output_dir+str(cluster)+".jpg", "wb") as fh:
                fh.write(base64.b64decode(image_base64))

"""
for i in range(1,len(cluster_video_timestamp['cluster'].unique())):
    show_images_cluster(i)
"""

cluster_video_timestamp = pd.read_csv("./outputs/27_01_2023_07_53_00/clustered_data.csv")

for cluster in cluster_video_timestamp['cluster'].unique():
    show_images_cluster(cluster, cluster_video_timestamp, "./src/app/clusters/", sample_size=1)


