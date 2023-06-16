import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

def convert_h5py_to_dataframe(video_id,  embeddings):
    # keys  = t,  y
    frame_amount  = len(embeddings)

    columns = ["video_id","timestamp"]

    for frame_count in range(0,len(embeddings[0])):
        columns.append("dim" + str(frame_count))

    for frame_count in range(0,frame_amount):
        video_frame_list = [(video_id,) + tuple(np.append(frame_count, embeddings[frame_count]))]
        
        if frame_count == 0:
            video_dataframe = pd.DataFrame(video_frame_list, columns=columns)
        else:
            frame_dataframe = pd.DataFrame(video_frame_list, columns=columns)
            video_dataframe = pd.concat([video_dataframe,frame_dataframe])
    
    return video_dataframe

def get_embeddings(config):
    
    for video_count, video_path in enumerate(Path(config["embeddings"]).glob('*/')): 
        logging.info("Transforming video: {}".format(video_count))

        video_path = str(video_path).replace("\\","/")
        video_id = video_path.replace("resources/embeddings/","")

        file = h5py.File(video_path)
        embeddings = file[config["embedding_algorithm"]] # embedding index = image_id

        if video_count == 0:
            source_df =  convert_h5py_to_dataframe(video_id, embeddings)
        else:
            video_df = convert_h5py_to_dataframe(video_id, embeddings)
            source_df = pd.concat([source_df,video_df])
        
    
    source_df.to_csv("./resources/transformed_embeddings/" + config["embedding_algorithm"] +".csv")


