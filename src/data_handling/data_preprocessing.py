import pandas as pd
import numpy as np
import h5py
from pathlib import Path

def iterate_directory(main_path):
    return Path(main_path).glob('*/')


def convert_h5py_to_dataframe(video_id,  file):
    # keys  = t,  y
    frame_amount  = len(file["y"])

    columns = ["video_id","timestamp"]
    for frame_count in range(0,512):
        columns.append("clip_dim" + str(frame_count))

    for frame_count in range(0,frame_amount):
        video_frame_list = [(video_id,) + tuple(np.append([file["t"][frame_count]],file["y"][frame_count]))]
        
        if frame_count == 0:
            video_dataframe = pd.DataFrame(video_frame_list, columns=columns)
        else:
            frame_dataframe = pd.DataFrame(video_frame_list, columns=columns)
            video_dataframe = pd.concat([video_dataframe,frame_dataframe])
    
    return video_dataframe


def get_embeddings(source_path):

    for video_count, video_path in enumerate(Path(source_path).glob('*/')): 
        print("Transforming video: {}".format(video_count))

        video_path = str(video_path).replace("\\","/")
        video_id = video_path.replace("resources/bildtv/","").replace("resources/compacttv/","").replace("resources/tagesschau/","")

        file = h5py.File(video_path + "\clip.hdf5")

        if video_count == 0:
            source_df =  convert_h5py_to_dataframe(video_id, file)
        else:
            video_df = convert_h5py_to_dataframe(video_id, file)
            source_df = pd.concat([source_df,video_df])

    return source_df

bild_clip_df = get_embeddings("./resources/tagesschau/")
#bild_clip_df.to_parquet("./resources/transformed_embeddings/bildtv.parquet.gzip",compression="gzip")
bild_clip_df.to_csv("./resources/transformed_embeddings/tagesschau.csv")
