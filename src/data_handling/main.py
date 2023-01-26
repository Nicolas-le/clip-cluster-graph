import data_preprocessing as dp
import pca 
import clusterting as clust
from graph_handler import ClusterGraph
import pandas as pd


def create_cluster_graph(clip_path):
    clip_embeddings  = dp.get_embeddings(clip_path)

    clip_embeddings_dim_reduces = pca.reduce_dimension(clip_embeddings)
    frames_with_clusters = clust.attach_clusters(clip_embeddings_dim_reduces)

    return ClusterGraph(frames_with_clusters)
    

if __name__ == "__main__":
    tagessschau_graph = create_cluster_graph("./resources/clips.hdf")
    bild_graph = create_cluster_graph("./resources/clips.hdf")
    compact_graph = create_cluster_graph("./resources/clips.hdf")
