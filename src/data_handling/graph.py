import networkx as nx
from networkx.algorithms import community
from collections import defaultdict
import pandas as pd
from itertools import chain
import os
import json
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class ClusterGraph():

    def __init__(self,
    frames_with_clusters,
    low_cluster_filter,
    community_resolution,
    edge_threshold=0.2):

        self.frames_with_clusters = pd.read_csv(frames_with_clusters)
        self.edge_threshold  = edge_threshold
        self.low_cluster_filter = low_cluster_filter
        self.community_resolution = community_resolution
        self.listOfEdges = self.transform_data()
        self.networkx_graph  = self.create_networkx_graph()
        #self.analytics = self.get_analytics()
        self.communities = self.get_communities()

    def transform_data(self):

        clusters = list(self.frames_with_clusters['cluster'].unique())
        videos = list(self.frames_with_clusters['video_id'].unique())
        self.frames_with_clusters = self.frames_with_clusters.drop(columns=["Unnamed: 0.1","Unnamed: 0", "timestamp"])

        self.filter_low_cluster_counts(videos)
        cluster_videos_listing = self.attach_videos_to_clusters(clusters)
        same_appearance_counter = self.count_intersection_between_clusters(cluster_videos_listing)

        list_of_edges = self.transform_counts_to_edges(same_appearance_counter)
        list_of_edges_norm_weights = self.normalize_weights(list_of_edges)
        
        return list_of_edges_norm_weights

    def filter_low_cluster_counts(self,videos):
        for video_id in videos:
            cluster_counts = self.frames_with_clusters[self.frames_with_clusters["video_id"] == video_id].value_counts()
            for identifier, count in cluster_counts.items():
                if count < self.low_cluster_filter:
                    self.frames_with_clusters = self.frames_with_clusters.drop(self.frames_with_clusters[(self.frames_with_clusters.video_id == identifier[0]) & (self.frames_with_clusters.cluster == identifier[1])].index)

    def attach_videos_to_clusters(self,clusters):
        cluster_videos_listing = defaultdict(list)
        for cluster_id in clusters:
            cluster_videos_listing[cluster_id] = list(self.frames_with_clusters["video_id"][self.frames_with_clusters["cluster"]==cluster_id])
        
        return cluster_videos_listing

    def count_intersection_between_clusters(self, cluster_videos_listing):
        counter = defaultdict(lambda: defaultdict(int))
        for cluster_id, videos in cluster_videos_listing.items():
            for cluster_id2, videos2 in cluster_videos_listing.items():
                if cluster_id == cluster_id2:
                    continue
                counter[cluster_id][cluster_id2] = len(list(set(videos).intersection(videos2)))
        
        return counter

    def transform_counts_to_edges(self, same_appearance_counter):
        list_of_edges = []
        for cluster, counter_dict in same_appearance_counter.items():
            for link_cluster, count in counter_dict.items():
                # no intersection = now edge
                if count == 0.0:
                    continue
                # quadratic boost of high counts
                list_of_edges.append((cluster,link_cluster,count**2))

        return list_of_edges

    def normalize_weights(self,listOfEdges):
        
        weights = []
        for edge in listOfEdges:
            weights.append(edge[2])

        minimum, maximum = min(weights), max(weights)

        normalized_list = []
        for edge in listOfEdges:
            norm_w = (edge[2]-minimum) / (maximum-minimum)
            if norm_w == 0.0 or norm_w < self.edge_threshold:
                continue
            normalized_list.append((edge[0],edge[1], norm_w))

        return normalized_list
        
    def create_networkx_graph(self):
        graph = nx.DiGraph()
        e = self.listOfEdges
        graph.add_nodes_from(self.frames_with_clusters['cluster'].unique())
        graph.add_weighted_edges_from(e)

        return graph

    def get_analytics(self):
        return {
            "degreeC": nx.degree_centrality(self.networkx_graph),
            "closenessC": nx.closeness_centrality(self.networkx_graph),
            "betweenessC": nx.betweenness_centrality(self.networkx_graph),
            "pagerank": nx.pagerank(self.networkx_graph, weight="weight")
        }

    def get_communities(self):
        communities = community.greedy_modularity_communities(self.networkx_graph,
        weight="weight",
        resolution=self.community_resolution)

        self.attach_communities_to_graph(communities)

        return communities

    def attach_communities_to_graph(self, communities):
        for comm_number, nodes in enumerate(communities):
            for node in nodes:
                self.networkx_graph.nodes[node].update({"community": comm_number})

    def save_to_json(self):
        with open(data + "/graph_communities.json", "w") as f:
            json.dump(nx.node_link_data(self.networkx_graph), f, cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
data = "./outputs/27_05_2023_15_38_55/"

g = ClusterGraph(data + "clustered_data.csv",
    low_cluster_filter = 20,
    community_resolution = 1.2,
    edge_threshold=0.1
    )

g.save_to_json()






