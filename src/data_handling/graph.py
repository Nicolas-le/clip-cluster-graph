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
from scipy.spatial import distance


class ClusterGraph():

    def __init__(self,
    frames_with_clusters,
    low_cluster_filter,
    community_resolution,
    columns_for_clustering,
    edge_threshold=0.2):

        self.run_path = frames_with_clusters
        #with open(self.run_path + "/cluster_centroids.json") as json_file:
        #    self.cluster_centroids = json.load(json_file)

        self.frames_with_clusters = pd.read_csv(frames_with_clusters + "clustered_data.csv")
        self.edge_threshold  = edge_threshold
        self.low_cluster_filter = low_cluster_filter
        self.community_resolution = community_resolution
        self.columns_for_clustering = columns_for_clustering
        self.listOfEdges = self.transform_data()
        self.networkx_graph  = self.create_networkx_graph()
        #self.analytics = self.get_analytics()
        self.communities = self.get_communities()

    def transform_data(self):

        clusters = list(self.frames_with_clusters['cluster'].unique())
        videos = list(self.frames_with_clusters['video_id'].unique())
        self.frames_with_clusters = self.frames_with_clusters.drop(columns=["Unnamed: 0.1","Unnamed: 0", "timestamp"])

         #list_of_edges = self.attach_inter_cluster_edges()
        list_of_edges = []
        list_of_edges = self.attach_distance_cluster_edges(list_of_edges)

        list_of_edges_norm_weights = self.normalize_weights(list_of_edges)
        
        return list_of_edges_norm_weights

    def attach_inter_cluster_edges(self):
        list_of_edges = []

        for game_title in self.frames_with_clusters['video_id'].unique():
            game_df = self.frames_with_clusters.loc[self.frames_with_clusters["video_id"] == game_title]

            for cluster_one in game_df['cluster'].unique():
                cluster_one_name = game_title + "_" + str(cluster_one)

                for cluster_two in game_df['cluster'].unique():
                    cluster_two_name = game_title + "_" + str(cluster_two)
                    list_of_edges.append((cluster_one_name,cluster_two_name,10))

        return list_of_edges

    def attach_distance_cluster_edges(self, list_of_edges):
        already_computed = []

        for game_title_one in self.frames_with_clusters['video_id'].unique():
            print("-"*100)
            print(game_title_one,flush=True)
            game_df_one = self.frames_with_clusters.loc[self.frames_with_clusters["video_id"] == game_title_one]
            for game_title_two in self.frames_with_clusters['video_id'].unique():

                if game_title_two in already_computed:
                    continue

                print(game_title_two,flush=True)

                game_df_two = self.frames_with_clusters.loc[self.frames_with_clusters["video_id"] == game_title_two]

                for cluster_one in game_df_one['cluster'].unique():
                    game_df_one_cluster =  game_df_one.loc[ game_df_one["cluster"] == cluster_one][self.columns_for_clustering]
                    for cluster_two in game_df_two['cluster'].unique():

                        #coord_one = self.cluster_centroids[game_title_one][str(cluster_one)]
                        #coord_two = self.cluster_centroids[game_title_two][str(cluster_two)]
                        #euc_distance = distance.euclidean(coord_one,coord_two) # euclidean distance between cluster centroids

                        game_df_two_cluster = game_df_two.loc[game_df_two["cluster"] == cluster_two][self.columns_for_clustering]

                        distances = []
                        for i in range(len(game_df_one_cluster)):
                            point1 = game_df_one_cluster.iloc[i].values
                            for j in range(len(game_df_two_cluster)):
                                point2 =  game_df_two_cluster.iloc[j].values
                                distances.append(distance.euclidean(point1, point2))
            
                        euc_distance = sum(distances)/len(distances)

                        if euc_distance > 28:
                            continue

                        if game_title_one != game_title_two:
                            euc_distance = euc_distance * 0.6

                        cluster_one_name = game_title_one + "_" + str(cluster_one) # different names for each game cluster
                        cluster_two_name = game_title_two + "_" + str(cluster_two)
                        list_of_edges.append((cluster_one_name,cluster_two_name,euc_distance))
            
            already_computed.append(game_title_one)

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
            normalized_list.append((edge[0],edge[1], 1-norm_w))

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
        with open(self.run_path + "/graph_communities.json", "w") as f:
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

"""
data = "outputs/17_06_2023_11_28_35/"

g = ClusterGraph(data,
    low_cluster_filter = 5,
    community_resolution = 1.2,
    edge_threshold=0
    )

g.save_to_json()
"""


