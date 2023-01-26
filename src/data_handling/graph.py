import networkx as nx
from collections import defaultdict
import pandas as pd

# cluster embeddings and attach specific cluster to embedding and -> image

class ClusterGraph():

    def __init__(self,
    frames_with_clusters):

        self.frames_with_clusters = pd.read_csv(frames_with_clusters)
        self.listOfEdges = self.transform_data()
        self.networkx_graph  = self.create_networkx_graph()
        self.analytics = self.get_analytics()

    def transform_data(self):

        clusters = list(self.frames_with_clusters['cluster'].unique())
        videos = list(self.frames_with_clusters['video_id'].unique())

        self.frames_with_clusters = self.frames_with_clusters.drop(columns=["Unnamed: 0.1","Unnamed: 0", "timestamp"])

        for video_id in videos:
            cluster_counts = self.frames_with_clusters[self.frames_with_clusters["video_id"] == video_id].value_counts()
        
            for identifier, count in cluster_counts.items():
                if count < 50:
                    self.frames_with_clusters = self.frames_with_clusters.drop(self.frames_with_clusters[(self.frames_with_clusters.video_id == identifier[0]) & (self.frames_with_clusters.cluster == identifier[1])].index)

        cluster_videos_listing = defaultdict(list)

        for cluster_id in clusters:
            cluster_videos_listing[cluster_id] = list(self.frames_with_clusters["video_id"][self.frames_with_clusters["cluster"]==cluster_id])

        counter = defaultdict(lambda: defaultdict(int))

        for cluster_id, videos in cluster_videos_listing.items():
            for cluster_id2, videos2 in cluster_videos_listing.items():
                if cluster_id == cluster_id2:
                    continue
                counter[cluster_id][cluster_id2] = len(list(set(videos).intersection(videos2))) # normalize weight
        
        listOfEdges = []
        for cluster, counter_dict in counter.items():
            for link_cluster, count in counter_dict.items():
                listOfEdges.append((cluster,link_cluster,count))  
        
        return self.normalize_weights(listOfEdges)

    def normalize_weights(self,listOfEdges):
        
        weights = []
        for edge in listOfEdges:
            weights.append(edge[2])

        minimum, maximum = min(weights), max(weights)

        normalized_list = []
        for edge in listOfEdges:
            norm_w = (edge[2]-minimum) / (maximum-minimum)
            normalized_list.append((edge[0],edge[1], norm_w))

        return normalized_list

    def create_networkx_graph(self):
        graph = nx.Graph()
        e = self.listOfEdges
        graph.add_weighted_edges_from(e)

        return graph

    def get_analytics(self):
        return {
            "degreeC": nx.degree_centrality(self.networkx_graph),
            "closenessC": nx.closeness_centrality(self.networkx_graph),
            "betweenessC": nx.betweenness_centrality(self.networkx_graph),
            "pagerank": nx.pagerank(self.networkx_graph)
        }


g = ClusterGraph("./outputs/26_01_2023_12_46_22/clustered_data.csv")
print(g.networkx_graph)
print(sorted(g.analytics["pagerank"].items(), key=lambda x:x[0]))
print(g.analytics["degreeC"])