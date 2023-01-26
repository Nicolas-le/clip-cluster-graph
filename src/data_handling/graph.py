import networkx as nx

# cluster embeddings and attach specific cluster to embedding and -> image

class ClusterGraph():

    def __init__(self,
    frames_with_clusters):

        self.frames_with_clusters = frames_with_clusters
        self.networkx_graph  = 
        self.analytics = self.get_analytics()

    def create_networkx_graph(self):
        G = nx.Graph()

    def get_analytics(self):
        return {
            degreeC: nx.degree_centrality(self.networkx_graph),
            closenessC: nx.closeness_centrality(self.networkx_graph),
            betweenessC: nx.betweenness_centrality(self.networkx_graph),
            pagerank = nx.pagerank(self.G)
        }


