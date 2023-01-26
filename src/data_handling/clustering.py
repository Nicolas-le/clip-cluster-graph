import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("./resources/transformed_embeddings/tagesschau_pca_20.csv")

columns_for_clustering = [e for e in list(df.columns) if e not in ('Unnamed: 0', "video_id", "timestamp")]

kmeans = KMeans(n_clusters=25, n_init=3, max_iter=3000, random_state=2)
kmeans = kmeans.fit(df[columns_for_clustering])
df.loc[:,"cluster"] = kmeans.labels_

only_cluster_df = df.drop(columns=columns_for_clustering)

#print(only_cluster_df)
#print(only_cluster_df["cluster"].value_counts())

only_cluster_df.to_csv("./resources/clustered_embeddings/tagesschau_kmeans_25_pca20.csv")

