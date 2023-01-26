from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_for_pca(path):
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"])

    X  = df.drop(columns=["video_id", "timestamp"])
    y = df.loc[:, "video_id":"timestamp"]

    scale = StandardScaler()
    x_scaled = pd.DataFrame(scale.fit_transform(X.values), columns=X.columns, index=X.index)

    return X, y, x_scaled


def perform_pca(x_scaled, y):

    principal_component_count = 20
    principal_components = ["PCA" + str(i) for i in range(1,principal_component_count+1)]

    pca = PCA(n_components=principal_component_count)

    pca_features = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(data=pca_features,columns=principal_components)
    
    pca_df["video_id"] =  y["video_id"]
    pca_df["timestamp"] =  y["timestamp"]
   
    return pca_df

def pca_main():
    _, y, data =  preprocess_for_pca("./resources/transformed_embeddings/tagesschau.csv")

    print("preprocessed...")

    pca_data = perform_pca(data, y)

    pca_data.to_csv("./resources/transformed_embeddings/tagesschau_pca_20.csv")

pca_main()