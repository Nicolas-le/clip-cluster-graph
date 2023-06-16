from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def reduce_to_500_rows_per_video(df):

    video_id_counts = df['video_id'].value_counts()

    # Create an empty DataFrame to store the transformed data
    transformed_df = pd.DataFrame()

    # Iterate over unique video_ids
    for video_id in df['video_id'].unique():
        # Get the first 500 rows for the current video_id
        selected_rows = df[df['video_id'] == video_id].head(500)
        
        # Append the selected rows to the transformed DataFrame
        transformed_df = transformed_df.append(selected_rows)

    # Reset the index of the transformed DataFrame
    transformed_df.reset_index(drop=True, inplace=True)

    return transformed_df
def split_Xy_scaling(path):

    df = reduce_to_500_rows_per_video(pd.read_csv(path)).drop(columns=["Unnamed: 0"])

    X  = df.drop(columns=["video_id", "timestamp"])
    y = df.loc[:, "video_id":"timestamp"]

    scale = StandardScaler()
    X_scaled = pd.DataFrame(scale.fit_transform(X.values), columns=X.columns, index=X.index)

    return X, y, X_scaled

def perform_pca(x_scaled, y, principal_component_count):

    principal_components_names = ["PCA" + str(i) for i in range(1,principal_component_count+1)]
    pca = PCA(n_components=principal_component_count)
    pca_features = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(data=pca_features,columns=principal_components_names)
    
    pca_df["video_id"] =  y["video_id"]
    pca_df["timestamp"] =  y["timestamp"]
   
    return pca_df

def pca_main(transformed_data_path, config):
    logging.info("Scaling data...")
    _, y, X_scaled = split_Xy_scaling(transformed_data_path)

    logging.info("Perform PCA...")
    pca_data = perform_pca(X_scaled, y, config["principal_components"])

    pca_data.to_csv(config["output_directory"]+ config["embedding_algorithm"] + "_pca_transformed_data.csv")

