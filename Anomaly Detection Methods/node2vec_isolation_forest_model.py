# import relevant libraries

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from node2vec.node2vec import Node2Vec
from sklearn.preprocessing import StandardScaler


# Extract final node and edge tables below

nodes = pd.read_csv('final_connectome_node_table.csv')
edges = pd.read_csv('final_connectome_edge_table (1).csv')

def filter_by_layer(nodes, edges, layer):
    '''
    Extract subgraphs from June connectome data by layer for analysis
    Parameters: 
        nodes (Pandas dataframe) final june node dataframe
        edges (Pandas dataframe) final june edge dataframe 
        layer (string) layer of interest to extract
    Return: 
        node table subgraph (Pandas dataframe) by layer
        edge table subgraph (Pandas dataframe) by layer
        Subgraph G (NetworkX graph object)
    '''

    # Filter entire node table to only those in layer 
    rslt_synapse_df = nodes[nodes['layer'] == layer] 
    
    # Get list of unique neurons
    list_of_neurons = set(rslt_synapse_df['pt_root_id'].to_list())

    # Filter the synapse tables  extract all unique connections between nodes
    filtered = edges[edges['Source'].isin(list_of_neurons)]
    filtered2 = filtered[filtered['Target'].isin(list_of_neurons)]

    # create graph object
    G = nx.from_pandas_edgelist(filtered2, source="Source", target="Target", create_using=nx.DiGraph, edge_attr=["Euclidean_Distance"])
    list_of_current_nodes = set(list(G.nodes))
    missing_nodes = list(list_of_current_nodes ^ list_of_neurons)
    G.add_nodes_from(missing_nodes)
    
    return rslt_synapse_df, filtered2, G

# Apply filter_by_layer on each layer and store Graph objects
node_L1, bedge_L1, G_L1 = filter_by_layer(nodes, edges, 'L1')  # sample run for subgraph L1
node_L23, bedge_L23, G_L23 = filter_by_layer(nodes, edges, 'L23')  # sample run for subgraph L23
node_L4, bedge_L4, G_L4 = filter_by_layer(nodes, edges, 'L4')  # sample run for subgraph L4
node_L5, bedge_L5, G_L5 = filter_by_layer(nodes, edges, 'L5')  # sample run for subgraph L5
node_L6, bedge_L6, G_L6 = filter_by_layer(nodes, edges, 'L6')  # sample run for subgraph L6
node_LWM, bedge_LWM, G_WM = filter_by_layer(nodes, edges, 'WM')  # sample run for subgraph LWM

# Read in the node2vec embeddings and store in dataframes
embedding_L1 = pd.read_csv('L1_embeddings.csv')
embedding_L23 = pd.read_csv('L23_embeddings.csv')
embedding_L4 = pd.read_csv('L4_node2vec.csv')
embedding_L5 = pd.read_csv('L5_node2vec.csv')
embedding_L6 = pd.read_csv('L6_node2vec.csv')
embedding_WM = pd.read_csv('LWM_embeddings.csv')


# Apply IsolationForest to the embeddings 
from sklearn.ensemble import IsolationForest

def extract_anomalous_nodes(embedding, node_table):
    # fit the model
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features = 30)
    model.fit(embedding)
    y_pred = model.predict(embedding)

    embedding['Anomaly_Prediction'] = y_pred

    outliers = embedding.loc[embedding['Anomaly_Prediction']==-1]

    list_of_outliers = outliers['Unnamed: 0'].to_numpy()

    filtered_outliers = node_table[node_table['pt_root_id'].isin(list_of_outliers)]

    return filtered_outliers

# run the function and save data results for further analysis
rslt1 = extract_anomalous_nodes(embedding_L1, node_L1)
rslt23 = extract_anomalous_nodes(embedding_L23, node_L23)
rslt4 = extract_anomalous_nodes(embedding_L4, node_L4)
rslt5 = extract_anomalous_nodes(embedding_L5, node_L5)
rslt6 = extract_anomalous_nodes(embedding_L6, node_LWM)
rsltWM = extract_anomalous_nodes(embedding_WM, node_LWM)


# Here, we save all rslts (of anomalous nodes) using node2vec embedding and random forest
rslt1.to_csv('Node2vec_rslts_L1.csv')
rslt23.to_csv('Node2vec_rslts_L23.csv')
rslt4.to_csv('Node2vec_rslts_L4.csv')
rslt5.to_csv('Node2vec_rslts_L5.csv')
rslt6.to_csv('Node2vec_rslts_L6.csv')
rsltWM.to_csv('Node2vec_rslts_LWM.csv')


