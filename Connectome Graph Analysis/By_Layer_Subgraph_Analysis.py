'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By-Layer Subgraph Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-> Goal: Extract dataframe by layer and return graph object
-> Apply attributes to the node-neuron model

Outline 
1. Apply these functions for the entire matching graph and df
2. Run PCA and create viz for each layer and graph by node attributes
3. Compute graph statistics for each sublayer and compare
4. Apply node2vec embedding for each layer and look for pattern among 
   the anomalies outputted by each sublayer
5. Create more viz to represent the data
'''


# Import relevant libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caveclient import CAVEclient
import networkx as nx
from cortical_layers.LayerPredictor import LayerClassifier
from MinnieData_Table_Extraction import get_nodes_table
from Node_Attributes import allens_prediction, gac_prediction, in_out_degs, cell_type
from MinnieData_Table_Extraction import extract_full_nodes
from node2vec.node2vec import Node2Vec


# Extract data from api
client = CAVEclient('minnie65_phase3_v1', pool_block=True)
synapses = client.materialize.query_table('synapses_pni_2')


def separate_by_layer (synapses, layer):
    '''
    Separate synapse_pni_2 table by layer and generate graph object from it

        Parameters:
                node_df (Pandas dataframe): synapses_pni_2 dataframe: query using CAVE
                layer (str: 'L1','L23','L4','L5','L6','WM')

        Returns:
                node_df (Pandas dataframe) reduced node df from defined layer
                edge_df (Pandas dataframe) reduced edge df from defined layer
                G (networkx graph object) generated graph object from node_df and edge_df
    '''

    synapses['ctr_pt_position_nm']=synapses.apply(lambda row:row.ctr_pt_position*np.array([4,4,40]),axis=1)
    points_nm = np.stack(synapses['ctr_pt_position_nm'].values) # soma locations, synapse locations, etc. 
    c = LayerClassifier(data="smooth_minnie65_full_prediction.json")  # an aligned volume name
    synapses["Synapse_Allen_layers"] = c.predict(points_nm) 

    '''
    synapses['pre_pt_nm']=synapses.apply(lambda row:row.pre_pt_position*np.array([4,4,40]),axis=1)
    points_nm = np.stack(synapses['pre_pt_nm'].values) # soma locations, synapse locations, etc. 
    c = LayerClassifier(data="smooth_minnie65_full_prediction.json")  # an aligned volume name
    synapses["Pre_Allen_Layers"] = c.predict(points_nm) 

    synapses['post_pt_nm']=synapses.apply(lambda row:row.post_pt_position*np.array([4,4,40]),axis=1)
    points_nm = np.stack(synapses['post_pt_nm'].values) # soma locations, synapse locations, etc. 
    c = LayerClassifier(data="smooth_minnie65_full_prediction.json")  # an aligned volume name
    synapses["Post_Allen_Layers"] = c.predict(points_nm) "
    '''

    rslt_synapse_df = synapses[synapses['Synapse_Allen_layers'] == layer] 

    new_df = get_nodes_table(rslt_synapse_df)
    from Node_Attributes import allens_prediction, gac_prediction, in_out_degs, cell_type
    x = allens_prediction(new_df)
    df = in_out_degs(rslt_synapse_df,x)
    df = gac_prediction(df)
    df2 = cell_type(df)


    #synapses = synapses.drop(columns = ['valid','pre_pt_supervoxel_id','post_pt_supervoxel_id','ctr_pt_position_nm','pre_pt_nm','post_pt_nm'])

    dfa2, dfb2, G2 = extract_full_nodes(df2, rslt_synapse_df)
    dfa2 = dfa2.reset_index()
    dfa2 = dfa2.drop_duplicates(subset = ['seg_ids'])

    return dfa2, dfb2, G2


def structual_embeddings(G):
    '''
    Generate node2vec structural embeddings generation for an input graph G

        Parameters:
                node_df (Pandas dataframe): synapses_pni_2 dataframe: query using CAVE
                layer

        Returns:
                node_df (Pandas dataframe) with 2 additional columns: the pt_coords in nm and the layer prediction
    '''


    node2vec_of_G = Node2Vec(G, dimensions=65, walk_length=45, num_walks=150, p=0.3, q=0.7)
    model = node2vec_of_G.fit(window=10, min_count=1, batch_words=4)  
    # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed 
    # (from the Node2Vec constructor)

    node_embedding_dict = {}

    list_of_nodes = list(G.nodes)

    for i in range(len(list_of_nodes)):
        node = str(list_of_nodes[i])
        vector = model.wv[node]
        node_embedding_dict[node] = vector

    df_embeddings = pd.DataFrame.from_dict(node_embedding_dict, orient='index')

    return df_embeddings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_generation(df_embeddings):
    '''
    PCA analysis for an input networkx graph object

        Parameters:
                df_embeddings (dataframe for node2vec embeddings)

        Returns:
                PCA analysis plot
    '''

    scaler = StandardScaler()

    scaled_df = pd.DataFrame(scaler.fit_transform(df_embeddings), columns = df_embeddings.columns)
    scaled_df.head()

    pca = PCA(n_components=20).fit(scaled_df)

    fig_1 = plt.figure(figsize = (8,6), facecolor='w',edgecolor='k')
    plt.plot(np.arange(1,21), pca.explained_variance_ratio_,'o-')
    plt.title('Explained Variance Ratio vs number of principal components')
    plt.xlabel('number of principle components')
    plt.ylabel('Explained variance ratio')
    plt.xticks([2,4,6,8,10,12,14,16,18,20])
    plt.show()

