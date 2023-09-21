'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MinnieData Table Extractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-> Goal: Extract synapse table into a node-neuron model 
-> Apply attributes to the node-neuron model
'''

import os
import numpy as np 
import json 
from caveclient import CAVEclient
import numpy as np
import pandas as pd
import networkx as nx


client=CAVEclient()
client = CAVEclient('minnie65_phase3_v1')
client.materialize.get_tables()

# extract data file for full minnie65 predictions
DATA_DIR = os.path.join(os.getcwd())
f = open(os.path.join(DATA_DIR, 'smooth_minnie65_full_prediction.json'))
data = json.load(f)


def get_nodes_table(df):

    '''
    Returns Nodes - only dataframe given synapse table 

    Parameters:
            self.df (Pandas dataframe): synapse_pni_2 data table

    Returns:
            node_df (Pandas dataframe) with 2 additional columns: the pt_coords in nm and the layer prediction
    '''

    presynapse_neurons = (df['pre_pt_root_id'].to_numpy()).reshape(-1,1)
    postsynapse_neurons = (df[['post_pt_root_id']].to_numpy()).reshape(-1,1)
    neurons = np.concatenate((presynapse_neurons,postsynapse_neurons))

    '''
        # convert extended position dimensions to a coordinate 
        pre_pt_position_x = (df["pre_pt_position_x"].to_numpy())
        pre_pt_position_y = (df["pre_pt_position_y"].to_numpy())
        pre_pt_position_z = (df["pre_pt_position_z"].to_numpy())
        pre_pt_position = []
        for i in range(pre_pt_position_x.shape[0]):
            pre_position = (pre_pt_position_x[i],pre_pt_position_y[i],pre_pt_position_z[i])
            pre_pt_position.append(pre_position)
        
        #post_pt_postion combined
        post_pt_position_x = (df["post_pt_position_x"].to_numpy())
        post_pt_position_y = (df["post_pt_position_y"].to_numpy())
        post_pt_position_z = (df["post_pt_position_z"].to_numpy())
        post_pt_position = []
        for i in range(post_pt_position_x.shape[0]):
            post_position = (post_pt_position_x[i],post_pt_position_y[i],post_pt_position_z[i])
            post_pt_position.append(post_position)

        df.insert(loc=6, column="pre_pt_position", value=pre_pt_position)
        df.insert(loc=12, column="post_pt_position", value=post_pt_position)
    '''

    pre_coords = df[['pre_pt_root_id', 'pre_pt_position']]
    post_coords = df[['post_pt_root_id', 'post_pt_position']]

    data1 = pd.DataFrame()
    data1['seg_ids'] = pre_coords['pre_pt_root_id']
    data1['coord_positions'] = pre_coords['pre_pt_position']

    data2 = pd.DataFrame()
    data2['seg_ids'] = post_coords['post_pt_root_id']
    data2['coord_positions'] = post_coords['post_pt_position']

    updated_nodes_df = pd.concat([data1, data2], ignore_index=False, axis=0)
    return updated_nodes_df.drop_duplicates(subset=['seg_ids'])


def get_coords(df):
    # Need to call function above with only node attribute calculations
    table_call = get_nodes_table(df)
    table_call.loc[table_call['pt_root_id'] == 864691136041487043, 'pt_position']
    return table_call

def extract_full_nodes(node_df, synapses):
    '''
    Returns Nodes - only dataframe given synapse table 

    Parameters:
            node_df (Pandas dataframe): node_df which contains all attributes already (after running functions in Node_Attributes)
            synapses (Pandas dataframe): synapses_pni_2 data table

    Returns:
            reduced_nodes_df (Pandas dataframe) which contains the reduced number of nodes from matchings
            reduced_edges_df (Pandas dataframe) which contains the reduced number of edges from matchings
            Subgraph_G (networkx Graph object) for the 
    '''

    
    node_df['Cell Type'] = node_df['Cell Type'].replace({'Null':np.nan})
    node_df['Cell Classification'] = node_df['Cell Classification'].replace({'Null':np.nan})
    node_df['Cell Classification'] = node_df['Cell Classification'].replace({'aibs_coarse_excitatory':'e'})
    node_df['Cell Classification'] = node_df['Cell Classification'].replace({'aibs_coarse_inhibitory':'i'})

    reduced_nodes_df = node_df.dropna(axis = 0, how='any')

    list_of_neurons = reduced_nodes_df['seg_ids'].to_numpy()

    distances = []

    for index, row in synapses.iterrows():
        x1 = row['pre_pt_position'][0]
        x2 = row['post_pt_position'][0]
        y1 = row['pre_pt_position'][1]
        y2 = row['post_pt_position'][1]
        z1 = row['pre_pt_position'][2]
        z2 = row['post_pt_position'][2]
    
        distance=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        distances.append(distance)
    synapses['Euclidean_Distance'] = distances 

    G = nx.from_pandas_edgelist(synapses, source="pre_pt_root_id", target="post_pt_root_id", create_using=nx.DiGraph, edge_attr=["size", "Euclidean_Distance"])

    Subgraph_G = G.subgraph(list_of_neurons)
    reduced_edges_df = nx.to_pandas_edgelist(Subgraph_G) 

    return reduced_nodes_df, reduced_edges_df, Subgraph_G

def set_node_attributes(G,node_df):
    '''
    Returns graph object G with set node attributes 
    
    Parameters:
            G (networkx graph object): Non-node-attributed graph
            node_df (Pandas dataframe): One-hot encoded node dataframe from extract_full_nodes 

    Returns:
            G (networkx graph object): Node attributes added to graph object
    '''
    
    node_df_copy=node_df.copy()
    node_df_copy=node_df_copy.drop(columns=['seg_ids','coord_positions','pt_position_nm'])
    node_attributes=node_df_copy.columns.tolist()
    for index, row in node_df.iterrows():
        node_attr_dict={k: float(row.to_dict()[k]) for k in node_attributes}
        G.nodes[row['seg_ids']].update(node_attr_dict)
    return G
