'''
Minnie Connectome Graph Analysis
June Data 

Output csv files for computed node and edge attributes/tables


Goal: Given the cleaned june data file, extract the all the nodes and edges from the file to create a Graph
Save results into csv files
'''

# import relevant libraries 
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from caveclient import CAVEclient
from joblib import Parallel, delayed
import numpy as np

# Read in the pickle data from June
df = pd.read_pickle("neurons_synapse_target_for_filtered_updated_neuron_only_table_after_proofreading_with_metadata_060922_65029_neurons_stitched_with_pcm.pkl")
df = df.drop(['classification_system','valid','cell_type','pt_supervoxel_id','cell_type_to_use','acceptable_out_degree','acceptable_in_degree','acceptable_out_connections_neighbors','acceptable_in_connections_neighbors','acceptable_out_synperconn','acceptable_in_synperconn'], axis = 1)

# Extract all list of nodes in the data
list_of_neurons = df['pt_root_id'].to_numpy()

def extract_neighbors(seg_id):
    '''
    Extract all neighbors of input seg_id
    Parameters: seg_id (int65) A given seg_id from the list_of_neurons
    Output: synapse_edge_df (pandas dataframe) 
    '''
    pre_syn_target_ids = df.loc[df['pt_root_id'] == seg_id, 'pre_syn_target_ids'].to_numpy()[0]
    seg_id_repeated = [seg_id] * len(pre_syn_target_ids)
    edge_table_1 = pd.DataFrame(
    {'Source': pre_syn_target_ids,
     'Target': seg_id_repeated,
    })
    post_syn_target_ids = df.loc[df['pt_root_id'] == seg_id, 'post_syn_target_ids'].to_numpy()[0]
    seg_id_repeated_2 = [seg_id] * len(post_syn_target_ids)
    edge_table_2 = pd.DataFrame(
    {'Source': seg_id_repeated_2,
     'Target': post_syn_target_ids,
    })
    synapse_edge_df = pd.concat([edge_table_1,edge_table_2])
    return synapse_edge_df


# Append all edge files to a single dataframe 
synapses_table = pd.DataFrame()

for i in tqdm(list_of_neurons):
    edge_table = extract_neighbors(i)
    synapses_table = pd.concat([edge_table, synapses_table])

# Extract synapses that exist for list of nodes
dropped_pre_df = synapses_table[synapses_table['Source'].isin(list_of_neurons)]
synapses_table = dropped_pre_df[dropped_pre_df['Target'].isin(list_of_neurons)]


# create node table
df_nodes = df.copy()
df_nodes = df_nodes.drop(['pre_syn_target_ids','post_syn_target_ids','post_syn_target_number_synapses','pre_syn_target_number_synapses'], axis = 1)

list_of_source_pos = []
list_of_target_pos = []

# organize in_nodes and out_nodes for each neuron in the list
for index, row in tqdm(synapses_table.iterrows()):
    seg_id_1 = row['Source']
    seg_id_2 = row['Target']
    source_pt_post = df_nodes.loc[df_nodes['pt_root_id']== seg_id_1, 'pt_position'].iloc[0]
    target_pt_post = df_nodes.loc[df_nodes['pt_root_id']== seg_id_2, 'pt_position'].iloc[0]
    list_of_source_pos.append(source_pt_post)
    list_of_target_pos.append(target_pt_post)

# Add source and target nodes to synapse table
synapses_table['Source_pt_position'] = list_of_source_pos
synapses_table['Target_pt_position'] = list_of_target_pos

# Add euclidean distance attribute to synapse table

distances = []

for index, row in synapses_table.iterrows():
    x1 = row['Source_pt_position'][0]
    x2 = row['Target_pt_position'][0]
    y1 = row['Source_pt_position'][1]
    y2 = row['Target_pt_position'][1]
    z1 = row['Source_pt_position'][2]
    z2 = row['Target_pt_position'][2]

    distance=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    distances.append(distance)
synapses_table['Euclidean_Distance'] = distances 

# save final results to csv files for further analysis
df_nodes.to_csv('final_connectome_node_table.csv')
synapses_table.to_csv('final_connectome_edge_table.csv')
