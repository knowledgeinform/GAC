'''
Edge Attributes Computations

This file contains functions for computing various edge attributes on the MICrONS Minnie dataset.

The following attributes are outlined and computed below.
Of note, the data used for these computations is a direct query of the synapse_pni_2 table using CaveClient
'''

# import relevant libraries
import numpy as np
import pandas as pd
from caveclient import CAVEclient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Computing Edge Attributes Table
cave_client = CAVEclient('minnie65_phase3_v1')

synapse_dataset = cave_client.materialize.query_table("synapses_pni_2")    #this dataset should be the synapses_pni_2 table 
def cell_class(seg_id):
    table_soma =  cave_client.materialize.query_table('allen_soma_coarse_cell_class_model_v1')
    classification = table_soma["classification_system"][table_soma[table_soma["pt_root_id"] == seg_id].index].to_numpy()[0]
    return classification

def cell_type(seg_id):
    table_soma =  cave_client.materialize.query_table('allen_soma_coarse_cell_class_model_v1')
    type1 = table_soma["cell_type"][table_soma[table_soma["pt_root_id"] == seg_id].index].to_numpy()[0]
    return type1

# Cell Type --> type of neurons 
# Classification --> E or I

def FromTo_Cell_type(synapseID):
    # define a dictionary for each of the type of neurons
    table_soma = cave_client.materialize.query_table('allen_soma_coarse_cell_class_model_v1')
    cell_type_lst = table_soma.cell_type.unique()  # list of cell_Types
    class_type_lst = table_soma.classification_system.unique() # E or I in list

    matchings_cellType = {}
    counter = 0
    for i in cell_type_lst:
        for j in cell_type_lst:
            matchings_cellType['{}_{}'.format(i,j)] = counter
            counter = counter + 1

    matchings_CellClass = {}
    counter1 = 0
    for i in class_type_lst:
        for j in class_type_lst:
            matchings_CellClass['{}_{}'.format(i,j)] = counter1
            counter1 = counter1 + 1

    df = cave_client.materialize.query_table('synapses_pni_2')
    pre_neuron = df.loc[df['id'] == synapseID, 'pre_pt_root_id']
    post_neuron = df.loc[df['id'] == synapseID, 'post_pt_root_id']
    pre_type = cell_type(pre_neuron)
    post_type = cell_type(post_neuron)
    pre_class = cell_class(pre_neuron)
    post_class = cell_class(post_neuron)
    keyA = pre_type + '_' + post_type
    keyB = pre_class + '_' + post_class
    return matchings_cellType[keyA], matchings_CellClass[keyB]


def euclidean_distance(synapse_df):

    '''
    Returns euclidean distance of a nucelus to nucleus connection
        Parameters:
            synapse_df (Pandas dataframe) of all synapse connections 
        Returns:
            synapse_df (Pandas dataframe) with column for euclidean distance
    '''

    distances = []

    for index, row in synapse_df.iterrows():
        x1=row['pre_pt_position_x']
        y1=row['pre_pt_position_y']
        z1=row['pre_pt_position_z']
        x2=row['post_pt_position_x']
        y2=row['post_pt_position_y']
        z2=row['post_pt_position_z']

        distance=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

        distances.append(distance)

    synapse_df['Euclidean Distance'] = distances

    return synapse_df