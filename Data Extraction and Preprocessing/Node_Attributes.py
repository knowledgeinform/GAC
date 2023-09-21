'''
Node Attributes Computations

This file contains functions for computing various node attributes on the MICrONS Minnie dataset.

The following attributes are outlined and computed below.
Of note, the data used for these computations is a direct query of the synapse_pni_2 table using CaveClient
'''

import os
import numpy as np 
import json 
from caveclient import CAVEclient
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from cortical_layers.LayerPredictor import LayerClassifier
import networkx as nx
import pcg_skel

client = CAVEclient('minnie65_phase3_v1', pool_block=True)

def allens_prediction(node_df):
    '''
    Returns Allen Institute's cortical layer prediction

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord

        Returns:
                node_df (Pandas dataframe) with 2 additional columns: the pt_coords in nm and the layer prediction
    '''

    node_df['pt_position_nm']=node_df.apply(lambda row:row.coord_positions*np.array([4,4,40]),axis=1)
    points_nm = np.stack(node_df['pt_position_nm'].values) # soma locations, synapse locations, etc. 
    c = LayerClassifier(data="smooth_minnie65_full_prediction.json")  # an aligned volume name
    node_df["Allen_layers"] = c.predict(points_nm) 
    return node_df

def gac_prediction(node_df):

    '''
    Returns GAC's cortical layer prediction

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord

        Returns:
                node_df (Pandas dataframe) with 2 additional columns: the pt_coords in nm and the layer prediction
    '''

    DATA_DIR = os.path.join(os.getcwd())
    f = open(os.path.join(DATA_DIR, 'smooth_minnie65_full_prediction.json'))
    data = json.load(f)
    cols_nm = np.array(data['cols_nm'])
    bounds_nm = np.array(data['bounds'])*(10**6)

    # Define the features x and z for linear regression
    x1 = cols_nm[:,0,0]
    x2 = cols_nm[:,0,2]
    x_lab = np.array([x1,x2]).T
    
    predictions = []

    X_trainA, X_test, y_trainA, y_test = train_test_split(x_lab, bounds_nm[:,4], test_size = 0.05, random_state = 0)
    modelA = LinearRegression().fit(X_trainA, y_trainA)
    X_trainB, X_test, y_trainB, y_test = train_test_split(x_lab, bounds_nm[:,3], test_size = 0.05, random_state = 0)
    modelB = LinearRegression().fit(X_trainB, y_trainB)
    X_trainC, X_test, y_trainC, y_test = train_test_split(x_lab, bounds_nm[:,2], test_size = 0.05, random_state = 0)
    modelC = LinearRegression().fit(X_trainC, y_trainC)
    X_trainD, X_test, y_trainD, y_test = train_test_split(x_lab, bounds_nm[:,1], test_size = 0.05, random_state = 0)
    modelD = LinearRegression().fit(X_trainD, y_trainD)
    X_trainE, X_test, y_trainE, y_test = train_test_split(x_lab, bounds_nm[:,0], test_size = 0.05, random_state = 0)
    modelE = LinearRegression().fit(X_trainE, y_trainE)

    for index, row in node_df.iterrows():
        x = row['pt_position_nm'][0]
        z = row['pt_position_nm'][2]
        y = row['pt_position_nm'][1]

        x_pred = np.array([x,z]).reshape(1,-1)
        #Bound 1  top most   highest....
        y_A = modelA.predict(x_pred)  # this will return a predicted y i.e. the vertical dimension -- along the layers
        #Bound 2 
        y_B = modelB.predict(x_pred)
        #Bound 3
        y_C = modelC.predict(x_pred)
        #Bound 4
        y_D = modelD.predict(x_pred)
        #Bound 5
        y_E = modelE.predict(x_pred)

        if y >= y_A:
            label = 'WM'
        elif y_B <= y < y_A:
            label = 'L6'
        elif y_C <= y < y_B:
            label = 'L5'
        elif y_D <= y < y_C:
            label = 'L4'
        elif y_E <= y < y_D:
            label = 'L23'
        else:
            label = 'L1'

        predictions.append(label)

    node_df['Gac Layers'] = predictions

    return node_df


def in_out_degs(synapse_df, node_df):

    '''
    Returns In and Out Degree

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord
                synapse_df (Pandas dataframe): synapse_pni_2 dataframe

        Returns:
                node_df (Pandas dataframe) with 2 additional columns: the pt_coords in nm and the layer prediction
    '''

    G = nx.from_pandas_edgelist(synapse_df, source="pre_pt_root_id", target="post_pt_root_id", create_using=nx.DiGraph)
    in_degree = list(dict(G.in_degree()).values())
    out_degree = list(dict(G.out_degree()).values())
    node_df['in degree'] = in_degree
    node_df['out degree'] = out_degree
    return node_df


def cell_type(node_df):

    '''
    Returns cell type and classification

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord

        Returns:
                node_df (Pandas dataframe) with 2 additional columns: the cell type and cell classification
    '''

    table_soma = client.materialize.query_table('allen_soma_coarse_cell_class_model_v1')

    data_outputs_for_e_i = []
    data_outputs_for_type = []
    
    for index, row in node_df.iterrows():
        seg_id = row['seg_ids']
        if 1 <= sum(table_soma['pt_root_id'].isin([seg_id])):
            data_outputs_for_e_i.append(table_soma["classification_system"][table_soma[table_soma["pt_root_id"] == seg_id].index].to_numpy()[0])
            data_outputs_for_type.append(table_soma["cell_type"][table_soma[table_soma["pt_root_id"] == seg_id].index].to_numpy()[0])

        else:
            data_outputs_for_e_i.append('Null')
            data_outputs_for_type.append('Null')
    
    node_df['Cell Classification'] = data_outputs_for_e_i
    node_df['Cell Type'] = data_outputs_for_type

    return node_df


def get_num_soma(node_df):

    '''
    Returns number of soma for each defined segmentation id

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord

        Returns:
                node_df (Pandas dataframe) with 1 additional column for number of soma
    '''

    somas = []

    for index, row in node_df.iterrows():
        root_id = row['seg_ids']
        cave_client = CAVEclient('minnie65_phase3_v1')
        soma = cave_client.materialize.query_table(
            'nucleus_neuron_svm',
            filter_equal_dict={'pt_root_id': root_id}
        )
    somas.append(len(soma))
    node_df['Num_of_soma'] = somas
    return node_df


def get_skeleton(root_id, **kwargs):
    """Find euclidean-space skeleton end point vertices from the pychunkedgraph
    Parameters
    ----------
    root_id : uint64
        Root id of the neuron to skeletonize
    client : caveclient.CAVEclientFull or None, optional
        Pre-specified cave client for the pcg. If this is not set, datastack_name must be provided. By default None
    datastack_name : str or None, optional
        If no client is specified, a CAVEclient is created with this datastack name, by default None
    cv : cloudvolume.CloudVolume or None, optional
        Prespecified cloudvolume instance. If None, uses the client info to make one, by default None
    refine : 'all', 'ep', 'bp', 'epbp', 'bpep', or None, optional
        Selects how to refine vertex locations by downloading mesh chunks. Unrefined vertices are placed in the
        center of their chunk in euclidean space.
        * 'all' refines all vertex locations. (Default)
        * 'ep' refines end points only
        * 'bp' refines branch points only
        * 'bpep' or 'epbp' refines both branch and end points.
       * None refines no points.
        * 'chunk' Keeps things in chunk index space.
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    root_point_search_radius : int, optional
        Distance in euclidean space to look for segmentation when finding the root vertex, by default 300
    collapse_soma : bool, optional,
        If True, collapses vertices within a given radius of the root point into the root vertex, typically to better
        represent primary neurite branches. Requires a specified root_point. Default if False.
    collapse_radius : float, optional
        Max distance in euclidean space for soma collapse. Default is 10,000 nm (10 microns).
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    return_mesh : bool, optional
        If True, returns the mesh in chunk index space, by default False
    return_l2dict : bool, optional
        If True, returns the tuple (l2dict, l2dict_r), by default False.
        l2dict maps all neuron level2 ids to skeleton vertices. l2dict_r maps skeleton indices to their direct level 2 id.
    return_l2dict_mesh : bool, optional
        If True, returns the tuple (l2dict_mesh, l2dict_mesh_r), by default False.
        l2dict_mesh maps neuron level 2 ids to mesh vertices, l2dict_r maps mesh indices to level 2 ids.
    return_missing_ids : bool, optional
        If True, returns level 2 ids that were missing in the chunkedgraph, by default False. This can be useful
        for submitting remesh requests in case of errors.
    nan_rounds : int, optional
        Maximum number of rounds of smoothing to eliminate missing vertex locations in the event of a
        missing level 2 mesh, by default 20. This is only used when refine=='all'.
    segmentation_fallback : bool, optional
        If True, uses the segmentation in cases of missing level 2 meshes. This is slower but more robust.
        Default is True.
    cache : str or None, optional
        If set to 'service', uses the online l2cache service (if available). Otherwise, this is the filename of a sqlite database with cached lookups for l2 ids. Optional, default is None.
    n_parallel : int, optional
        Number of parallel downloads passed to cloudvolume, by default 1
    Returns
    -------
    end_points : TrackedArray
        Skeleton end point vertices in euclidean space
    """
    sk_l2 = pcg_skel.pcg_skeleton(root_id, **kwargs)
    end_points = sk_l2.vertices[sk_l2.end_points, :]
    return sk_l2, sk_l2.edges, sk_l2.vertices

def total_run_length(node_df):

    '''
    Returns total run length for each seg_id in the node table

        Parameters:
                node_df (Pandas dataframe): node dataframe which contains object seg_id and its associated coord

        Returns:
                node_df (Pandas dataframe) with 1 additional column for total run length
    '''

    run_lengths = []

    for index, row in node_df.iterrows():
        seg_id = row['seg_ids']
        skel_edges=get_skeleton(seg_id, client = CAVEclient('minnie65_phase3_v1'))[1]
        skel_vertices=get_skeleton(seg_id, client = CAVEclient('minnie65_phase3_v1'))[2]
        node_col1=[]
        node_col2=[]
        coordset_col1=[]
        coordset_col2=[]
        distance=[]
        for i in range(skel_edges.shape[0]):
            node_col1.append(skel_edges[i][0])
            node_col2.append(skel_edges[i][1])
        
        for j in range(skel_edges.shape[0]):
            coordset_col1.append(skel_vertices[node_col1[j]])
            coordset_col2.append(skel_vertices[node_col2[j]])
        for k in range(skel_edges.shape[0]):
            x1=coordset_col1[k][0]
            y1=coordset_col1[k][1]
            z1=coordset_col1[k][2]
            x2=coordset_col2[k][0]
            y2=coordset_col2[k][1]
            z2=coordset_col2[k][2]
    
            dist=np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            distance.append(dist)
        total_distance=np.sum(distance)
        run_lengths.append(total_distance)

    node_df['Run_length'] = run_lengths

    return node_df