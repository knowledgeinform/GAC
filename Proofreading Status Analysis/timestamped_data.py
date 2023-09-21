import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('default')
from caveclient import CAVEclient
from datetime import datetime, timedelta
from meshparty import trimesh_io
from meshparty import trimesh_vtk
from meshparty import skeleton_o
import pcg_skel

client = CAVEclient("minnie65_phase3_v1")

def get_dates(months_back):
    '''
        Function to get timestamps and synapses_pni_2 at those timestamps

        Parameter:
            months_back (int): Number of montlhy timestamps you want to generate going back x months.
            Works best going no more than 12 months back.
        Outputs:
            date (datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])):
            timestamp for CAVEclient table
            dates_df (pandas dataframe): synapses_pni_2 dataframe queried at a specific timestamp
    '''
    current_time = datetime.now()
    dates = []
    for i in range(months_back):
        current_time = current_time - timedelta(days=30)
        dates.append(current_time)
    dates_df = {}
    for i in range(len(dates)):
        dates_df[i] = client.materialize.query_table('synapses_pni_2', timestamp=dates[i])
    return dates, dates_df

def get_stats(synapse_table):
    '''
        Function that creates a node list from the input synapses table, adds node attributes, 
        and then uses those attributes to get stats

        Parameter: 
            synapse_table (pandas dataframe): synapses_pni_2 table
        Output:
            updated_df_stats (list): List of stats - Number of Nodes, Average In Degree, Average Out Degree,
            Density, Transitivity, Average Clustering, Average Degree Centrality,
            Number of Strongly Connected Components, Average Connectivity
    '''
    pre_coords = synapse_table[['pre_pt_root_id', 'pre_pt_position']]
    post_coords = synapse_table[['post_pt_root_id', 'post_pt_position']]

    data1 = pd.DataFrame()
    data1['seg_ids'] = pre_coords['pre_pt_root_id']
    data1['coord_positions'] = pre_coords['pre_pt_position']

    data2 = pd.DataFrame()
    data2['seg_ids'] = post_coords['post_pt_root_id']
    data2['coord_positions'] = post_coords['post_pt_position']

    updated_nodes_df = pd.concat([data1, data2], ignore_index=False, axis=0)
    updated_nodes_df = updated_nodes_df.drop_duplicates(subset=['seg_ids'])

    from MinnieData_Table_Extraction import get_nodes_table
    from Node_Attributes import allens_prediction, gac_prediction, in_out_degs, cell_type, get_num_soma

    # The following functions compute various attributes for internal analysis
    # This returns a dataframe with all relevant attributes from synapses

    x = allens_prediction(updated_nodes_df)
    df = in_out_degs(synapse_table, x)
    df2 = gac_prediction(x)
    updated_df = cell_type(df2)
    #updated_df = get_num_soma(updated_df)
    
    G = nx.from_pandas_edgelist(synapse_table, source="pre_pt_root_id", target="post_pt_root_id", create_using=nx.DiGraph)

    updated_df_stats = []
    updated_df_stats.append(len(G.nodes()))
    avg_in = np.average(updated_df["in degree"].to_numpy())
    updated_df_stats.append(avg_in)
    avg_out = np.average(updated_df["out degree"].to_numpy())
    updated_df_stats.append(avg_out)
    density = nx.density(G)
    updated_df_stats.append(density)
    transitivity = nx.transitivity(G)
    updated_df_stats.append(transitivity)
    avg_clustering = nx.average_clustering(G)
    updated_df_stats.append(avg_clustering)
    degree_centrality = nx.degree_centrality(G)
    vals = np.fromiter(degree_centrality.values(), dtype=float)
    avg_degree_centrality = np.average(vals)
    updated_df_stats.append(avg_degree_centrality)
    #num_strongly_connected_components = nx.number_strongly_connected_components(G)
    #updated_df_stats.append(num_strongly_connected_components)
    #avg_connectivity = nx.average_node_connectivity(G)
    #updated_df_stats.append(avg_connectivity)

    return updated_df_stats
    
def get_dated_stats(seg_id, date):
    '''
        Function to track the progress of seg_ids across different time stamped dataframes

        Parameters:
            seg_id: pt_root_id (pre or post) from synapse table
            date (datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])):
            timestamp for CAVEclient table
        Outputs:
            returns: pre_synapses_count, post_synapses_count, soma_count, total_distance
    '''
    #Synapse Count
    cave_client = CAVEclient('minnie65_phase3_v1')
    pre_synapses = cave_client.materialize.query_table('synapses_pni_2', timestamp=date,
        filter_in_dict={'pre_pt_root_id': [seg_id]},
        select_columns=['ctr_pt_position', 'pre_pt_root_id']
    )
    post_synapses = cave_client.materialize.query_table(
        'synapses_pni_2', timestamp=date,
        filter_in_dict={'post_pt_root_id': [seg_id]},
        select_columns=['ctr_pt_position', 'post_pt_root_id']
    )
    
    #Soma Count
    client = CAVEclient('minnie65_phase3_v1')
    soma = client.materialize.query_table(
        'nucleus_neuron_svm', timestamp=date,
        filter_equal_dict={'pt_root_id': seg_id}
    )

    return len(pre_synapses), len(post_synapses), len(soma)
#########
    #Run Length
    sk_l2 = pcg_skel.pcg_skeleton(seg_id, client = CAVEclient('minnie65_phase3_v1'))
    end_points = sk_l2.vertices[sk_l2.end_points, :]
    skel_edges=sk_l2.edges
    skel_vertices=sk_l2.vertices
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
  
    return len(pre_synapses), len(post_synapses), len(soma), total_distance
###########

def lineage_graph(seg_id):
    '''
        Function to get the lineage graph of any seg_id and visualize it's seg_id progression
        as proofreading as gone on

        Parameters:
            seg_id: pt_root_id (pre or post) from synapse table
          
        Outputs:
            lineage_dict (dictionary): Dictionary giving a seg_id and its associated 
                pre_synapses_count, post_synapses_count, soma_count
            lineage_graph (networkx graph): Visualization that shows how the seg_ids have transformed over time
    '''
    
    G = client.chunkedgraph.get_lineage_graph(seg_id, as_nx_graph=True)
    lineage = client.chunkedgraph.get_lineage_graph(seg_id)
    lineage_ids = list(G.nodes())
    lineage_ids_info = []
    for i in range(len(lineage_ids)):
        if (datetime.utcfromtimestamp(lineage["nodes"][i]["timestamp"]) < datetime(2021, 7, 1, 9, 14, 42, 277291)):
            try:
                lineage_ids_info.append(
                    get_dated_stats(lineage_ids[i], datetime(2021, 7, 1, 9, 14, 42, 277291)))
            except ValueError:
                lineage_ids_info.append((0, 0, 0))
        else:
            #lineage_ids_info.append(prog(lineage_ids[i], datetime.utcfromtimestamp(lineage["nodes"][i]["timestamp"])))
            try:
                lineage_ids_info.append(
                    get_dated_stats(lineage_ids[i], datetime(2022, 7, 1, 9, 14, 42, 277291)))
            except ValueError:
                lineage_ids_info.append((0, 0, 0))
    
    lineage_dict = dict()
    for i in range(len(lineage_ids)):
        lineage_dict[lineage_ids[i]] = lineage_ids_info[i]
    
    plt.figure(figsize=(10, 6))
    lineage_graph = nx.draw_networkx(G, node_size=200, edge_color="#1f78b4", font_size=8)
    
    return lineage_dict, lineage_graph


def plot_progression(seg_id):
    '''
        Function to visualize how the in and out degree of a seg_id and its lineage have progressed with time

        Parameters:
            seg_id: pt_root_id (pre or post) from synapse table
          
        Outputs:
            lineage_dict (dictionary): Dictionary giving a seg_id and its associated 
                pre_synapses_count, post_synapses_count, soma_count
            lineage_graph (networkx graph): Visualization that shows how the seg_ids have transformed over time
            In_Out_Degree_graph: Visualization that shows how in and out degree of the seg_ids have transformed over time
    '''

    Timestamp = []
    In_degree = []
    Out_degree = []

    G = client.chunkedgraph.get_lineage_graph(seg_id, as_nx_graph=True)
    lineage = client.chunkedgraph.get_lineage_graph(seg_id)
    lineage_ids = list(G.nodes())
    lineage_dict = lineage_graph(seg_id)[0]
    for i in range(len(lineage_dict.keys())):
        Timestamp.append(datetime.utcfromtimestamp(
            lineage["nodes"][i]["timestamp"]))
    for i in lineage_dict.keys():
        In_degree.append(lineage_dict[i][0])
    for i in lineage_dict.keys():
        Out_degree.append(lineage_dict[i][1])

    plt.figure(figsize=(9, 8))
    plt.subplot(2, 1, 1)
    plt.plot(Timestamp, In_degree)
    plt.xlabel("Timestamp")
    plt.ylabel("In Degree")
    plt.subplot(2, 1, 2)
    plt.plot(Timestamp, Out_degree)
    plt.xlabel("Timestamp")
    plt.ylabel("Out Degree")

    return lineage_dict, plt.show()




