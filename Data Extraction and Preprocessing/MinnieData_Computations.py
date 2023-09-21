'''
This file analyzes the synapses_pni_2 table and returns the complete neuron-only node and edge tables

1. The synapses_pni_2 table is the most up-to-date proofread materialized version of the Minnie Dataset
2. This notebook references and calls other files to computer relevant attributes
3. The output of this file should be Graph Object File, reduced_node csv, reduced_edge csv

Dependencies 
1. caveclient with premissions/access
2. pandas 
3. networkx
4. smooth_minnie65_full_prediction.json
5. Node_Attributes.py
6. MinnieData_Table_Extraction.py
7. cortical_layers
8. sklearn
'''

import pandas as pd
from caveclient import CAVEclient
import networkx as nx

# No not limit to 200,000 rows --> query the entire table (all 337,312,429)
client = CAVEclient('minnie65_phase3_v1', pool_block=True)
synapses = client.materialize.query_table('synapses_pni_2')

# Check number of self-loops
count = 0
count1 = 0
for index, row in synapses.iterrows():
    pre_seg = row['pre_pt_root_id']
    post_seg = row['post_pt_root_id']

    if pre_seg == post_seg:
        count = count + 1
    else:
        count1 = count1 + 1

# Report this result for documentation 
print("Number of self loops: ", count)

from MinnieData_Table_Extraction import get_nodes_table
from Node_Attributes import allens_prediction, gac_prediction, in_out_degs, cell_type, total_run_length 

# The following functions compute various attributes for internal analysis
# This returns a dataframe with all relevant attributes from synapses

new_df = get_nodes_table(synapses)
x = allens_prediction(new_df)
df = in_out_degs(synapses,x)
df = gac_prediction(df)
df2 = cell_type(df)
df3 = total_run_length(df2)

from MinnieData_Table_Extraction import extract_full_nodes
node_df, edge_df, G = extract_full_nodes(df3, synapses)

node_df.to_csv('Reduced_Node_Table.csv')
edge_df.to_csv('Reduced_Edge_Table.csv')

nx.write_gml(G, "Full_Graph_Object", stringizer=None)
