# Connectome Graph Analysis Description

The June MICrONS data is the Full Connectome Data as of 06/09/2022. This dataset contains all accurately indentified nuerons and synapses.
Node and edge attributes were computed and stored in this dataset and were used for analysis.
Here we demonstrate a simple process pipeline for analyzing a large scale attribute-rich connectome graph, leveraging many different techniques.

### Attributes Extracted and Embedded in the Connectome Graph
#### Node Attributes
1. Cell Classification (E/I)
2. Cell Type
3. Cell in-synapse count
4. Cell out-synapse count
5. Cortical Layer 

#### Edge Attributes
1. Soma-Soma Euclidean Distance

## Process Pipeline
1. Generate Node and Edge tables (csvs) from the june dataset and extract relevant attributes for each node and edge (june_Minnie_connectome.py)
2. Create Graph object given node and edge tables
3. Apply Node2vec Embedding and Random Forest Anomaly Detection algorithm
4. Store resulting anomalous nodes into csvs for further analysis (located in respective folders for each method applied)
5. Sorted the anomalous nodes data to isolate the most reoccuring nodes accross majority of the methods
6. Extracted features of these filtered anomalous nodes to uncover and learn connectivity patterns
7. Applied Eigenvector centrality and Pagerank algorithms to determine "important" nodes in each Subgraph
8. Extracted feature of these importance nodes to uncover and learn additional connectivity patterns

## Files Descriptions
1. june_Minnie_Connectome.py --> Extract node and edge table from the cleaned june dataset
2. node2vec_analysis_example.ipynb --> example use-case for Layer 1 subgraph consisting of node2vec embedding and kmeans analysis
3. Connectome_Data_Analysis_June.ipynb --> Graph metrics and data analysis on the June Connectome
4. AD_comparison.ipynb --> Compares the anomalous nodes detected by all the anomaly detection methods
5. By_layer_subgraph_analysis.py -> Extract dataframe by layer and return graph object using the synapse_pni_2 query

