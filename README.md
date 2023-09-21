<h1 align="center"> CRIMSON: Graph Analytics for Nanoscale Connectomics (GAC) </h1>

<p align="center">
<img width="650" alt="Layer 1 Visualization" src="https://user-images.githubusercontent.com/97049854/184223712-c68276b4-7a87-4bf8-bde0-4b216ccb4820.png">
</p>

<p align="center">
Cortical Layer 1 Connectome Graph Structural Visualization
</p>

## Objective 
The field of connectomics is dedicated to mapping brain connectivity and topology, allowing researchers to study circuitry motifs, intricate synaptic connections, and cell morphology–all of which are integral building blocks to enable all the complex capabilities of the mind. By studying a 1.4mm x .87mm x .84mm region of a mouse’s visual cortex imaged as part of the MICrONS project (the 1 mm3 dataset), the application of graph theory for anomaly detection may yield novel discoveries about major processing points in the brain and hidden patterns of communication. [[1]](#one) In the future, structural and connectivity comparisons between neurotypical and neurodivergent brains by even reveal contributing factors to the onset and progression of neurological diseases and disorders. Furthermore, these studies continue to fill in the picture of a highly detailed, complete model of brain wiring, opening the door to enhanced artificial neural network design and advancing the study of computing chips based on neuro-biological architectures to greatly increase computing and learning ability. 

This project aims to present a comprehensive scheme to streamline the graphical anomaly detection process by providing compatible functions for data preprocessing and extraction, accessible methods for efficient attributed graph object creation, and straightforward implementations of four varying anomaly detection algorithms, hand-picked for their unique advantages. The process is demonstrated through an in-depth analysis on the MICrONS connectome as of June 2022, including graph analytics over the extended course of proofreading.

Graph Representation of the Connectome:
    Graphs = Nodes + Edges
    Graph of Connectome = Neurons + Synapses
    
## Process Outline

1. Node and Edge Attribute Computation
    * Developed functions for cortical layer classification, neuronal run length, Euclidean distance between neurons, etc.
2. Data Cleaning and Graph Creation
    * Eliminated non-neural cells and falsely identified connectivity to generate graph object representing neural network
3. Embedding and Anomaly Detection on Graph Object
    * Applied both Node2vec embedding with predefined parameteres as well as GNN-based appraoches such as MADAN, DONE, and DOMINANT, some of which outputted predicted anomaly scores
4. Pattern and Relationship Examination for Anomaly Detection
    * Deployed PCA, TSNE, and k-means to visualize abnormal clusters of nodes as well as inspecting feature distribution, PageRank, Eigenvector Centrality, and Neighborhood Analysis 
5. By-Layer Subgraph Analysis on June Connectome Data
    * Studied connectivity patterns acros each cortical layer and compared anomalous nodes
6. Proof-read Neuron Tracking
    * Tracked connectome segments through proof-reading history to calculate graph metrics like in degree, out degree, and orphan count
    

## File Organization and Documentation

### Anomaly Detection Methods
* Notebooks for anomaly detection analysis on Cortical Layers 1, 2/3, 4, 5, 6, and White Matter
    * Anomaly detection methods include DOMINANT [[2]](#two) , DONE [[2]](#two) , MADAN [[3]](#three) , and Node2vec [[4]](#four)
### Connectome Graph Analysis 
* **AD Anomaly Detection By Layer Directories:** Complete csv's for all identified anomalies for each anomaly detection method filtered by cortical layer
* **README:** Process pipeline for using directory Resources
    * Reference for further documentation
### Data Extraction and Preprocessing 
* Scripts for node and edge attribute computations:
  
  **Node Attributes**
    * Cortical layer prediction by Allen Institute [[5]](#five)
    * In and out degree
    * Cell type
    * Cell Classification
    * Number of soma
    * Neuron skeletonization
    * Total neuron run length 
    * Cortical layer prediction by GAC
 
  **Edge Attributes**  
    * Euclidean distance between somas
    * Synaptic size
    
**Note:** In the June Connectome Analysis, only Cortical layer (Allen), in/out degree, cell type, and e/i cell classification were used as embedded attributes. The repository contains functions for number of soma, neuron skeletonization, total neuron run length, euclidean distance between somas, and synaptic size attributes which may be embedded in the future.

* Script for variations on node table, edge table, and graph object extraction 
    * General extraction of node table from synapse table
    * Extraction of pure neurons and synapses from synapse table 
    * Graph object creation only containing neurons and synapses with corresponding node and edge tables
    * Subgraph object creation by cortical layer (with and without orphans) with corresponding node and edge tables

* **MinnieData_Computations.py:** example script for integrating all edge, node, and table extraction attribute scripts together

### Proofreading
* Script for studying proofreading metric trends over time
    * Retrieve timestamped data tables from CaveClient
    * Calculate metrics on a timestamped table: number of nodes, average in/out degree, density, transitivity, average clsutering, average degree centrality, number of strongly connected components, and average connectivity
    * Track progression of seg_id metrics across timestamped dataframes
    * Visualize lineage graph of seg_ids over time to see merging/splitting history
    * Visualize changes in in/out degree of a seg_id over time
* Visualization examples of lineage graphs and in/out degree changes over time
### Results and Discussion
* Final Project Poster
    * Overview of GAC complete with written results and figures
* Anomaly Detection Results 
    * Supplementary Visualizations of Anomaly Detection metrics and compared feature distributions
* Graph Metrics Results
    * Supplementary Visualizations of Graph metric distributions

## Scheme
<p align="center">
<img width="650" alt="Scheme" src=https://user-images.githubusercontent.com/97049854/184978885-94eeea22-0292-47aa-b12b-9ec79822bfa6.jpg>
</p>

## Anomaly Detection Method Descriptions
### Node2vec
The Node2vec framework generates low dimensional representations of each node in a graph. Node2vec leverages 2nd order biased random walks to learn the graph embeddings. The biased random walks are dependent on 2 parameters: p and q. P is the return probability parameter and q is the in-out probability parameter. A large p and small q supports exploration of the graph and generates further (depth first) walks in the Graph. On the other hand, a small p and large q supports backtrack or small walks that remain close to the starting node. [[6]](#six) In this Connectome Graph Analysis, various combinations of p and q were evaluated to determine the most optimal and relevant values for p and q. The parameters selected for this model are [dimensions=30, walk_length=45, num_walks=150, p=0.3, q=0.7]. From the generated node2vec embedding spaces for each node, Isolation Forest was deployed to classify nodes as anomalous in the graph of interest. [[7]](#seven)
### Deep Anomaly Detection on Attributed Networks (DOMINANT)
DOMINANT follows an autoencoder framework comprised of three components: first, a graph convolutional network (GCN) which encodes node attributes and topology followed by a structure reconstruction decoder to reconstruct the original network structure and finished with an attribute reconstruction decoder to reconstruct the original nodal attributes using the learned embeddings. The magnitude of the reconstruction error is then taken as a significant indicator of abnormality—large reconstruction errors point to a major deviation from overall patterns. The use of a GCN alleviates the effect of network sparsity and more accurately captures data nonlinearity and complex relationships between nodes. [[8]](#eight)
### Deep Outlier aware attributed Network Embedding (DONE)
DONE similarly utilizes a deep autoencoder framework where two parallel autoencoders are used for network structure and node attributes. After being trained to minimize reconstruction errors and preserve homophily (assumption that connected nodes likely have similar attributes), nodes are assigned an anomaly score. Again, this approach abides by the logic that anomalous nodes will be difficult to reconstruct due to non-conformation to standard behavior. DONE minimizes the influence of community outliers in the learning process and preserves various orders of proximities by accounting for the bias through appropriate Loss functions. [[9]](#nine) 
### Multi-scale Anomaly Detection on Attributed Networks (MADAN)
MADAN first creates a Gaussian weighted graph, reinforcing network homophily and applies a heat kernel, essentially acting as a smoothing filter to characterize the graph structure. Anomalies are then detected based on their concentration—the magnitude of their normalized, filtered unit signal. High concentration nodes tend to be poorly connected, and are assigned an anomalous score based on standard deviation across nodes.  MADAN ranks and localizes anomalous nodes with respect to node attributes and network structure at all scales of the network, not only limited to local, global, or community scales. To these ends, MADAN aims to identify anomalous nodes that may emerge in a specific context but later disappear in a different scale which is often a challenge in real world networks.[[10]](#ten)

## Reference
<a name="one">[[1]](https://www.biorxiv.org/content/10.1101/2021.07.28.454025v2.full.pdf)</a>
MICrONs Consortium et al. Functional connectomics spanning multiple areas of mouse 
visual cortex. bioRxiv 2021.07.28.454025; doi: 
https://doi.org/10.1101/2021.07.28.454025.

<a name="two">[[2]](https://github.com/pygod-team/pygod.git)</a>
Liu, K., Dou, Y., Zhao, Y., Ding, X., Hu, X., Zhang, R., Ding, K., Chen, C., Peng, H., Shu, K., Sun, L., Li, J., Chen, G.H., Jia, Z., and Yu, P.S. (2022). PyGOD[Source code]. https://github.com/pygod-team/pygod.git. 

<a name="three">[[3]](https://github.com/leoguti85/MADAN.git)</a>
Gutiérrez-Gómez, L., Bovet, A., & Delvenne, J.-C. (2021). MADAN[Source code]. https://github.com/leoguti85/MADAN.git. 

<a name="four">[[4]](https://github.com/aditya-grover/node2vec)</a>
Grover, A. (2017). Node2vec[Source code]. https://github.com/aditya-grover/node2vec.

<a name="five">[[5]](https://github.com/AllenInstitute/MicronsBinder.git)</a>
AllenInstitute (2021) MicronsBinder[Source code]. https://github.com/AllenInstitute/MicronsBinder.git.

<a name="six">[[6]](https://arxiv.org/pdf/1607.00653.pdf)</a>
Grover, A., Leskovec, J., 2016, July. Node2vec: Scalable Feature Learning for Networks. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. bioRxiv 1607.00653; doi: https://doi.org/10.48550/arXiv.1607.00653.

<a name="seven">[[7]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781136)</a>
F. T. Liu, K. M. Ting and Z. Zhou, "Isolation Forest," 2008 Eighth IEEE International Conference on Data Mining, 2008, pp. 413-422, doi: 10.1109/ICDM.2008.17.

<a name="eight">[[8]](https://epubs.siam.org/doi/epdf/10.1137/1.9781611975673.67)</a>
Ding, K., Li, J., Bhanushali, R. and Liu, H., 2019, May. Deep anomaly detection on attributed networks. In Proceedings of the SIAM International Conference on Data Mining (SDM).

<a name="nine">[[9]](http://eprints.iisc.ac.in/64704/1/WSDM%2020.pdf)</a>
Bandyopadhyay, S., Vivek, S.V. and Murty, M.N., 2020, January. Outlier resistant unsupervised deep architectures for attributed network embedding. In Proceedings of the International Conference on Web Search and Data Mining (WSDM).

<a name="ten">[[10]](https://ojs.aaai.org//index.php/AAAI/article/view/5409)</a>
Gutiérrez-Gómez, L., Bovet, A., & Delvenne, J.-C. (2020). Multi-Scale Anomaly Detection on Attributed Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 34(01), 678-685. 



