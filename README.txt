Hello! Thank you for your interest in this work. We hope you find it valuable. Here are descriptions of what you will find in this repository. See the requirements.txt for the required Python modules. Instructions:

If you only have a list of peptide sequences without other information, it is still recommended to use '0 Data Curation Step 1- Filter by Library Design' before moving onto 3 Make Map by Dim Reduction. Otherwise, the resulting sequence space map may just separate your peptides based on library design.

========== All data from our publication is here in this repository in these folders:
-------	1 Data Curation Step 1- Filter by Library Design
	This jupyter notebook will filter your list of peptides by regular expressions to isolate peptides that fit a specific design (e.g., X12K). It is highly recommended to use this even if you only have a list of peptides, and just skip the last filtering step. Otherwise, you may end up with sequence space map that just separates your peptides based on library design

-------	2 Data Curation Step 2- Remove Seq Isomers
	This jupyter notebook is AS-MS specific, completely optional, and will rigorously remove sequence isomers from the dataset.

-------	3 Make Map by Dim Reduction
	This is the main jupyter notebook of all this work.
	It takes a list of peptide sequences, encodes them with whichever encoding method you choose, and performs dimensionality reduction to prepare the sequence space map. Random peptides sampled from your library can be added to improve interpretability.

-------	4 Scout Cluster Detection Parameters
	After making your sequence space map, you may see apparent clusters. Herein, python files are provided to scan the parameters of DBSCAN and AggomerativeClustering to detect your apparent clusters based on the density of points in the clusters.

-------	5 Detect Cluster on Map and Logo
	Last, this notebook will take the scount parameters from (4) and prepare a figure quality plot of your detected clusters.
	It will also isolate a consensus sequence, logo plot, and centroid sequence(s) from each detected cluster

-------	Extra Pieces
	Herein is a dedicated logo plot script prepared using Logomaker.

This repository uses Jupyter 6.4.8 (Python 3.9.12), but see the requirements.txt for all module specific details. 
Also, additional details are provided for specialized modules at the beginning of each notebook. 
Installation of all modules including a fresh install of anaconda should take <1 hour if not <10 minutes. 

Please follow the annotate Jupyter notebook for further instructions of how to use this code. Any laptop can run this code in <1 hour, and the expected output is shown in the notebook

Note about reproducibility: PCA is deterministic and will always produce the same output. However, UMAP is stochastic and varies slightly from computer to computer. To limit this variation, use the data provided, which has been randomized once) and keep the random seed set.
