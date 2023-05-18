Hello! Thank you for your interest in this work. We hope you find it valuable. Here are descriptions of what you will find in this repository.


If you only have a list of peptide sequences without other information, it is still recommended to use '0 Data Curation Step 1- Filter by Library Design' before moving onto 3 Make Map by Dim Reduction. Otherwise, the resulting sequence space map may just separate your peptides based on library design.


========== Folders:
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