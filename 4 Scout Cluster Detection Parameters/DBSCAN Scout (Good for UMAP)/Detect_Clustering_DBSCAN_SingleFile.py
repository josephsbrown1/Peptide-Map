# This script helps scout parameters for the detection of apparent clusters in 2D data

import os, math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import pandas as pd

### Folders and files, inputs, and outputs
os.chdir(os.path.dirname(os.path.abspath(__file__))) #this makes sure the current working directory is where the python file is

# Use this if you want to iterate through multiple .csv files
#for filename in os.listdir(os.getcwd()):
#    if filename.endswith('.csv'):
filename = 'UMAP Fingerprint.csv'
print(' ')
print(filename)

modelname = 'DBSCAN'
                        
# Process inputs
df1 = pd.read_csv(filename)

all_column_list = df1.columns.to_list()
C1_col_name = [s for s in all_column_list if 'C1' in s][0]
C2_col_name = [s for s in all_column_list if 'C2' in s][0]
X = np.asarray([(float(x), float(y)) for x, y in zip(df1[C1_col_name], df1[C2_col_name])])
print('Extracted embeddings points for graph and cluster detection')

# Outputs
outputfolder = os.path.join(os.getcwd(),f'{modelname} {filename[:-4]}')
file = os.path.join(os.getcwd(),filename)
save_title = f'{filename[:-4]}'
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)

# Define range for parameters eps (DBSCAN only) and min_samples (DBSCAN and AggCl), multiple options:

    #eps_range_start = np.logspace(math.log10(0.06),math.log10(0.2),10).tolist()
eps_range_start = np.linspace(0.1,0.2,25).tolist()
eps_range = [round(num,4) for num in eps_range_start]

min_samp_range_start = np.linspace(10,40,31).tolist()
#min_samp_range_start = [14,16,18,20,22,24,26,28,30]

for count1,epsy in enumerate(eps_range):
    for count2,min_samp in enumerate(min_samp_range_start):
        if modelname == 'DBSCAN':
            model = DBSCAN(eps=epsy, min_samples=min_samp) # define the model
            print(f'DBSCAN detection, eps is {str(epsy)}, sample_min {str(min_samp)}')

        if modelname == 'AggCl':
            model = AgglomerativeClustering(n_clusters=min_samp) # define the model
            print(f'AggCl detection, n_clusters is {str(min_samp)}')
            epsy = ''
              
        yhat = model.fit_predict(X) # fit model, predict clusters. Every row has a cluster associated.
        yhat = yhat + 1 # This is done because clusters start at 0, and we want to label them starting at 1.
        clusters = np.unique(yhat)
        
        # Make titles for use in labeling
        if modelname == 'AggCl': cluster_out_title = f'{save_title} {str(min_samp)} clusters'  
        if modelname == 'DBSCAN': cluster_out_title = f'{save_title} eps {str(epsy)} sample_min {str(min_samp)} {str(len(clusters)-1)} clusters'
        
        Center_ave = np.zeros([len(clusters)+1,2]) # +1 is to make both DBSCAN (which has a noise cluster) and AggCl (which does not have a noise cluster) comptabile
        
        
        fig,ax = plt.subplots()
        fig.set_size_inches(12, 9)
        for j,i in enumerate(clusters):
            row_ix = np.where(yhat == i)  # get row indexes for samples with this cluster
            if i != 0:
                # Plot the data
                plt.scatter(X[row_ix, 0], X[row_ix, 1], alpha = 0.3)
                # Plot points that are at the center of each cluster, and annotate them with their autonomous number label
                Center_ave[i,0] = np.average(X[row_ix, 0])
                Center_ave[i,1] = np.average(X[row_ix, 1])
                plt.scatter(Center_ave[i,0], Center_ave[i,1], alpha=0.8, s=50, color='black')
                plt.annotate(i, (Center_ave[i,0], Center_ave[i,1]), fontsize = 30, alpha=1)
                
            if i == 0:
                plt.scatter(X[row_ix, 0], X[row_ix, 1], alpha = 0.3, color='gray') # plot the noise points less intensely

        plt.title(cluster_out_title,fontsize=20)
        plt.xlabel(C1_col_name,fontsize=20)
        plt.ylabel(C2_col_name,fontsize=20)

        file_save_title = f'{str(count1).zfill(2)} {str(count2).zfill(2)} {cluster_out_title}'
        #print(outputfolder,file_save_title)
        fig.savefig(os.path.join(outputfolder,f'{file_save_title}.png'), dpi=120) # save the figure as an image
        plt.close()