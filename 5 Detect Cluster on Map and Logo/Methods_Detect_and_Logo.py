import os, math, glob, collections
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import logomaker as lm
from enum import Enum
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline


def centroider(
    seqlist,
    X_list,
    Y_list,
    num_of_reported_seqs,
    cluster_num,
    C1_col_name_cent,
    C2_col_name_cent,
    outputfile_cent,
    generateplot,
    savecentroidplot
):
    cluster_title_str = f'Cluster {cluster_num}'
    if type(X_list[0]) != float: X_list = [float(datapoints) for datapoints in X_list]
    if type(Y_list[0]) != float: Y_list = [float(datapoints) for datapoints in Y_list]
    tempdf = pd.DataFrame()
    tempdf['Peptide'] = seqlist
    tempdf[C1_col_name_cent] = X_list
    tempdf[C2_col_name_cent] = Y_list
    CenterX = sum(X_list)/len(X_list)
    CenterY = sum(Y_list)/len(Y_list)
    XY_zip = zip(X_list,Y_list)
    
    # calculate and record the distance of each sequence from the cluster center
    Dist = []
    for x,y in XY_zip: Dist.append(math.dist([x,y], [CenterX, CenterY]))
    tempdf['Dist from Center'] = Dist 
    tempdf = tempdf.sort_values(by=['Dist from Center']) # sort the sequences by their distance from the cluster center
    tempdf.reset_index(drop=True,inplace=True) # reset index
    
    # Now that they are sorted, pull centroid sequences using the dataframe index.
    if len(seqlist) < 10: max_centroid_dist_rank = 0.7
    else: max_centroid_dist_rank = 0.5 # Range = 0-1
    # The furthest index pulled for centroid seq reporting will be this fraction of the index length
    
    if num_of_reported_seqs > len(tempdf):
        print(f'The number of centroid sequences requested {num_of_reported_seqs} is greater \
than the cluster size of {len(tempdf)} sequences. The maximum amount of centroid sequences \
will be reported instead {math.floor(len(tempdf)*max_centroid_dist_rank)}. More centroid sequences can be reported if \
max_centroid_dist_rank is increased, though this is not recommended')
        num_of_reported_seqs = math.floor(len(tempdf)*max_centroid_dist_rank)
    
    index_list = np.linspace(0,len(tempdf)*max_centroid_dist_rank,int(num_of_reported_seqs))
    rounded_index_list = [int(round(i,0)) for i in index_list]
    tempdfout = tempdf.iloc[rounded_index_list]
    label_list = [cluster_title_str]*len(tempdfout)
    tempdfout['Labels'] = label_list
    
    end_peps = tempdf.iloc[rounded_index_list]['Peptide'].to_list()
    end_peps_list = [*set(end_peps)] #remove duplicates if they exist
    end_str = str(end_peps_list)[1:][:-1]
    
    print(f'{cluster_title_str}, Number of seqeuence(s) reported is {int(num_of_reported_seqs)}; Centroid sequence(s): {end_str}')
    
    C1_extras = tempdf.iloc[rounded_index_list][C1_col_name_cent].to_list()
    C2_extras = tempdf.iloc[rounded_index_list][C2_col_name_cent].to_list()
    
    if generateplot:
        plt.scatter(X_list,Y_list)
        plt.scatter(CenterX,CenterY)
        plt.title(f'{cluster_title_str} centroid plot')
        plt.xlabel(f'{cluster_title_str} C1')
        plt.ylabel(f'{cluster_title_str} C2')
        plt.scatter(C1_extras,C2_extras, marker = 'x', color = 'black', s = 200)
        for i,seq in enumerate(tempdf.iloc[rounded_index_list]['Peptide']):
            plt.annotate(seq, (C1_extras[i]+0.02, C2_extras[i]+0.02), fontsize = 10)
        if savecentroidplot:
            plt.savefig(os.path.join(outputfile_cent,f'{cluster_title_str} centroid plot.png'), dpi=300)
        plt.show()
    return end_str, tempdfout

def large_centroid_seq_plot(df_centroids,
                            original_df,
                            C1_col_name_centp,
                            C2_col_name_centp,
                            save_loc_name_centp
):
    cent_clusters = np.unique(original_df['Cluster labels'])
    my_colorblind_colors = [(0.902,0.624,0),(0.337,0.706,0.914),(0,0.62,0.451),(0,0.447,0.698),(0.835,0.369,0),(0.8,0.475,0.655)]
    my_colorblind_colors_set = my_colorblind_colors*1000 # good up to 6000 clusters, which is excessive
    fig, axs = plt.subplots(1,1,figsize=(30,25))
    for j,i in enumerate(cent_clusters):
        X1 = original_df.loc[original_df['Cluster labels'] == i,[C1_col_name_centp]]
        Y1 = original_df.loc[original_df['Cluster labels'] == i,[C2_col_name_centp]]
        X2 = df_centroids.loc[df_centroids['Labels'] == f'Cluster {i}',[C1_col_name_centp]]
        Y2 = df_centroids.loc[df_centroids['Labels'] == f'Cluster {i}',[C2_col_name_centp]]        
        if i != 0:
            plt.scatter(X1,Y1,color=my_colorblind_colors_set[j], s = 200, alpha = 0.5)
            plt.scatter(X2,Y2, marker = 'x', color = 'black', s = 1000, alpha = 0.5)
        if i == 0: # noise points
            plt.scatter(X1,Y1,color='gray', s = 200, alpha = 0.5)
        cent_seqs = df_centroids.loc[df_centroids['Labels'] == f'Cluster {i}',['Peptide']]['Peptide'].to_list()
        for i,seq in enumerate(cent_seqs):
            plt.annotate(seq, (X2.iloc[i,0], Y2.iloc[i,0]), fontsize = 30, alpha=1)
    plt.title(f'{C1_col_name_centp[:-3]} big centroid plot', fontsize = 30)
    plt.xlabel(C1_col_name_centp, fontsize = 30)
    plt.ylabel(C2_col_name_centp, fontsize = 30)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.savefig(save_loc_name_centp, dpi=300)
    plt.show()

def strconverter(s):
    new = "" # initialization of string to ""
    for x in s: new += x # return string 
    return new

def consensus_seq(seqs,cutoff):
    # Prepares the consensus sequence of a list of sequences, with no alignment(!), reporting the residue if it appears
    # the cutoff of all sequences. This unelegant function requires the 'strconverter'
    if type(seqs) != list: print('Input error: the input of the consensus_seq function must be a list of sequences')
    temp_sequence = np.zeros((len(seqs),len(seqs[0])), dtype=object) # Prepare a matrix of the residues
    for l in range(0,len(seqs[0])): 
        for k,seq in enumerate(seqs):
            temp_sequence[k,l] = seq[l]
    col_sequence = temp_sequence.T # Transpose to access a 'column' of all the residues in each position
    consensus_letter = np.zeros(len(col_sequence),dtype=object)
    for o in range(len(col_sequence)): # Iterate through each column and see if the most common letter is above the cutoff %
        letter = collections.Counter(col_sequence[o]).most_common(1)[0][0]
        count = collections.Counter(col_sequence[o]).most_common(1)[0][1]
        if count/len(col_sequence[o]) > cutoff: consensus_letter[o] = letter
        else: consensus_letter[o] = 'X'
    consensus_seq_fix = strconverter(consensus_letter.tolist())
    return consensus_seq_fix[:-1]

class Y_AXIS_UNIT(Enum):
    COUNTS = 1
    BITS = 2

def generate_logos(
    seqs,
    save_loc_and_name, 
    displaysavefilename_astitle, 
    y_axis_units=Y_AXIS_UNIT.BITS
):
    
    fig, axs = plt.subplots(1,1,figsize=(2.5,0.8)) ## Change Weblogo size
    if displaysavefilename_astitle == True:
        title_wl = os.path.splitext(os.path.basename(save_loc_and_name))[0]
        axs.set_title(title_wl, size=8)
    axs.set_xticks([])
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    if y_axis_units == Y_AXIS_UNIT.BITS:
        if len(seqs) < 15: pseudocnt = 0.1
        else: pseudocnt = 0.2
        counts_mat = lm.alignment_to_matrix(seqs, to_type='information',characters_to_ignore='-',pseudocount=pseudocnt)
        the_y_max = math.ceil(counts_mat.sum(axis=1).to_numpy().max())
        if the_y_max > 4.4: the_y_max = 4.4
        logo = lm.Logo(counts_mat, ax=axs, color_scheme="hydrophobicity")
        axs.set_ylim([0,the_y_max])
        plt.yticks([0,the_y_max], fontsize = 10) 
        plt.ylabel('Bits', size = 10, labelpad=-10)
        # plt.xlabel("Alignment Position")
        plt.savefig(save_loc_and_name,dpi=300,bbox_inches='tight')
        plt.show()
    elif y_axis_units == Y_AXIS_UNIT.COUNTS:
        if len(seqs) < 15: pseudocnt = 0.1
        else: pseudocnt = 0.2
        counts_mat = lm.alignment_to_matrix(seqs, to_type='counts',characters_to_ignore='-',pseudocount=pseudocnt)
        the_y_max = math.ceil(counts_mat.sum(axis=1).to_numpy().max())
        logo = lm.Logo(counts_mat, ax=axs, color_scheme="hydrophobicity")
        axs.set_ylim([0,the_y_max])
        plt.yticks([0,the_y_max], fontsize = 10) 
        plt.ylabel('Residue Counts', size = 10, labelpad=-10)
        # plt.xlabel("Alignment Position")
        plt.savefig(save_loc_and_name,dpi=300,bbox_inches='tight')
        plt.show()
    
def aligner(chosen_sequences_aln,cluster_aln,outputfile_aln):
    with open(os.path.join(outputfile_aln,f'cluster{cluster_aln}.fa'),'w') as ofile: # Open a temporary fasta file
        for i,seq in enumerate(chosen_sequences_aln):
            ofile.write(">" + str(i) + "\n" + seq + "\n") # write all sequences to fasta file
    fin1 = open(os.path.join(outputfile_aln,f'cluster{cluster_aln}.fa')) # open and read in the newly written fasta file for clustalw2

    print(f'Aligning sequences from cluster {cluster_aln}') # Multiple sequence alignment using ClustalW
    aln_loc = f'{fin1.name[:-3]}.aln' # Generate alignment file         # Call ClustalW 2.1 (separately installed .exe) Installation instructions here: http://www.clustal.org/clustal2/
    cmd = ClustalwCommandline('clustalw2',infile=fin1.name,gapopen=10000) # Call ClustalW 
    fin1.close()
    stdout, stderr = cmd()
    align = AlignIO.read(aln_loc, "clustal")
    aligned_seqs = []
    for seq in align:
        aligned_seqs.append(str(seq.seq))
    for f in glob.glob(os.path.join(outputfile_aln,f'*.fa')): os.remove(f)
    for f in glob.glob(os.path.join(outputfile_aln,f'*.aln')): os.remove(f)
    for f in glob.glob(os.path.join(outputfile_aln,f'*.dnd')): os.remove(f)
    return aligned_seqs