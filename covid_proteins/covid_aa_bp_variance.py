import sys,os
import random
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC
import glob

from Bio import SeqIO

#========================================================================================
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
# TO RUN: 		singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py
       
base_pairs = ['A','T','G','C']

table = { 
'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
'NNN':'X'
} 
# Swarm aligned file 
msa_file = root_dir+"covid_genome_full_aligned.fasta"
ref_file = root_dir+"wuhan_ref.fasta"

encoding_ranges =	{
			'NSP2' : [(806-1,2719-1)],
			'NSP3' : [(2720-1,8554-1)],
			'NSP12' : [(13442-1,13468-1),(13468-1,16236-1)],
			'NSP14a2' : [(18040-1,19620-1)],
			'NSP16' : [(20659-1,21552-1)],
			'S' : [(21563-1,25384-1)],
			'ORF3a' : [(25393-1,26220-1)],
			'E' : [(26245-1,26472-1)],
			'N' : [(28274-1,29533-1)],
			'Full' : [(266-1,29674-1)]
			}



# Load list of bp positions and corresponding aa variance
aa_files =  glob.glob(root_dir+'*_aa_column.npy')
print(aa_files)
positions = []
for aa_file in aa_files:
	positions.append(int(aa_file.strip(root_dir+'_aa_column.npy')))
print('Considering bp postitions: ',positions)

for bp_pos in positions:

	aa = np.load('%d_aa_column.npy'%bp_pos)
	bp = np.load('%d_bp_column.npy'%bp_pos)

	unique, counts = np.unique(aa, return_counts=True)
	bp_unique, bp_counts = np.unique(bp, return_counts=True)


	aa_total = sum(counts)
	bp_total = sum(bp_counts)
	print('\n#------------------------------------------------#')
	print('             Position %d                       '%bp_pos)
	#print('%d bp count, %d aa count'%(bp_total,aa_total))
	print('#------------------------------------------------#')

	print('#---------------- BP ----------------------------#')
	for i,bp in enumerate(bp_unique):
		print('%s frequency: %f'%(bp,bp_counts[i]/float(bp_total)))
	print('#------------------------------------------------#')


	print('#----------------- AA ---------------------------#')
	for i,aa in enumerate(unique):
		print('%s frequency: %f'%(aa,counts[i]/float(aa_total)))
	print('#------------------------------------------------#\n')
	#print(unique)	
	#print(counts)	
	#print(bp_unique)	
	#print(bp_counts)	


	


