import sys,os
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
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

#========================================================================================
data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'


# Swarm aligned file 
msa_file = root_dir+"covid_genome_full_aligned.fasta"
ref_file = root_dir+"wuhan_ref.fasta"

region = sys.argv[1]
clade = sys.argv[2]
msa_file = sys.argv[3]
msa_file = root_dir+msa_file

#--------------------------------------------------------------------------------------#
#------------------------------------ Utility Functions -------------------------------#
#-------------------------- Sourced from genome_data_processing.py --------------------#
#--------------------------------------------------------------------------------------#
from genome_data_processing import *
from Bio import SeqIO


#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#

print("Loading Fasta file: %s \n	Going to generate incidence plots"%(msa_file))

gap_seqs = .2
gap_cols = .2


# Load Genome Sequence Set
s = []
with open(msa_file,"r") as handle:
	for record in SeqIO.parse(handle, "fasta"):
		s.append([char for char in str(record.seq).upper()])
s = np.asarray(s) 
print(s.shape)


tpdb = 0 # Reference sequence should be first sequence ALWAYS
gap_pdb = s[tpdb] =='-' # returns True/False for gaps/no gaps

s = s[:,~gap_pdb] # removes gaps  

bad_cols = find_bad_cols(s,gap_cols)

s, tpdb = remove_bad_seqs(s,tpdb,gap_seqs) # removes all sequences (rows) with >gap_seqs gap %

# Create s_index to keep track of original indexing
s_index = np.arange(s.shape[1])

#------ Run through same preprocessing done in simulation -----#
# replace 'Z' by 'Q' or 'E' with prob
s = find_and_replace(s,'Z',np.array(['Q','E']))

# replace 'B' by Asparagine (N) or Aspartic (D)
s = find_and_replace(s,'B',np.array(['N','D']))

# replace 'X' as amino acids with prob
base_pairs = np.array(['A','T','G','C'])
s = find_and_replace(s,'X',base_pairs)
#--------------------------------------------------------------#
N_frac = 50
incidence_thresholds = np.linspace(.75,1.,N_frac)
print('Looping throught incidence %d thresholds:\n'%len(incidence_thresholds),incidence_thresholds)

protein_ranges = {}				#  buffer of 265 --> [0, 264]
protein_ranges['ORF1ab']	= [266,21555] 	#  21290 	# 21289
protein_ranges['S'] 		= [21563,25384] #  3822 	# 3821
if 0:
	protein_ranges['ORF3a'] 	= [25393,26220] #  828 		# 827
	protein_ranges['ORF3b'] 	= [25765,26220] #     	 	# 455
	protein_ranges['E'] 		= [26245,26472] #  228		# 227 
	protein_ranges['M'] 		= [26523,27191] #  669		# 668 
	protein_ranges['ORF6']	 	= [27202,27387] #  186		# 185
	protein_ranges['ORF7a'] 	= [27394,27759] #  366		# 365
	protein_ranges['ORF7b'] 	= [27756,27887] #  132		# 131
	protein_ranges['ORF8'] 		= [27894,28259] #  193		# 265
	protein_ranges['N'] 		= [28274,29533] #  908		# 1259
	protein_ranges['ORF10'] 	= [29558,29674] #  117		# 116

def get_incidence(i0, s, conserved_fracs, bad_cols, s_index, protein_ranges, region):
	print('parallel run %d'%i0)
	conserved_frac = conserved_fracs[i0]

	# Find find and remove conserved cols
	conserved_cols = find_conserved_cols(s,conserved_frac)
	removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))
	try:
		s_cons = np.delete(s,removed_cols,axis=1)
		s_index_cons = np.delete(s_index,removed_cols)
	except(IndexError):
		print('No columns to remove: remvode_cols has length %d'%len(removed_cols))
		s_cons = s		
		s_index_cons = s_index

	# Loop through regions
	print('after removing %f conserved columns we will investigate the inidences in regions %s'%(conserved_frac,region))

	protein_range = protein_ranges[region]
	print('%s region has the following index range\n'%region,protein_range)
	print('length of s_index_cons: ', len(s_index_cons))
	try:
		index_start = min( [i for i in  s_index_cons if i > protein_range[0]])
		index_end = max( [i for i in  s_index_cons if i < protein_range[1]])
	except(ValueError):
		print('value error. pass to the next loop')
		region_incidence = 0
		return (region_incidence, conserved_frac)
	
	index_range = ( np.where(s_index_cons==index_start)[0][0], np.where(s_index_cons==index_end)[0][0] )
	region_incidence = index_range[1] - index_range[0]
	return (region_incidence, conserved_frac)

# parallel
incidence_pairs = Parallel(n_jobs = 16)(delayed(get_incidence)\
	(i0, s, incidence_thresholds, bad_cols, s_index, protein_ranges,region)\
        for i0 in range(len(incidence_thresholds)))

print(incidence_pairs)


np.save('%s_%s_incidence.npy'%(clade,region), incidence_pairs)

if 0:
	region_incidences = []
	for conserved_frac in incidence_thresholds:

		# Find find and remove conserved cols
		conserved_cols = find_conserved_cols(s,conserved_frac)
		removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))
		try:
			s_cons = np.delete(s,removed_cols,axis=1)
			s_index_cons = np.delete(s_index,removed_cols)
		except(IndexError):
			print('No columns to remove: remvode_cols has length %d'%len(removed_cols))
			s_cons = s		
			s_index_cons = s_index

		# Loop through regions
		print('after removing %f conserved columns we will investigate the inidences in regions %s'%(conserved_frac,region))

		protein_range = protein_ranges[region]
		print('%s region has the following index range\n'%region,protein_range)
		print('length of s_index_cons: ', len(s_index_cons))
		try:
			index_start = min( [i for i in  s_index_cons if i > protein_range[0]])
			index_end = max( [i for i in  s_index_cons if i < protein_range[1]])
		except(ValueError):
			print('value error. pass to the next loop')
			region_incidences.append((conserved_frac,0))
			continue
		
		index_range = ( np.where(s_index_cons==index_start)[0][0], np.where(s_index_cons==index_end)[0][0] )
		region_incidence = index_range[1] - index_range[0]
		region_incidences.append((conserved_frac,region_incidence))














