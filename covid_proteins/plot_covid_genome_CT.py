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


def distance_restr_sortedDI(site_pair_DI_in, s_index=None):
	#print(site_pair_DI_in[:10])
	restrained_DI= dict()
	for site_pair, score in site_pair_DI_in:
		# if s_index exists re-index sorted pair
		if s_index is not None:
			pos_0 = s_index[site_pair[0]]
			pos_1 = s_index[site_pair[1]]
			#print('s_index positions: (%d, %d)'%(pos_0,pos_1))
		else:
			pos_0 = site_pair[0]
			pos_1 = site_pair[1]
		indices = (pos_0 , pos_1)

		if abs(pos_0- pos_1)<5:
			restrained_DI[indices] = 0
		else:
			restrained_DI[indices] = score
	sorted_DI  = sorted(restrained_DI.items(), key = lambda k : k[1], reverse=True)
	print(sorted_DI[:10])
	return sorted_DI

def delete_sorted_DI_duplicates(sorted_DI):
	temp1 = []
	#print(sorted_DI[:10])
	DI_out = dict() 
	for (a,b), score in sorted_DI:
		if (a,b) not in temp1 and (b,a) not in temp1: #to check for the duplicate tuples
			temp1.append(((a,b)))
			if a>b:
				DI_out[(b,a)]= score
			else:
				DI_out[(a,b)]= score
	print('Sorting DI values')
	DI_out = sorted(DI_out.items(), key = lambda k : k[1], reverse=True)
	#DI_out.sort(key=lambda x:x[1],reverse=True) 
	print(DI_out[:10])
	return DI_out 




pf_dict_file = root_dir + 'cov_genome_DP.pickle'
# Load Simulation Files
with open(pf_dict_file, 'rb') as f:
	pf_dict = pickle.load(f)
f.close()

cov_DI_file = root_dir+'cov_genome_DI.pickle'
with open(cov_DI_file, 'rb') as f:
    er_gen_DI = pickle.load( f)
f.close()


s_index  = pf_dict['s_index']
s_ipdb  = pf_dict['s_ipdb']
cols_removed  = pf_dict['cols_removed']

print('\n\n#--------------------------- Plotting Covid Genome Contacts ------------------------#')
print('%d columns removed'%(len(cols_removed)))
print('%d columns left'%(len(s_index)))
print('#-----------------------------------------------------------------------------------#\n')

print('\n#-----------------------------------------------------------------------------------#')
print('Un-Processed DI')
print(er_gen_DI[:10])

print('\nSorting DI\nRestraining guesses Linear Distance')
print('Sorted-DR DI')
sorted_DI = distance_restr_sortedDI(er_gen_DI, s_index)
print(sorted_DI[:10])

print('\nDeleting DI Duplicates')
sorted_DI = delete_sorted_DI_duplicates(sorted_DI)
print('Final DI:')
print(sorted_DI[:10])

with open(root_dir+'cov_genome_DI_processed.pickle', 'wb') as f:
    pickle.dump(sorted_DI, f)
f.close()


print('#-----------------------------------------------------------------------------------#\n')























