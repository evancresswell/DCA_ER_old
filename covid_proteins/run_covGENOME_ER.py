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

cpus_per_job = int(sys.argv[1])
print("Calculating DI for Sars-Cov-2 using %d (of %d) threads"%(cpus_per_job-4,cpus_per_job))

msa_file = root_dir+"aligned_cov_genome.fasta"
ref_file = root_dir+"cov_genome_ref.fasta"

alignment_file_name = "aligned_cov_genome" 
alignment_file_name = "ncbi_cov_gen" 

input_data_file = "cov_genome_DP.pickle"
if os.path.exists(input_data_file):    
	print('\n\nUsing existing pre-processed FASTA files\n\n')
	with open(input_data_file,"rb") as f:
		pfam_dict =  pickle.load(f)
	f.close()
	cols_removed = pfam_dict['cols_removed']
	s_index= pfam_dict['s_index']
	s_ipdb = pfam_dict['s_ipdb']
else:
	print('\n\nPre-Processing MSA with muscle alignment\n\n')
	# Preprocess data using ATGC
	
	# data processing
	s0,cols_removed,s_index,s_ipdb = gdp.data_processing(data_path,alignment_file_name,0,\
					gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)

	# Save processed data
	msa_outfile, ref_outfile = gdp.write_FASTA(s0,'COV_GENOME',s_ipdb,path='pfam_ecc/')	
	pf_dict = {}
	pf_dict['s0'] = s0
	pf_dict['s_index'] = s_index
	pf_dict['s_ipdb'] = s_ipdb
	pf_dict['cols_removed'] = cols_removed
	pfam_dict['s_ipdb'] = s_ipdb

	with open(input_data_file, 'wb') as f:
		pickle.dump(pf_dict, f)
	f.close()

# data processing


print(s0.shape)

n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)

#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1

#-------------------------------
# parallel
res = Parallel(n_jobs = 16)(delayed(predict_w)\
        (s,i0,i1i2,niter_max=10,l2=100.0)\
        for i0 in range(n_var))

#-------------------------------
for i0 in range(n_var):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
       
    h01 = res[i0][0]
    w1 = res[i0][1]

    h0[i1:i2] = h01    
    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]

# make w to be symmetric
w = (w + w.T)/2.
di = direct_info(s0,w)

er_gen_DI = sort_di(di)

for site_pair, score in er_gen_DI[:5]:
    print(site_pair, score)

with open(root_dir+'cov_genome_DI.pickle', 'wb') as f:
    pickle.dump(erdca_DI, f)
f.close()


