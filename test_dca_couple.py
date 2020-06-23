import sys
import numpy as np
import pickle
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from meanfield_dca import meanfield_dca
import expectation_reflection as ER
from direct_info import direct_info
from joblib import Parallel, delayed
import data_processing as dp

from direct_info import sort_di

#========================================================================================
# Set data path
#========================================================================================
data_path = '/home/eclay/Pfam-A.full'

np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]


#========================================================================================
# Process data and write fast file for ER and MF
#========================================================================================

# Load PDB structure 
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

#---------- Pre-Process Structure Data ----------------#
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

# data processing THESE SHOULD BE CREATED DURING DATA GENERATION
ipdb = 0
s0,cols_removed,s_index,s_ipdb,s_old = dp.data_processing(data_path,pfam_id,ipdb,\
				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)

#========================================================================================
#                   DCA
#========================================================================================
msa_outfile = '/home/eclay/DCA_ER/pfam_ecc/MSA_%s.fa'%(pfam_id) 
# use msa fasta file generated in data_processing
msa_outfile = './pfam_ecc/MSA_%s.fa'%(pfam_id) 

# MF instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    msa_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,
)
reg_fi = mfdca_inst.get_reg_single_site_freqs()
reg_fij = mfdca_inst.get_reg_pair_site_freqs()
corr_mat = mfdca_inst.construct_corr_mat(reg_fi, reg_fij)
couplings = mfdca_inst.compute_couplings(corr_mat)
print(couplings)
print(len(couplings))


#========================================================================================
# Compute DCA scores 
#========================================================================================
print('Computing sorted DI')
sorted_DI_mf = mfdca_inst.compute_sorted_DI()
print('Done')

#print('MF DI: ', sorted_DI_mf)


#========================================================================================
# Compute ER 
#                   ER
#========================================================================================

# number of positions
n_var = s0.shape[1]
n_var_old = s_old.shape[1]

mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]
print ('Plotting Protein Famility ', pfam_id)

mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

mx_old = np.array([len(np.unique(s_old[:,i])) for i in range(n_var_old)])
mx_cumsum_old = np.insert(mx_old.cumsum(),0,0)
i1i2_old = np.stack([mx_cumsum_old[:-1],mx_cumsum_old[1:]]).T 


#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)
s_old = onehot_encoder.fit_transform(s_old)

mx_sum = mx.sum()
my_sum = mx.sum()
mx_sum_old = mx_old.sum()
my_sum_old = mx_old.sum()


w = np.zeros((mx_sum,mx_sum))
w_old = np.zeros((mx_sum_old,mx_sum_old))

print('processed w shape:',w.shape)
print('unprocessed w shape:',w_old.shape)

print('processed s0 shape:',s0.shape)
print('unprocessed s shape:',s.shape)

print('DCA couplings shape:',couplings.shape)

h0 = np.zeros(my_sum)

#========================================================================================
#========================================================================================
#========================================================================================
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

sorted_DI_er = sort_di(di)
#print('ER DI: ', sorted_DI_er)

