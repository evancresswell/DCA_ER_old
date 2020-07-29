import os,sys

sys.path.append(os.path.dirname('/data/cresswellclayec/DCA_ER/biowulf/'))
from Bio import SeqIO
from Bio.PDB import *
import pickle
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca
from pydca.plmdca import plmdca
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import ecc_tools as tools
from glob import glob
import data_processing as dp
import inspect

#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1

#-------------------------------

#-------------------------------
#=========================================================================================
def predict_w_couplings(s,i0,i1i2,niter_max,l2,couplings):
    #print('i0:',i0)
    #print('i1i2: length = number of positions: ',len(i1i2))
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
    #print(s.shape,': shape of s')
    #print(couplings.shape,': shape of couplings')
    #print('coupling matrix is symmetric:',np.allclose(couplings, couplings.T, rtol=1e-5, atol=1e-8))


    #print('predict_w, s_onehot: shape', s.shape)
    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]
    y_couplings = np.delete(couplings,[range(i1,i2)],0)					# remove subject rows  from original coupling matrix 
    y_couplings = np.delete(y_couplings,[range(i1,i2)],1)					# remove subject columns from original coupling matrix 
    #print('y_couplings shape: ',y_couplings.shape, ' x-column size: ',x.shape[1])	# Should be same dimensions as x column size as a result

    #print('predict_w, x: shape', x.shape)
    #print('predict_w, y: shape', y.shape)

    h01,w1 = ER.fit(x,y,niter_max,l2,y_couplings)

    return h01,w1

#========================================================================================

cov_list = np.load('covid_protein_list.npy')

np.random.seed(1)

pfam_id = sys.argv[1]
if pfam_id not in cov_list:
	print('protein not in covid protein data list: covid_protein_list.npy incorrect')
	sys.exit()

print("RUNNING SIM FOR %s"%(pfam_id))

#------- DCA Run -------#
msa_outfile = '%s/MSA_%s.fa'%(pfam_id,pfam_id) 

# MF instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    msa_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,
)

# Compute DCA scores 
sorted_DI_mf = mfdca_inst.compute_sorted_DI()

with open('%s/DI_DCA.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_mf, f)
f.close()
#-----------------------#
#------- PLM Run -------#

# PLM instance
plmdca_inst = plmdca.PlmDCA(
    msa_outfile,
    'protein',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads = 16,
    max_iterations = 500,
)

# Compute DCA scores 
sorted_DI_plm = plmdca_inst.compute_sorted_DI()

with open('%s/DI_PLM.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_plm, f)
f.close()


if os.path.exists('%s/DP.pickle'%(pfam_id)):
	with open('%s/DP.pickle'%(pfam_id), "rb") as f:\
		pf_dict = pickle.load(f)
	f.close()
else:
	print('no MSA data procesed DP.pickle DNE')
	sys.exit()

ipdb =0
#print(s0[0])
s0 = pf_dict['s0'] 
s_index = pf_dict['s_index'] 
s_ipdb = pf_dict['s_ipdb']
cols_removed = pf_dict['cols_removed']

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

with open('%s/DI_ER.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_er, f)
f.close()


