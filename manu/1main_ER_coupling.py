import sys
import pickle
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import ecc_tools as tools

#========================================================================================
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]


input_data_file = "pfam_ecc/%s_DP_ER.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
f.close()
#s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
#				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
ipdb = 0
#s_index = pfam_dict['s_index']	
s_ipdb = pfam_dict['s_ipdb']	
#cols_removed = pfam_dict['cols_removed']

# LOADING S0
#s0 = pfam_dict['s0']	
#s0 = np.loadtxt('pfam_ecc/%s_s0.txt'%(pfam_id))\

# For testing to fix DCA-coupling in erdca..
trimmed_data = pfam_dict['processed_msa']

#----- generate data for erdca to calculate couplings -----#
s0 = []
for sequence_data in trimmed_data:
	s0.append([char for char in sequence_data[1]])
s0 = np.array(s0)
print('\nMSA ref-trimmed\ns0: \n',s0[:10],'\n\n')
print(s0.shape)



n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)


#========================================================================================
# Compute ER couplings using MF initialization
#========================================================================================

seqs_weight = tools.compute_sequences_weight(alignment_data = s0, seqid = .8)
np.save('pfam_ecc/%s_seqs_weight.npy'%(pfam_id),np.array(seqs_weight))

single_site_freqs = tools.compute_single_site_freqs(alignment_data = s0,seqs_weight=seqs_weight,mx= mx)
np.save('pfam_ecc/%s_single_site_freqs.npy'%(pfam_id),np.array(single_site_freqs))

reg_single_site_freqs = tools.get_reg_single_site_freqs(
	    single_site_freqs = single_site_freqs,
	    seqs_len = n_var,
	    mx = mx,
	    pseudocount = .5) # default pseudocount value used in regularization
#print (len(reg_single_site_freqs))
#print(reg_single_site_freqs[0])

pair_site_freqs = tools.compute_pair_site_freqs_serial(alignment_data=s0, mx=mx,seqs_weight=seqs_weight)
np.save('pfam_ecc/%s_pair_site_freqs.npy'%(pfam_id),np.array(pair_site_freqs))

corr_mat =  tools.construct_corr_mat(reg_fi = reg_single_site_freqs, reg_fij = pair_site_freqs, seqs_len = n_var, mx = mx)
np.save('pfam_ecc/%s_corr_mat.npy'%(pfam_id),corr_mat)

couplings = tools.compute_couplings(corr_mat = corr_mat)
np.save('pfam_ecc/%s_couplings.npy'%(pfam_id),couplings)
print('couplings first row:\n', couplings[0])
print('couplings (1MAIN, DCA_tools_coupling) shape: ', couplings.shape)
             
#========================================================================================



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


#========================================================================================
# ER - COUPLINGS
#========================================================================================
mx_sum = mx.sum()
w = np.zeros((mx_sum,mx_sum))
print('full weights matrix: shape: ',w.shape)

h0 = np.zeros(mx_sum)


#-------------------------------
# parallel
res_couplings = Parallel(n_jobs = 4)(delayed(predict_w_couplings)\
		(s,i0,i1i2,niter_max=10,l2=100.0,couplings=couplings)\
		for i0 in range(n_var))
#-------------------------------
for i0 in range(n_var):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
       
    h01 = res_couplings[i0][0]
    w1 = res_couplings[i0][1]

    h0[i1:i2] = h01    
    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]

# make w to be symmetric
w = (w + w.T)/2.
di = direct_info(s0,w)

sorted_DI_er_couplings = sort_di(di)
print('sorted DI: ',sorted_DI_er_couplings[:10])

with open('DI/ER/er_couplings_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_er_couplings, f)
f.close()
#print('ER DI: ', sorted_DI_er)



