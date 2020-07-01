import sys,os
import numpy as np
import pickle
from scipy import linalg
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca
import expectation_reflection as ER
from direct_info import direct_info
from joblib import Parallel, delayed
import data_processing as dp
import time
import ecc_tools as tools
#import er_setup as er_tools # these functions were moved to ecc_tools 6/20/2020

import matplotlib.pyplot as plt
from direct_info import sort_di

#========================================================================================
# Set data path
#========================================================================================
data_path = '/home/eclay/Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'

np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'

input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
f.close()
#s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
#				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
ipdb = 0
s0 = pfam_dict['s0']	
# use Feb version of s0
s0 = np.loadtxt('pfam_ecc/%s_s0.txt'%(pfam_id))
print(s0.shape)
s_index = pfam_dict['s_index']	
s_ipdb = pfam_dict['s_ipdb']	
cols_removed = pfam_dict['cols_removed']



#========================================================================================
# Process data and write fast file for ER and MF
#========================================================================================

n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 



# number of positions
n_var = s0.shape[1]


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
    print('y_couplings shape: ',y_couplings.shape, ' x-column size: ',x.shape[1])	# Should be same dimensions as x column size as a result

    #print('predict_w, x: shape', x.shape)
    #print('predict_w, y: shape', y.shape)

    h01,w1 = ER.fit(x,y,niter_max,l2,y_couplings)

    return h01,w1

#========================================================================================

#========================================================================================
# Compute ER 
#                   ER
print('\n\ncomputing curated q-states ER\n\n')
er_start = time.time()
#========================================================================================
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]
print ('Investigating Protein Famility ', pfam_id)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)
print('s0: \n',s0[0][0])
print('s0 OneHot -> s: \n',s[0][:mx[0]])



computing_DI = False
computing_DI = True
if computing_DI:

	#========================================================================================
	#                   DCA
	#========================================================================================
	computing_DCA = False
	computing_DCA = True
	if computing_DCA:
		msa_outfile = '/home/eclay/DCA_ER/pfam_ecc/MSA_%s.fa'%(pfam_id) 
		# use msa fasta file generated in data_processing
		msa_outfile = './pfam_ecc/MSA_%s.fa'%(pfam_id) 
		msa_outfile = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/MSA_%s.fa'%(pfam_id) 

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

		sorted_DI_mf = mfdca_inst.compute_sorted_DI()
		#with open('DI/MF/mf_DI_%s.pickle'%(pfam_id), 'wb') as f:
		with open('./mf_DI_%s.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(sorted_DI_mf, f)
		f.close()

	#========================================================================================
	# Get coupling matrix
	computing_coupling = False
	computing_coupling = True
	if computing_coupling:
		# seqid is similar to find_conserved_cols' fc in data_processing (fc = .8)

		# Compute Sequence Weight Array
		if os.path.exists('%s_seqs_weight.npy'%(pfam_id)):
			seqs_file = open('%s_seqs_weight.npy'%(pfam_id), 'rb')
			seqs_weight = np.load(seqs_file,allow_pickle=True)
			seqs_file.close()
		else:
			seqs_weight = tools.compute_sequences_weight(alignment_data = s0, seqid = .8)
			np.save('%s_seqs_weight.npy'%(pfam_id),np.array(seqs_weight))
		#print (seqs_weight)
		#print (seqs_weight.shape)

		# Compute Single Site Frequency Matrix
		if os.path.exists('%s_single_site_freqs.npy'%(pfam_id)):
			site_freqs_file = open('%s_single_site_freqs.npy'%(pfam_id), 'rb')
			single_site_freqs = np.load(site_freqs_file,allow_pickle=True)
			site_freqs_file.close()
		else:
			single_site_freqs = tools.compute_single_site_freqs(alignment_data = s0,seqs_weight=seqs_weight,mx= mx)
			np.save('%s_single_site_freqs.npy'%(pfam_id),np.array(single_site_freqs))
		#print (len(single_site_freqs))
		#print(single_site_freqs[0])

		# Regularize Single Site Frequency Matrix
		for i,states in enumerate(mx):
			if states != len(single_site_freqs[i]):
				print('order of single_sites_freqs (%d) does not match mx (%d)'%(states,len(single_site_freqs[i])))
		reg_single_site_freqs = tools.get_reg_single_site_freqs(
			    single_site_freqs = single_site_freqs,
			    seqs_len = n_var,
			    mx = mx,
			    pseudocount = .5) # default pseudocount value used in regularization
		#print (len(reg_single_site_freqs))
		#print(reg_single_site_freqs[0])

		# Compute Pair Site Frequency Matrix
		if os.path.exists('%s_pair_site_freqs.npy'%(pfam_id)):
			pair_site_file = open('%s_pair_site_freqs.npy'%(pfam_id), 'rb')
			pair_site_freqs = np.load(pair_site_file,allow_pickle=True)
			pair_site_file.close()
		else:
			pair_site_freqs = tools.compute_pair_site_freqs_serial(alignment_data=s0, mx=mx,seqs_weight=seqs_weight)
			np.save('%s_pair_site_freqs.npy'%(pfam_id),np.array(pair_site_freqs))
		#print (len(pair_site_freqs))
		#print(pair_site_freqs[0][0][0])

		#print(mx.cumsum())
		#print(len(mx.cumsum()))
		# Compute Correlation Matrix
		if os.path.exists('%s_corr_mat.npy'%(pfam_id)):
			corr_file = open('%s_corr_mat.npy'%(pfam_id), 'rb')
			corr_mat = np.load(corr_file,allow_pickle=True)
			print('Correlation Matrix loaded: shape ',corr_mat.shape,':\n',corr_mat)
			corr_file.close()
		else:
			corr_mat =  tools.construct_corr_mat(reg_fi = reg_single_site_freqs, reg_fij = pair_site_freqs, seqs_len = n_var, mx = mx)
			np.save('%s_corr_mat.npy'%(pfam_id),corr_mat)
		#print(corr_mat.shape)
		#print('row 1:\n', corr_mat[0][:40])
		#print('row 14:\n',corr_mat[13][:40])
		#print(len(corr_mat))

		# Compute Coupling Matrix
		if os.path.exists('%s_couplings.npy'%(pfam_id)):
			couplings_file = open('%s_couplings.npy'%(pfam_id), 'rb')
			couplings = np.load(couplings_file,allow_pickle=True)
			print('Couplings Matrix loaded: \n',couplings)


			s_av = np.mean(s,axis=0)
			ds = s - s_av
			l,n = s.shape

			l2 = 100.
			# calculate covariance of s (NOT DS) why not???
			s_cov = np.cov(s,rowvar=False,bias=True)
			# tai-comment: 2019.07.16:  l2 = lamda/(2L)
			s_cov += l2*np.identity(n)/(2*l)
			s_inv = linalg.pinvh(s_cov)
			print('s_inv shape: ', s_inv.shape)



			print('vs cov-inv: ',s_inv)
			couplings_file.close()

		else:
			couplings = tools.compute_couplings(corr_mat = corr_mat)
			np.save('%s_couplings.npy'%(pfam_id),couplings)
		print('Coupling Matrix Generated:')
		print(couplings.shape)
		print('row1:\n',couplings[0][:30])
		print('row 14:\n',couplings[13][:30])

		#print(np.unique(s0[:,20]))
	
	couplings_file = open('%s_couplings.npy'%(pfam_id), 'rb')
	couplings = np.load(couplings_file,allow_pickle=True)
	couplings_file.close()

	#========================================================================================
	# ER - COUPLINGS
	#========================================================================================
	w = np.zeros((mx_sum,mx_sum))
	print('full weights matrix: shape: ',w.shape)

	h0 = np.zeros(my_sum)


	#-------------------------------
	# parallel
	res_couplings = Parallel(n_jobs = 12)(delayed(predict_w_couplings)\
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

	with open('DI/ER/er_couplings_DI_%s.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(sorted_DI_er_couplings, f)
	f.close()
	#print('ER DI: ', sorted_DI_er)


	er_end = time.time()

	#========================================================================================
	#========================================================================================
	# ER - NO COUPLINGS
	#========================================================================================
	# parallel
	res = Parallel(n_jobs = 12)(delayed(predict_w)\
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

	with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(sorted_DI_er, f)
	f.close()

	#print('ER DI: ', sorted_DI_er)



	#========================================================================================

	w = np.zeros((mx_sum,mx_sum))
	print('full weights matrix: shape: ',w.shape)

	h0 = np.zeros(my_sum)

	#========================================================================================
	# ER - COV-COUPLINGS
	#========================================================================================

	s_av = np.mean(s,axis=0)
	ds = s - s_av
	l,n = s.shape

	l2 = 100.
	# calculate covariance of s (NOT DS) why not???
	s_cov = np.cov(s,rowvar=False,bias=True)
	# tai-comment: 2019.07.16:  l2 = lamda/(2L)
	s_cov += l2*np.identity(n)/(2*l)
	s_inv = linalg.pinvh(s_cov)
	print('s_inv shape: ', s_inv.shape)

	w = np.zeros((mx_sum,mx_sum))
	print('full weights matrix: shape: ',w.shape)

	h0 = np.zeros(my_sum)


	#-------------------------------
	# parallel
	res_cov_couplings = Parallel(n_jobs = 8)(delayed(predict_w_couplings)\
			(s,i0,i1i2,niter_max=10,l2=100.0,couplings=s_inv)\
			for i0 in range(n_var))
	#-------------------------------
	for i0 in range(n_var):
	    i1,i2 = i1i2[i0,0],i1i2[i0,1]
	       
	    h01 = res_cov_couplings[i0][0]
	    w1 = res_cov_couplings[i0][1]

	    h0[i1:i2] = h01    
	    w[:i1,i1:i2] = w1[:i1,:]
	    w[i2:,i1:i2] = w1[i1:,:]

	# make w to be symmetric
	w = (w + w.T)/2.
	di = direct_info(s0,w)

	sorted_DI_er_cov_couplings = sort_di(di)

	with open('DI/ER/er_cov_couplings_DI_%s.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(sorted_DI_er_cov_couplings, f)
	f.close()
	#print('ER DI: ', sorted_DI_er)


	er_end = time.time()


	#========================================================================================

#========================================================================================
# Calculate ROC curves
#========================================================================================
plotting = True
#if plotting and not computing_DI:
if plotting :
	print ('Plotting Protein Famility ', pfam_id)

	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

	#---------- Pre-Process Structure Data ----------------#
	# delete 'b' in front of letters (python 2 --> python 3)
	pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
	for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

	ct = tools.contact_map(pdb,ipdb,cols_removed,s_index)
	ct_distal = tools.distance_restr(ct,s_index,make_large=True)



	#---------------------- Load DI -------------------------------------#
	print("Unpickling DI pickle files for %s"%(pfam_id))
	file_obj = open("DI/ER/er_DI_%s.pickle"%(pfam_id),"rb")
	sorted_DI_er = pickle.load(file_obj)
	file_obj.close()

	file_obj = open("DI/MF/mf_DI_%s.pickle"%(pfam_id),"rb")
	sorted_DI_mf = pickle.load(file_obj)
	file_obj.close()

	file_obj =open("DI/ER/er_couplings_DI_%s.pickle"%(pfam_id),"rb")
	sorted_DI_er_coupling = pickle.load(file_obj)
	file_obj.close()
	
	file_obj = open("DI/ER/er_cov_couplings_DI_%s.pickle"%(pfam_id),"rb")
	sorted_DI_er_cov_coupling = pickle.load(file_obj)
	file_obj.close()

	#sorted_DI_er = dp.delete_sorted_DI_duplicates(sorted_DI_ER_redundant)

	from data_processing import delete_sorted_DI_duplicates # newly generated for this python notebook.
	sorted_DI_er = delete_sorted_DI_duplicates(sorted_DI_er)
	sorted_DI_mf = delete_sorted_DI_duplicates(sorted_DI_mf)
	sorted_DI_er_coupling = delete_sorted_DI_duplicates(sorted_DI_er_coupling)
	sorted_DI_er_cov_coupling = delete_sorted_DI_duplicates(sorted_DI_er_cov_coupling)

	restrain_distance = True
	if restrain_distance:
		sorted_DI_er = tools.distance_restr_sortedDI(sorted_DI_er)
		sorted_DI_mf = tools.distance_restr_sortedDI(sorted_DI_mf)
		sorted_DI_er_coupling = tools.distance_restr_sortedDI(sorted_DI_er_coupling)
		sorted_DI_er_cov_coupling = tools.distance_restr_sortedDI(sorted_DI_er_cov_coupling)

	print("\nPrint top 10 Non-Redundant pairs")
	sorted_DIs = [sorted_DI_er,sorted_DI_mf,sorted_DI_er_coupling,sorted_DI_er_cov_coupling]
	for sorted_DI in sorted_DIs:	
		for x in sorted_DI[:10]:
			print(x)
		print('\n\n')
	#--------------------------------------------------------------------#


	#---------------------- Plot Contact Map ----------------------------#

	#plt.title('Contact Map')
	plt.imshow(ct_distal,cmap='rainbow_r',origin='lower')
	plt.xlabel('i')
	plt.ylabel('j')
	plt.colorbar(fraction=0.045, pad=0.05)
	plt.close()
	#--------------------------------------------------------------------#

	n_seq = max([coupling[0][0] for coupling in sorted_DI_er]) 
	di_er = np.zeros((n_var,n_var))
	di_mf = np.zeros((n_var,n_var))
	di_er_coupling = np.zeros((n_var,n_var))
	di_er_cov_coupling = np.zeros((n_var,n_var))
	for coupling in sorted_DI_er:
		#print(coupling[1])
		di_er[coupling[0][0],coupling[0][1]] = coupling[1]
		di_er[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in sorted_DI_mf:
		di_mf[coupling[0][0],coupling[0][1]] = coupling[1]
		di_mf[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in sorted_DI_er_coupling:
		di_er_coupling[coupling[0][0],coupling[0][1]] = coupling[1]
		di_er_coupling[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in sorted_DI_er_cov_coupling:
		di_er_cov_coupling[coupling[0][0],coupling[0][1]] = coupling[1]
		di_er_cov_coupling[coupling[0][1],coupling[0][0]] = coupling[1]


	#--------------------------------------------------------------------#

	#----------------- Generate Optimal ROC Curve -----------------------#
	# find optimal threshold of distance for both DCA and ER
	ct_thres = np.linspace(1.5,10.,18,endpoint=True)
	n = ct_thres.shape[0]

	auc_mf = np.zeros(n)
	auc_er = np.zeros(n)
	auc_er_coupling = np.zeros(n)
	auc_er_cov_coupling = np.zeros(n)

	for i in range(n):
		p,tp,fp = tools.roc_curve(ct_distal,di_mf,ct_thres[i])
		auc_mf[i] = tp.sum()/tp.shape[0]
		
		################################3 need to update singularity container p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
		p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
		auc_er[i] = tp.sum()/tp.shape[0]
		
		p,tp,fp = tools.roc_curve(ct_distal,di_er_coupling,ct_thres[i])
		auc_er_coupling[i] = tp.sum()/tp.shape[0]
	
		p,tp,fp = tools.roc_curve(ct_distal,di_er_cov_coupling,ct_thres[i])
		auc_er_cov_coupling[i] = tp.sum()/tp.shape[0]


	i0_mf = np.argmax(auc_mf)
	i0_er = np.argmax(auc_er)
	i0_er_coupling = np.argmax(auc_er_coupling)
	i0_er_cov_coupling = np.argmax(auc_er_cov_coupling)


	p0_mf,tp0_mf,fp0_mf = tools.roc_curve(ct_distal,di_mf,ct_thres[i0_mf])
	##################################### need to update singularity container   p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
	p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
	p0_er_coupling,tp0_er_coupling,fp0_er_coupling = tools.roc_curve(ct_distal,di_er_coupling,ct_thres[i0_er_coupling])
	p0_er_cov_coupling,tp0_er_cov_coupling,fp0_er_cov_coupling = tools.roc_curve(ct_distal,di_er_cov_coupling,ct_thres[i0_er_cov_coupling])


	#========================================================================================
	# Plot ROC for optimal DCA vs optimal ER

	#========================================================================================
	print("Optimal Contact threshold for (mf, er, er_coupling, er_cov_coupling) = (%f, %f, %f, %f)"%(ct_thres[i0_mf],ct_thres[i0_er],ct_thres[i0_er_coupling],ct_thres[i0_er_cov_coupling]))
	print("Maximal AUC for (mf, er, er_coupling) = (%f, %f, %f, %f)"%(auc_mf[i0_mf], auc_er[i0_er], auc_er_coupling[i0_er_coupling], auc_er_cov_coupling[i0_er_cov_coupling]))


	plt.subplot2grid((1,3),(0,0))
	plt.title('ROC ')
	plt.plot(fp0_er,tp0_er,'b-',label="er")
	plt.plot(fp0_mf,tp0_mf,'r-',label="mf")
	plt.plot(fp0_er_coupling,tp0_er_coupling,'g-',label="er_coupling")
	plt.plot(fp0_er_cov_coupling,tp0_er_cov_coupling,color='#F39C12',label="er_cov_coupling")
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()

	# Plot AUC for DCA and ER
	plt.subplot2grid((1,3),(0,1))
	plt.title('AUC')
	plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
	plt.plot(ct_thres,auc_er,'b-',label="er")
	plt.plot(ct_thres,auc_mf,'r-',label="mf")
	plt.plot(ct_thres,auc_er_coupling,'g-',label="er_coupling")
	plt.plot(ct_thres,auc_er_cov_coupling,color='#F39C12',label="er_cov_coupling")
	plt.ylim([min(auc_er.min(),auc_mf.min(),auc_er_coupling.min(),auc_er_cov_coupling.min())-0.05,max(auc_er.max(),auc_mf.max(),auc_er_coupling.max(),auc_er_cov_coupling.max())+0.05])
	plt.xlim([ct_thres.min(),ct_thres.max()])
	plt.xlabel('distance threshold')
	plt.ylabel('AUC')
	plt.legend()

	# Plot Precision of optimal DCA and ER
	plt.subplot2grid((1,3),(0,2))
	plt.title('Precision')
	plt.plot( p0_er,tp0_er / (tp0_er + fp0_er),'b-',label='er')
	plt.plot( p0_mf,tp0_mf / (tp0_mf + fp0_mf),'r-',label='mf')
	plt.plot( p0_er_coupling,tp0_er_coupling / (tp0_er_coupling + fp0_er_coupling),'g-',label='er_coupling')
	plt.plot( p0_er_cov_coupling,tp0_er_cov_coupling / (tp0_er_cov_coupling + fp0_er_cov_coupling),color='#F39C12',label='er_cov_coupling')
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0,1])
	#plt.ylim([0,1])
	plt.ylim([.4,.8])
	plt.xlabel('Recall (Sensitivity - P)')
	plt.ylabel('Precision (PPV)')
	plt.legend()

	plt.tight_layout(h_pad=1, w_pad=1.5)

	plt.show()



	#========================================================================================


