import os,sys

sys.path.append(os.path.dirname('/data/cresswellclayec/DCA_ER/biowulf/'))
from Bio import SeqIO
from Bio.PDB import *
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
from glob import glob
import data_processing as dp

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


#========================================================================================
# Loop Throught covid proteins
#========================================================================================

# Check format of s0 
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
pfam_id ='PF00186'
#s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
#for line in s:
#	print(line)


pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
ipdb = 0

# convert bytes to str (python 2 to python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
     for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
#print(pdb)
tpdb = int(pdb[ipdb,1])
#tpdb is the sequence #
#print(tpdb)
#sys.exit()

# Generate MSA numpy array
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins'
dir_list = glob(root_dir+"/*/")
#print(dir_list)
string ="/data/cresswellclayec/DCA_ER/covid_proteins/QHD"
covid_protein_list = []

parser = PDBParser()

for pfam_dir in dir_list:
	pfam_id = pfam_dir[44:-1]
	print('Generating MSA array for ',pfam_id)
	# only work on QHD directoies
	if pfam_id[:2] != 'QH':
		continue

	# Create msa.npy
	try:
		with open(pfam_dir+"/MSA/protein.aln", 'r') as infile:
			MSA = infile.readlines()
			#print('\n\n',MSA,'\n\n')
		#try:
	#		msa = np.chararray((len(MSA),len(MSA[1])-1))
#		except(IndexError):
#			print('MSA doesent have more than 1 row')
#			continue
		msa = []
		for i,line in enumerate(MSA):
			#print(line)
			msa.append(list(line)[:-1])
			print(list(line)[:-1])
			#print(msa[i,:])
		np.save(pfam_id+'/msa.npy',np.asarray(msa))

		#print(type(msa))
		#print(type(msa[0]))
		#print(msa[0])
		covid_protein_list.append(pfam_id)	
	except(FileNotFoundError):
		print('No MSA file')
		continue

	try:
		# Create pdb_refs.npy	
		#print( pfam_dir[:-1]+'.pdb')
		structure = parser.get_structure(id =pfam_id, file = pfam_dir[:-1]+'.pdb')
		if 0:	
			for model in structure:
				for chain in model:
					for residue in chain:
						for a in residue:
							print(a.get_name())
							print(a.get_id())
							print(a.get_coord())      # atomic coordinates
							print(a.get_occupancy())  # occupancy
							print(a.get_altloc())     # alternative location specifier
	except(FileNotFoundError):
		print('No PDB structure')
	print('\n\n\n\n')
print(covid_protein_list)
np.save('covid_protein_list.npy',covid_protein_list)

cov_list = np.load('covid_protein_list.npy')
for pfam_id in cov_list:
	print('\n\n\n',pfam_id,'\n')
	# data processing
	ipdb =0
	s0,cols_removed,s_index,s_ipdb = dp.data_processing_covid(root_dir,pfam_id,ipdb,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
	#print(s0[0])
	pf_dict = {}
	pf_dict['s0'] = s0
	pf_dict['s_index'] = s_index
	pf_dict['s_ipdb'] = s_ipdb
	pf_dict['cols_removed'] = cols_removed

	with open('%s/DP.pickle'%(pfam_id), 'wb') as f:
		pickle.dump(pf_dict, f)
	f.close()
	
	np.random.seed(1)
	#pfam_id = 'PF00025'

	#print(s0.shape,'\n\n')

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

	with open('%s/DI.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(sorted_DI_er, f)
	f.close()
