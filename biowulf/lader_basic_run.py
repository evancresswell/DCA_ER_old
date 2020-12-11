import sys,os
from direct_info import direct_info
from direct_info import sort_di

import expectation_reflection as ER
import data_processing as dp
from joblib import Parallel, delayed
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

from pydca.fasta_reader import fasta_reader
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC

# Import Bio data processing features 
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning

warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
"""
RUN COMMAND:
singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/ER_clean/ /data/cresswellclayec/DCA_ER/erdca_regularized.simg python er_basic_run.py PF00186 $SLURM_CPUS_PER_TASK $SLURM_JOB_ID
"""

#-------------------------------------------------------------------------------------------------------#
#------------------------------- LAD ER Implementation -------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
def fit_LAD(x,y_onehot,niter_max,l2,couplings = None):      
   
    #print(niter_max)    
    n = x.shape[1]
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    #------------------ EigenValue Reg ------------------------@
    # Calculate Eigenvalue of covariance matrix
    # Eigenvalue replaces l2 for regularization strength
    cov_eigen = np.linalg.eigvalsh(c)
    eig_hist, eig_ranges = np.histogram(cov_eigen) 

    #cov_eiv = max(cov_eigen)						# largest eigenvalue
    cov_eiv = min(eig_ranges[eig_ranges > 1e-4]) 			# smallest non-zero eigenvalue
    #cov_eiv = sorted(list(set(cov_eigen.flatten().tolist())))[-5]	# 5th largest eigenvalue
    #print('Regularizing using EV of Cov Mat: ',cov_eiv)

    l2 = cov_eiv # replace l2 (fit_LAD passed parmeter)
    #----------------------------------------------------------@

    # 2019.07.16:  
    c += l2*np.identity(n) / (2*len(y_onehot))

    H0 = np.zeros(m)
    W = np.zeros((n,m))
    #print('y_onehot shape: ',y_onehot.shape)


    for i in range(m):
        y = y_onehot[:,i]  # y = {0,1}
        y1 = 2*y - 1       # y1 = {-1,1}

        # initial values
        h0 = 0.
    
        # If couplings (ie initial weight state) is passed, use it otherwise random.
        if couplings is not None: 
            w = couplings[:,i]
        else: 
            w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
       
        cost = np.full(niter_max,100.)
        #for iloop in range(niter_max):
        # instead of n_iter times, we iterate through scalling values of regu
        regu_scalling_vals = [0.2,0.4,0.6,1.]
        cost = np.full(len(regu_scalling_vals),100.) # redefine cost to be regu scalling iterations
        for iloop,regu_coef in enumerate(regu_scalling_vals):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    
            #print('h shape ', h.shape)
    
            # stopping criterion
            h_too_neg = h < -15
            h[h_too_neg] = -15.
            p = 1/(1+np.exp(-h))                
            #print('p shape ', p.shape)
            #print('y shape ', y.shape)
            cost[iloop] = ((p-y)**2).mean()
    
            if iloop>0 and cost[iloop] >= cost[iloop-1]: break
    
            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]
            # 2019.12.26: 
            h0,w = infer_LAD(x,h[:,np.newaxis],regu = regu_coef*l2)

        H0[i] = h0
        W[:,i] = w

    return H0,W

def infer_LAD(x, y, regu=0.1,tol=1e-8, max_iter=5000):
## 2019.12.26: Jungmin's code    
    #weights_limit = sperf(1e-10)*1e10
    weights_limit = (1e-10)*1e10
    
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    
    mu = np.zeros(x.shape[1])

    w_sol = 0.0*(np.random.rand(s_pred,s_target) - 0.5)
    b_sol = np.random.rand(1,s_target) - 0.5

    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1


            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
            sigma_w = np.std(w_sol[:,index])
                
            w_eq_0 = np.abs(w_sol[:,index]) < 1e-10
            mu[w_eq_0] = 2./np.sqrt(np.pi)
        
            #mu[~w_eq_0] = sigma_w*sperf(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
            mu[~w_eq_0] = sigma_w*(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
                                                        
            w_sol[:,index] = np.linalg.solve(cov_xx + regu * np.diag(mu),cov_xy).reshape(s_pred)
        
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std((weights))
            error = np.mean(np.abs(weights))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit

            #weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            weights[~weights_eq_0] = sigma*(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            
            weights /= np.mean(weights) #now the mean weight is 1.0
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                         ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
    return b_sol[0][0],w_sol[:,0] # for only one target case

def predict_w_LADER(s,i0,i1i2,niter_max,l2,couplings = None):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    # h01,w1 = er_fit(x,y,niter_max,l2) # old inference
    if couplings is not None:
        y_couplings = np.delete(couplings,[range(i1,i2)],0)	# remove subject rows  from original coupling matrix 
        y_couplings = np.delete(y_couplings,[range(i1,i2)],1)   # remove subject columns from original coupling matrix 

        h01,w1 = fit_LAD(x,y,niter_max,l2,couplings = y_couplings)

    else:
        h01,w1 = fit_LAD(x,y,niter_max,l2)

    return h01,w1

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

def delete_sorted_DI_duplicates(sorted_DI):
	temp1 = []
	print(sorted_DI[:10])
	DI_out = dict() 
	for (a,b), score in sorted_DI:
		if (a,b) not in temp1 and (b,a) not in temp1: #to check for the duplicate tuples
			temp1.append(((a,b)))
			if a>b:
				DI_out[(b,a)]= score
			else:
				DI_out[(a,b)]= score
	DI_out = sorted(DI_out.items(), key = lambda k : k[1], reverse=True)
	#DI_out.sort(key=lambda x:x[1],reverse=True) 
	return DI_out 
	    

# from pydca/pydca/erdca.py
def replace_gaps(s0,ref_seq):
	amino_acid_ints = [ 1,  2,  3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15,16,  17, 18, 19, 20]
	s0_nogap = s0
	for i,seq in enumerate(s0):
		#print('\n',seq)
		for ii,aa in enumerate(seq):
			if aa  == 21:
				#s0_nogap[i,ii] = random.choice(s0[:,ii][s0[:,ii]!= 21])
				s0_nogap[i,ii] = refseq[ii]
	#print('at %d replace %d with %d '% (ii,aa,self.__refseq[ii]))
	#print(s0_nogap[i],'\n')
	return s0_nogap

data_path = '/home/eclay/Pfam-A.full'
preprocess_path = '/home/eclay/DCA_ER/biowulf/pfam_ecc/'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/ER_clean/pfam_ecc/'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'



#pfam_id = 'PF00025'
pfam_id = sys.argv[1]
cpus_per_job = int(sys.argv[2])
job_id = sys.argv[3]
print("Calculating DI for %s using %d (of %d) threads (JOBID: %s)"%(pfam_id,cpus_per_job-4,cpus_per_job,job_id))

# Read in Reference Protein Structure
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))                                                                                                                   
# convert bytes to str (python 2 to python 3)                                                                       
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])      for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
ipdb = 0
tpdb = int(pdb[ipdb,1])
print('Ref Sequence # should be : ',tpdb-1)

# Load Multiple Sequence Alignment
s = dp.load_msa(data_path,pfam_id)

# Load Polypeptide Sequence from PDB as reference sequence
print(pdb[ipdb,:])
pdb_id = pdb[ipdb,5]                                                                              
pdb_chain = pdb[ipdb,6]                                                                           
pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])                                             
pdb_range = [pdb_start-1, pdb_end]
#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        

#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    


pfam_dict = {}
#---------------------------------------------------------------------------------------------------------------------#            
#--------------------------------------- Create PDB-PP Reference Sequence --------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
ppb = PPBuilder().build_peptides(chain)                                                       
#    print(pp.get_sequence())
print('peptide build of chain produced %d elements\n\n'%(len(ppb)))                               

matching_seq_dict = {}
poly_seq = list()
for i,pp in enumerate(ppb):
    for char in str(pp.get_sequence()):
        poly_seq.append(char)                                     
print('PDB Polypeptide Sequence: \n',poly_seq)

poly_seq_range = poly_seq[pdb_range[0]:pdb_range[1]]
print('PDB Polypeptide Sequence (In Proteins PDB range len=%d): \n'%len(poly_seq_range),poly_seq_range)

pp_msa_file_range, pp_ref_file_range = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/',nickname='range')

pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/')
#---------------------------------------------------------------------------------------------------------------------#            


#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------- PreProcess FASTA Alignment -------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
trimmed_data_outfile = preprocess_path+'MSA_%s_Trimmed.fa'%pfam_id
print('Pre-Processing MSA')
try:
	print('\n\nPre-Processing MSA with Range PP Seq\n\n')
	trimmer = msa_trimmer.MSATrimmer(
	    pp_msa_file_range, biomolecule='PROTEIN', 
	    refseq_file=pp_ref_file_range
	)
	pfam_dict['ref_file'] = pp_ref_file_range
except:
	print('\nDidnt work, using full PP seq\nPre-Processing MSA wth PP Seq\n\n')
	sys.exit()
	"""
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    pp_msa_file, biomolecule='protein', 
	    refseq_file=pp_ref_file
	)
	pfam_dict['ref_file'] = pp_ref_file
	"""
# Adding the data_processing() curation from tools to erdca.
try:
	trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)
	print('\n\nTrimmed Data: \n',trimmed_data[:10])

except(MSATrimmerException):
	ERR = 'PPseq-MSA'
	print('Error with MSA trimms\n%s\n'%ERR)
	sys.exit()
#write trimmed msa to file in FASTA format
with open(trimmed_data_outfile, 'w') as fh:
	for seqid, seq in trimmed_data:
		fh.write('>{}\n{}\n'.format(seqid, seq))
fh.close()
s_index = list(np.arange(len(''.join(seq))))
#---------------------------------------------------------------------------------------------------------------------#            


# Load trimmed data (the way its done in pydca) for ER calculation
ref_seq_array = fasta_reader.get_alignment_int_form(pp_ref_file_range,biomolecule='Protein')
refseq = ref_seq_array[0]

sequences = fasta_reader.get_alignment_int_form(trimmed_data_outfile,biomolecule='Protein')

s0 = np.asarray(sequences)
print('\ns0 first row:\n',s0[0])
print('s0 shape:',s0.shape,'\n')

# replace gaps (as done in erdca)
s0 = replace_gaps(s0,refseq)
print('    after replacing gaps...\n    s0 shape:',s0.shape)

n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum


#---------------------------------------------------------------------------------------------------------------------#   
#------------------- Generate DCA Couplings - as defined in PYDCA ----------------------------------------------------#       
#---------------------------------------------------------------------------------------------------------------------#        
num_site_states = 21 # 21 possible aa states (including '-')
# MF instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    trimmed_data_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,
)

reg_fi = mfdca_inst.get_reg_single_site_freqs()
reg_fij = mfdca_inst.get_reg_pair_site_freqs()
corr_mat = mfdca_inst.construct_corr_mat(reg_fi, reg_fij)
couplings = mfdca_inst.compute_couplings(corr_mat)
print('DCA couplings shape: ', couplings.shape)
fields_ij = mfdca_inst.compute_two_site_model_fields(couplings, reg_fi)
print('DCA fields shape: ', fields_ij.shape)
#---------------------------------------------------------------------------------------------------------------------#       
#---------------------------------------------------------------------------------------------------------------------#  
#---------------------------------------------------------------------------------------------------------------------#   



#-- Remove rows/cols for aa's (states) which do not exist in positions (MSA column/sequence position) --#
unique_states = np.array([np.unique(s0[:,i]) for i in range(n_var)])
non_states = []
unique_aminos = []
for states in unique_states: 
	unique_aminos.append(states[states!=21]) #DCA couplings calculation doesn't include '-' states
#print('uniqe_aminos[0]:\n',unique_aminos[0])


col = 0
for i in range(n_var):
	for j in range(1,num_site_states):
		if j not in unique_aminos[i]:
			non_states.append(col)
		col += 1
print('deleting %d colums/rows of %d, column/row count should be %d '%(len(non_states),col,s.shape[1]))
er_couplings = np.delete(couplings,non_states,axis=0)
er_couplings = np.delete(er_couplings,non_states,axis=1)

#np.save('pfam_ecc/%s_ER_couplings.npy'%(pfam_id),er_couplings)
print('ER couplings shape: ', er_couplings.shape)
#---------------------------------------------------------------------------------------------------------------------#            


#---------------------------------------------------------------------------------------------------------------------#  
#----------------------------------------- Run Simulation ERDCA ------------------------------------------------------# 
#---------------------------------------------------------------------------------------------------------------------# 

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)

#=========================================================================================

#-------------------------------
# parallel
res = Parallel(n_jobs = cpus_per_job - 4)(delayed(predict_w_LADER)\
        (s,i0,i1i2,niter_max=10,l2=100.0,couplings=er_couplings)\
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

sorted_DI_er = delete_sorted_DI_duplicates(sorted_DI_er)



with open('DI/ER/lader_clean_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_er, f)
f.close()# Save processed data dictionary and FASTA file
print('s shape (msa): ',s.shape)
print('s_index shape: ',len(s_index))
pfam_dict['msa'] = s  
pfam_dict['s_index'] = s_index
pfam_dict['processed_msa'] = trimmed_data 
pfam_dict['s_ipdb'] = tpdb
pfam_dict['cols_removed'] = []

input_data_file = preprocess_path+"%s_DP_LADER_clean.pickle"%(pfam_id)
with open(input_data_file,"wb") as f:
	pickle.dump(pfam_dict, f)
f.close()

# Print resulting DI score
print('Top 10  DIs')
for site_pair, score in sorted_DI_er[:10]:
    print(site_pair, score)

print('\n\tComputing DCA direct information')
try:
	unsorted_DI = mfdca_inst.compute_direct_info(
	    couplings = couplings,
	    fields_ij = fields_ij,
	    reg_fi = reg_fi,
	    seqs_len = equences_len,
	    num_site_states = num_site_states,
	)
	print(unsorted_DI)
except:
	e = sys.exc_info()[0]
	print('could not generate DI.. ERROR: ',e)
#---------------------------------------------------------------------------------------------------------------------# 



plotting = False
if plotting:
	# Print Details of protein PDB structure Info for contact visualizeation
	print('Using chain ',pdb_chain)
	print('PDB ID: ', pdb_id)

	from pydca.contact_visualizer import contact_visualizer

	visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = pp_ref_file,
	sorted_dca_scores = sorted_DI_er,
	linear_dist = 4,
	contact_dist = 8.)

	contact_map_data = visualizer.plot_contact_map()
	plt.show()
	#plt.close()
	tp_rate_data = visualizer.plot_true_positive_rates()
	plt.show()
	#plt.close()
	#print('Contact Map: \n',contact_map_data[:10])
	#print('TP Rates: \n',tp_rate_data[:10])

	with open(preprocess_path+'ER_%s_contact_map_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(contact_map_data, f)
	f.close()

	with open(preprocess_path+'ER_%s_tp_rate_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(tp_rate_data, f)
	f.close()



