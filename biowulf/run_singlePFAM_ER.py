import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

from pydca.erdca import erdca
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

#========================================================================================
data_path = '/home/eclay/Pfam-A.full'
preprocess_path = '/home/eclay/DCA_ER/biowulf/pfam_ecc/'


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
if len(poly_seq_range) < 10:
	print('PP sequence overlap with PDB range is too small.\nWe will find a match\nBAD PDB-RANGE')
	poly_seq_range = poly_seq
else:
	pp_msa_file_range, pp_ref_file_range = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/',nickname='range')

pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/')
#---------------------------------------------------------------------------------------------------------------------#            


#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------- PreProcess FASTA Alignment -------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
preprocessing = True
preprocessing = False
if preprocessing:
	try:
		preprocessed_data_outfile = preprocess_path+'MSA_%s_PreProcessed.fa'%pfam_id
		print(preprocessed_data_outfile)
		print('\n\nPre-Processing MSA with Range PP Seq\n\n')
		trimmer = msa_trimmer.MSATrimmer(
		    pp_msa_file_range, biomolecule='PROTEIN', 
		    refseq_file=pp_ref_file_range
		)
		pfam_dict['ref_file'] = pp_ref_file_range
	except:
		print('\nDidnt work, using full PP seq\nPre-Processing MSA wth PP Seq\n\n')
		# create MSATrimmer instance 
		trimmer = msa_trimmer.MSATrimmer(
		    pp_msa_file, biomolecule='protein', 
		    refseq_file=pp_ref_file
		)
		pfam_dict['ref_file'] = pp_ref_file
	# Adding the data_processing() curation from tools to erdca.
	try:
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms\n%s\n'%ERR)
		sys.exit()

	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
		for seqid, seq in preprocessed_data:
			fh.write('>{}\n{}\n'.format(seqid, seq))
	fh.close()
else:
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
		# create MSATrimmer instance 
		trimmer = msa_trimmer.MSATrimmer(
		    pp_msa_file, biomolecule='protein', 
		    refseq_file=pp_ref_file
		)
		pfam_dict['ref_file'] = pp_ref_file
	# Adding the data_processing() curation from tools to erdca.
	try:
		trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)
		print('\n\nTrimmed Data: \n',trimmed_data[:10])


		#----- generate data for erdca to calculate couplings -----#
		s0 = []
		for sequence_data in trimmed_data:
			s0.append([char for char in sequence_data[1]])
		s0 = np.array(s0)
		print('\ns0: \n',s0[:10],'\n\n')
		print(s0.shape)

		n_var = s0.shape[1]
		mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
		mx_cumsum = np.insert(mx.cumsum(),0,0)
		i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
		#----------------------------------------------------------#

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

#---------------------------------------------------------------------------------------------------------------------#            
#----------------------------------------- Run Simulation ERDCA ------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            

#========================================================================================
# Compute ER couplings using MF initialization
#========================================================================================
if 1:
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
else:
	#========================================================================================
	# ER - COV-COUPLINGS
	#========================================================================================
        
	onehot_encoder = OneHotEncoder(sparse=False)
        
	s = onehot_encoder.fit_transform(s0)
	
	s_av = np.mean(s,axis=0)
	ds = s - s_av
	l,n = s.shape

	l2 = 100.
	# calculate covariance of s (NOT DS) why not???
	s_cov = np.cov(s,rowvar=False,bias=True)
	# tai-comment: 2019.07.16:  l2 = lamda/(2L)
	s_cov += l2*np.identity(n)/(2*l)
	s_inv = linalg.pinvh(s_cov)
	couplings = s_inv
	print('couplings (s_inv) shape: ', s_inv.shape)
     
#========================================================================================






try:
	print('s_index : ',s_index,'\n')
	if preprocessing:
		print('Initializing ER instance\n\n')
		# Compute DI scores using Expectation Reflection algorithm
		erdca_inst = erdca.ERDCA(
		    preprocessed_data_outfile,
		    #trimmed_data_outfile,
		    'PROTEIN',
		    s_index = s_index,
		    pseudocount = 0.5,
		    num_threads = cpus_per_job-4,
		    seqid = 0.8)
	else:
		print('Initializing ER instance\n\n')
		# Compute DI scores using Expectation Reflection algorithm
		erdca_inst = erdca.ERDCA(
		    #preprocessed_data_outfile,
		    trimmed_data_outfile,
		    'PROTEIN',
		    s_index = s_index,
		    pseudocount = 0.5,
		    num_threads = cpus_per_job-4,
		    seqid = 0.8)

except:
	ref_seq = s[tpdb,:]
	print('Using PDB defined reference sequence from MSA:\n',ref_seq)
	msa_file, ref_file = tools.write_FASTA(ref_seq, s, pfam_id, number_form=False,processed=False,path=preprocess_path)
	pfam_dict['ref_file'] = ref_file

	print('Re-trimming MSA with pdb index defined ref_seq')
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    msa_file, biomolecule='protein', 
	    refseq_file=ref_file
	)

	preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
		for seqid, seq in preprocessed_data:
			fh.write('>{}\n{}\n'.format(seqid, seq))

	erdca_inst = erdca.ERDCA(
		    preprocessed_data_outfile,
		    'PROTEIN',
		    s_index = s_index,
		    pseudocount = 0.5,
		    num_threads = cpus_per_job-4,
		    seqid = 0.8)


print('Running ER simulation\n\n')
# Compute average product corrected Frobenius norm of the couplings
start_time = timeit.default_timer()
erdca_DI = erdca_inst.compute_sorted_DI(LAD=False,init_w = couplings)
run_time = timeit.default_timer() - start_time
print('ER run time:',run_time)

for site_pair, score in erdca_DI[:5]:
    print(site_pair, score)

with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(erdca_DI, f)
f.close()

# Save processed data dictionary and FASTA file
pfam_dict['msa'] = s  
pfam_dict['s_index'] = s_index
if preprocessing:
	pfam_dict['processed_msa'] = preprocessed_data
	pfam_dict['cols_removed'] = cols_removed 
	pfam_dict['s_ipdb'] = s_ipdb
else:
	pfam_dict['processed_msa'] = trimmed_data 
	pfam_dict['s_ipdb'] = tpdb
	pfam_dict['cols_removed'] = []

input_data_file = preprocess_path+"%s_DP_ER.pickle"%(pfam_id)
with open(input_data_file,"wb") as f:
	pickle.dump(pfam_dict, f)
f.close()





#---------------------------------------------------------------------------------------------------------------------#            



plotting = False
if plotting:
	# Print Details of protein PDB structure Info for contact visualizeation
	print('Using chain ',pdb_chain)
	print('PDB ID: ', pdb_id)

	from pydca.contact_visualizer import contact_visualizer

	visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = pp_ref_file,
	sorted_dca_scores = erdca_DI,
	linear_dist = 4,
	contact_dist = 8.)

	contact_map_data = visualizer.plot_contact_map()
	#plt.show()
	#plt.close()
	tp_rate_data = visualizer.plot_true_positive_rates()
	#plt.show()
	#plt.close()
	#print('Contact Map: \n',contact_map_data[:10])
	#print('TP Rates: \n',tp_rate_data[:10])

	with open(preprocess_path+'ER_%s_contact_map_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(contact_map_data, f)
	f.close()

	with open(preprocess_path+'ER_%s_tp_rate_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(tp_rate_data, f)
	f.close()



