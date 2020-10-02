#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""pydca demo

Author: Evan Cresswell-Clay
"""
import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC
import matplotlib.pyplot as plt

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


pfam_id = 'PF07073'
pfam_id = 'PF03068' # MUSCLE removes all msa alignments
pfam_id = 'PF01583'
pfam_id = 'PF10401'
pfam_id = 'PF14806'
pfam_id = 'PF00186'


data_path = '../../../Pfam-A.full'
data_path = '/home/eclay/Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'

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
#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        

#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    
#---------------------------------------------------------------------------------------------------------------------#            
chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
ppb = PPBuilder().build_peptides(chain)                                                       
#    print(pp.get_sequence())
print('peptide build of chain produced %d elements'%(len(ppb)))                               

found_match = True
matching_seq_dict = {}
poly_seq = list()
for i,pp in enumerate(ppb):
    for char in str(pp.get_sequence()):
        poly_seq.append(char)                                     
print('PDB Polypeptide Sequence: \n',poly_seq)
#check that poly_seq matches up with given MSA
    
pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False)


muscling  = False
muscling  = True

preprocessing = False
preprocessing = True

computing_DI = False
computing_DI = True

compute_ROC = False
compute_ROC = True


muscle_msa_file = 'PP_muscle_msa_'+pfam_id+'.fa'
if muscling:    
	#just add using muscle:
	#https://www.drive5.com/muscle/manual/addtomsa.html
	#https://www.drive5.com/muscle/downloads.htmL
	os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
	print("PP sequence added to alignment via MUSCLE")


preprocessed_data_outfile = 'MSA_%s_PreProcessed.fa'%pfam_id
if preprocessing:
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    muscle_msa_file, biomolecule='PROTEIN', 
	    refseq_file=pp_ref_file
	)


	# Adding the data_processing() curation from tools to erdca.
	try:
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms (%s)'%ERR)
		sys.exit()

	# Save processed data dictionary and FASTA file
	pfam_dict = {}
	pfam_dict['s0'] = s
	pfam_dict['msa'] = preprocessed_data
	pfam_dict['s_index'] = s_index
	pfam_dict['s_ipdb'] = s_ipdb
	pfam_dict['cols_removed'] = cols_removed 

	input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
	with open(input_data_file,"wb") as f:
		pickle.dump(pfam_dict, f)
	f.close()

	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
	    for seqid, seq in preprocessed_data:
	        fh.write('>{}\n{}\n'.format(seqid, seq))
else:
	input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
	with open(input_data_file,"rb") as f:
		pfam_dict =  pickle.load(f)
	f.close()
	cols_removed = pfam_dict['cols_removed']
	s_index= pfam_dict['s_index']
	s_ipdb = pfam_dict['s_ipdb']



if computing_DI:
	# Compute DI scores using Expectation Reflection algorithm
	erdca_inst = erdca.ERDCA(
	    preprocessed_data_outfile,
 	    'PROTEIN',
	    s_index = s_index,
	    pseudocount = 0.5,
	    num_threads = 40,
	    seqid = 0.8)

	# Compute average product corrected Frobenius norm of the couplings
	start_time = timeit.default_timer()
	erdca_DI = erdca_inst.compute_sorted_DI()
	run_time = timeit.default_timer() - start_time
	print('ER run time:',run_time)

	for site_pair, score in erdca_DI[:5]:
	    print(site_pair, score)

	with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(erdca_DI, f)
	f.close()
else:
	with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'rb') as f:
	    erdca_DI = pickle.load( f)
	f.close()

# Works
if compute_ROC:
	# For muscle and poly-seq error debugging
	"""
	MUST be a PFAM from job 163:
	pfam_id = 'PF00186'
	pfam_id = 'PF03068'
	pfam_id = 'PF07073'
	pfam_id = 'PF10401'
	pfam_id = 'PF14806'
	pfam_id = 'PF01583'
	"""

	# Jobload info from text file 
	prep_df_file_er = 'ER_job-53759610_swarm_ouput_setup.pkl'
	job_id_er = '53759610_163'


	# Get dataframe of job_id
	df_prep_er = pickle.load(open(prep_df_file_er,"rb"))
	df_jobID_er = df_prep_er.copy()
	df_jobID_er = df_jobID_er.loc[df_jobID_er.Jobid == job_id_er]
	roc_jobID_df = add_ROC(df_jobID_er,prep_df_file_er,data_path=data_path,pfam_id_focus = pfam_id)
	with open('./%s_roc_DF.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(roc_jobID_df, f)
	f.close()

else:	
	with open('./%s_roc_DF.pickle'%(pfam_id), 'rb') as f:
	   roc_jobID_df =  pickle.load(f)
	f.close()

plotting = True
if plotting:
	if 1:
		# Print Details of protein PDB structure Info for contact visualizeation
		print('Using chain ',pdb_chain)
		print('PDB ID: ', pdb_id)
		print(roc_jobID_df['OptiDist'])

		from pydca.contact_visualizer import contact_visualizer

		erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
		refseq_file = pp_ref_file,
		sorted_dca_scores = erdca_DI,
		linear_dist = 4,
		contact_dist = 8.)

		er_contact_map_data = erdca_visualizer.plot_contact_map()
		plt.show()
		plt.savefig('contact_map_%s.pdf'%pfam_id)
		plt.close()
		er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
		plt.show()
		plt.savefig('TP_rate_%s.pdf'%pfam_id)
		plt.close()


	tp = np.asarray(roc_jobID_df['TP'])[0]
	fp =np.asarray(roc_jobID_df['FP'])[0]
	p = np.asarray(roc_jobID_df['P'])[0]
	print('len tp= %d len fp = %d '%(len(tp),len(fp)))
	print(tp)
	print(fp)
	from matplotlib.backends.backend_pdf import PdfPages
	with PdfPages("./%s_ROC.pdf"%pfam_id) as pdf:
		plt.subplot2grid((1,2),(0,0))
		plt.title('ROC ')
		plt.plot(fp,tp,'b-',label="er")
		plt.plot([0,1],[0,1],'k--')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()
		print('%s AUC: %f'%(pfam_id,np.asarray(roc_jobID_df['AUC'])[0]))

		
		# Plot Precision of optimal DCA and ER
		plt.subplot2grid((1,2),(0,1))
		plt.title('Precision')
		plt.plot( p,tp / (tp + fp),'b-',label='er')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylim([.4,.8])
		plt.xlabel('Recall (Sensitivity - P)')
		plt.ylabel('Precision (PPV)')
		plt.legend()

		plt.tight_layout(h_pad=.25, w_pad=.1)
		pdf.attach_note("ROC")  # you can add a pdf note to
		plt.show()
		plt.close()
		#----------------------------------------------------------------------------#

