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


processed_data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files'
working_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'

#---------------------------------------------------------------------------------------------------------------------# 
# RUN WITH: singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/erdca.simg python run_covPROTEIN_ER.py cov_fasta_files/NSP1_aligned.fasta cov_fasta_files/NSP1_ref.fasta $SLURM_CPUS_PER_TASK



# File Names from MSA-PDB matching and Muscling
msa_file = sys.argv[1]
ref_file = sys.argv[2]
num_threads = int(sys.argv[3])- 4
pfam_id = 'spike'

print('Preprocessing FASTA files %s and %s '%(msa_file,ref_file))
try:
    	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    msa_file, biomolecule='PROTEIN',
	    refseq_file=ref_file
	)
	preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False,conserved_cols=.95)
	    
except(MSATrimmerException):
	ERR = 'PPseq-MSA'
	print('Error with MSA trimms (%s)'%ERR)
	sys.exit()
except(ValueError):
	ERR = 'PPseq-MSA'
	print('Error with MSA trimms (%s)'%ERR)
	sys.exit()


#write trimmed msa to file in FASTA format
preprocessed_data_outfile = processed_data_path + 'MSA_%s_PreProcessed.fa'%pfam_id
with open(preprocessed_data_outfile, 'w') as fh:
	for seqid, seq in preprocessed_data:
		fh.write('>{}\n{}\n'.format(seqid, seq))

#---------------------------------------------------------------------------------------------------------------------# 
# Save processed data dictionary and FASTA file
pfam_dict = {}
pfam_dict['s0'] = s
pfam_dict['msa'] = preprocessed_data
pfam_dict['s_index'] = s_index
pfam_dict['s_ipdb'] = s_ipdb
pfam_dict['cols_removed'] = cols_removed

input_data_file = processed_data_path+ "%s_DP.pickle"%(pfam_id)
with open(input_data_file,"wb") as f:
	pickle.dump(pfam_dict, f)
f.close()
#---------------------------------------------------------------------------------------------------------------------# 


# Compute DI scores using Expectation Reflection algorithm
erdca_inst = erdca.ERDCA(
    preprocessed_data_outfile,
    'PROTEIN',
    s_index = s_index,
    pseudocount = 0.5,
    num_threads = num_threads,
    seqid = 0.2)

# Compute average product corrected Frobenius norm of the couplings
start_time = timeit.default_timer()
erdca_DI = erdca_inst.compute_sorted_DI()
run_time = timeit.default_timer() - start_time
print('ER run time: %f \n\n'%run_time)
pfam_dict['runtime'] = run_time


di_filename = working_dir+'ER_DI_%s.pickle'%(pfam_id)
print('\n\nSaving file as ', di_filename)
with open(di_filename, 'wb') as f:
	pickle.dump(erdca_DI, f)
f.close()

#---------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------------------------------------------------------------------# 

input_data_file = processed_data_path+ "%s_DP.pickle"%(pfam_id)
with open(input_data_file,"wb") as f:
	pickle.dump(pfam_dict, f)
f.close()

#---------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------------------------------------------------------------------# 
#---------------------------------------------------------------------------------------------------------------------#            

