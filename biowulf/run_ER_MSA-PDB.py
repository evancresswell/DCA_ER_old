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

pfam_id = sys.argv[1]
num_threads = int(sys.argv[2])

processed_data_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'
working_dir = '/data/cresswellclayec/DCA_ER/biowulf/'

#---------------------------------------------------------------------------------------------------------------------# 
# File Names from MSA-PDB matching and Muscling
pp_msa_file_range = processed_data_path+'MSA_'+pfam_id+'_range.fa'
pp_msa_file_match = processed_data_path+'MSA_'+pfam_id+'_match.fa'

pp_ref_file_range = processed_data_path+'PP_ref_'+pfam_id+'_range.fa'
pp_ref_file_match = processed_data_path+'PP_ref_'+pfam_id+'_match.fa'

pp_msa_file = processed_data_path+'MSA_'+pfam_id+'_.fa'
pp_ref_file = processed_data_path+'PP_ref_'+pfam_id+'_.fa'


muscle_msa_file = processed_data_path+ 'PP_muscle_msa_'+pfam_id+'.fa'

print('Preprocessing MUSCLE DATA')
try:

    # create MSATrimmer instance 
	if os.path.exists(muscle_msa_file):
		if os.path.exists(pp_msa_file_match):
			trimmer = msa_trimmer.MSATrimmer(
			    muscle_msa_file, biomolecule='PROTEIN',
			    refseq_file=pp_ref_file
			)
		else:
			trimmer = msa_trimmer.MSATrimmer(
			    muscle_msa_file, biomolecule='PROTEIN',
			    refseq_file=pp_ref_file
			)
		pp_ref_file = pp_ref_file_match
		muscling = True # so we use the pp_range-MSAmatched and muscled file!!
	elif os.path.exists(pp_msa_file_match):
		trimmer = msa_trimmer.MSATrimmer(
		    pp_msa_file_match, biomolecule='PROTEIN',
		    refseq_file=pp_ref_file_match
		)  
		muscling = False
		pp_ref_file = pp_ref_file_match
	else:
		trimmer = msa_trimmer.MSATrimmer(
		    pp_msa_file_range, biomolecule='PROTEIN',
		    refseq_file=pp_ref_file_range
		)  
		pp_ref_file = pp_ref_file_range
		muscling = False

	# Adding the data_processing() curation from tools to erdca.
	preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
    
except(MSATrimmerException):
	print('           MUSCLE DATA did not work. Trying orginal PDB-Range DATA\nMSATrimmerException')
	try:
		muscling = False
		# Re-Write references file as original pp sequence with pdb_refs-range
		trimmer = msa_trimmer.MSATrimmer(
			pp_msa_file_range, biomolecule='PROTEIN',
			refseq_file=pp_ref_file_range
		    )
		pp_ref_file = pp_ref_file_range

		# Adding the data_processing() curation from tools to erdca.
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms (%s)'%ERR)
		sys.exit()
except(ValueError):
	print('           MUSCLE created empty file. Trying orginal PDB-Range DATA\nValueError')
	try:
		muscling = False
		# Re-Write references file as original pp sequence with pdb_refs-range
		trimmer = msa_trimmer.MSATrimmer(
			pp_msa_file_range, biomolecule='PROTEIN',
			refseq_file=pp_ref_file_range
		    )
		pp_ref_file = pp_ref_file_range

		# Adding the data_processing() curation from tools to erdca.
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
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
pfam_dict['ref_seq_file'] = pp_ref_file
pfam_dict['muscle'] = muscling

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
    seqid = 0.8)

# Compute average product corrected Frobenius norm of the couplings
start_time = timeit.default_timer()
erdca_DI = erdca_inst.compute_sorted_DI()
run_time = timeit.default_timer() - start_time
print('ER run time: %f \n\n'%run_time)
pfam_dict['runtime'] = run_time


di_filename = working_dir+'DI/ER/ER_DI_%s.pickle'%(pfam_id)
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

