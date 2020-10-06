import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-DCA module
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca

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

#========================================================================================
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'

data_path = '/home/eclay/Pfam-A.full'
preprocess_path = '/home/eclay/DCA_ER/biowulf/pfam_ecc/'


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
#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        

#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    
#---------------------------------------------------------------------------------------------------------------------#            
chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
ppb = PPBuilder().build_peptides(chain)                                                       
#    print(pp.get_sequence())
print('peptide build of chain produced %d elements'%(len(ppb)))                               

poly_seq = list()
for i,pp in enumerate(ppb):
    for char in str(pp.get_sequence()):
        poly_seq.append(char)                                     
print('PDB Polypeptide Sequence: \n',poly_seq)
#check that poly_seq matches up with given MSA
    
pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False,path=preprocess_path)


preprocessed_data_file = 'MSA_%s_Trimmed.fa'%pfam_id
if os.path.exists(preprocess_path+preprocessed_data_file):    
	print('Using existing pre-processed FASTA files\n')
	trimmed_data_outfile = preprocess_path+preprocessed_data_file
else:
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(pp_msa_file, biomolecule='protein', refseq_file=pp_ref_file)
	# Adding the data_processing() curation from tools to erdca.
	try:
		trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms\n%s\n'%ERR)
		sys.exit()
	#write trimmed msa to file in FASTA format
	trimmed_data_outfile = preprocess_path+preprocessed_data_file
	with open(trimmed_data_outfile, 'w') as fh:
		for seqid, seq in trimmed_data:
			fh.write('>{}\n{}\n'.format(seqid, seq))
		fh.close()

# Compute DI scores using Expectation Reflection algorithm
# MF instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    trimmed_data_outfile,
    'PROTEIN',
    pseudocount = 0.5,
    seqid = 0.8,
)

# Compute average product corrected Frobenius norm of the couplings
start_time = timeit.default_timer()
# Compute DCA scores 
sorted_DI_mf = mfdca_inst.compute_sorted_DI()
run_time = timeit.default_timer() - start_time
print('DCA run time:',run_time)

for site_pair, score in sorted_DI_mf[:5]:
    print(site_pair, score)

with open('DI/MF/mf_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_mf, f)
f.close()

# Print Details of protein PDB structure Info for contact visualizeation
print('Using chain ',pdb_chain)
print('PDB ID: ', pdb_id)

from pydca.contact_visualizer import contact_visualizer

visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
refseq_file = pp_ref_file,
sorted_dca_scores = sorted_DI_mf,
linear_dist = 4,
contact_dist = 8.)

contact_map_data = visualizer.plot_contact_map()
plt.show()
plt.close()
tp_rate_data = visualizer.plot_true_positive_rates()
plt.show()
plt.close()

