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
from pydca.dca_utilities import dca_utilities
import numpy as np

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


pfam_id = 'PF03068'
pfam_id = 'PF00186'



data_path = '../../../Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
data_path = '/home/eclay/Pfam-A.full'

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
# Incorporate SequenceBackmapper to see if PP sequence is in the MSA already. 
#Or if theres a close enough match
    
#just add using muscle:
#https://www.drive5.com/muscle/manual/addtomsa.html
#https://www.drive5.com/muscle/downloads.htmL
muscle_msa_file = 'PP_muscle_msa_'+pfam_id+'.fa'
os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
print("PP sequence added to alignment via MUSCLE")


# create MSATrimmer instance 
trimmer = msa_trimmer.MSATrimmer(
    muscle_msa_file, biomolecule='protein', 
    refseq_file=pp_ref_file
)

# Adding the data_processing() curation from tools to erdca.
preprocessed_data,s_index, cols_removed,s_ipdb = trimmer.get_preprocessed_msa(printing=True, saving = False)

#write trimmed msa to file in FASTA format
preprocessed_data_outfile = 'MSA_PF00186_PreProcessed.fa'
with open(preprocessed_data_outfile, 'w') as fh:
    for seqid, seq in preprocessed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))
        


# Compute DI scores using Expectation Reflection algorithm
erdca_inst = erdca.ERDCA(
    preprocessed_data_outfile,
    'protein',
    pseudocount = 0.5,
    num_threads = 20,
    seqid = 0.8)

# Compute average product corrected Frobenius norm of the couplings
start_time = timeit.default_timer()
erdca_DI = erdca_inst.compute_sorted_DI()
run_time = timeit.default_timer() - start_time
print('ER run time:',run_time)

for site_pair, score in erdca_DI[:5]:
    print(site_pair, score)

with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_er, f)
f.close()


if plotting:
    ipdb = 0

    #------------------------ Load PDB--------------------------#
    pdb = np.load('PF00186_pdb_refs.npy')
    # Pre-Process Structure Data
    # delete 'b' in front of letters (python 2 --> python 3)
    pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])          for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
    print(pdb[ipdb,:])
    pdb_id = pdb[ipdb,5]
    pdb_chain = pdb[ipdb,6]
    seq_num = int(pdb[ipdb,1])
    pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
    #-----------------------------------------------------------#
    # Print Details of protein PDB structure Info for contact visualizeation
    print('Using chain ',pdb_chain)
    print('PDB ID: ', pdb_id)

    from pydca.contact_visualizer import contact_visualizer

    erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
        refseq_file = protein_refseq_file,
        sorted_dca_scores = erdca_DI,
        linear_dist = 4,
        contact_dist = 8.0)

    er_contact_map_data = erdca_visualizer.plot_contact_map()
    er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()

