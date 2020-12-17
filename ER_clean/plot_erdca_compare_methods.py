import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-MF module
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
import matplotlib
#matplotlib.use('Qt4Agg')
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

# import pydca for plmDCA
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities


def distance_restr_sortedDI(site_pair_DI_in, s_index=None):
	print(site_pair_DI_in[:10])
	restrained_DI= dict()
	for site_pair, score in site_pair_DI_in:
		# if s_index exists re-index sorted pair
		if s_index is not None:
			pos_0 = s_index[site_pair[0]]
			pos_1 = s_index[site_pair[1]]
		else:
			pos_0 = site_pair[0]
			pos_1 = site_pair[1]
    
		indices = (pos_0 , pos_1)
    
		if abs(pos_0- pos_1)<5:
			restrained_DI[indices] = 0
		else:
			restrained_DI[indices] = score
	sorted_DI  = sorted(restrained_DI.items(), key = lambda k : k[1], reverse=True)
	print(sorted_DI[:10])
	return sorted_DI
    
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
	print(DI_out[:10])
	return DI_out 
    



data_path = '../../Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'



pfam_id = 'PF02146'
pfam_id = 'PF13414'
pfam_id = 'PF17795'
pfam_id = 'PF00186'
pfam_id = sys.argv[1]

#---------------------------------------------------------------------------------------------------------------------#            
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
#check that poly_seq matches up with given MSA
    
pp_msa_file, pp_ref_outfile = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/')

#---------------------------------------------------------------------------------------------------------------------#            


input_data_file = "pfam_ecc/%s_DP_ER.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
f.close()
#s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
#				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
s0 = pfam_dict['processed_msa']	
# change s0 (processed_msa) from record list to character array		
s = []
for seq_record in s0:
	s.append([char for char in seq_record[1]])		
s0 = np.array(s)



msa = pfam_dict['msa']	
s_index = pfam_dict['s_index']	
s_ipdb = pfam_dict['s_ipdb']	
cols_removed = pfam_dict['cols_removed']


# number of positions
n_var = s0.shape[1]
print('Number of ERDCA index positions..')
print('erdca reference sequence (%d): '%(len(s0[s_ipdb])),''.join(s0[s_ipdb]))

#
#---------------------- Load DI -------------------------------------#
print("Unpickling DI pickle files for %s"%(pfam_id))
with open("%ser_DI_%s.pickle"%(er_directory,pfam_id),"rb") as f:
	DI_er = pickle.load(f)
f.close()
with open("%splm_DI_%s.pickle"%(plm_directory,pfam_id),"rb") as f:
	DI_plm = pickle.load(f)
f.close()
with open("%smf_DI_%s.pickle"%(mf_directory,pfam_id),"rb") as f:
	DI_mf = pickle.load(f)
f.close()


erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = pp_ref_outfile,
	sorted_dca_scores = DI_er,
	linear_dist = 4,
	contact_dist = 8.0,
)
mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = pp_ref_outfile,
	sorted_dca_scores = DI_mf,
	linear_dist = 4,
	contact_dist = 8.0,
)

plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = pp_ref_outfile,
	sorted_dca_scores = DI_plm,
	linear_dist = 4,
	contact_dist = 8.0,
)

biomol_info,er_pdb_seq = erdca_visualizer.pdb_content.pdb_chain_sequences[erdca_visualizer.pdb_chain_id]
biomol_info,plm_pdb_seq = plmdca_visualizer.pdb_content.pdb_chain_sequences[plmdca_visualizer.pdb_chain_id]
biomol_info,mf_pdb_seq = mfdca_visualizer.pdb_content.pdb_chain_sequences[mfdca_visualizer.pdb_chain_id]
print('\n\nERDCA-Visualizer pdb seq')
print('ERDCA-pdb (%d) :\n'%(len(er_pdb_seq)),er_pdb_seq)
print('PLM-pdb (%d) :\n'%(len(plm_pdb_seq)),plm_pdb_seq)
print('MF-pdb (%d) :\n'%(len(mf_pdb_seq)),mf_pdb_seq)
print('\n\n')

er_tp_rate_data = erdca_visualizer.plot_contact_map()
plt.show()

sys.exit()

mf_tp_rate_data = mfdca_visualizer.plot_contact_map()
plt.show()
plm_tp_rate_data = plmdca_visualizer.plot_contact_map()
plt.show()

er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
plt.show()
mf_tp_rate_data = mfdca_visualizer.plot_true_positive_rates()
plt.show()
plm_tp_rate_data = plmdca_visualizer.plot_true_positive_rates()
plt.show()
