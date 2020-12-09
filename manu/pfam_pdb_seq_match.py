import ecc_tools as tools
import sys,os
import pandas as pd
import matplotlib.pyplot as plt
#import emachine as EM
import pickle as pkl
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from pydca.contact_visualizer import contact_visualizer
import pickle
import data_processing as dp

jobid = os.getenv('SLURM_ARRAY_JOB_ID')

#-- Read in Protein structure --#
data_path = '../../hoangd2_data/Pfam-A.full'
# works
if len(sys.argv[:]) >1:
	pfam_id = sys.argv[1]
else:
	pfam_id = 'PF00186'
ipdb = 0
#-------------------------------#a


#------------------------ Load PDB--------------------------#
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

# make sure swarm file and directories exist
df_filepath = "verified_pdb_refs/swarm/%s.pkl"%jobid
if os.path.exists(df_filepath):
	df = pd.read_pickle(df_filepath)
else:
	if not os.path.exists('verified_pdb_refs'):
		os.system('mkdir verified_pdb_refs')
	if not os.path.exists('verified_pdb_refs/swarm'):
		os.system('mkdir verified_pdb_refs/swarm')

# Pre-Process Structure Data
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
print(pdb[ipdb,:])
pdb_id = pdb[ipdb,5]
pdb_chain = pdb[ipdb,6]
seq_num = int(pdb[ipdb,1])
pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
#-----------------------------------------------------------#



# Original Processed data
s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
					gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9,printing=False)

ref_outfile = 'pfam_ecc/ref_%s.fa'%(pfam_id) 
# check where/how made 
# Save un-processed data

tpdb = int(pdb[ipdb,1])
msa = dp.load_msa(data_path,pfam_id)

gap_pdb = msa[tpdb] =='-' # returns True/False for gaps/no gaps
msa = msa[:,~gap_pdb] # removes gaps  
subject = msa[tpdb]

msa_outfile, ref_outfile = dp.write_FASTA(msa,pfam_id,tpdb,number_form=False,processed=False,path='pfam_ecc/')	

#-- Load DI pairs generated for contact mapping --#
with open('DI/MF/mf_DI_%s.pickle'%(pfam_id), 'rb' ) as f:
	sorted_DI_mf = pickle.load(f)
	f.close()
with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'rb' ) as f:
	sorted_DI_er = pickle.load(f)
	f.close()
with open('DI/PLM/plm_DI_%s.pickle'%(pfam_id), 'rb' ) as f:
	sorted_DI_plm = pickle.load(f)
	f.close()
#-------------------------------------------------#

#-- Load DF generated for DI simulations --#
with open('pfam_ecc/%s_DP.pickle'%(pfam_id), 'rb' ) as f:
	pfam_df = pickle.load(f)
	f.close()
#------------------------------------------#
s_index = pfam_df['s_index']
print('\n s_index: ',s_index,'\n')
# sort DI site pairs using original reference index
print('\n\n#--- Loading and sorting ER DI pairs ---#')
print(sorted_DI_er[:10])
sorted_DI_er = tools.distance_restr_sortedDI(sorted_DI_er, s_index=s_index)
print(sorted_DI_er[:10])
sorted_DI_er = dp.delete_sorted_DI_duplicates(sorted_DI_er)
print(sorted_DI_er[:10])
print('#---------------------------------------#\n\n')

#--- Load Contact Visualizer ---#
#print('Making DCA visualizer instance for pdb id (passed as pdb_file): ',pdb[ipdb,5])
erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
    refseq_file = 'pfam_ecc/'+ref_outfile,
    sorted_dca_scores = sorted_DI_er,
    linear_dist = 4,
    contact_dist = 8.0,
)


# Plot Contact Map
er_contact_map_data = erdca_visualizer.plot_contact_map()
er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
#print(er_contact_map_data)
#print(er_tp_rate_data)
#plt.show()

ref_seq = erdca_visualizer.get_matching_refseq_to_biomolecule()
mapped_res_pos, res_not_found = erdca_visualizer.map_pdbseq_to_refseq()

print('\n\n\npdb alignment: ',mapped_res_pos)
print('ref_seq: ',ref_seq)

mapped_residues, residues_not_found_in_pdb = erdca_visualizer.get_mapped_pdb_contacts()
#print(residues_not_found_in_pdb)

ct = tools.contact_map(pdb,tpdb,cols_removed,s_index)
ct_distal = tools.distance_restr(ct,s_index,make_large=True)
print("\n\n\nour contact map: ")
for i,distance_row in enumerate(ct_distal):
	for j,distance_val in enumerate(distance_row):
		print('Local Dist = ',distance_val)
		print('PYDCA Dist = ',mapped_residues[(i,j)][3])
		print('PYDCA Dist = ',mapped_residues[(j,i)][3])
print('PYDCA contact map: ')
#for key in mapped_residues.keys():
	#print(type(key))	
	#print(mapped_residues[key])
	#print(mapped_residues[key][3])


n_var = msa.shape[1]
di_er = np.zeros((n_var,n_var))
for coupling in sorted_DI_er:
	di_er[coupling[0][0],coupling[0][1]] = coupling[1]
	di_er[coupling[0][1],coupling[0][0]] = coupling[1]


ct_thres = np.linspace(1.5,10.,18,endpoint=True)
n = ct_thres.shape[0]
auc_er = np.zeros(n)
for i in range(n):
	p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
	auc_er[i] = tp.sum()/tp.shape[0]
i0_er = np.argmax(auc_er)
p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])

plt.plot(fp0_er,tp0_er,'b-',label="er")
plt.title('ROC ')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()





#-------------------------------#
#-------------------------- Match Sequence data PDB/MSA -----------------------#
s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
print('unprocessed msa has shape: ',s.shape)
s = np.array([s[t,i].decode('UTF-8') for t in range(s.shape[0]) \
     for i in range(s.shape[1])]).reshape(s.shape[0],s.shape[1])

# Subject Sequence info
gap_pdb = s[seq_num] =='-' # returns True/False for gaps/no gaps
#print("removing gaps...")
s_subject_trimmed = s[:,~gap_pdb] # removes gaps  
print(s.shape)
s_index = np.arange(s.shape[1])
n_vals = 10

print('Given subject sequence (# %d): '%seq_num, s_subject_trimmed[seq_num] )
print('Subject seq length = ',len(s_subject_trimmed[seq_num]))

print('\n\n\nNaively Finding subject sequence by matching first and last 10 values\n\n\n')

pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')
chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain]
ppb = PPBuilder().build_peptides(chain)
print('peptide build of chain produced %d elements'%(len(ppb)))
for k,pp in enumerate(ppb):
	print('chain ',k)
	for i,pp in enumerate(ppb):
		poly_seq = [char for char in str(pp.get_sequence())]
		#poly_seq = poly_seq[pdb_start-1:pdb_end]
		print('PDB seq: ', poly_seq)
		print('PDB seq length =  ', len(poly_seq))

		# Try to find match for poly_seq in msa
		for j,sequence in enumerate(s):
			gap_pdb = sequence =='-' # returns True/False for gaps/no gaps
			sequence_trimmed = sequence[~gap_pdb] # removes gaps  
			sequence_trimmed = [x.upper() for x in sequence_trimmed]
			if j == 69:
				print('SUB seq: ',sequence_trimmed)
				print('Seuence trimmed seq length =  ', len(sequence_trimmed))


			if len(poly_seq) == len(sequence_trimmed):
				if all([a == poly_seq[ii] for ii,a in enumerate(sequence_trimmed[:n_vals])]) and all([a == poly_seq[-n_vals+ii] for ii,a in enumerate(sequence_trimmed[-n_vals:])]):
				#if all([s == poly_seq[i] for i,s in enumerate(s_trimmed[:10])]) :
					print('\n\n')
					print(s_subject_trimmed)
					print('Sequence %d Matches!! '%j)
					
					print('\n\n')
			
#------------------------------------------------------------------------------#










