import os,sys
import numpy as np
import pandas as pd
import data_processing as dp
#--------------------------------------------------------------#

pfam_id = sys.argv[1]

# create pfam folder
os.system('mkdir pfam_ecc/')
os.system('mkdir DI')
os.system('mkdir DI/MF')
os.system('mkdir DI/ER')
os.system('mkdir DI/PLM')

print("Generating Data for Protein:\n",pfam_id)

# Create list file for missing pdb structures
if not os.path.exists('missing_PDB.txt'):
	file_missing_pdb = open("missing_PDB.txt",'w')
	file_missing_pdb.close()

def generate_pfam_data(pfam_id):
	data_path = '../../hoangd2_data/Pfam-A.full'
	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

	# Pre-Process Structure Data
	# delete 'b' in front of letters (python 2 --> python 3)
	print(pdb.shape)	
	print(pdb)	
	if len(pdb) == 0:
		print("Missing PDB structure")
		file_missing_pdb = open("missing_PDB.txt",'a')
		file_missing_pdb.write("%s\n"% pfam_id)
		file_missing_pdb.close()
	else:	
		pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
				 for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

		# Create pandas dataframe for protein structure
		df = pd.DataFrame(pdb,columns = ['PF','seq','id','uniprot_start','uniprot_start',\
										 'pdb_id','chain','pdb_start','pdb_end'])
		print(df.head())

		ipdb = 0
		print('seq:',int(pdb[ipdb,1]))

		# data processing
		s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
						gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
		# Save processed data
		msa_outfile, ref_outfile = dp.write_FASTA(s0,pfam_id,s_ipdb,path='pfam_ecc/')	
		pf_dict = {}
		pf_dict['s0'] = s0
		pf_dict['s_index'] = s_index
		pf_dict['s_ipdb'] = s_ipdb
		pf_dict['cols_removed'] = cols_removed

		with open('pfam_ecc/%s_DP.pickle'%(pfam_id), 'wb') as f:
			pickle.dump(pf_dict, f)
		f.close
		return

#-------------------------------


generate_pfam_data(pfam_id)


