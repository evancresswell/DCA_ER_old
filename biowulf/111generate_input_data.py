import os
import numpy as np
import pandas as pd
import data_processing as dp
#--------------------------------------------------------------#
# create pfam folder
os.system('mkdir pfam_ecc/')
os.system('mkdir DI')
os.system('mkdir DI/MF')
os.system('mkdir DI/ER')
os.system('mkdir DI/PLM')

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')
s = np.loadtxt('pfam_ecc.txt',dtype='str')

n = s.shape[0]
pfam_list = s[:,0]
print("Generating Data for Proteins:\n",pfam_list)

for pfam_id in pfam_list:
	data_path = '../../Pfam-A.full'
	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

	# Pre-Process Structure Data
	# delete 'b' in front of letters (python 2 --> python 3)
	print(pdb)	
	pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
			 for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

	# Print number of pdb structures in Protein ID folder
	npdb = pdb.shape[0]
	print('number of pdb structures:',npdb)

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

	np.savetxt('pfam_ecc/%s_s0.txt'%(pfam_id),s0)	
	np.savetxt('pfam_ecc/%s_s_index.txt'%(pfam_id),s_index)	







