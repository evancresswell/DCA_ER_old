import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'

from pydca.contact_visualizer import contact_visualizer
from Bio import SeqIO

import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


method_map = {0: 'ER', 1:'MF',2:'PLM'}


#--- Load PDB sequence from contact_visualizer ---#
#biomol_info,er_pdb_seq = erdca_visualizer.pdb_content.pdb_chain_sequences[erdca_visualizer.pdb_chain_id]
#print('ERDCA-pdb (%d) :\n'%(len(er_pdb_seq)),er_pdb_seq)

def get_score(pfam_id, data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'):
	print('\n\nPfam: %s\n\n'%pfam_id)
	#--------------------------------------------------------------------------------#
	# pdb_ref should give list of
	# 0) accession codes,
	# 1) number of residues,
	# 2) number of sequences,
	# 3) and number of pdb references
	# Read in Reference Protein Structure

	# Load or generate structural information
	# Load PDB Strucutre	
	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
	# delete 'b' in front of letters (python 2 --> python 3)
	pdb = np.array([pdb[t,ii].decode('UTF-8') for t in range(pdb.shape[0]) \
	for ii in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

	ipdb = 0
	pdb_id = pdb[ipdb,5]
	pdb_chain = pdb[ipdb,6]

	tpdb = int(pdb[ipdb,1])
	#--------------------------------------------------------------------------------#

	#--------------------------------------------------------------------------------#

	#------------------------------- Load Simulation Data ---------------------------#
	# Load ER data file #
	er_data_file = preprocess_path+"%s_DP_ER.pickle"%(pfam_id)
	with open(er_data_file,"rb") as f:
		ER_pfam_dict = pickle.load(f)
	f.close()

	ER_s0 = ER_pfam_dict['processed_msa']	
	processed_msa = ER_pfam_dict['msa']
	s_index = ER_pfam_dict['s_index']	
	ER_s_ipdb = ER_pfam_dict['s_ipdb']	 # this is actually seq num
	ER_seq_row = ER_s_ipdb 
	cols_removed = ER_pfam_dict['cols_removed']
	ER_ref_file = ER_pfam_dict['ref_file']

	# change s0 (processed_msa) from record list to character array		
	s = []
	for seq_record in ER_s0:
		s.append([char for char in seq_record[1]])		
	ER_s0 = np.array(s)
	er_n_var = ER_s0.shape[1]
	er_num_seqs =ER_s0.shape[0]


	# Load References file (from both sim output and saved fasta)
	with open(ER_ref_file,"r") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			sequence = str(record.seq).upper()
			er_ref_seq = np.array([char for char in sequence])
	er_ref_seq_s0 = ER_s0[ER_s_ipdb]	
	er_rs_s0_full = []
	for i,char in enumerate(er_ref_seq):
		if i in s_index:
			er_rs_s0_full.append(char) 
		else:
			er_rs_s0_full.append('-')
	#-------------------#



	# Load DCA data file #
	dca_data_file = preprocess_path+"%s_DP.pickle"%(pfam_id)
	with open(dca_data_file,"rb") as f:
		pfam_dict = pickle.load(f)
	f.close()

	dca_s0 = pfam_dict['processed_msa']	
	dca_s_ipdb = pfam_dict['s_ipdb']	 # this is actually seq num
	dca_seq_row = dca_s_ipdb 
	dca_ref_file = pfam_dict['ref_file']

	# change s0 (processed_msa) from record list to character array		
	s = []
	for seq_record in dca_s0:
		s.append([char for char in seq_record[1]])		
	dca_s0 = np.array(s)
	dca_n_var = dca_s0.shape[1]
	dca_num_seqs =dca_s0.shape[0]

	with open(dca_ref_file,"r") as handle:
		for record in SeqIO.parse(handle, "fasta"):
			sequence = str(record.seq).upper()
			dca_ref_seq = np.array([char for char in sequence])
	dca_ref_seq_s0 = dca_s0[dca_s_ipdb]	
	#--------------------#

	#--------------------------------------------------------------------------------#

	# Print Specs
	print("#------- ER Specs -------#")
	print('reference sequence: \n',''.join(er_ref_seq_s0),'\n',''.join(er_rs_s0_full),'\n',''.join(er_ref_seq)) 
	print('number positions: \n',er_n_var) 
	print('number sequences: \n',er_num_seqs) 
	print("#------------------------#")
	print("#------- DCA Specs ------#")
	print('reference sequence: \n',''.join(dca_ref_seq_s0),'\n',''.join(dca_ref_seq)) 
	print('number positions: \n',dca_n_var) 
	print('number sequences: \n',dca_num_seqs) 

	print("#------------------------#")
	#--------------------------------------------------------------------------------#

	#--------------------------------------------------------------------------------#
	try:
		with open("DI/ER/er_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_er = pickle.load(f)
		f.close()
		with open("DI/PLM/plm_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_plm = pickle.load(f)
		f.close()
		with open("DI/MF/mf_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_mf = pickle.load(f)
		f.close()
	except(FileNotFoundError):
		print('Not all methods have a DI!!!')
	#--------------------------------------------------------------------------------#

	#--------------------------------------------------------------------------------#
	erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = ER_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_er,
	linear_dist = 5,
	contact_dist = 8. )


	mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = dca_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_mf,
	linear_dist = 5,
	contact_dist = 8. )

	plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = dca_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_plm,
	linear_dist = 5,
	contact_dist = 8. )

	# Print Ranked DIs size by side A
	print('\n#-------- ER ------------------------------------ MF --------------------------- PLM ----------------#')
	for i,pair in enumerate(DI_er[:25]):
		print(pair ,'  ',DI_mf[i],'  ',DI_plm[i])	
	print('#----------------------------------------------------------------------------------------------------#\n')
	#--------------------------------------------------------------------------------#
	plotting = False

	contact_instances = [erdca_visualizer, mfdca_visualizer, plmdca_visualizer]
	scores = []

	for visualizer in contact_instances:
		contact_categories_dict = visualizer.contact_categories()
		tp = contact_categories_dict['tp']	
		fp = contact_categories_dict['fp']	
		p = contact_categories_dict['pdb']	
		missing = contact_categories_dict['missing']	
		print('TP: ',len(tp))
		print('FP: ',len(fp))
		
		tp_rate_dict = er_tp_rate_data = visualizer.compute_true_positive_rates()
		score = np.trapz(tp_rate_dict['dca'],dx=1) / np.trapz(tp_rate_dict['pdb'],dx=1)
		print('\nMethod Score: AU-TPR_method / AUTPR_pdb',score,'\n')
		scores.append(score)

		if plotting:
			er_tp_rate_data = visualizer.plot_contact_map()
			plt.show()

	print('Scores:\n%s %f %f %f %d\n\n' % (pfam_id , scores[0], scores[1], scores[2], dca_num_seqs))
	# write results of Pfam to txt file
	f = open('%s.txt'%pfam_id,'w')
	f.write('%s %f %f %f %d' % (pfam_id , scores[0], scores[1], scores[2], dca_num_seqs) )    
	f.close()



def main():
	#-----------------------------------------------------------------------------------#
	#------------------------------- Create .txt file of scores ------------------------#
	#----------------------------------- Single DI Swarm Job ---------------------------#
	#-----------------------------------------------------------------------------------#

	# Full list of all PFams  being scored (passed to swarm file)
	# Must have DI for all 3 methods
	pfam_id = sys.argv[1]

	get_score(pfam_id, data_path = data_path)
	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#

if __name__ == '__main__':
	main()


