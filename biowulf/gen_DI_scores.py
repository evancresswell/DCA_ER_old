import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool
from pydca.contact_visualizer import contact_visualizer
from Bio import SeqIO
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning

warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt



"""
RUN COMMAND:
singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_regularized.simg python gen_DI_scores.py PF00186
"""

method_map = {0:'LADER_clean',1:'ER_clean',2:'ER_INIT',3: 'ER', 4:'MF',5:'PLM'}


#--- Load PDB sequence from contact_visualizer ---#
#biomol_info,er_pdb_seq = erdca_visualizer.pdb_content.pdb_chain_sequences[erdca_visualizer.pdb_chain_id]
#print('ERDCA-pdb (%d) :\n'%(len(er_pdb_seq)),er_pdb_seq)

def get_score(pfam_id, data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full',preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'):
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
	#try:
	# Load ER data file #
	er_data_file = preprocess_path+"%s_DP_laderdca.pickle"%(pfam_id)
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
	ref_seq_temp = list(er_ref_seq_s0)
	print(len(er_ref_seq))
	print('passed ref seq (%d): '%(len(er_ref_seq)),''.join(er_ref_seq))
	print('s0 ref seq (%d):'%(len(er_ref_seq_s0)),''.join(ref_seq_temp))
	print('s_index: ',s_index)
	print(len(s_index))
	er_rs_s0_full = []
	for indx in range(len(er_ref_seq)):
		if indx in s_index:
			#print('popping indx ',np.where(s_index==indx)[0][0])
			er_rs_s0_full.append(ref_seq_temp.pop(0)) 
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
	#except(FileNotFoundError):
#		print('Not all methods have a DP!!!')
#		sys.exit()
	#--------------------------------------------------------------------------------#

	# Print Specs
	print("#------- ER Specs -------#")
	print('s_ipdb: ',ER_s_ipdb)
	print('reference sequence: \n','s0[s_ipdb]   ',''.join(er_ref_seq_s0),'\n','s0[s_ipdb]   ',''.join(er_rs_s0_full),'\n','er_ref_seq   ',''.join(er_ref_seq)) 
	print('ref file: ',ER_ref_file)
	print('number positions: \n',er_n_var) 
	print('number sequences: \n',er_num_seqs) 
	print("#------------------------#")
	print("#------- DCA Specs ------#")
	print('s_ipdb: ',dca_s_ipdb)
	print('reference sequence: \n','s0[s_ipdb]   ',''.join(dca_ref_seq_s0),'\n','dca_ref_seq  ',''.join(dca_ref_seq)) 
	print('ref file: ',dca_ref_file)
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
		with open("DI/ER/erdca_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_erdca = pickle.load(f)
		f.close()
		with open("DI/ER/laderdca_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_laderdca = pickle.load(f)
		f.close()

		"""
		with open("DI/ER/er_clean_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_er_clean = pickle.load(f)
		f.close()
		with open("DI/ER/lader_clean_DI_%s.pickle"%(pfam_id),"rb") as f:
			DI_lader_clean = pickle.load(f)
		f.close()
		"""



	except(FileNotFoundError):
		print('Not all methods have a DI!!!')
		sys.exit()
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

	erdca_new_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = ER_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_erdca,
	linear_dist = 5,
	contact_dist = 8. )

	laderdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = ER_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_laderdca,
	linear_dist = 5,
	contact_dist = 8. )

	"""
	er_clean_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = ER_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_er_clean,
	linear_dist = 5,
	contact_dist = 8. )

	lader_clean_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = ER_ref_file, # eith _match or _range depending on how pre-processing went
	sorted_dca_scores = DI_lader_clean,
	linear_dist = 5,
	contact_dist = 8. )
	"""
	# Print Ranked DIs size by side A
	print('\n#-------- ER ---------------------- MF --------------------------- PLM ------------------------ ER new ----------------#')
	for i,pair in enumerate(DI_er[:25]):
		print(pair ,'  ',DI_mf[i],'  ',DI_plm[i], '  ',DI_erdca[i])	
	print('#------------------------------------------------------------------------------------------------------------------------#\n')
	#--------------------------------------------------------------------------------#


	plotting = True
	plotting = False



	#---------------------------------------------------------------------------------------------------------------------#
	# Get AUTPR using PYDCA plotting
	#---------------------------------------------------------------------------------------------------------------------#
	"""
	contact_instance_labels = ['lader_clean_visualizer','er_clean_visualizer','erdca_new_visualizer','erdca_visualizer', 'mfdca_visualizer', 'plmdca_visualizer']
	contact_instances = [lader_clean_visualizer,er_clean_visualizer, erdca_new_visualizer,erdca_visualizer, mfdca_visualizer, plmdca_visualizer]
	"""
	contact_instance_labels = ['laderdca_visualizer','erdca_new_visualizer','erdca_visualizer', 'mfdca_visualizer', 'plmdca_visualizer']
	contact_instances = [laderdca_visualizer,erdca_new_visualizer,erdca_visualizer, mfdca_visualizer, plmdca_visualizer]

	scores = []

	for i,visualizer in enumerate(contact_instances):
		print(contact_instance_labels[i])
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
			tp_rate_data = visualizer.plot_contact_map()
			plt.show()
			tp_rate_data = visualizer.plot_true_positive_rates()
			plt.show()
	#---------------------------------------------------------------------------------------------------------------------#





	#---------------------------------------------------------------------------------------------------------------------#
	# Get AUC using roc_curve from tools..
	#---------------------------------------------------------------------------------------------------------------------#            
	"""
	DIs = [DI_lader_clean, DI_er_clean, DI_erdca, DI_er, DI_mf, DI_plm] # SHOULD HAVE SAME ORDER AS contact_instances!!!!
	"""
	DIs = [DI_laderdca, DI_erdca, DI_er, DI_mf, DI_plm] # SHOULD HAVE SAME ORDER AS contact_instances!!!!
	AUCs = []

	#-------------------------- Get PP sequences coordinates ------------------------#            
	# Read in Reference Protein Structure
	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))                                                            
	# convert bytes to str (python 2 to python 3)                                                                       
	pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])      for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
	ipdb = 0

	# Load Polypeptide Sequence from PDB as reference sequence
	print('\n\n',pdb[ipdb,:],'\n\n')
	pdb_id = pdb[ipdb,5]                                                                              
	pdb_chain = pdb[ipdb,6]                                                                           
	pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])                                             
	pdb_range = [pdb_start-1, pdb_end]
	#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        
	pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              


	chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
	ppb = PPBuilder().build_peptides(chain)                                                       
	#    print(pp.get_sequence())
	print('peptide build of chain produced %d elements\n\n'%(len(ppb)))                               

	coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
	ca_residues = np.array([a.get_name()=='CA' for a in chain.get_atoms()])
	ca_coords = coords_all[ca_residues]

	poly_seq = list()
	for i,pp in enumerate(ppb):
		for char in str(pp.get_sequence()):
			poly_seq.append(char)                                     

	print('%d ca coordinates for %d polypeptide aa\'s '%(len(ca_coords),len(poly_seq)))
	if len(poly_seq) != len(ca_coords):
		print('\n\nCa-coords and polypeptides aa sequence DO NOT MATCH\n\n')



	ct_thres = np.linspace(1.5,10.,18,endpoint=True)
	n = ct_thres.shape[0]

	distance_mat = distance_matrix(ca_coords,ca_coords)
	#--------------------------------------------------------------------------------#            


	for i,DI in enumerate(DIs):
		di_matrix = np.zeros(distance_mat.shape)
		for coupling in DI:                                                                          
			di_matrix[coupling[0][0],coupling[0][1]] = coupling[1]                                               
			di_matrix[coupling[0][1],coupling[0][0]] = coupling[1]

		auc_ct_thresh = np.zeros(n)
		for ii,ct_threshold in enumerate(ct_thres):
			#ct = distance_mat[distance_mat < ct_threshold]
			p,tp,fp = tools.roc_curve(distance_mat , di_matrix ,ct_threshold)
			auc_ct_thresh[ii] = tp.sum()/tp.shape[0] 
		i0 = np.argmax(auc_ct_thresh)

		p,tp,fp = tools.roc_curve(distance_mat , di_matrix ,ct_thres[i0])
		AUCs.append(tp.sum()/tp.shape[0])
	#---------------------------------------------------------------------------------------------------------------------#            


		
	# Save scores and AUC
	"""
	print('Scores:\n%s %f %f %f %f %f %f %d\n\n' % (pfam_id , scores[0], scores[1], scores[2], scores[3],  scores[4], scores[5], dca_num_seqs))
	# write results of Pfam to txt file
	f = open('DI/%s.txt'%pfam_id,'w')
	f.write('%s %f %f %f %f %f %f %d' % (pfam_id , scores[0], scores[1], scores[2], scores[3], scores[4],  scores[5], dca_num_seqs) )    
	f.close()

	print('AUC: %s %f %f %f %f %f %f %d' % (pfam_id , AUCs[0], AUCs[1], AUCs[2], AUCs[3], AUCs[4], AUCs[5], dca_num_seqs) )
	# write results of Pfam to txt file
	f = open('DI/%s_auc.txt'%pfam_id,'w')
	f.write('%s %f %f %f %f %f %f %d' % (pfam_id , AUCs[0], AUCs[1], AUCs[2], AUCs[3], AUCs[3], AUCs[5], dca_num_seqs) )    
	f.close()
	"""
	# got rid of lader and er clean DIs just the og erdca.
	print('Scores:\n%s %f %f %f %f %f %d\n\n' % (pfam_id , scores[0], scores[1], scores[2], scores[3], scores[4], dca_num_seqs))
	# write results of Pfam to txt file
	f = open('DI/%s.txt'%pfam_id,'w')
	f.write('%s %f %f %f %f %f %d' % (pfam_id , scores[0], scores[1], scores[2], scores[3], scores[4], dca_num_seqs) )    
	f.close()

	print('AUC: %s %f %f %f %f %f %d' % (pfam_id , AUCs[0], AUCs[1], AUCs[2], AUCs[3],AUCs[4], dca_num_seqs) )
	# write results of Pfam to txt file
	f = open('DI/%s_auc.txt'%pfam_id,'w')
	f.write('%s %f %f %f %f %f %d' % (pfam_id , AUCs[0], AUCs[1], AUCs[2], AUCs[3],AUCs[4], dca_num_seqs) )    
	f.close()

def main():
	#-----------------------------------------------------------------------------------#
	#------------------------------- Create .txt file of scores ------------------------#
	#----------------------------------- Single DI Swarm Job ---------------------------#
	#-----------------------------------------------------------------------------------#

	# Full list of all PFams  being scored (passed to swarm file)
	# Must have DI for all 3 methods
	pfam_id = sys.argv[1]


	preprocess_path = '/home/eclay/DCA_ER/biowulf/pfam_ecc/'
	data_path = '/home/eclay/Pfam-A.full'
	preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'
	data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'


	get_score(pfam_id, data_path = data_path,preprocess_path=preprocess_path)


	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#

if __name__ == '__main__':
	main()


