import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'


#------------------------------- Create ROC Curves ---------------------------------#
def add_ROC(df,filepath,data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full',pfam_id_focus = None):
	
	TPRs = []
	TPRs_pdb = []
	scores = []
	Ps = []
	TPs = []
	FPs = []
	AUCs = []
	I0s =[]
	DIs = []
	ODs = []
	pfams =[]
	seq_lens = []
	num_seqs = []
	ERRs = []
	bad_index_ref_seqs = []
	for i,row in df.iterrows():
		pfam_id = row['Pfam'] 
		if pfam_id_focus is not None:
			if pfam_id != pfam_id_focus:
				print('Only want %s, %s is not it'%(pfam_id_focus,pfam_id))
				continue
		pfams.append(pfam_id)
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
		# Load processed data from smulation
		input_data_file = preprocess_path+"%s_DP_ER.pickle"%(pfam_id)
		with open(input_data_file,"rb") as f:
			pfam_dict = pickle.load(f)
		f.close()

		if filepath[:2] =="ER":
			s0 = pfam_dict['processed_msa']	
			processed_msa = pfam_dict['msa']
			s_index = pfam_dict['s_index']	
			s_ipdb = pfam_dict['s_ipdb']	 # this is actually seq num
			seq_row =s_ipdb 
			cols_removed = pfam_dict['cols_removed']
			#pp_ref_file = pfam_dict['ref_seq_file']
		else:
			s0 = pfam_dict['processed_msa']	
			s_ipdb = pfam_dict['s_ipdb']	 # this is actually seq num
			seq_row =s_ipdb 
			cols_removed = pfam_dict['cols_removed']
			#pp_ref_file = pfam_dict['ref_seq_file']

		# change s0 (processed_msa) from record list to character array		
		s = []
		for seq_record in s0:
			s.append([char for char in seq_record[1]])		
		s0 = np.array(s)


		n_var = s0.shape[1]
		seq_lens.append(n_var)
		num_seqs.append(s0.shape[0]) 
		#--------------------------------------------------------------------------------#



		df.loc[df.Pfam== pfam_id,'PDBid'] = pdb_id 
		print('\n\n\n\n\n\n#-----------------------------------------------------------------------#\nAnalysing  %s\n#-----------------------------------------------------------------------#\n'%pfam_id,df.loc[df.Pfam== pfam_id])
		#print('\n s_index',s_index,'\nlen(s_index)=%d\n'%len(s_index))
		#print('\n cols_removed',cols_removed,'\nlen(s_index)=%d\n'%len(cols_removed))
	

		print('Ref Sequence # should be : ',tpdb-1)


	
		# Load Contact Map
		try:
			#---------------------- Load DI -------------------------------------#
			#print("Unpickling DI pickle files for %s"%(pfam_id))
			if filepath[:3] =="cou":
				with open("DI/ER/er_couplings_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[:3] =="cov":
				with open("DI/ER/er_cov_couplings_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[:2] =="ER":
				with open("DI/ER/er_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[:3] =="PLM":
				with open("DI/PLM/plm_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[:2] == "MF":
				with open("DI/MF/mf_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			else:
				print("File Method-Prefix %s of %s  not recognized"%(filepath[:6],filepath))
				sys.exit()
			#--------------------------------------------------------------------#

			# there was an error!!! this is fixed for future version (i think)
			pp_ref_file = preprocess_path+'PP_ref_'+pfam_id+'_range.fa'

			from pydca.contact_visualizer import contact_visualizer
			erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
			refseq_file = pp_ref_file, # eith _match or _range depending on how pre-processing went
			sorted_dca_scores = DI,
			linear_dist = 5,
			contact_dist = 8. )




			contact_categories_dict = erdca_visualizer.contact_categories()
			print(contact_categories_dict.keys())
			tp = contact_categories_dict['tp']	
			fp = contact_categories_dict['fp']	
			p = contact_categories_dict['pdb']	
			missing = contact_categories_dict['missing']	

			
			#er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
			tp_rate_dict = er_tp_rate_data = erdca_visualizer.compute_true_positive_rates()
			score = np.trapz(tp_rate_dict['dca'],dx=1) / np.trapz(tp_rate_dict['pdb'],dx=1)
			print('Method Score: AU-TPR_method / AUTPR_pdb',score,'\n\n')


			TPs.append(tp)		
			FPs.append(fp)		
			Ps.append(p)		
			TPRs.append(tp_rate_dict['dca'])	
			TPRs_pdb.append(tp_rate_dict['pdb'])	
			scores.append(score)
			DIs.append(DI)
				
			ERRs.append('None')
			if pfam_id_focus is not None:
				#old version will have to be updated with tp-rate and score 
				df_temp = df.copy()
				df_temp = df_temp.loc[df['Pfam']==pfam_id_focus]
				df_temp = df_temp.assign(ERR = 'None')
				df_temp = df_temp.assign(P = [p0])
				df_temp = df_temp.assign(TP = [tp0])
				df_temp = df_temp.assign(FP = [fp0])
				df_temp = df_temp.assign(DI = [DI])
				df_temp = df_temp.assign(AUC = [auc[i0]])
				df_temp = df_temp.assign(OptiDist = [ct_thres[i0]])
				df_temp = df_temp.assign(seq_len = [n_var])
				df_temp = df_temp.assign(num_seq = [s0.shape[0]])
				return df_temp.copy()


		except  FileNotFoundError as e:
			print("ERROR!!\n Prediction: PDB File not found by BioPython")
			print(str(e))
			#<Example file>'/data/cresswellclayec/DCA_ER/biowulf/uz/pdb5uz5.ent'
			if '.ent' in str(e):
				ERRs.append('No PDB')
			#<Example file>'DI/ER/er_DI_PF17859.pickle'
			elif '.pickle' in str(e):
				ERRs.append('No DI')
			else:
				ERRs.append(str(e))
			print("\n\nAdding empty row\n\n")
			Ps.append([])	
			TPs.append([])	
			FPs.append([])	
			AUCs.append(-1)	
			DIs.append([])
			ODs.append(-1)
			pass
		except  ValueError as e:
			print("ERROR!!\n Prediction:max() arg is an empty sequence")
			print("\n\nAdding empty row\n\n")
			print(str(e))
			ERRs.append('ValueErr')
			Ps.append([])	
			TPs.append([])	
			FPs.append([])	
			AUCs.append(-1)	
			DIs.append([])
			ODs.append(-1)
		except IndexError as e:
			print("\n\n#--------------------ERROR------------------------#\n Indexing error, check DI or PDB for %s"%(pfam_id))
			print('MSA subject: ',ref_seq)
			print('MSA subject length: ',len(ref_seq))
			print('Max s_index value: ',max(s_index))
			print('length of s_index: ', len(s_index))
			print("\n\nAdding empty row\n\n")
			print(str(e))
			bad_index_ref_seqs.append(ref_seq)
			print("\n\n#-------------------------------------------------#\n")
			pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
			if len(ref_seq) != len(poly_seq_curated) and pdb_end-pdb_start-1 <= max(max(s_index),max(cols_removed)):
				print('pdb start (%d) and end (%d) coincide with length of unprocessed erdca ref_seq (%d)\n POTENTIAL ISSUE WITH POLY SEQ READIN'%(pdb_start,pdb_end,max(max(s_index),max(cols_removed))))
				ERRs.append('PDB-MSA')
			else:
				ERRs.append('Indexing_CT')
			Ps.append([])	
			TPs.append([])	
			FPs.append([])	
			AUCs.append(-1)	
			DIs.append([])
			ODs.append(-1)
		

		if 0:
			try:
				ct,ct_full,n_amino_full,poly_seq = tools.contact_map(pdb,ipdb,cols_removed,s_index)	
				poly_seq_curated = np.delete(poly_seq,cols_removed)
				print(pfam_id, ': Full PP sequence\n',poly_seq,'\nlen %d\n'%len(poly_seq))
				print('		Curated PP sequence\n',poly_seq_curated,'\nlen %d\n'%len(poly_seq_curated))
				ref_seq = [char for char in processed_msa[s_ipdb][1]]
				print('		curated ref sequence\n',ref_seq,'\nlen %d\n'%len(ref_seq))
				print('contact map shape: ',ct.shape)
				print('contact map (full) shape: ',ct_full.shape)
				#ct_distal = tools.distance_restr_ct(ct,s_index,make_large=True)
				#ct_distal = tools.distance_restr_ct(ct_full,s_index,make_large=True)
				#DI already sorted
				#DI_dup = dp.delete_sorted_DI_duplicates(DI)	
				#print('Deleted DI index duplicates')
				#sorted_DI = tools.distance_restr_sortedDI(DI_dup,s_index)
				# sorted_DI has no duplicates, incorporates s_index and distance restraint
				#print('Distance restraint enforced')
				#--------------------------------------------------------------------#
				
				#--------------------- Generate DI Matrix ---------------------------#
				di = np.zeros((n_amino_full,n_amino_full))
				for coupling in DI:
					#print(coupling[1])
					di[coupling[0][0],coupling[0][1]] = coupling[1]
					di[coupling[0][1],coupling[0][0]] = coupling[1]
				#--------------------------------------------------------------------#

				print('\n\n #------------------ Calculating ROC Curve --------------------#')
			
				#----------------- Generate Optimal ROC Curve -----------------------#
				# find optimal threshold of distance for both DCA and ER
				ct_thres = np.linspace(1.5,10.,18,endpoint=True)
				n = ct_thres.shape[0]
				
				auc = np.zeros(n)


				# Generate DI matrix of our predictions only	
				di_predict = np.zeros((len(s_index),len(s_index)))
				for coupling in DI:
					#print(coupling[1])
					if coupling[0][0] in s_index and coupling[0][1] in s_index:
						di_predict[np.where(s_index==coupling[0][0]),np.where(s_index==coupling[0][1])] = coupling[1]
						di_predict[np.where(s_index==coupling[0][1]),np.where(s_index==coupling[0][0])] = coupling[1]


				print('before distance restr on contact map predicitons\ncontact map shape: ',ct.shape)
				# Want to only ROC-analyze positions we made predictions for !!!
				ct_predict = np.asarray(tools.distance_restr_ct(ct,s_index,make_large = True))
				print('Prediction Contact map shape (distance enforced): ',ct_predict.shape)
				
				for i in range(n):
					#p,tp,fp = tools.roc_curve(ct_distal,di,ct_thres[i])
					p,tp,fp = tools.roc_curve(ct_predict,di_predict,ct_thres[i])
					auc[i] = tp.sum()/tp.shape[0]
				i0 = np.argmax(auc)
				print('Optimal Distance: ',ct_thres[i0])

				
				# set true positivies, false positives and predictions for optimal distance
				p0,tp0,fp0 = tools.roc_curve(ct_predict,di_predict,ct_thres[i0])
				print(tp0)
				print(fp0)
				
				
				Ps.append(p0)	
				TPs.append(tp0)	
				FPs.append(fp0)	
				AUCs.append(auc[i0])	
				DIs.append(DI)
				ODs.append(ct_thres[i0])
				ERRs.append('None')
				if pfam_id_focus is not None:
					df_temp = df.copy()
					df_temp = df_temp.loc[df['Pfam']==pfam_id_focus]
					df_temp = df_temp.assign(ERR = 'None')
					df_temp = df_temp.assign(P = [p0])
					df_temp = df_temp.assign(TP = [tp0])
					df_temp = df_temp.assign(FP = [fp0])
					df_temp = df_temp.assign(DI = [DI])
					df_temp = df_temp.assign(AUC = [auc[i0]])
					df_temp = df_temp.assign(OptiDist = [ct_thres[i0]])
					df_temp = df_temp.assign(seq_len = [n_var])
					df_temp = df_temp.assign(num_seq = [s0.shape[0]])
					return df_temp.copy()

				# Fill Data Frame with relavent info
				#print("adding TP =",tp0)
				#df.loc[i,'TP'] = tp0.tolist()
				#print("df[i, TP] = ",df['TP'])
				#df.loc[i,'FP'] = fp0.tolist()
				#df.loc[i,'AUC'] = np.argmax(auc)
				#df.loc[i,'DI'] = sorted_DI 
				#df.loc[i,'OptiDist'] = ct_thres[i]  
				#print("here")
				print("Done\n\n\n")
			except  FileNotFoundError as e:
				print("ERROR!!\n Prediction: PDB File not found by BioPython")
				print(str(e))
				#<Example file>'/data/cresswellclayec/DCA_ER/biowulf/uz/pdb5uz5.ent'
				if '.ent' in str(e):
					ERRs.append('No PDB')
				#<Example file>'DI/ER/er_DI_PF17859.pickle'
				elif '.pickle' in str(e):
					ERRs.append('No DI')
				else:
					ERRs.append(str(e))
				print("\n\nAdding empty row\n\n")
				Ps.append([])	
				TPs.append([])	
				FPs.append([])	
				AUCs.append(-1)	
				DIs.append([])
				ODs.append(-1)
				pass
			except  ValueError as e:
				print("ERROR!!\n Prediction:max() arg is an empty sequence")
				print("\n\nAdding empty row\n\n")
				print(str(e))
				ERRs.append('ValueErr')
				Ps.append([])	
				TPs.append([])	
				FPs.append([])	
				AUCs.append(-1)	
				DIs.append([])
				ODs.append(-1)
			except IndexError as e:
				print("\n\n#--------------------ERROR------------------------#\n Indexing error, check DI or PDB for %s"%(pfam_id))
				print('MSA subject: ',ref_seq)
				print('MSA subject length: ',len(ref_seq))
				print('Max s_index value: ',max(s_index))
				print('length of s_index: ', len(s_index))
				print("\n\nAdding empty row\n\n")
				print(str(e))
				bad_index_ref_seqs.append(ref_seq)
				print("\n\n#-------------------------------------------------#\n")
				pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
				if len(ref_seq) != len(poly_seq_curated) and pdb_end-pdb_start-1 <= max(max(s_index),max(cols_removed)):
					print('pdb start (%d) and end (%d) coincide with length of unprocessed erdca ref_seq (%d)\n POTENTIAL ISSUE WITH POLY SEQ READIN'%(pdb_start,pdb_end,max(max(s_index),max(cols_removed))))
					ERRs.append('PDB-MSA')
				else:
					ERRs.append('Indexing_CT')
				Ps.append([])	
				TPs.append([])	
				FPs.append([])	
				AUCs.append(-1)	
				DIs.append([])
				ODs.append(-1)
			

	#print("df: ",len(df))
	#print("P vec: ",len(Ps))
	if pfam_id_focus is None:

		df = df.assign(ERR = ERRs)
		df = df.assign(P = Ps)
		df = df.assign(TP = TPs)
		df = df.assign(FP = FPs)
		df = df.assign(DI = DIs)
		#df = df.assign(AUC = AUCs)
		#df = df.assign(OptiDist = ODs)
		df = df.assign(seq_len = seq_lens)
		df = df.assign(num_seq = num_seqs)
		df = df.assign(ERR = ERRs)

		df = df.assign(TPR_Method = TPRs)
		df = df.assign(TPR_PDB = TPRs_pdb)
		df = df.assign(Score = scores)

		for bad_seq in bad_index_ref_seqs:
			print(bad_seq)
		# Print duplicate Pfams rows
		print("Duplicates:")
		print(df[df.duplicated(['Pfam'])])
		#print(df)
		return df.copy()
               
	
#-----------------------------------------------------------------------------------#

def main():
	#-----------------------------------------------------------------------------------#
	#------------------------------- Create Pandas DataFrame ---------------------------#
	#----------------------------------- Single DI Swarm Job ---------------------------#
	#-----------------------------------------------------------------------------------#
	# Jobload info from text file 
	prep_df_file = sys.argv[1]
	job_id = sys.argv[2]

	# Get dataframe of job_id
	#df_prep = pd.load(open(prep_df_file,"rb"))
	df_prep = pd.read_pickle(prep_df_file)
	print(df_prep)
	df_jobID = df_prep.copy()
	df_jobID = df_jobID.loc[df_jobID.Jobid == job_id]
	print(df_jobID)

	roc_jobID_df = add_ROC(df_jobID,prep_df_file)


	print(roc_jobID_df)
	print(roc_jobID_df['ERR'])
	if not os.path.exists('./job_ROC_dfs/'): 
		print('job_ROC_dfs/ DNE.. Make directory and rerun')
		sys.exit()

	df_jobID_filename = 'job_ROC_dfs/%s.pkl'%job_id
	print ('saving file: ' + df_jobID_filename)

	roc_jobID_df.to_pickle(df_jobID_filename)


	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#
	#-----------------------------------------------------------------------------------#

if __name__ == '__main__':
	main()

