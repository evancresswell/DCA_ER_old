import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool
#------------------------------- Create ROC Curves ---------------------------------#
def add_ROC(df):
	data_path = '../../../hoangd2_data/Pfam-A.full'

	Ps = []
	TPs = []
	FPs = []
	AUCs = []
	I0s =[]
	DIs = []
	ODs = []
	for i,row in df.iterrows():
		pfam_id = row['Pfam'] 

		# Load or generate structural information
			# data processing THESE SHOULD BE CREATED DURING DATA GENERATION
		pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
		#---------- Pre-Process Structure Data ----------------#
		# delete 'b' in front of letters (python 2 --> python 3)
		pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
		for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

		# ONLY CONSIDERING FIRST PDB-ID IN PFAM (FOR NOW)
		ipdb = 0

		input_data_file = "../pfam_ecc/%s_DP.pickle"%(pfam_id)
		with open(input_data_file,"rb") as f:
			pfam_dict = pickle.load(f)
		f.close()
		s0 = pfam_dict['s0']	
		s_index = pfam_dict['s_index']	
		s_ipdb = pfam_dict['s_ipdb']	
		cols_removed = pfam_dict['cols_removed']
		n_var = s0.shape[1]
   		 
		pdb_id = pdb[ipdb,5]
		df.loc[df.Pfam== pfam_id,'PDBid'] = pdb_id 
		
		# Load Contact Map
		try:
			ct = tools.contact_map(pdb,ipdb,cols_removed,s_index)
			ct_distal = tools.distance_restr(ct,s_index,make_large=True)
			#---------------------- Load DI -------------------------------------#
			#print("Unpickling DI pickle files for %s"%(pfam_id))
			if filepath[3:5] =="ER":
				with open("../DI/ER/er_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[3:6] =="PLM":
				with open("../DI/PLM/plm_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			elif filepath[3:5] == "MF":
				with open("../DI/MF/mf_DI_%s.pickle"%(pfam_id),"rb") as f:
					DI = pickle.load(f)
				f.close()
			else:
				print("File Method-Prefix %s not recognized"%(filepath[3:6]))
				sys.exit()

			DI_dup = dp.delete_sorted_DI_duplicates(DI)	
			sorted_DI = tools.distance_restr_sortedDI(DI_dup)
			#--------------------------------------------------------------------#
			
			#--------------------- Generate DI Matrix ---------------------------#
			n_seq = max([coupling[0][0] for coupling in sorted_DI]) 
			di = np.zeros((n_var,n_var))
			for coupling in sorted_DI:
				#print(coupling[1])
				di[coupling[0][0],coupling[0][1]] = coupling[1]
				di[coupling[0][1],coupling[0][0]] = coupling[1]
			#--------------------------------------------------------------------#
			
			#----------------- Generate Optimal ROC Curve -----------------------#
			# find optimal threshold of distance for both DCA and ER
			ct_thres = np.linspace(1.5,10.,18,endpoint=True)
			n = ct_thres.shape[0]
			
			auc = np.zeros(n)
			
			for i in range(n):
				p,tp,fp = tools.roc_curve(ct_distal,di,ct_thres[i])
				auc[i] = tp.sum()/tp.shape[0]
			i0 = np.argmax(auc)
			
			# set true positivies, false positives and predictions for optimal distance
			p0,tp0,fp0 = tools.roc_curve(ct_distal,di,ct_thres[i0])
			
			
			Ps.append(p0)	
			TPs.append(tp0)	
			FPs.append(fp0)	
			AUCs.append(auc[i0])	
			DIs.append(sorted_DI)
			ODs.append(ct_thres[i0])
			# Fill Data Frame with relavent info
			#print("adding TP =",tp0)
			#df.loc[i,'TP'] = tp0.tolist()
			#print("df[i, TP] = ",df['TP'])
			#df.loc[i,'FP'] = fp0.tolist()
			#df.loc[i,'AUC'] = np.argmax(auc)
			#df.loc[i,'DI'] = sorted_DI 
			#df.loc[i,'OptiDist'] = ct_thres[i]  
			#print("here")
		except  FileNotFoundError as e:
			print("ERROR!!\n Prediction: PDB File not found by BioPython")
			print(str(e))
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
			print(str(e))
			Ps.append([])	
			TPs.append([])	
			FPs.append([])	
			AUCs.append(-1)	
			DIs.append([])
			ODs.append(-1)
			print("\n\nAdding empty row\n\n")
			

	#print("df: ",len(df))
	#print("P vec: ",len(Ps))
	df = df.assign(P = Ps)
	df = df.assign(TP = TPs)
	df = df.assign(FP = FPs)
	df = df.assign(DI = DIs)
	df = df.assign(AUC = AUCs)
	df = df.assign(OptiDist = ODs)
	#print(df)
	return df.copy()
#-----------------------------------------------------------------------------------#

#-------------------------- Parallelize Data Frame Generation ----------------------#
def parallelize_dataframe(df, func, n_cores=60):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split),sort=False)
    pool.close()
    pool.join()
    return df
#-----------------------------------------------------------------------------------#




#-----------------------------------------------------------------------------------#
#------------------------------- Create Pandas DataFrame ---------------------------#
#-----------------------------------------------------------------------------------#

# Jobload info from text file 
filepath = sys.argv[1]
#file_out = sys.argv[2]

#if os.path.exists(filepath[:-4]+'.pkl'):
#	df = pd.read_pickle(filepath[:-4]+'.pkl')
#	print(df)
#else:
# Read in DataFrame
df = pd.read_csv(filepath, sep='\s+', engine='python',header=0)
#print(df)

# Polish MemReq Column
df.MemReq = df.MemReq.str.extract(r'(\d+\.+\d+)(GB/node)' ,expand=False) 
df['MemReq'] =pd.to_numeric(df['MemReq'],downcast='float')
df = df.rename(columns={'MemReq': 'GB/node'})

# Polish MemReq Column
df.MemUsed = df.MemUsed.str.extract(r'(\d+\.+\d+)(GB)' ,expand=False) 
df['MemUsed'] =pd.to_numeric(df['MemUsed'],downcast='float')

#print(df)

# Iterate Through Jobs and add to DataFrame
jobs = df.Jobid
for i,job_id in enumerate(jobs):
	#print(job_id)
	with open("swarm_%s.o"%(job_id)) as f:
		families = re.findall(r'PF+\d+',f.read())
	df = df.append([df.loc[df['Jobid']==job_id]]*(len(families)-1), ignore_index=True)	
	df.loc[df.Jobid == job_id,'Pfam'] = families

print(df)
#print("Generating ROC curves for %d Pfams"%(len(df)))

# Genreate ROC / AUC / Precisoin / DI Dataframe
df_roc = parallelize_dataframe(df, add_ROC)
#print(df_roc)
roc_filename = filepath[:-4]+'_roc.pkl' 
df_roc.to_pickle(roc_filename)
print ('saving file: ' + roc_filename)


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


