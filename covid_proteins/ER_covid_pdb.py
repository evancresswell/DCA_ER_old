import os,sys

sys.path.append(os.path.dirname('/data/cresswellclayec/DCA_ER/biowulf/'))
sys.path.append(os.path.dirname('/home/eclay/DCA_ER/biowulf/'))
from Bio import SeqIO
from Bio.PDB import *
import pickle
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import ecc_tools as tools
from glob import glob
import data_processing as dp
import inspect

#========================================================================================
# Loop Throught covid proteins
#========================================================================================
if 0:
	# Check format of s0 
	data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
	pfam_id ='PF00186'
	#s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
	#for line in s:
	#	print(line)

	pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
	ipdb = 0

	#a convert bytes to str (python 2 to python 3)
	pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
	     for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
	print(pdb[0,:])
	#print(pdb)
	tpdb = int(pdb[ipdb,1])
	#tpdb is the sequence #
	#print(tpdb)
	#sys.exit()

	# Generate MSA numpy array
	root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins'
	dir_list = glob(root_dir+"/*/")
	#print(dir_list)
	string ="/data/cresswellclayec/DCA_ER/covid_proteins/QHD"
	covid_pdb_list = []

parser = PDBParser()
pdb_files = glob('*.pdb')
pdb_structures = [pdb_name[:-4] for pdb_name in pdb_files]
#print(pdb_structures)

# Dictionary of pdb residue ranges (from https://zhanglab.ccmb.med.umich.edu/COVID-19/ ):
unknown_pdb_ranges = ['QHD43420', 'QHD43415_2', 'QHD43417','QHD43415_13', 'QHD43415_12', 'QHD43418', 'QHD43415_5', 'QHD43421','QHD43422', 'QHD43419','QHD43415_6', 'QHD43415_1']

pdb_ranges = { 'QHD43423':[('6M3M',50,174)], 'QHD43415_8':[('6M71',84,132)], 'QHD43415_7':[('6M71',1,83)], 'QHD43415_3':[('6W6Y',207,379),('6W9C',748,1060)],  'QHD43415_4':[('6LU7',1,306)],  'QHD43415_14':[('6VWW',1,346)], 'QHD43415_9':[('6W4B',1,113)],   'QHD43415_15':[('6W75',1,298)], 'QHD43415_11':[('6M71',1,932)], 'QHD43415_10':[('6W75',1, 139)], 'QHD43416':[('6VYB' ,1,1273),('6VXX',1,1273)]}#('6LXT',912,988,1164,1202)]} #DYNAMIC RANGE IS NOT INCORPRATED>>> NOT PLOTTING 3RD TYPE


#for i,pdb_file in enumerate( pdb_files):
	#pfam_id = pdb_structures[i] 
for i,pdb_key in enumerate( pdb_ranges.keys()):
	pfam_id = pdb_key
	print('\n\nGenerating MSA array for ',pfam_id)
	# only work on QHD directoies

	# Desired Format
	#	['PF00186' '69' 'Q5KZ26_GEOKA' '1'                           '160'                          '1ZDR'                   'B' '1'                            '160'                        ]
	pdb =   [ [pfam_id , '0', pfam_id       , pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]  ,pdb_ranges[pfam_id][a][0]  ,'A', pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]] for a in range(0,len(pdb_ranges[pfam_id])) ]
	print(pdb)

	print('\n')

	try:
		print("		Unpickling DI pickle files for %s"%(pfam_id))
		file_obj = open("%s/DI.pickle"%(pfam_id),"rb")
		DI_er = pickle.load(file_obj)
		file_obj.close()
	except(FileNotFoundError):
		print("		No DI file found on to the next one")
		continue
		
	# Remove bad distances and Duplicates
	DI_er_dup = dp.delete_sorted_DI_duplicates(DI_er)	
	sorted_DI_er = tools.distance_restr_sortedDI(DI_er_dup)
	#print(sorted_DI_er[0:10])
	#print(cols_removed)

	file_obj =  open('%s/DP.pickle'%(pfam_id), 'rb') 
	DP = pickle.load(file_obj)
	file_obj.close()
	s0 = DP['s0']
	s_index = DP['s_index']
	cols_removed = DP['cols_removed']	




	ipdb = 0

	"""	
	# Plot Contact Map
	# Code from : ecc_tools.contact_map(pdb,ipdb,cols_removed,s_index)
	if 0:
		# Get coords from pdb structure
		try:
			# Create pdb_refs.npy	
			#print( pfam_dir[:-1]+'.pdb')
			structure = parser.get_structure(id =pfam_id, file = pdb_file)
			for j,model in enumerate(structure):
				if j>0:
					print('Too many models')
					sys.exit()
				for i,chain in enumerate(model):
					if i>0:
						print('Too many chains')
						sys.exit()
					chain_id = chain.get_id()
		except(FileNotFoundError):
			print('No PDB structure')


		pdb_id = pdb[ipdb,5]
		pdb_chain = pdb[ipdb,6]
		pdb_start,pdb_end = int(pdb[ipdb][7]),int(pdb[ipdb][8])

		attrs = vars(structure)
		chain = structure[0][chain_id]
		coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
		coords = coords_all#[pdb_start-1:pdb_end]
		#print(coords)
		#print('original pdb:')
		#print(coords_all.shape)
		#print(coords.shape)
		#print(s_index.shape)



		coords_remain = np.delete(coords,cols_removed,axis=0)
		print('		Residue Position Coords:' ,coords_remain.shape)
		#print(coords_remain.shape)

		ct = distance_matrix(coords_remain,coords_remain)
		ct_distal = tools.distance_restr(ct,s_index,make_large=True)
	"""
	# Actual Call of contact_map with hand-entered pdb- ref dictionary
	# Plot Contact Map

	pdb_start,pdb_end = int(pdb[ipdb][7]),int(pdb[ipdb][8])
	n_var = s0.shape[0]
	print("s0[0]: ", s0[0])
	print( s0.shape[0])
	print("s_sindex: ", s_index.shape[0])
	print( len(s_index))

	ct = tools.contact_map(np.asarray(pdb),ipdb,cols_removed,s_index)
	#ct_distal = tools.distance_restr(ct,s_index,make_large=True)
	try:
		n_seq = max([coupling[0][0] for coupling in sorted_DI_er]) 
		di_er = np.zeros((n_var,n_var))
		for coupling in sorted_DI_er:
			#print(coupling[1])
			di_er[coupling[0][0],coupling[0][1]] = coupling[1]
			di_er[coupling[0][1],coupling[0][0]] = coupling[1]
		ct_distal = tools.distance_restr(di_er,s_index,make_large=True)
	except(IndexError):
		print('given range %d - %d does not match coupling indices (bad experimental pdb range most likely)'%(pdb_start,pdb_end))
		continue
		
	#----------------- Generate Optimal ROC Curve -----------------------#
	# find optimal threshold of distance for both DCA and ER
	ct_thres = np.linspace(1.5,10.,18,endpoint=True)
	n = ct_thres.shape[0]

	auc_er = np.zeros(n)

	for i in range(n):
		p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
		auc_er[i] = tp.sum()/tp.shape[0]
	i0_er = np.argmax(auc_er)

	p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])

	#plt.title('Contact Map')
	plt.imshow(ct_distal,cmap='rainbow_r',origin='lower')
	plt.xlabel('i')
	plt.ylabel('j')
	plt.title(pfam_id)
	plt.colorbar(fraction=0.045, pad=0.05)
	#plt.show()
	plt.savefig('%s_contact_map.pdf'%pfam_id, format='pdf', dpi=100)	
	plt.close()
	#--------------------------------------------------------------------#

	plt.subplot2grid((1,2),(0,0))
	plt.title('ROC ')
	plt.plot(fp0_er,tp0_er,'b-',label="er")
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()

	# Plot AUC for DCA and ER
	plt.subplot2grid((1,2),(0,1))
	plt.title('AUC')
	plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
	plt.plot(ct_thres,auc_er,'b-',label="er")
	print(auc_er)
	plt.ylim([auc_er.min()-0.05,auc_er.max()+0.05])
	plt.xlim([ct_thres.min(),ct_thres.max()])
	plt.xlabel('distance threshold')
	plt.ylabel('AUC')
	plt.legend()
	plt.savefig('%s_ROC.pdf'%pfam_id, format='pdf', dpi=100)	
	#plt.show()
	plt.close()




	print('\n\n')
	break
