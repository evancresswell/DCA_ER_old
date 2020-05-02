import os,sys
import datetime
import numpy as np
on_pc = False
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

from direct_info import sort_di

import ecc_tools as tools
import data_processing as dp
import Bio.PDB
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()

# import inference_dca for mfDCA
from inference_dca import direct_info_dca

data_path = '../../Pfam-A.full'
data_path = '../../hoangd2_data/Pfam-A.full'

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'


pfam_id = sys.argv[1]

print ('Calculate Variance of Residue Pairs in Protein Famility ', pfam_id)
# Load PDB structure 
#------------------ Load PDB -----------------------------------#
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

#---------- Pre-Process Structure Data ----------------#
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
#---------------------------------------------------------------#

#------------------ Load MSA -----------------------------------#
s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
print("shape of s (import from msa.npy):\n",s.shape)

# convert bytes to str
try:
	s = np.array([s[t,i].decode('UTF-8') for t in range(s.shape[0]) \
	     for i in range(s.shape[1])]).reshape(s.shape[0],s.shape[1])
	if printing:
	    print("shape of s (after UTF-8 decode):\n",s.shape)
except:
	print("UTF not decoded, pfam_id: %s \n "%pfam_id,s.shape)
	# Create list file for missing pdb structures
	if not os.path.exists('missing_MSA.txt'):
	    file_missing_msa = open("missing_MSA.txt",'w')
	    file_missing_msa.write("%s\n"% pfam_id)
	    file_missing_msa.close()
	else:
	    file_missing_msa = open("missing_MSA.txt",'a')
	    file_missing_msa.write("%s\n"% pfam_id)
	    file_missing_msa.close()
#---------------------------------------------------------------#
print(s)
sys.exit()

input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
#------------------------------------------------------#

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
#--------------------------------------------------------------------#

#---------------------- Load Pre-Processing Infor ----------------#
input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
f.close()
#s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
#				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
s0 = pfam_dict['s0']	
s_index = pfam_dict['s_index']	
s_ipdb = pfam_dict['s_ipdb']	
cols_removed = pfam_dict['cols_removed']
#-----------------------------------------------------------------#


#----------------- Load DI Dictionary ----------------------------#
DIs = {}
DI_er_dup = dp.delete_sorted_DI_duplicates(DI_er)	
DIs["ER"] = tools.distance_restr_sortedDI(DI_er_dup)
#di_matrix = gen_DI_matrix(sorted_DI)

DI_mf_dup = dp.delete_sorted_DI_duplicates(DI_mf)	
DIs["MF"] = tools.distance_restr_sortedDI(DI_mf_dup)
#di_matrix = gen_DI_matrix(sorted_DI)

DI_plm_dup = dp.delete_sorted_DI_duplicates(DI_plm)	
DIs["PLM"] = tools.distance_restr_sortedDI(DI_plm_dup)
#di_matrix = gen_DI_matrix(sorted_DI)
#-----------------------------------------------------------------#


#--------------------------------------------------------------------#
# create pfam_pair_distances list to store distances per ordered pair across all pdb entries in this Pfam
pfam_pair_distances = {"MF":[],"PLM":[], "ER":[]}
for method in DIs.keys():
	for tuple_row in DIs[method]:	
		pfam_pair_distances[method].append((tuple_row[0],tuple_row[1],[]))

print("s_index: ",s_index)
bad_pdb = []
good_pdb = []
for ipdb in range(len(pdb)):
	pdb_id = pdb[ipdb,5]
	#print(pdb_id)
	pdb_chain = pdb[ipdb,6]
	#print("PDB_chain: ",pdb_chain)
	pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
	print("PDB start and end (%d, %d) => sequence len: %d "%(pdb_start,pdb_end,pdb_end-pdb_start))
	pdb_file = pdb_list.retrieve_pdb_file(pdb_id,file_format='pdb')
	chain = pdb_parser.get_structure(pdb_id,pdb_file)[0][pdb_chain]
	coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
	coords = coords_all[pdb_start-1:pdb_end]
	coords_remain = np.delete(coords,cols_removed,axis=0)
	#print("Shape of total coords : ",coords.shape)
	#print("Shape of coords w/ cols rem: ", coords_remain.shape)
	#print("s_index length = ",len(s_index))
	if(len(s_index) != coords_remain.shape[0]):
		bad_pdb.append(pdb_id)
		print('.. is bad .. ')
	else:
		good_pdb.append(pdb_id)
		for method in pfam_pair_distances.keys():
			print("Method: ",method)
			for i,residue_pair in enumerate(pfam_pair_distances[method]):
				if(i<10):
					print(residue_pair[0]," ",residue_pair[1])
				coord1 = coords[s_index[residue_pair[0][0]]]
				coord2 = coords[s_index[residue_pair[0][1]]]
				pair_distance = np.sqrt( (coord1[0] - coord2[0])**2. + (coord1[1] - coord2[1])**2. + (coord1[2] - coord2[2])**2. )
				residue_pair[2].append(pair_distance)
			print("%s top-ranked distances: "%(method),pfam_pair_distances[method][:,3])

print('bad pdbs: ', bad_pdb)
print('good pdbs: ', good_pdb)

with open('DI/%s_pair_distances_%s.pickle'%(method,pfam_id), 'wb') as f:
    pickle.dump(pfam_pair_distances, f)
f.close()

fig_mean, ax_mean = plt.subplots()
fig_var, ax_var = plt.subplots()
plotting = True
if plotting:
	for method in pfam_pair_distances.keys():
		avg_distances = []
		var_distances = []
		for residue_pair in pfam_pair_distances[method][:50]:
			avg_distances.append(np.mean(residue_pair[2]))
			var_distances.append(np.var(residue_pair[2]))
		ax_mean.errorbar(range(len(avg_distances)),avg_distances,var_distances,label=method)
		ax_var.plot(range(len(var_distances)),var_distances,label=method)
	plt.legend()	
	plt.show()
	

