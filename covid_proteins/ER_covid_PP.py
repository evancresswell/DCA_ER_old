import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC

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

#========================================================================================
on_pc = True 

#========================================================================================
# Loop Throught covid proteins
#========================================================================================

# Generate MSA numpy array
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins'
if on_pc:
	root_dir = '/home/eclay/DCA_ER/covid_proteins'
dir_list = glob(root_dir+"/*/")
#print(dir_list)
string ="/data/cresswellclayec/DCA_ER/covid_proteins/QHD"
if on_pc:
	string ="/home/eclay/DCA_ER/covid_proteins/QHD"
covid_protein_list = []

parser = PDBParser()

pdb_ranges = { 'QHD43423':[('6M3M',50,174)], 'QHD43415_8':[('6M71',84,132)], 'QHD43415_7':[('6M71',1,83)], 'QHD43415_3':[('6W6Y',207,379),('6W9C',748,1060)],  'QHD43415_4':[('6LU7',1,306)],  'QHD43415_14':[('6VWW',1,346)], 'QHD43415_9':[('6W4B',1,113)],   'QHD43415_15':[('6W75',1,298)], 'QHD43415_11':[('6M71',1,932)], 'QHD43415_10':[('6W75',1, 139)], 'QHD43416':[('6VYB' ,1,1273),('6VXX',1,1273)]}#('6LXT',912,988,1164,1202)]} #DYNAMIC RANGE IS NOT INCORPRATED>>> NOT PLOTTING 3RD TYPE

generating_msa = False
if generating_msa:
	for pfam_dir in dir_list:
		pfam_id = pfam_dir[44:-1]
		print('Generating MSA array for ',pfam_id)
		# only work on QHD directoies
		if pfam_id[:2] != 'QH':
			continue

		# Load MSA file
		try:
			#print("		Unpickling DI pickle files for %s"%(pfam_id))
			#file_obj = open("%s/DI.pickle"%(pfam_id),"rb")
			msa_dir = pfam_dir+"/MSA/"
			with open(msa_dir+"protein.aln", 'r') as infile:
				MSA = infile.readlines()
				#print('\n\n',MSA,'\n\n')
			#try:
				#msa = np.chararray((len(MSA),len(MSA[1])-1))
			#except(IndexError):
				#print('MSA doesent have more than 1 row')
				#continue
			msa = []
			for i,line in enumerate(MSA):
				#print(line)
				msa.append(list(line)[:-1])
				print(list(line)[:-1])
				#print(msa[i,:])
			np.save(pfam_id+'/msa.npy',np.asarray(msa))
			#print(type(msa))
			#print(type(msa[0]))
			print('msa: ',msa[0])
			covid_protein_list.append(pfam_id)	
		except(FileNotFoundError):
			print('No MSA file')
			continue


		# MUSCLE MSA with polypeptide sequence from pdb structure ( if it exists)
		try:

			# Define pdb data (from hardcoded pdb_ranges)
			pdb =   [ [pfam_id , '0', pfam_id       , pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]  ,pdb_ranges[pfam_id][a][0]  ,'A', pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]] for a in range(0,len(pdb_ranges[pfam_id])) ]

			# Set BioPython parameters to begin extracting PP sequence
			pdb_id = pdb[ipdb,5]                                                                              
			pdb_chain = pdb[ipdb,6]                                                                           
			pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])                                             

			pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
			chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
			ppb = PPBuilder().build_peptides(chain)                                                       

			pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, msa, pfam_id, number_form=False,processed=False,path=msa_dir)



			muscle_msa_file = preprocess_path+'PP_muscle_msa_'+pfam_id+'.fa'
			if os.path.exists(muscle_msa_file):    
				print('Using existing muscled FASTA files\n\n')
			else:
				# Add using muscle:
				#	- https://www.drive5.com/muscle/manual/addtomsa.html
				#	- https://www.drive5.com/muscle/downloads.htmL
				os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
				print("PP sequence added to alignment via MUSCLE")
		except(KeyError):
			pdb_ranges[pfam_id] = [('SIM',0,0)]	
			no_existing_pdb = True

	print(covid_protein_list) # list of covid proteins with MSA
	np.save('covid_protein_list.npy',covid_protein_list)


cov_list = np.load('covid_protein_list.npy')

for pfam_id in cov_list:
	if not generating_msa:
		msa = 	np.load(pfam_id+'/msa.npy')
	no_existing_pdb = False
	print('\n\n\n',pfam_id,'\n')
	# data processing
	ipdb =0
	s0,cols_removed,s_index,s_ipdb = dp.data_processing_covid(root_dir,pfam_id,ipdb,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
	#print(s0[0])
	pf_dict = {}
	pf_dict['s0'] = s0
	pf_dict['s_index'] = s_index
	pf_dict['s_ipdb'] = s_ipdb
	pf_dict['cols_removed'] = cols_removed

	with open('%s/DP.pickle'%(pfam_id), 'wb') as f:
		pickle.dump(pf_dict, f)
	f.close()

	print("Writing FASTA")
	path = "/data/cresswellclayec/DCA_ER/covid_proteins/%s/"%(pfam_id)
	if on_pc:
		path = "/home/eclay/DCA_ER/covid_proteins/%s/"%(pfam_id)
	s_ipdb =  0
	dp.write_FASTA(s0,pfam_id,s_ipdb,number_form=True,processed = True,path=path)



	try:	
		pdb =   [ [pfam_id , '0', pfam_id       , pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]  ,pdb_ranges[pfam_id][a][0]  ,'A', pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]] for a in range(0,len(pdb_ranges[pfam_id])) ]
	except(KeyError):
		pdb_ranges[pfam_id] = [('SIM',0,0)]	
		no_existing_pdb = True

	try:
		subject = msa[0]
		L = len(subject)

		if 0 :	
			# Create pdb_refs.npy	
			#print( pfam_dir[:-1]+'.pdb')
			structure = parser.get_structure(id =pfam_id, file =pfam_id+'.pdb')
			ppb=PPBuilder()
			cppb = CaPPBuilder()
			for pp in ppb.build_peptides(structure):
				print(pp.get_sequence())
			#for cp in cppb.build_peptides(structure):
			#	print('cp: ' ,cp)
			#	print(cp[0])
				#print( inspect.getmembers(cp[0]))


			for model in structure:
				for chain in model:
					#print( inspect.getmembers(chain))
					print(chain.get_full_id())
					print(chain.get_residues())
					for residue in  chain.get_residues():
						#print( inspect.getmembers(residue))
						print(residue)
						print(residue.get_full_id())
						for i,a in enumerate(residue):
							#print(a.get_name())
							#print(a.get_id())
							print(a.get_coord())      # atomic coordinates
							#print(a.get_occupancy())  # occupancy
							#print(a.get_altloc())     # alternative location specifier
						print( inspect.getmembers(a))
		print('subject sequence: ', subject)
	except(FileNotFoundError):
		print('No PDB structure')
	print('\n\n\n\n')

simulating = True
np.random.seed(1)
if simulating:
	for pfam_id in cov_list:

		print("RUNNING SIM FOR %s"%(pfam_id))

		#------- DCA Run -------#
		msa_outfile = '%s/MSA_%s.fa'%(pfam_id,pfam_id) 

		# MF instance 
		mfdca_inst = meanfield_dca.MeanFieldDCA(
		    msa_outfile,
		    'protein',
		    pseudocount = 0.5,
		    seqid = 0.8,
		)

		# Compute DCA scores 
		sorted_DI_mf = mfdca_inst.compute_sorted_DI()

		with open('%s/DI_DCA.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(sorted_DI_mf, f)
		f.close()
		#-----------------------#
		#------- PLM Run -------#

		# PLM instance
		plmdca_inst = plmdca.PlmDCA(
		    msa_outfile,
		    'protein',
		    seqid = 0.8,
		    lambda_h = 1.0,
		    lambda_J = 20.0,
		    num_threads = 16,
		    max_iterations = 500,
		)

		# Compute DCA scores 
		sorted_DI_plm = plmdca_inst.compute_sorted_DI()

		with open('%s/DI_PLM.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(sorted_DI_plm, f)
		f.close()

		n_var = s0.shape[1]
		mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
		mx_cumsum = np.insert(mx.cumsum(),0,0)
		i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

		#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
		onehot_encoder = OneHotEncoder(sparse=False)

		s = onehot_encoder.fit_transform(s0)

		mx_sum = mx.sum()
		my_sum = mx.sum() #!!!! my_sum = mx_sum

		w = np.zeros((mx_sum,my_sum))
		h0 = np.zeros(my_sum)


		#-------------------------------
		# parallel
		res = Parallel(n_jobs = 16)(delayed(predict_w)\
			(s,i0,i1i2,niter_max=10,l2=100.0)\
			for i0 in range(n_var))

		#-------------------------------
		for i0 in range(n_var):
		    i1,i2 = i1i2[i0,0],i1i2[i0,1]
		       
		    h01 = res[i0][0]
		    w1 = res[i0][1]

		    h0[i1:i2] = h01    
		    w[:i1,i1:i2] = w1[:i1,:]
		    w[i2:,i1:i2] = w1[i1:,:]

		# make w to be symmetric
		w = (w + w.T)/2.
		di = direct_info(s0,w)

		sorted_DI_er = sort_di(di)

		with open('%s/DI_ER.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(sorted_DI_er, f)
		f.close()


