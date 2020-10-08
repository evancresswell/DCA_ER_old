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
from glob import glob

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
cpus_per_job = sys.argv[1]
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
covid_pdb_list = []

parser = PDBParser()

pdb_ranges = { 'QHD43423':[('6M3M',50,174)], 'QHD43415_8':[('6M71',84,132)], 'QHD43415_7':[('6M71',1,83)], 'QHD43415_3':[('6W6Y',207,379),('6W9C',748,1060)],  'QHD43415_4':[('6LU7',1,306)],  'QHD43415_14':[('6VWW',1,346)], 'QHD43415_9':[('6W4B',1,113)],   'QHD43415_15':[('6W75',1,298)], 'QHD43415_11':[('6M71',1,932)], 'QHD43415_10':[('6W75',1, 139)], 'QHD43416':[('6VYB' ,1,1273),('6VXX',1,1273)]}#('6LXT',912,988,1164,1202)]} #DYNAMIC RANGE IS NOT INCORPRATED>>> NOT PLOTTING 3RD TYPE
generating_msa = True
if generating_msa:
	for pfam_dir in dir_list:
		if on_pc:
			pfam_id = pfam_dir[34:-1]
		else:
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
				#print(list(line)[:-1])
				#print(msa[i,:])
			np.save(pfam_id+'/msa.npy',np.asarray(msa))
			#print(type(msa))
			#print(type(msa[0]))
			#print('msa: ',msa[0])
			covid_protein_list.append(pfam_id)	
		except(FileNotFoundError):
			print('No MSA file')
			continue


		# MUSCLE MSA with polypeptide sequence from pdb structure ( if it exists)
		try:
			# Start by assuming there is a PDB structure
			no_existing_pdb = False

			# Define pdb data (from hardcoded pdb_ranges)
			pdb =   [ [pfam_id , '0', pfam_id       , pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]  ,pdb_ranges[pfam_id][a][0]  ,'A', pdb_ranges[pfam_id][a][1]  ,  pdb_ranges[pfam_id][a][2]] for a in range(0,len(pdb_ranges[pfam_id])) ]

			# Set BioPython parameters to begin extracting PP sequence
			ipdb = 0
			pdb_id = pdb[ipdb][5]                                                                              
			pdb_chain = pdb[ipdb][6]                                                                           
			pdb_start,pdb_end = int(pdb[ipdb][7]),int(pdb[ipdb][8])                                             

			pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
			chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
			ppb = PPBuilder().build_peptides(chain)                                                       

			# Define Polypeptide-Reference Sequence
			poly_seq = list()
			for i,pp in enumerate(ppb):
				for char in str(pp.get_sequence()):
					poly_seq.append(char)                                     
			print('PDB Polypeptide Sequence: \n',poly_seq)
			   
			# Write MSA and PP-ref seq to FASTA 

			print("Writing FASTA")
			pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, msa, pfam_id, number_form=False,processed=False,path=msa_dir)
			# Muscle add PP-ref seq to MSA 
			muscle_msa_file = msa_dir+'PP_muscle_msa_'+pfam_id+'.fa'
			#if os.path.exists(muscle_msa_file):    
			#	print('Using existing muscled FASTA files\n\n')
			#else:
			# Add using muscle:
			#	- https://www.drive5.com/muscle/manual/addtomsa.html
			#	- https://www.drive5.com/muscle/downloads.htmL
			os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
			print("PP sequence added to alignment via MUSCLE\n\n\n")
			covid_pdb_list.append(pfam_id)	
		except(KeyError):
			pdb_ranges[pfam_id] = [('SIM',0,0)]	
			no_existing_pdb = True
			subject = msa[0]
			L = len(subject)
			print('subject sequence: ', subject)
			pp_msa_file, pp_ref_file = tools.write_FASTA(subject, msa, pfam_id, number_form=False,processed=False,path=msa_dir)
			print('No PDB structure (KeyError)\n Using MSA subject sequence\n\n\n')
			continue

		except(FileNotFoundError):
			pdb_ranges[pfam_id] = [('SIM',0,0)]	
			no_existing_pdb = True
			subject = msa[0]
			L = len(subject)
			print('subject sequence: ', subject)
			pp_msa_file, pp_ref_file = tools.write_FASTA(subject, msa, pfam_id, number_form=False,processed=False,path=msa_dir)
			print('No PDB structure (FileNotFoundError)\n Using MSA subject sequence\n\n\n')
			continue
	#print(covid_protein_list) # list of covid proteins with MSA
	print(covid_protein_list)
	print(covid_pdb_list)
	np.save('covid_protein_list.npy',covid_protein_list)
	np.save('covid_pdb_list.npy',covid_pdb_list)


cov_list = np.load('covid_protein_list.npy')
cov_pdb_list = np.load('covid_pdb_list.npy')
print('There as %d proteins with MSA and %d with PDB structures'%(len(cov_list),len(cov_pdb_list)))
for pfam_id in cov_list:
	msa_dir = root_dir+'/'+pfam_id+'/MSA/'

	msa_outfile = msa_dir+'MSA_'+pfam_id+'.fa'
	ref_outfile = msa_dir+'PP_ref_'+pfam_id+'.fa'
	muscle_msa_file = msa_dir+'PP_muscle_msa_'+pfam_id+'.fa'

	print('\n\n\n',pfam_id,'\n')
	# data processing
	ipdb =0
	if pfam_id in cov_pdb_list:
		trimmer = msa_trimmer.MSATrimmer(
		    muscle_msa_file, biomolecule='PROTEIN', 
		    refseq_file=pp_ref_file
		)
	else:
		trimmer = msa_trimmer.MSATrimmer(
		    msa_outfile, biomolecule='PROTEIN', 
		    refseq_file=pp_ref_file
		)
		
	# Adding the data_processing() curation from tools to erdca.
	try:
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms\n%s\n'%ERR)
		sys.exit()

	# Save processed data dictionary and FASTA file
	pfam_dict = {}
	pfam_dict['s0'] = s
	pfam_dict['msa'] = preprocessed_data
	pfam_dict['s_index'] = s_index
	pfam_dict['s_ipdb'] = s_ipdb
	pfam_dict['cols_removed'] = cols_removed 

	input_data_file = msa_dir+"%s_DP.pickle"%(pfam_id)
	with open(input_data_file,"wb") as f:
		pickle.dump(pfam_dict, f)
	f.close()

	preprocessed_data_outfile = msa_dir+'MSA_%s_PreProcessed.fa'%pfam_id
	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
	    for seqid, seq in preprocessed_data:
	        fh.write('>{}\n{}\n'.format(seqid, seq))


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
		    num_threads = cpus_per_job-4,
		    max_iterations = 500,
		)

		# Compute DCA scores 
		sorted_DI_plm = plmdca_inst.compute_sorted_DI()

		with open('%s/DI_PLM.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(sorted_DI_plm, f)
		f.close()

		print('Initializing ER instance\n\n')
		# Compute DI scores using Expectation Reflection algorithm
		erdca_inst = erdca.ERDCA(
		    preprocessed_data_outfile,
		    'PROTEIN',
		    s_index = s_index,
		    pseudocount = 0.5,
		    num_threads = cpus_per_job-4,
		    seqid = 0.8)

		print('Running ER simulation\n\n')
		# Compute average product corrected Frobenius norm of the couplings
		start_time = timeit.default_timer()
		erdca_DI = erdca_inst.compute_sorted_DI()
		run_time = timeit.default_timer() - start_time
		print('ER run time:',run_time)

		with open('%s/DI_ER.pickle'%(pfam_id), 'wb') as f:
		    pickle.dump(erdca_DI, f)
		f.close()


