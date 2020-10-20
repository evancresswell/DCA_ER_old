import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from pydca.erdca import erdca
from pydca.plmdca import plmdca 
from pydca.meanfield_dca import meanfield_dca
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

muscling = True
muscling = False

generating_msa = True
generating_msa = False

preprocessing_data = True
preprocessing_data = False

simulating = True
simulating = False

plotting = False
plotting = True
#========================================================================================
# Loop Throught covid proteins
#========================================================================================
# Generate MSA numpy array
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins'
if on_pc:
	root_dir = '/home/eclay/DCA_ER/covid_proteins'

ipdb = 0


covid_protein = sys.argv[1]
pfam_dir = root_dir+'/%s/'%covid_protein
zhang_pdb_file = root_dir+'/%s.pdb'%covid_protein
zhang_chained_file = root_dir+'/zhang_%s.pdb'%covid_protein

# String to strip from dir list name to get covid_protein
string ="/data/cresswellclayec/DCA_ER/covid_proteins/QHD"
if on_pc:
	string ="/home/eclay/DCA_ER/covid_proteins/QHD"


cpus_per_job = int(sys.argv[2])

covid_protein_list = []
covid_pdb_list = []

parser = PDBParser()
pdb_ranges = { 'QHD43423':[('6M3M',50,174)], 'QHD43415_8':[('6M71',84,132)], 'QHD43415_7':[('6M71',1,83)], 'QHD43415_3':[('6W6Y',207,379),('6W9C',748,1060)],  'QHD43415_4':[('6LU7',1,306)],  'QHD43415_14':[('6VWW',1,346)], 'QHD43415_9':[('6W4B',1,113)],   'QHD43415_15':[('6W75',1,298)], 'QHD43415_11':[('6M71',1,932)], 'QHD43415_10':[('6W75',1, 139)], 'QHD43416':[('6VYB' ,1,1273),('6VXX',1,1273)]}#('6LXT',912,988,1164,1202)]} #DYNAMIC RANGE IS NOT INCORPRATED>>> NOT PLOTTING 3RD TYPE
pdb_ranges = { 'QHD43415_5':[('6LU7',1,306,'A')],'QHD43415_7':[('6M71',1,83,'C')],'QHD43415_9':[('6W4B',1,113,'B')]} # Test set pdb_refs 

if generating_msa:
	print('Generating MSA array for ',covid_protein)
	# only work on QHD directoies

	# Load MSA file
	try:
		#print("		Unpickling DI pickle files for %s"%(covid_protein))
		#file_obj = open("%s/DI.pickle"%(covid_protein),"rb")
		msa_dir = pfam_dir+"/MSA/"
		with open(msa_dir+"protein.aln", 'r') as infile:
			MSA = infile.readlines()
		infile.close()
		print('\n\n MSA shape',np.shape(MSA),'\n\n')
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
		np.save(covid_protein+'/msa.npy',np.asarray(msa))
		#print(type(msa))
		#print(type(msa[0]))
		#print('msa: ',msa[0])
		covid_protein_list.append(covid_protein)	
	except(FileNotFoundError):
		print('No MSA file')
		sys.exit()
		


	# MUSCLE MSA with polypeptide sequence from pdb structure ( if it exists)
	try:
		# Start by assuming there is a PDB structure
		no_existing_pdb = False

		# Define pdb data (from hardcoded pdb_ranges)
		pdb =   [ [covid_protein , '0', covid_protein       , pdb_ranges[covid_protein][a][1]  ,  pdb_ranges[covid_protein][a][2]  ,pdb_ranges[covid_protein][a][0]  ,'A', pdb_ranges[covid_protein][a][1]  ,  pdb_ranges[covid_protein][a][2]] for a in range(0,len(pdb_ranges[covid_protein])) ]

		# Set BioPython parameters to begin extracting PP sequence
		#pdb_id = pdb[ipdb][5]                                                                              
		pdb_id = pdb_ranges[covid_protein][ipdb][0]
		#pdb_chain = pdb[ipdb][6]                                                                           
		pdb_chain = pdb_ranges[covid_protein][ipdb][3]
		pdb_start,pdb_end = int(pdb_ranges[covid_protein][ipdb][1]),int(pdb_ranges[covid_protein][ipdb][2])                                             

		pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
		chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
		ppb = PPBuilder().build_peptides(chain)                                                       

		# Define Polypeptide-Reference Sequence
		poly_seq = list()
		for i,pp in enumerate(ppb):
			for char in str(pp.get_sequence()):
				poly_seq.append(char)                                     
		print('PDB Polypeptide Sequence: \n',poly_seq,'\nlength: %d\n\n'%len(poly_seq))

		# Load Zhang PDB strucutre
		from Bio.PDB.Chain import Chain
		print('Loading Zhang PDB strucutre from : '+zhang_pdb_file)
		zhang_chained_file = root_dir+'/zhang_%s.pdb'%covid_protein
		#os.system("cp %s %s "%(zhang_pdb_file,zhang_chained_file))
			#os.system("sed -i 's/^\(ATOM.\{17ed }\) /\1%s/' %s "%(pdb_chain,zhang_chained_file))
		#os.system("sed -i 's/ /%s/17' %s "%('CHAIN INPUT',zhang_chained_file))
		#os.system("awk"%(pdb_chain,zhang_chained_file))

		zhang_chain = pdb_parser.get_structure(str(covid_protein),zhang_chained_file)[ipdb][pdb_chain]
		
		ppb = PPBuilder().build_peptides(zhang_chain)                                                       

		# Define Polypeptide-Reference Sequence
		zhang_poly_seq = list()
		for i,pp in enumerate(ppb):
			for char in str(pp.get_sequence()):
				zhang_poly_seq.append(char)                                     
		print('Zhang PDB Polypeptide Sequence: \n',zhang_poly_seq,'\nlength: %d\n\n'%len(zhang_poly_seq))
		   
		# Write MSA and PP-ref seq to FASTA 

		print("Writing FASTA")
		pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, msa, covid_protein, number_form=False,processed=False,path=msa_dir)
		if muscling:
			# Muscle add PP-ref seq to MSA 
			muscle_msa_file = msa_dir+'PP_muscle_msa_'+covid_protein+'.fa'
			#if os.path.exists(muscle_msa_file):    
			#	print('Using existing muscled FASTA files\n\n')
			#else:
			# Add using muscle:
			#	- https://www.drive5.com/muscle/manual/addtomsa.html
			#	- https://www.drive5.com/muscle/downloads.htmL
			os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
			print("%s: PP sequence added to alignment via MUSCLE\n\n\n"%covid_protein)
			covid_pdb_list.append(covid_protein)	
	except(KeyError):
		print('KeyError!!! \n\n')
		pdb_ranges[covid_protein] = [('SIM',0,0)]	
		no_existing_pdb = True
		subject = msa[0]
		L = len(subject)
		print('subject sequence: ', subject)
		pp_msa_file, pp_ref_file = tools.write_FASTA(subject, msa, covid_protein, number_form=False,processed=False,path=msa_dir)
		print('No PDB structure (KeyError)\n Using MSA subject sequence\n\n\n')
		sys.exit()

	except(FileNotFoundError):
		print('FileNotFoundError!!! \n\n')
		pdb_ranges[covid_protein] = [('SIM',0,0)]	
		no_existing_pdb = True
		subject = msa[0]
		L = len(subject)
		print('subject sequence: ', subject)
		pp_msa_file, pp_ref_file = tools.write_FASTA(subject, msa, covid_protein, number_form=False,processed=False,path=msa_dir)
		print('No PDB structure (FileNotFoundError)\n Using MSA subject sequence\n\n\n')
		sys.exit()
		#print(covid_protein_list) # list of covid proteins with MSA


print('There as %d proteins with MSA and %d with PDB structures'%(len(covid_protein_list),len(covid_pdb_list)))
if preprocessing_data:
	msa_dir = root_dir+'/'+covid_protein+'/MSA/'

	msa_outfile = msa_dir+'MSA_'+covid_protein+'.fa'
	ref_outfile = msa_dir+'PP_ref_'+covid_protein+'.fa'
	muscle_msa_file = msa_dir+'PP_muscle_msa_'+covid_protein+'.fa'

	print('\n\n\n',covid_protein,'\n',msa_outfile,'\n',ref_outfile)
	# data processing
	ipdb =0
	if covid_protein in covid_pdb_list:
		if muscling:
			trimmer = msa_trimmer.MSATrimmer(
			    muscle_msa_file, biomolecule='PROTEIN', 
			    refseq_file=ref_outfile
			)
		else:
			trimmer = msa_trimmer.MSATrimmer(
			    msa_outfile, biomolecule='PROTEIN', 
			    refseq_file=ref_outfile
			)
	else:
		try:
			trimmer = msa_trimmer.MSATrimmer(
			    msa_outfile, biomolecule='PROTEIN', 
			    refseq_file=ref_outfile
			)
		except(ValueError):
			print('\n\n%s: Empty protein.aln! Moving On ..\n\n'%covid_protein)
			sys.exit()
	# Adding the data_processing() curation from tools to erdca.
	try:
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	except(MSATrimmerException):
		ERR = 'PPseq-MSA'
		print('Error with MSA trimms\n%s\nUsing MSA[0]'%ERR)
		if covid_protein not in covid_pdb_list:
			print('%s: BAD MSA\n\n\n'%covid_protein)
			sys.exit()

		# Because MUSCLED MSA yields uselfss 
		with open(msa_dir+"protein.aln", 'r') as infile:
			MSA = infile.readlines()
		infile.close()
		msa = []
		for i,line in enumerate(MSA):
			msa.append(list(line)[:-1])
		subject = msa[0]
		print('\n%s: New subject sequence (msa[0]): '%covid_protein, subject)
		msa_file, ref_file = tools.write_FASTA(subject, msa, covid_protein, number_form=False,processed=False,path=msa_dir)

		trimmer = msa_trimmer.MSATrimmer(
		    msa_file, biomolecule='PROTEIN', 
		    refseq_file=ref_file
		)
		preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
		

	# Save processed data dictionary and FASTA file
	pfam_dict = {}
	pfam_dict['s0'] = s
	pfam_dict['msa'] = preprocessed_data
	pfam_dict['s_index'] = s_index
	pfam_dict['s_ipdb'] = s_ipdb
	pfam_dict['cols_removed'] = cols_removed 

	input_data_file = msa_dir+"%s_DP.pickle"%(covid_protein)
	with open(input_data_file,"wb") as f:
		pickle.dump(pfam_dict, f)
	f.close()

	preprocessed_data_outfile = msa_dir+'MSA_%s_PreProcessed.fa'%covid_protein
	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
		for seqid, seq in preprocessed_data:
			fh.write('>{}\n{}\n'.format(seqid, seq))


if simulating:
	msa_dir = root_dir+'/'+covid_protein+'/MSA/'
	preprocessed_data_outfile = msa_dir+'MSA_%s_PreProcessed.fa'%covid_protein

	print("RUNNING SIM FOR %s"%(covid_protein))

	#------- DCA Run -------#
	msa_outfile = msa_dir+'MSA_'+covid_protein+'.fa'
	ref_outfile = msa_dir+'PP_ref_'+covid_protein+'.fa'
	muscle_msa_file = msa_dir+'PP_muscle_msa_'+covid_protein+'.fa'

	if covid_protein in covid_pdb_list:
		# MF instance 
		mfdca_inst = meanfield_dca.MeanFieldDCA(
		    muscle_msa_file,
		    'protein',
		    pseudocount = 0.5,
		    seqid = 0.8,
		)
	else:
		try:
			# MF instance 
			mfdca_inst = meanfield_dca.MeanFieldDCA(
			    msa_outfile,
			    'protein',
			    pseudocount = 0.5,
			    seqid = 0.8,
			)
		except(ValueError):
			print('\n\n%s: Empty protein.aln! Moving On ..\n\n'%covid_protein)
			sys.exit()
	# Compute DCA scores 
	sorted_DI_mf = mfdca_inst.compute_sorted_DI()

	with open('%s/DI_DCA.pickle'%(covid_protein), 'wb') as f:
	    pickle.dump(sorted_DI_mf, f)
	f.close()
	#-----------------------#
	#------- PLM Run -------#

	if covid_protein in covid_pdb_list:
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

	else:
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

	with open('%s/DI_PLM.pickle'%(covid_protein), 'wb') as f:
	    pickle.dump(sorted_DI_plm, f)
	f.close()

	input_data_file = msa_dir+"%s_DP.pickle"%(covid_protein)
	with open(input_data_file,"rb") as f:
		pfam_dict = pickle.load(f)
	f.close()
	# Save processed data dictionary and FASTA file
	s = pfam_dict['s0']
	preprocessed_data = pfam_dict['msa']
	s_index = pfam_dict['s_index']
	s_ipdb = pfam_dict['s_ipdb']
	cols_removed = pfam_dict['cols_removed']


	print('Initializing ER instance\n\n')
	erdca_inst = erdca.ERDCA(
	    preprocessed_data_outfile,
	    'PROTEIN',
	    s_index = s_index,
	    pseudocount = 0.5,
	    num_threads = cpus_per_job-4,
	    seqid = 0.8)

	# Compute DI scores using Expectation Reflection algorithm
	print('Running ER simulation\n\n')
	# Compute average product corrected Frobenius norm of the couplings
	start_time = timeit.default_timer()
	erdca_DI = erdca_inst.compute_sorted_DI()
	run_time = timeit.default_timer() - start_time
	print('ER run time:',run_time)

	with open('%s/DI_ER.pickle'%(covid_protein), 'wb') as f:
	    pickle.dump(erdca_DI, f)
	f.close()

if plotting:
	pdb_id = pdb_ranges[covid_protein][ipdb][0]
	#pdb_chain = pdb[ipdb][6]                                                                           
	pdb_chain = pdb_ranges[covid_protein][ipdb][3]
	pdb_start,pdb_end = int(pdb_ranges[covid_protein][ipdb][1]),int(pdb_ranges[covid_protein][ipdb][2])                                             


	#------- Data Files ----#
	msa_dir = root_dir+'/'+covid_protein+'/MSA/'

	msa_outfile = msa_dir+'MSA_'+covid_protein+'.fa'
	ref_outfile = msa_dir+'PP_ref_'+covid_protein+'.fa'
	muscle_msa_file = msa_dir+'PP_muscle_msa_'+covid_protein+'.fa'

	preprocessed_data_outfile = msa_dir+'MSA_%s_PreProcessed.fa'%covid_protein
	#-----------------------#

	# Print Details of protein PDB structure Info for contact visualizeation
	print('Using chain ',pdb_chain)
	print('PDB ID: ', pdb_id)

	from pydca.contact_visualizer import contact_visualizer

	with open('%s/DI_ER.pickle'%(covid_protein), 'rb') as f:
	    er_DI = pickle.load(f)
	f.close()

	with open('%s/DI_DCA.pickle'%(covid_protein), 'rb') as f:
	    mf_DI = pickle.load(f)
	f.close()

	with open('%s/DI_PLM.pickle'%(covid_protein), 'rb') as f:
	    plm_DI = pickle.load(f)
	f.close()
	if 0:
		visualizer = contact_visualizer.DCAVisualizer('protein',pdb_chain,pdb_id,
			refseq_file = ref_outfile,
			sorted_dca_scores = er_DI,
			linear_dist = 4,
			contact_dist = 8.0)

		contact_map_data = visualizer.plot_contact_map()
		plt.show()
		plt.close()
		tp_rate_data = visualizer.plot_true_positive_rates()
		plt.show()
		plt.close()


	zhang_visualizer = contact_visualizer.DCAVisualizer('protein',pdb_chain,zhang_chained_file,
		refseq_file = ref_outfile,
		sorted_dca_scores = er_DI,
		linear_dist = 5,
		contact_dist = 10.0)
	contact_map_data = zhang_visualizer.plot_contact_map()
	plt.show()
	plt.close()

	zhang_visualizer = contact_visualizer.DCAVisualizer('protein',pdb_chain,zhang_chained_file,
		refseq_file = ref_outfile,
		sorted_dca_scores = mf_DI,
		linear_dist = 5,
		contact_dist = 10.0)
	contact_map_data = zhang_visualizer.plot_contact_map()
	plt.show()
	plt.close()

	zhang_visualizer = contact_visualizer.DCAVisualizer('protein',pdb_chain,zhang_chained_file,
		refseq_file = ref_outfile,
		sorted_dca_scores = plm_DI,
		linear_dist = 5,
		contact_dist = 10.0)
	contact_map_data = zhang_visualizer.plot_contact_map()
	plt.show()
	plt.close()



	#print('Contact Map: \n',contact_map_data[:10])
	#print('TP Rates: \n',tp_rate_data[:10])



