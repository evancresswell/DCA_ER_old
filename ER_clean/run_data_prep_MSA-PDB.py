"""pydca demo

Author: Evan Cresswell-Clay
"""
import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC
import matplotlib.pyplot as plt

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

pfam_id = sys.argv[1]
num_threads = int(sys.argv[2])-2

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
processed_data_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'

# pdb_ref should give list of
# 0) accession codes,
# 1) number of residues,
# 2) number of sequences,
# 3) and number of pdb references
# Read in Reference Protein Structure
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
# convert bytes to str (python 2 to python 3)                                                                       
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])      for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
ipdb = 0
tpdb = int(pdb[ipdb,1])
print('Ref Sequence # should be : ',tpdb-1)

# Load Multiple Sequence Alignment
s = dp.load_msa(data_path,pfam_id)

# Load Polypeptide Sequence from PDB as reference sequence
print(pdb[ipdb,:])
pdb_id = pdb[ipdb,5]
pdb_chain = pdb[ipdb,6]
pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        
pdb_range = [pdb_start-1, pdb_end]
#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    

chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain]
ppb = PPBuilder().build_peptides(chain)
#    print(pp.get_sequence())
print('peptide build of chain produced %d elements'%(len(ppb)))

found_match = True
matching_seq_dict = {}
poly_seq = list()
for i,pp in enumerate(ppb):
	for char in str(pp.get_sequence()):
		        poly_seq.append(char)    
print('PDB Polypeptide Sequence (len=%d): \n'%len(poly_seq),poly_seq)
poly_seq_range = poly_seq[pdb_range[0]:pdb_range[1]]
print('PDB Polypeptide Sequence (In Proteins PDB range len=%d): \n'%len(poly_seq_range),poly_seq_range)
if len(poly_seq_range) < 10:
	print('PP sequence overlap with PDB range is too small.\nWe will find a match\nBAD PDB-RANGE')
	poly_seq_range = poly_seq

#check that poly_seq matches up with given MSA
    
pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path=processed_data_path,nickname='range')

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
#Add correct range of PP sequence to MSA using muscle
#https://www.drive5.com/muscle/manual/addtomsa.html
muscle_msa_file = processed_data_path + 'PP_muscle_msa_'+pfam_id+'.fa'

# Search MSA for for PP-seq (in range) match.
slice_size = 10
matches = np.zeros(s.shape[0])
#print(poly_seq_range)
try:
	# MSA Sequence which best matches PP Sequence
	for i in range(len(poly_seq_range)-slice_size + 1):
		poly_seq_slice = poly_seq_range[i:i+slice_size]
		#print(''.join(poly_seq_slice))
		#print(s.shape)
		for row in range(s.shape[0]):
		#print(''.join(s[row,:]).replace('-','').upper())
			if ''.join(poly_seq_slice).upper() in ''.join(s[row,:]).replace('-','').upper():
				#print('Match found on Row %d : \n'%row,''.join(s[row,:]).replace('-','').upper() )
				matches[row]+=1    
	    

	m = max(matches)
	if m > 0:
	    best_matches_indx = [i for i,j in enumerate(matches) if j==m]
	print('PP sequence: \n',''.join(poly_seq_range).upper())
	print('best match in msa at indices: ',best_matches_indx)
	match_ref = ''.join(s[best_matches_indx[0],:]).replace('-','').upper()
	print('MSA sequence: \n',match_ref,'\n\n Finding index mapping...')

	import re
	largest_match = [0,-1]
	for i in range(len(poly_seq_range)-slice_size + 1):
		poly_seq_slice = poly_seq_range[i:i+slice_size]
		pp_slice_string = ''.join(poly_seq_slice).upper()
		#print(''.join(s[row,:]).replace('-','').upper())
		if pp_slice_string in match_ref:
			matching = True
			match_start = re.search(pp_slice_string,match_ref).start()
			#rint('finding length of match starting at ',match_start)
			ii = match_start+1
			while(matching):
				ii = ii + 1
				if ii > len(poly_seq_range)-slice_size:
					match_end = regex_match.end()
					break
				poly_seq_slice = poly_seq_range[ii:ii+slice_size]
				pp_slice_string = ''.join(poly_seq_slice).upper()
				if pp_slice_string in match_ref:
					regex_match = re.search(pp_slice_string,match_ref)
					match_end = regex_match.end()
				else:
					if match_end - match_start > largest_match[1] - largest_match[0]:
						largest_match = [match_start,match_end]
						print('                      ...\n\n')
					matching = False
					i = match_end+1

	print('\n\nMSA match range', largest_match)

	# Find indices corresponding for MSA's matching reference and PP sequence.
	new_match_ref = match_ref[largest_match[0]:largest_match[1]]
	print('MSA match ref in match range: ',new_match_ref)
	regex_polyseq = re.search(''.join(new_match_ref),''.join(poly_seq_range) )
	new_poly_seq = ''.join(poly_seq_range)[regex_polyseq.start(): regex_polyseq.end()] 

	print('PP match range [%d, %d]\n\nMatched sequences: \n' %(regex_polyseq.start(), regex_polyseq.end()))

	print(''.join(new_poly_seq))
	print(''.join(new_match_ref))

	# Write new poly peptide sequence to fasta for alignment    
	# save the matched poly seq for use in the simulation
	pp_msa_file_match, pp_ref_file_match = tools.write_FASTA(new_poly_seq, s, pfam_id, number_form=False,processed=False,path=processed_data_path,nickname='match')

	# save the matched PP poly seq for adding to muscle file
	pp_msa_file, pp_ref_file = tools.write_FASTA(new_poly_seq, s, pfam_id, number_form=False,processed=False,path=processed_data_path)
	print('\n\nWriting MSA and Ref FASTA files:\n%s\n%s\n\n'%(pp_msa_file,pp_ref_file))



except(NameError):

	print('No Matches found.. using Original PDB range')

	# save the PDB-range PP poly seq for adding to muscle file
	pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path=processed_data_path)
	print('\n\nWriting MSA and Ref FASTA files:\n%s\n%s\n\n'%(pp_msa_file,pp_ref_file))

except(AttributeError):
	print('Match not in PDB range PP sequence')



