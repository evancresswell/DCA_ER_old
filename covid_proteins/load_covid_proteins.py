import sys,os
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC

data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'

data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'


# Load fasta file for all covid proteins
from Bio import SeqIO

import sys
from Bio import SeqIO

nucleotide_letters = np.array(['A','C','G','T'])

def value_with_prob(a,p1):
    """ generate a value (in a) with probability
    input: a = np.array(['A','B','C','D']) and p = np.array([0.4,0.5,0.05,0.05]) 
    output: B or A (likely), C or D (unlikely)
    """
    p = p1.copy()
    # if no-specific prob --> set as uniform distribution
    if p.sum() == 0:
        p[:] = 1./a.shape[0] # uniform
    else:
        p[:] /= p.sum() # normalize

    ia = int((p.cumsum() < np.random.rand()).sum()) # cordinate

    return a[ia]    

#------------------------------
def find_and_replace(s,z,a):
	""" find positions of s having z and replace by a with a probality of elements in s column
	input: s = np.array([['A','Q','A'],['A','E','C'],['Z','Q','A'],['A','Z','-']])
	   z = 'Z' , a = np.array(['Q','E'])    
	output: s = np.array([['A','Q','A'],['A','E','C'],['E','Q','A'],['A','Q','-']]           
	"""  
	xy = np.argwhere(s == z)
	try:
		for it in range(xy.shape[0]):
			t,i = xy[it,0],xy[it,1]

			na = a.shape[0]
			p = np.zeros(na)    
			for ii in range(na):
				p[ii] = (s[:,i] == a[ii]).sum()

			s[t,i] = value_with_prob(a, p)
	except(IndexError):
		print(xy)
	return s 


nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U'])
def sequuence_alignment_prep(fasta_file):
	# Create our hash table to add the sequences
	sequences = {}
	sequence_list = []
	# Using the Biopython fasta parse we can read our fasta input
	for seq_record in SeqIO.parse(fasta_file, "fasta"):
		# Take the current sequence
		sequence = str(seq_record.seq).upper()
		sequence_list = [char for char in sequence]

		sequence_array = np.array(sequence_list)
		for i,char in enumerate(sequence_array):
			if char not in nucleotide_letters_full:
				print('bad character!!: at pos %d '%i,char)
				print(seq_record)
				sequence_array = find_and_replace(sequence_array, char, np.array(['N']))

		sequence = ''.join(sequence_array)
		# hash table, the sequence and its id are going to be in the hash
		if sequence not in sequences:
			sequences[sequence] = seq_record.id
		

	# Write the clean sequences
	# Create a file in the same directory where you ran this script
	with open("cleaned_" + fasta_file, "w+") as output_file:
		# Just read the hash table and write on the file as a fasta format
		for sequence in sequences:
			output_file.write(">" + sequences[sequence] + "\n" + sequence + "\n")
		
		print("CLEAN!!!\nPlease check clear_" + fasta_file)

	return "cleaned_" + fasta_file

def sequence_cleaner(fasta_file, min_length=0, por_n=100):
	# Create our hash table to add the sequences
	sequences = {}
	sequence_list = []
	# Using the Biopython fasta parse we can read our fasta input
	for seq_record in SeqIO.parse(fasta_file, "fasta"):
		bad_seq = False
		# Take the current sequence
		sequence = str(seq_record.seq).upper()
		sequence_list.append( [char for char in sequence])
	sequence_list = np.array(sequence_list)
	# Find and replace nucleotide characters with random delineation from:
	# https://www.bioinformatics.org/sms/iupac.html
	sequence_list = find_and_replace(sequence_list, 'N', nucleotide_letters)
	sequence_list = find_and_replace(sequence_list, 'R', np.array(['A','G']))
	sequence_list = find_and_replace(sequence_list, 'Y', np.array(['C','T']))
	sequence_list = find_and_replace(sequence_list, 'S', np.array(['G','C']))
	sequence_list = find_and_replace(sequence_list, 'W', np.array(['A','T']))
	sequence_list = find_and_replace(sequence_list, 'K', np.array(['G','T']))
	sequence_list = find_and_replace(sequence_list, 'M', np.array(['A','C']))
	sequence_list = find_and_replace(sequence_list, 'B', np.array(['C','G','T']))
	sequence_list = find_and_replace(sequence_list, 'D', np.array(['A','G','T']))
	sequence_list = find_and_replace(sequence_list, 'H', np.array(['A','C','T']))
	sequence_list = find_and_replace(sequence_list, 'V', np.array(['A','C','G']))

	for i,character in enumerate(sequence_list):
		if character not in ['A','C','G','T']:
			print(sequence_list)
			print('bad character!!: at pos %d '%i,character)
			print(seq_record)
			bad_seq = True

			sys.exit()
	# Check if the current sequence is according to the user parameters
	#if (len(sequence) >= min_length and (float(sequence.count("N")) / float(len(sequence))) * 100 <= por_n):
	if not bad_seq:
		sequence = ""
		sequence.join(sequence_list)
		# hash table, the sequence and its id are going to be in the hash
		if sequence not in sequences:
			sequences[sequence] = seq_record.id
		

	# Write the clean sequences

	# Create a file in the same directory where you ran this script
	with open("clear_" + fasta_file, "w+") as output_file:
		# Just read the hash table and write on the file as a fasta format
		for sequence in sequences:
			output_file.write(">" + sequences[sequence] + "\n" + sequence + "\n")
		
		print("CLEAN!!!\nPlease check clear_" + fasta_file)

	return "clear_" + fasta_file


print('\n\nCleaning File\n\n')
#clean_file = sequence_cleaner("covid_genome_sequences.fasta")
clean_file = sequuence_alignment_prep("covid_genome_sequences.fasta")



if 0:
	cov_proteins = {}

	for record in SeqIO.parse("allprot1017.fasta", "fasta"):
		record_list = record.id.split('|')
		pfam_id = record_list[0]
		if(pfam_id not in cov_proteins.keys()):
			print('New PDB ID, ',pfam_id) 
			cov_proteins[pfam_id] = []
		cov_proteins[pfam_id].append([char for char in str(record.seq)])

	print(cov_proteins.keys())
	for key in cov_proteins.keys():
		print (len(cov_proteins[key]))

	with open(root_dir+'cov_proteins_dict.pickle', 'wb') as f:
	    pickle.dump(cov_proteins, f)
	f.close()










