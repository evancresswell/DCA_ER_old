import sys,os
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
#matplotlib.rcParams['text.usetex'] = True
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

from Bio import SeqIO

#========================================================================================
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
# TO RUN: 		singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py
       
table = { 
'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
} 


def translate_sequence(seq,indices,subject_index): 

    protein =[]
    index_mapping = {}
    if len(seq)%3 == 0: 
        amino_index = 0
        for i in range(0, len(seq), 3): 
            # add index mapping from gene to amino acid array
            if subject_index ==indices[i] or subject_index == indices[i+1] or subject_index == indices[i+2]:
                index_tuple = (i,i+1,i+2)
            index_mapping[indices[i]] = amino_index
            index_mapping[indices[i+1]] = amino_index
            index_mapping[indices[i+2]] = amino_index
            codon = seq[i:i + 3] 
            protein.append( table[codon] )
            amino_index += 1
    return protein, index_mapping, index_tuple

# Swarm aligned file 
msa_file = root_dir+"covid_genome_full_aligned.fasta"
ref_file = root_dir+"wuhan_ref.fasta"

s_range = [(21563-1,25384-1)]
e_range = [(26245-1,26472-1)]
e_range = [(26245-1,26472-1)]
nsp12_range = [(13442-1,13468-1),(13468-1,16236-1)]
full_gene_range = [(266-1,29674-1)]

def convert_codon(gene_range = nsp12_range, aligned_file = root_dir+"covid_genome_full_aligned.fasta", ref_file= root_dir+"wuhan_ref.fasta"):
	with open(aligned_file,"r") as handle:


		subject_index = 14407
		subject_encoding_region = 'nsp12'
		column_aa = []


		for i,record in enumerate(SeqIO.parse(handle, "fasta")):
			if i==0:
				seq_array = [char for char in ''.join(record.seq).upper()]
				seq_indices = [i for i,char in enumerate(''.join(record.seq))]
				seq_range_array = []
				seq_range_indices = []
				for start,end in gene_range:
					seq_range_indices.extend( seq_indices[start:end+1])
					seq_range_array.extend(seq_array[start:end+1])

				if 0: # to compare against ncbi genes and proteins
					# compared against nsp12 data @ https://www.ncbi.nlm.nih.gov/protein/1802476815
					# WORKS
					print('record: ',record.id)
					print(seq_range_indices)
					print(seq_range_array)
					print('length: ', len(seq_range_array))
					print('length % 3 ', len(seq_range_array)%3)

				# get index mapping with reference sequence to use for the entire alignment
				protein_seq,codon_index_map,subject_codon_indices = translate_sequence(''.join(seq_range_array),seq_range_indices,subject_index)
				#print(protein_seq)
				print('\n\namino acid array len:', len(protein_seq))
				print('bp to amino acid mapping len: ',len(codon_index_map))

				print('\n#------------------------ 14408 Mapping -----------------------------#')
				print('#-----------------------  Reference Seq -----------------------------#')
				test_index = 14407
				print( '	14408 index in array: ',test_index)
				i1,i2,i3 = subject_codon_indices
				subject_codon_indices = (seq_range_indices[i1],seq_range_indices[i2],seq_range_indices[i3])
				bp1,bp2,bp3 = subject_codon_indices
				subject_codon = [seq_range_array[i1],seq_range_array[i2],seq_range_array[i3]]
				print( '	14408 codon indices: ',subject_codon_indices)
				print( '	14408 codon nucleotieds: ',subject_codon,' --> ',table[''.join(subject_codon)])
				test_amino_index = codon_index_map[test_index]
				print(	'	corresponding amino acid index and letter: %d, %s'%(test_amino_index, protein_seq[test_amino_index]))
				print('#--------------------------------------------------------------------#\n\n')
			# add aa corresponding to gene subject index
			amino_index = codon_index_map[subject_index]
			column_aa.append(protein_seq[amino_index])	
			print(	'	seq %d aa index and letter: %d, %s'%(i,test_amino_index, protein_seq[test_amino_index]))
			sys.exit()

	print('#--------------------------------------------------------------------#')
	print('\n\nSaving...')
	np.save('%d_aa_column.npy'%subject_index,column_aa)
	with open('%s_codon_index_map.pkl'%subject_encoding_region, 'wb') as f:
		pickle.dump(codon_index_map, f)
	f.close()
	print('...Done\n')

convert_codon()


