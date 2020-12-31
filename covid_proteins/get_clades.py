import sys, os
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

#singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python get_clades.py

# break aligned covid fasta file into clades
aligned_fasta = 'covid_genome_full_aligned.fasta'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'


#import re
#subject_genome = re.sub(".fasta","",subject_genome_file)

nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])

from Bio import SeqIO

clade_determinants = 	{\
			'GR':[(14407,'T'),(240,'T'),(3036,'T'),(23402,'G'),(28880,'A'),(28881,'A'),(28882,'C')],\
			'GH':[(14407,'T'),(240,'T'),(3036,'T'),(23402,'G'),(25562,'T')]\
			'S':[(8781,'T'),(28143, 'C')], \
			'V':[(11082,'T'),(26143, 'T')], \
			'G':[(14407,'T'),(240,'T'),(3036,'T'),(23402,'G')], \
			}
clade_records = {\
		'G':[],\
		'GR':[],\
		'GH':[]\
		'S':[],\
		'V':[],\
		}

with open(aligned_fasta,"r") as handle:
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):
		print('record %d '%(i))
		# determine which clade the record belongs to
		for clade in clade_determinants.keys():
			#print('Clade: ',clade)
			#print([(i,char) for (i,char) in clade_determinants[clade] ])
			#print([''.join(record.seq).upper()[i] for (i,char) in clade_determinants[clade] ])
			#print(  all ([''.join(record.seq).upper()[i]== char for (i,char) in clade_determinants[clade] ]))
			if all ([''.join(record.seq).upper()[i]== char for (i,char) in clade_determinants[clade] ]):
				print('Clade: ',clade)
				clade_records[clade].append(record)
		
handle.close()	
		
for clade in clade_determinants.keys():
	print('Clade: %s has %d sequences'%(clade,len(clade_records[clade])))
	print('writing...')		
	out_file1 = data_out+'%s_aligned.fasta'%clade
	with open(out_file1,"w") as output_handle:
		SeqIO.write(clade_records[clade],output_handle,"fasta")
	output_handle.close()

