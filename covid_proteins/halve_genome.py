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

#singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python halve_genome.py

# break aligned covid fasta file into clades
aligned_fasta = 'covid_genome_full_aligned.fasta'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'


#import re
#subject_genome = re.sub(".fasta","",subject_genome_file)

nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])

from Bio import SeqIO
import random 
def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]
# need to return two lists..

records = []
with open(aligned_fasta,"r") as handle:
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):
		records.append(record)
handle.close()	

split_lists = partition(records, 2)
print(len(split_lists[0]))
print(len(split_lists[1]))
#for listed in split_lists:
#	print('\n\n',[item.id for item in listed],'\n\n')
		
print('writing...')		
for i,partition in enumerate(split_lists):
	print('partion %s has %d sequences'%(i,len(partition)))
	out_file1 = data_out+'cov_genome_aligned_part%d.fasta'%i
	with open(out_file1,"w") as output_handle:
		SeqIO.write(partition,output_handle,"fasta")
	output_handle.close()

