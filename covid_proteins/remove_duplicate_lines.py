import sys, os
import genome_data_processing as gdp
import data_processing as dp
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

# Extracts all sequences whos ID contain a given name (protein_name)
#	- saves file for alignment
# 	- RUN WITH: exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/erdca.simg python remove_duplicate_lines.py cov_fasta_files/Spike.fasta spike_NoDup 

data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'


full_fasta_file = sys.argv[1]
protein_name = sys.argv[2]
print('Searching for %s proteins in %s ' %(r''.join(protein_name),full_fasta_file))

from Bio import SeqIO
record_list = []
records = []
with open(full_fasta_file,"r") as handle:
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):
		if ''.join(record.seq) not in record_list:
			print(''.join(record.id))
			print(''.join(record.seq))
			records.append(record)
			record_list.append(''.join(record.seq))
			
	out_file = data_out+protein_name+'.fasta'
	with open(out_file,"w") as output_handle:
		SeqIO.write(records,output_handle,"fasta")
	output_handle.close()
	


