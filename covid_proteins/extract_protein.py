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
#	- need to define protein name
# 	- RUN WITH:  singularity exec -B /data/cresswellclayec/DCA_ER/covid_proteins/,/data/cresswellclayec/DCA_ER/biowulf /data/cresswellclayec/DCA_ER/erdca.simg python extract_protein.py allprot1017.fasta

data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'


protein_name = 'Spike' 
protein_name = 'NSP3|' 
full_fasta_file = sys.argv[1]
print('Searching for %s proteins in %s ' %(r''.join(protein_name),full_fasta_file))


import re
regex_protein = re.compile(r''.join(protein_name))


from Bio import SeqIO
with open(full_fasta_file,"r") as handle:
	records = []
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):
		if ''.join(record.id)[:len(protein_name)]==protein_name:
			print(record.id)
			records.append(record)
	print('\n\nWriting file with %s Sequences'%(protein_name))	
			
	out_file = data_out+protein_name+'.fasta'
	with open(out_file,"w") as output_handle:
		SeqIO.write(records,output_handle,"fasta")
	output_handle.close()
	

