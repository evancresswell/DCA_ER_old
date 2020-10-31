import sys,os
import timeit
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import glob

data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'

data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'


# Load fasta file for all covid proteins
from Bio import SeqIO



# Load list of fasta files
aligned_cov_fasta_files =  glob.glob(root_dir+'cov_fasta_files/covid_genome_full_aligned*.fasta')
print('Concatenating %d aligned files:\n'%len(aligned_cov_fasta_files),aligned_cov_fasta_files[:10])

final_stock_id = '2020-04-20|Europe/UnitedKingdom/Wales|EPI_ISL_446782'

concat_list = []

for file_indx,aligned_file in enumerate(aligned_cov_fasta_files):
	if file_indx==0:
		# add first 200 records for first file.. 
		concat =True
	else:
		# after first file add all records after first 200
		concat = False

	# iterate through records in mini-alignment
	for i,record in enumerate(SeqIO.parse(aligned_file, "fasta")):
		# Once at to the end of orginal alignment:
		#	-- start concatenating
		if record.id == final_stock_id:
			print('starting concatenation at (file %d) '%file_indx,i)
			concat = True

		# add record to concatenated list
		if concat:	
			concat_list.append(record)
print('Concatenated records count= %d'%len(concat_list))

with open("covid_genome_full_aligned.fasta", "w") as output_handle:
	SeqIO.write(concat_list, output_handle, "fasta")
output_handle.close()
