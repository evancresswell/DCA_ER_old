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


# break up covid fasta file into smaller fasta for alignment with Wuhand Covide reference genome alignment.
# Write swarm file to muscle aligne genome 10 at a time
# 	- swarm file is cov_align.swarm
# 	- submit swarm job with following command:
# 		-	./submit_align_swarm.script

# Creates aligned files which then need to be concatenated..
#	- Aligned file sequences will be REDUNDANT in once you've concatenated all the swarm output file!!!!



# aligned file: cov_gen_aligned.fasta
#		- acquired by running: 
#  			- mafft --auto --keeplength --addfragments 09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa wuhan_ref.fasta > cov_gen_aligned.fasta
#		-Files used:
# 			- 09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa: FILE FROM---> from http://www.dilekbalik.com/SARS-CoV-2_ODOTool/
#			- wuhand_ref_fasta: FILE FROM----> covid reference genome: NC_045512.2

data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'

aligned_genome_file = 'cov_gen_aligned.fasta'
subject_genome_file = sys.argv[2]

import re
subject_genome = re.sub(".fasta","",subject_genome_file)

nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])

from Bio import SeqIO

f = open('cov_align.swarm','w')
with open(subject_genome_file,"r") as handle:
	records = []
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):

		records.append(record)

		if i%10 ==0:
			print('\n\nwriting file with sequences: \n')	
			for record_sub in records:
				print(record_sub.id)
			
			out_file1 = data_out+subject_genome+'_%d.fasta'%i
			with open(out_file1,"w") as output_handle:
				SeqIO.write(records,output_handle,"fasta")
		
			out_file2 = data_out+subject_genome+'_aligned_%d.fasta'%i
			#f.write("muscle -profile -in1 %s -in2 %s -out %s\n"%(aligned_genome_file, out_file1,out_file2))
			f.write("mafft --keeplength --add %s %s > %s\n"%(out_file1,aligned_genome_file, out_file2))
			# Empty records list for next batch
			records = []


handle.close()	
f.close()

