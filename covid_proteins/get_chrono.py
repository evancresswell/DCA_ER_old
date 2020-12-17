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

#singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python get_chrono.py G_aligned.fasta

#aligned_fasta = 'covid_genome_full_aligned.fasta'
aligned_fasta = sys.argv[1]  # clade aligned file
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
data_out = '/data/cresswellclayec/DCA_ER/covid_proteins/cov_fasta_files/'

#sample ID.. want to extract and bin bassed off of time.
#>2020-04-20|Europe/UnitedKingdom/Wales|EPI_ISL_446771

#import re
#subject_genome = re.sub(".fasta","",subject_genome_file)

nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])



def create_bins(lower_bound, width, quantity):
    """ create_bins returns an equal-width (distance) partitioning. 
        It returns an ascending list of tuples, representing the intervals.
        A tuple bins[i], i.e. (bins[i][0], bins[i][1])  with i > 0 
        and i < quantity, satisfies the following conditions:
            (1) bins[i][0] + width == bins[i][1]
            (2) bins[i-1][0] + width == bins[i][0] and
                bins[i-1][1] + width == bins[i][1]
    """
    

    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low,low+width))

def find_bin(value, bins):
    """ bins is a list of tuples, like [(0,20), (20, 40), (40, 60)],
        binning returns the smallest index i of bins so that
        bin[i][0] <= value < bin[i][1]
    """
    
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1    

from Bio import SeqIO
import time
import datetime
import math


dates = []
timestmp = []
stmps = []
bad_time_count = 0
with open(aligned_fasta,"r") as handle:
	for i,record in enumerate(SeqIO.parse(handle, "fasta")):
		#print('record %d '%(i))
		#print('record date: ', record.id[0:10])
		dates.append(record.id[0:10])
		try:
			timestmp.append((i,time.mktime(datetime.datetime.strptime(record.id[0:10],"%Y-%m-%d").timetuple())))
			stmps.append(time.mktime(datetime.datetime.strptime(record.id[0:10],"%Y-%m-%d").timetuple()))
		except(ValueError):
			try:
				timestmp.append((i,time.mktime(datetime.datetime.strptime(record.id[0:7],"%Y-%m").timetuple())))
				stmps.append(time.mktime(datetime.datetime.strptime(record.id[0:7],"%Y-%m").timetuple()))
			except:
				#print('unknown time in ID: ', record.id)
				bad_time_count += 1
				pass

	sorted_timestmp = sorted(timestmp,key=lambda x: x[1])
handle.close()	


chrono_bins = create_bins(int(min(stmps))-10, math.ceil(((max(stmps)+10)-(min(stmps)-10))/3.), 3)
from scipy.stats import binned_statistic


hist, bin_edges = np.histogram(stmps, bins=10, range=(int(min(stmps)), int(max(stmps))))
print(hist)
print(bin_edges)
print('with max %d and min %d timestamps we have the following chrono-bins:\n'%(max(stmps),min(stmps)),chrono_bins)
print('%d timed sequences'%len(stmps)) 

print('%d bad timestmps'%bad_time_count) 
for clade in clade_determinants.keys():
	print('Clade: %s has %d sequences'%(clade,len(clade_records[clade])))
	print('writing...')		
	out_file1 = data_out+'%s_aligned.fasta'%clade
	with open(out_file1,"w") as output_handle:
		SeqIO.write(clade_records[clade],output_handle,"fasta")
	output_handle.close()















