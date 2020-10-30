import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit

import numpy as np
import pickle
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import glob
import operator

genreating_txt = True
if genreating_txt:
	pfam_txt_files =  glob.glob('DI/PF*.txt')


	testing = True
	testing = False
	if testing:
		# make smaller dataframes
		print("\n\nMAKING TEST SIZE DATAFRAME\n\n")
		pfam_txt_files = pfam_txt_files[:50]

	pf_score = {}
	pf_num_seq = {}
	for i,filename in enumerate(pfam_txt_files):
		f = open(filename, "r")
		f_input = f.read().split()
		f.close()
		pfam_id = f_input[0]
		pf_score[pfam_id] = [float(f_input[1]),float(f_input[2]),float(f_input[3])]
		pf_num_seq[pfam_id] = int(f_input[4])
		if testing:
			print(f_input)
			print(pfam_id) 
			print(pf_score[pfam_id])
			print(pf_num_seq[pfam_id])

	print('recorded scores for %d pfams'%len(pf_score))
	with open('pfam_method_AUTPR.pickle', 'wb') as handle:
		pickle.dump(pf_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()

	with open('pfam_num_seq.pickle', 'wb') as handle:
		pickle.dump(pf_num_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()


# load PF dictionaries
with open('pfam_method_AUTPR.pickle','rb') as f:
	pf_scores = pickle.load(f)
f.close()
with open('pfam_num_seq.pickle','rb') as f:
	pf_num_seqs = pickle.load(f)
f.close()

print('Plotting for %d Pfams'%len(pf_score.keys()))
bm = []
for key in pf_scores:
	best_method_indx, score = max(enumerate(pf_scores[key]), key=operator.itemgetter(1))
	#print(pf_score[key])
	#print(best_method_indx)
	bm.append(best_method_indx)	

print('ER best for %d methods'%bm.count(0))
print('MF best for %d methods'%bm.count(1))
print('PLM best for %d methods'%bm.count(2))
