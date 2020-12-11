
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
import operator

"""
This driver goes through list of pairs (ie orf1ab_pairs_list) which is passed as argument
-  pair data  is generated in codon_mapping.py when two positions are passed.
-  list of pairs is made for during creation of swarm file in  make_pair_aa_swarm.py

This driver generates following ouput:
- gets the pair counts.
- finds the top aa in both positions
- counts the aa pairs  for top 2 (each position) and all others (combined)
"""


#indices are actual postions-1 ie 14408 (real position) --> 14407

pair_list_file = sys.argv[1]
# indices of coevolving with ORF1ab
orf1ab_pairs_list = 	[
		((3036, 'NSP3'),(14407,'NSP12')),
		((19290, 'NSP14a2'),(19558,'NSP14a2')),
		((1058, 'NSP2'),(25562, 'ORF3a')),
		((14804,'NSP12'),(26143, 'ORF3a')),
		((7539, 'NSP3'),(23400, 'S')),
		((14407,'NSP12'),(23403, 'S')),
		((3036, 'NSP3'),(23402, 'S')),
		((21253, 'NSP16'), (22226, 'S')),
		((18554,'NSP14a2'), (23400, 'S')),
		((21366, 'NSP16'), (21368, 'NSP16')),
		((1162, 'NSP2'), (23400, 'S')),
		]

pairs_list =np.load(pair_list_file)
print(pairs_list)

for pair_set in pairs_list:
	print('\n\nCounts for Pair: ',pair_set,'\n')
	pos1 = int(pair_set[0][0])
	region1 = pair_set[0][1]
	pos2 = int(pair_set[1][0])
	region2 = pair_set[1][1]

	aa_pairs = np.load('%d_%d_aa_pairs.npy'%(pos1,pos2))
	print('%d aa pairs \n'%len(aa_pairs))
	#print(np.unique(aa_counts,axis=0,return_counts=True),'\n\n')
	#print('aa_counts\n', aa_counts,'\n')
	aa_pairs, counts = np.unique(aa_pairs,axis=0,return_counts=True)


	#for i,aa_pair in enumerate(aa_pairs):
	#	print(aa_pair,':  ',counts[i])
	#print('\n\n')


	aa_pair_counts = list(zip(counts.tolist(), aa_pairs.tolist()))


	sorted_aa_pair_counts = sorted(aa_pair_counts,key=operator.itemgetter(0),reverse=True)
	print(sorted_aa_pair_counts)	
	
	for aa_pair_count in sorted_aa_pair_counts[:2]:
		print(aa_pair_count)

	pos1_aa = []
	pos2_aa = []
	for (count,aa_pair) in sorted_aa_pair_counts[:2]:
		pos1_aa.append(aa_pair[0])	
		pos2_aa.append(aa_pair[1])	
	print('Position 1 top 2 aa: ',pos1_aa)
	print('Position 2 top 2 aa: ',pos2_aa) 

	# find position 1 and position 2 alternatives (drop top 2 pos 2)
	print('\n\nAlternative counts:\n')
	for i,frequent_aa in enumerate(pos1_aa):
		pos1_alt = 0 
		pos2_alt = 0
		p1_aa = frequent_aa
		p2_aa = pos2_aa[i] 

		for aa_pair in sorted_aa_pair_counts:
			#print(aa_pair)
			if aa_pair[1][1] not in pos2_aa and aa_pair[1][0]==p1_aa:
				#print('pos1 alt! %s col'%p1_aa)
				pos1_alt += aa_pair[0]

			if aa_pair[1][0] not in pos1_aa and aa_pair[1][1]==p2_aa:
				#print('pos2 alt! %s row'%p2_aa)
				pos2_alt += aa_pair[0]

		print('pos1 alt=%s: %d\npos2 alt =%s: %d ' %(p1_aa, pos1_alt,p2_aa,pos2_alt))	

	# find find position 1/2 alternative (neither aa in top 2 of position)
	both_alt = 0
	for aa_pair in sorted_aa_pair_counts:
		if aa_pair[1][0] not in pos1_aa and aa_pair[1][1] not in pos2_aa:
			print(aa_pair[1],' not in either')
			both_alt += aa_pair[0]
	print('pos 1/2 alternative (neither in top 2): %d\n\n\n'%both_alt)
	#print('counts for rest of pairs: ' , sum([item[0] for item in sorted_aa_pair_counts[2:]]),'\n\n')	




