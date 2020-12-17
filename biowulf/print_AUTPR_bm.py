import sys,os
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

import timeit
import pandas as pd
import seaborn as sns

import numpy as np
import pickle
import glob
import operator




# Methods string to set best method. 
# ---> ORDER SHOULD MATCH WITH gen_DI_scores.py ORDER OF STORED SCORES IN .txt FILES !!!!
methods = ['LADER_clean','ER_clean','ER_coup','ER','MF','PLM']
methods = ['ERDCA','ER','MF','PLM']




genreating_txt = True
if genreating_txt:
	#pfam_txt_files =  glob.glob('DI/PF*.txt')
	file_in = sys.argv[1] 
	score_type = sys.argv[2] 
	pfam_txt_files = np.loadtxt(file_in,dtype='str')


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

		# ER_coup vs ER
		#pf_score[pfam_id] = [float(f_input[1]),float(f_input[2]),float(f_input[3]),float(f_input[4])]
		pf_score[pfam_id] = []
		for indx,method in enumerate(methods):
			pf_score[pfam_id].append(float(f_input[indx+1]))

		pf_num_seq[pfam_id] = int(f_input[-1])
		if testing:
			print(f_input)
			print(pfam_id) 
			print(pf_score[pfam_id])
			print(pf_num_seq[pfam_id])
		

	print('recorded scores for %d pfams'%len(pf_score))
	if score_type == 'AUTPR':
		with open('pfam_method_AUTPR.pickle', 'wb') as handle:
			pickle.dump(pf_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
		handle.close()
	elif score_type == 'AUC':
		with open('pfam_method_AUC.pickle', 'wb') as handle:
			pickle.dump(pf_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
		handle.close()
	else:
		print('\n\nsys.argv[2] Must define score type: (AUTPR, AUC)\n\n')
		sys.exit()

	with open('pfam_num_seq.pickle', 'wb') as handle:
		pickle.dump(pf_num_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)
	handle.close()

# load PF dictionaries
if score_type == 'AUTPR':
	with open('pfam_method_AUTPR.pickle','rb') as f:
		pf_scores = pickle.load(f)
	f.close()
elif score_type == 'AUC':
	with open('pfam_method_AUC.pickle','rb') as f:
		pf_scores = pickle.load(f)
	f.close()

with open('pfam_num_seq.pickle','rb') as f:
	pf_num_seqs = pickle.load(f)
f.close()

print('Plotting for %d Pfams'%len(pf_score.keys()))
bm = []
bm_str = []
max_score = 0


kill_index = [0,1,2] # No LADER/ER clean or ER coup
kill_index = [2] # ER coup is obsolete..
kill_index = [1] # get rid of original ER
kill_index = [] # keep all scores

print('REMOVING FOLLOWING SCORE INDICES FROM CONTENTION:')
for index in kill_index:
	print(index)
	
for key in pf_scores.keys():
	# remove score from contention
	for index in kill_index:
		pf_scores[key][index] = 0
	best_method_indx, score = max(enumerate(pf_scores[key]), key=operator.itemgetter(1))
	
	bm.append(best_method_indx)	

	bm_str.append(methods[best_method_indx])

	if best_method_indx == 0: #index should match ERDCA index in methods above
		if max_score<=score:
			best_pfam = key
			max_score = score 

#print('Best ER pfam: %s ,score; %f'%(best_pfam,max_score))
for indx,method in enumerate(methods):
	print('%s best for %d methods'%(method,bm.count(indx)))
print('ERDCA performs best for %s with score/AUC = %f'%(best_pfam,score))

EDCA__AUTPR = []
ER_AUTPR = []
MF_AUTPR = []
PLM_AUTPR = []

Scores = []
for indx,method in enumerate(methods):
	Scores.append([])
for key in pf_scores:
	for indx,method in enumerate(methods):
		Scores[indx].append(pf_scores[key][indx])


num_seq_bins = np.arange(min(pf_num_seq.values()),max(pf_num_seq.values()),1000)
#print('Divding scores by number of sequences:\n',num_seq_bins)


# create dictionary of scores by method to convert to dataframe (converting for ease of slicing)
method_score_dict = {}
for indx,method in enumerate(methods):
	method_score_dict[method] = Scores[indx]

method_score_dict['BM'] = bm_str
method_score_dict['num_seq'] = pf_num_seq.values()
df = pd.DataFrame(method_score_dict )
# above replaces following
#df = pd.DataFrame( {'LADER_clean': LADER_clean_AUTPR,'ER_clean': ER_clean_AUTPR, 'ER_coup': ER_coup_AUTPR, 'ER': ER_AUTPR, 'MF':MF_AUTPR, 'PLM':PLM_AUTPR,'BM':bm_str, 'num_seq':pf_num_seq.values() } )

df['num_seq_range'] = pd.cut(x=df['num_seq'], bins=num_seq_bins )

print(df.head())
method_bm_ranges = []
for indx,method in enumerate(methods):
	method_bm_ranges.append([])	

interval_ranges = pd.Series(df.num_seq_range.unique())
intervals = interval_ranges.sort_values()
#print(intervals)



for num_seq_range in intervals:
	df_temp = df.loc[df['num_seq_range'] ==num_seq_range ]
	#print('%d pfams in range: '%(len(df_temp)),num_seq_range)
	for indx,method in enumerate(methods):
		method_bm_ranges[indx].append(len(df_temp.loc[df['BM']==method]))



N = len(intervals)
ind = np.arange(N) 
width = 1./len(methods)       
for indx,method in enumerate(methods):
	plt.bar(ind + indx*width , method_bm_ranges[indx], width, label=method)

plt.ylabel('Best Method')
if score_type == 'AUTPR':
	plt.title('AUTPR by Number of Sequences')
elif score_type == 'AUC':
	plt.title('AUC by Number of Sequences')

plt.xticks(ind + width / 4, [str(interval) for interval in intervals])
plt.legend(loc='best')
plt.show()
	
	


