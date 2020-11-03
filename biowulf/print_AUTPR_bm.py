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



genreating_txt = True
if genreating_txt:
	#pfam_txt_files =  glob.glob('DI/PF*.txt')
	file_in = sys.argv[1] 
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
bm_str = []
for key in pf_scores:
	best_method_indx, score = max(enumerate(pf_scores[key]), key=operator.itemgetter(1))
	#print(pf_score[key])
	#print(best_method_indx)
	
	bm.append(best_method_indx)	
	if best_method_indx == 0:
		best_method = 'ER'
	if best_method_indx == 1:
		best_method = 'MF'
	if best_method_indx == 2:
		best_method = 'PLM'
	bm_str.append(best_method)

print('ER best for %d methods'%bm.count(0))
print('MF best for %d methods'%bm.count(1))
print('PLM best for %d methods'%bm.count(2))

ER_AUTPR = []
MF_AUTPR = []
PLM_AUTPR = []

for key in pf_scores:
	ER_AUTPR.append(pf_scores[key][0])
	MF_AUTPR.append(pf_scores[key][1])
	PLM_AUTPR.append(pf_scores[key][2])


num_seq_bins = np.arange(min(pf_num_seq.values()),max(pf_num_seq.values()),1000)
#print('Divding scores by number of sequences:\n',num_seq_bins)
df = pd.DataFrame( { 'ER': ER_AUTPR, 'MF':MF_AUTPR, 'PLM':PLM_AUTPR,'BM':bm_str, 'num_seq':pf_num_seq.values() } )
df['num_seq_range'] = pd.cut(x=df['num_seq'], bins=num_seq_bins )

print(df.head())
bm_ranges = {}
er_bm_ranges = []
mf_bm_ranges = []
plm_bm_ranges = []

interval_ranges = pd.Series(df.num_seq_range.unique())
intervals = interval_ranges.sort_values()
#print(intervals)

for num_seq_range in intervals:
	df_temp = df.loc[df['num_seq_range'] ==num_seq_range ]
	#print('%d pfams in range: '%(len(df_temp)),num_seq_range)
	bm_ranges[num_seq_range] = [len(df_temp.loc[df['BM']=='ER']),len(df_temp.loc[df['BM']=='MF']),len(df_temp.loc[df['BM']=='PLM'])]
	#print('Best method counts: ',bm_ranges[num_seq_range])

	er_bm_ranges.append(len(df_temp.loc[df['BM']=='ER']))
	mf_bm_ranges.append(len(df_temp.loc[df['BM']=='MF']))
	plm_bm_ranges.append(len(df_temp.loc[df['BM']=='PLM']))


N = len(intervals)
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, er_bm_ranges, width, label='ER')
plt.bar(ind + width, mf_bm_ranges, width,   label='MF')
plt.bar(ind + 2*width, plm_bm_ranges, width,   label='PLM')

plt.ylabel('Best Method')
plt.title('AUTPR by Number of Sequences')

plt.xticks(ind + width / 3, [str(interval) for interval in intervals])
plt.legend(loc='best')
plt.show()
	
	


