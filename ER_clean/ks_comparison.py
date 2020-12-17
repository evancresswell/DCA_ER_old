import sys,os
os.chdir('/data/cresswellclayec/DCA_ER/biowulf/')
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import combinations
import matplotlib
from matplotlib import pyplot as plt
from numpy import mean
import matplotlib.pylab as pylab
from scipy import stats
import ecc_tools as tools 

comparing = 'AUC'
comparing = 'Score'

df_diff_full =pd.read_pickle('df_pfam-bar.pkl')
bad_guess_pfams = np.load('bad_guess_pfams.npy')

methods = ['ER','PLM','MF']
"""
for method in methods:
	df = df_diff_full.loc[df_diff_full['method']==method]
	scores = df[comparing].values.tolist()
	print(scores,'\n\n',method,' scores')
	print(len(scores))
	print(len(df['Score'].values.tolist()))
"""
for method1,method2 in combinations(methods,2):
	df_1 = df_diff_full.loc[df_diff_full['method']==method1]
	df_2 = df_diff_full.loc[df_diff_full['method']==method2]

	method1_scores = df_1[comparing].values.tolist()
	method2_scores = df_2[comparing].values.tolist()
	
	#print(method2_scores,method2,' scores')

	ks_score, p_val = stats.ks_2samp(method1_scores,method2_scores)

	print(method1,' and ', method2, 'KS score and p-value =',ks_score,p_val)

np.random.seed(42)
plt.hist(df_diff_full.loc[df_diff_full['method']=='ER'][comparing],alpha=.4, density=True, bins=100,label='ER')  # `density=False` would make counts
plt.hist(df_diff_full.loc[df_diff_full['method']=='PLM'][comparing],alpha=.4, density=True, bins=100,label='PLM')  # `density=False` would make counts
plt.hist(df_diff_full.loc[df_diff_full['method']=='MF'][comparing],alpha=.4, density=True, bins=100,label='MF')  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Data')
plt.legend()
plt.show()


