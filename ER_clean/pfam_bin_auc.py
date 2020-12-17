import sys,os
os.chdir('/data/cresswellclayec/DCA_ER/biowulf/')
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import permutations
import matplotlib
from matplotlib import pyplot as plt
from numpy import mean
import matplotlib.pylab as pylab

xtick_params = {'xtick.labelsize':'small'}
pylab.rcParams.update(xtick_params)
print('using backend: ',matplotlib.get_backend())

logging_x = False
logging_x = True

method_list = ['ER','PLM','MF']
method_list = ['coupER','covER','ER','PLM','MF']
print('\n\n\nBinning AUC for  methods: ',method_list)
print('\n\n\n')

#------- load dataframes for plotting and binning -------#
tp_file = "df_TP_method_summary.pkl"
fp_file = "df_FP_method_summary.pkl"
p_file = "df_P_method_summary.pkl"
pr_file = "df_PR_method_summary.pkl"
aupr_file = "df_AUPR_method_summary.pkl"
auc_file = "df_AUC_method_summary.pkl"

df_TP = pd.read_pickle(tp_file)
df_FP = pd.read_pickle(fp_file)
df_P = pd.read_pickle(p_file)
df_PR = pd.read_pickle(pr_file)
df_AUPR = pd.read_pickle(aupr_file)
df_AUC = pd.read_pickle(auc_file)

#df_diff = df_AUPR.copy()
df_AUC_diff = df_AUC.copy()
df_AUPR_diff = df_AUPR.copy()

#print(df_AUPR_diff.head())

df_diff = df_AUC_diff.copy()
df_diff_full = df_AUC_diff.copy()

df_diff_full = df_diff_full[df_diff_full['method'].isin(method_list)]
df_diff = df_diff[df_diff['method'].isin(method_list)]

if logging_x:
	df_diff['log_num_seq']= np.log(df_diff['num_seq'])
	df_diff_full['log_num_seq']= np.log(df_diff_full['num_seq'])

#--------------------------------------------------------#



df_diff = df_diff_full.copy()
df_diff = df_diff.loc[df_diff['AUC']>.5]

print(df_diff_full.columns)
print(df_diff.columns)
print('\n\n')
# Seaborn bar plt with numseq
#print("plotting with columns: ",df_AUPR_diff.columns)


#-----------------------------------------------------------------#
#----------- Plot Method Counts across Sequence Ranges -----------#
#-----------------------------------------------------------------#
show_plot = False
show_plot = True

f, axes = plt.subplots(1, 2,sharey=True)
n_bins = 100
method_color={'covER':'purple','coupER':'red','ER':'green','PLM':'orange','MF':'blue'}
for method in method_list:
	x = df_diff_full.loc[df_diff_full['method']==method]['AUC'].to_list()
	axes[0].hist(x, bins=n_bins,edgecolor='black',color=method_color[method],alpha=.2,label=method)
	axes[0].legend()
	
for method in method_list:
	x = df_diff.loc[df_diff['method']==method]['AUC'].to_list()
	axes[1].hist(x, bins=n_bins,edgecolor='black',color=method_color[method],alpha=.2,label=method)
	axes[1].legend()
	
axes[0].set_title('Pfam Count  (All Pfam)')
axes[1].set_title('Pfam Count (AUC >.5)')
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
f.set_size_inches((11, 8.5), forward=False)
plt.savefig('method_bin_AUC.pdf',dpi=500)
if show_plot:
	plt.show()
plt.close()

if 0:

	sns.set(style="darkgrid")

	# set axes
	x,y,hue = 'auc_range', 'method','method' 
	value_y = 'AUC'
	value_x = 'AUC'
	value_hue = 'Method'

	for method in method_list:
		print('MIN %s AUC '%method,min(df_diff_full.loc[df_diff_full['method']==method]['AUC']))
		print('MEAN %s AUC '%method, np.mean(df_diff_full.loc[df_diff_full['method']==method]['AUC']))





	#---------- Group ---------#
	# group sequence ranges by best_method and get pfam counts
	#df_count= (df_diff[x].groupby(df_diff[hue]).value_counts().rename(y).reset_index())
	#df_count_full= (df_diff_full[x].groupby(df_diff_full[hue]).value_counts().rename(y).reset_index())
	#print(df_seq_range_count_full)
		
	if logging_x:
		df_diff['log_auc']= np.log(df_diff['AUC'])
		df_diff_full['log_auc']= np.log(df_diff_full['AUC'])

	df_diff['auc_range'] = pd.cut(df_diff['AUC'],np.arange(min(df_diff['AUC']),max(df_diff['AUC']),step=.05))
	df_diff_full['auc_range'] = pd.cut(df_diff_full['AUC'],np.arange(min(df_diff_full['AUC']),max(df_diff_full['AUC']),step=.1))
	print(df_diff['auc_range'].unique())
	#---------- Group ---------#
	# group sequence ranges by best_method and get pfam counts
	df_auc_range_count= (df_diff[x].groupby(df_diff[hue]).value_counts())
	df_auc_range_count_full= (df_diff_full[x].groupby(df_diff_full[hue]).value_counts())
	df_auc_range_count= (df_diff[x].groupby(df_diff[hue]).value_counts().rename(value_hue).reset_index())
	df_auc_range_count_full= (df_diff_full[x].groupby(df_diff_full[hue]).value_counts().rename(value_hue).reset_index())

	print(df_auc_range_count)

	#print(df_seq_range_count_full)

	# For the first plot only keep sequence number ranges where there are > 10 proteins
	#auc_order = df_auc_range_count_full.unique()
	auc_order = df_auc_range_count_full.loc[df_auc_range_count_full['Method']>10]['auc_range'].sort_values().unique()
	#print('only plotting sequence number ranges with counts > 10: \n',auc_order)

	# get total and method count ranges
	total = float(len(df_auc_range_count)) # one person per row 
	full_total = float(len(df_auc_range_count_full)) # one person per row 
	print('full count: ',full_total, ' vs >.5 count: ',total)
	#--------------------------#


	#--------------------------#



	# --plot mean value-- #
	f, axes = plt.subplots(1, 2,sharey=True)
	sns.barplot(x=x, y=value_hue, hue=hue,order = auc_order, data=df_auc_range_count_full,ax=axes[0])
	sns.barplot(x=x, y=value_hue, hue=hue,order = auc_order, data=df_auc_range_count,ax=axes[1])
	for ax in axes:
		ax.set_ylabel('Protein Count')
		for label in ax.xaxis.get_ticklabels()[0::2]:
			label.set_visible(False)

	axes[1].set_title('Pfam Count (AUC >.5)')
	axes[0].set_title('Pfam Count  (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_count_summary.pdf',dpi=500)
	if show_plot:
		plt.show()
	plt.close()



#-----------------------------------------------------------------#


