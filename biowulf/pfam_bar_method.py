# RUN : singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/dca_er.simg python pfam_bar_method.py

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

finding_best_method = False
finding_best_method = True
method_list = ['covER','coupER','ER','PLM','MF']
method_list = ['covER','PLM','MF']
method_list = ['ER','PLM','MF']
method_list = ['ER','MF']
print('\n\n\nFinding best method of: ',method_list)
print('\n\n\n')

if finding_best_method:
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
	print("df_diff_full shape: " ,df_diff_full.shape	)
	print("number of pfams: " ,len(df_diff_full['Pfam'].unique()))	
	print("df_diff_full methods: " ,df_diff_full['method'].unique())	
	df_diff_full = df_diff_full[df_diff_full['method'].isin(method_list)]
	print("df_diff_full shape: (after removing methods)" ,df_diff_full.shape)	
	print("number of pfams: " ,len(df_diff_full['Pfam'].unique()))	
	print("df_diff_full methods: " ,df_diff_full['method'].unique())	
	df_diff = df_diff[df_diff['method'].isin(method_list)]

	df_diff['num_seq_range'] = pd.cut(df_diff['num_seq'],np.arange(min(df_diff['num_seq']),max(df_diff['num_seq']),step=10000))
	df_diff['seq_len_range'] = pd.cut(df_diff['seq_len'],np.arange(min(df_diff['seq_len']),max(df_diff['seq_len']),step=100))

	df_diff_full['num_seq_range'] = pd.cut(df_diff_full['num_seq'],np.arange(min(df_diff_full['num_seq']),max(df_diff_full['num_seq']),step=10000))
	df_diff_full['seq_len_range'] = pd.cut(df_diff_full['seq_len'],np.arange(min(df_diff_full['seq_len']),max(df_diff_full['seq_len']),step=100))
	
	df_diff['best_method'] = 'None'
	df_diff_full['best_method'] = 'None'
	df_diff_full['Score'] = 0.
	df_diff['Score'] = 0.
	bad_guess_pfams = []
	for protein_family in df_diff_full['Pfam']:
		#print(protein_family)
		try:

			#df_protein = df_diff.loc[df_diff['Pfam']==protein_family]
			df_protein = df_diff_full.loc[df_diff_full['Pfam']==protein_family]

			# Use only methods defined in method list
			methods = df_protein['method'].unique()
			print(methods)

			AUC_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC']
			for method in methods:
				# score calculation AUC_meth - AUC_non_meth_max
				AUC_method = df_protein.at[df_protein[df_protein['method'] == method].index.tolist()[0],'AUC']
				non_method_df = df_protein.loc[df_protein['method'] != method]
				AUC_other = non_method_df.at[non_method_df[non_method_df['AUC'] \
				== non_method_df['AUC'].max()].index.tolist()[0],'AUC']
				score = (AUC_method - AUC_other) 

				# score calculation AUC_meth / AUC_max
				score = AUC_method / AUC_max
				#score = (AUC_max - AUC_min) * np.heaviside(AUC_max,.5)  

				print(method,' Score = ',AUC_method,' - ' ,AUC_other,' = ', score)

				df_diff_full['Score'].loc[df_diff_full['Pfam']==protein_family,df_diff_full['method']==method] = score
				df_diff['Score'].loc[df_diff['Pfam']==protein_family,df_diff['method']==method] = score

			# get best method
			method_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'method']
			#print('best method: ',method_max,' score:',df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'])

			if df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'] > .5:
				df_diff.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
			else:
				#print('for pfam ',protein_family,' method ',method_max,' has has best AUC: ,',df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'])
				bad_guess_pfams.append(protein_family)
				df_diff.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
			df_diff_full.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
		except IndexError:
			print('ERROR in : ',protein_family)
			print(df_diff.loc[df_diff['Pfam']==protein_family])
	#print(df_diff.loc[df_diff['Pfam']==protein_family])
	if len(method_list)>3:
		np.save('bad_guess_pfams_allMethods.npy',np.array(bad_guess_pfams))
		df_diff.to_pickle('df_pfam-bar_allMethods.pkl')	
		df_diff_full.to_pickle('df_pfam-bar_full_allMethods.pkl')	
	else:
		np.save('bad_guess_pfams.npy',np.array(bad_guess_pfams))
		df_diff.to_pickle('df_pfam-bar.pkl')	
		df_diff_full.to_pickle('df_pfam-bar_full.pkl')	

else:
	if len(method_list)>3:
		#df_diff =pd.read_pickle('df_pfam-bar.pkl')
		#df_diff_full =pd.read_pickle('df_best_method_full.pkl')
		df_diff_full =pd.read_pickle('df_pfam-bar_full_allMethods.pkl')
		bad_guess_pfams = np.load('bad_guess_pfams_allMethods.npy')
	else:
		df_diff_full =pd.read_pickle('df_pfam-bar_full.pkl')
		bad_guess_pfams = np.load('bad_guess_pfams.npy')


#because this routing and pfam_best_method_surface does not create df_diff with 'Score' Column we must add now
print('df_diff_full shape: ',df_diff_full.shape)
df_diff_full = df_diff_full[df_diff_full['method'].isin(method_list)]
df_diff = df_diff_full.copy()
df_diff = df_diff.loc[df_diff['AUC']>.5]
print("Using ",method_list," method set we have found best methods:" ,df_diff_full['best_method'].unique())
print('df_diff_full shape: ',df_diff_full.shape)

print(df_diff_full.columns)
print(df_diff.columns)
# Seaborn bar plt with numseq
#print("plotting with columns: ",df_AUPR_diff.columns)

plot_best_method_only = True
if plot_best_method_only:
	er_mean = df_diff_full.loc[df_diff_full['method']=='ER']['AUC'].mean()
	mf_mean = df_diff_full.loc[df_diff_full['method']=='MF']['AUC'].mean()
	plm_mean = df_diff_full.loc[df_diff_full['method']=='PLM']['AUC'].mean()
	df_diff = df_diff.loc[df_diff['best_method'] == df_diff['method']]
	df_diff_full = df_diff_full.loc[df_diff_full['best_method'] == df_diff_full['method']]
	print(df_diff.head())

	print("#----------------------------------------------------------------------#")
	print("\n\n\n%d Proteins"%(len(df_diff_full['Pfam'].unique())))
	print('ER best in %d Proteins'%(len(df_diff_full.loc[df_diff_full['method']=='ER'])))
	print('ER mean: %f'%(er_mean))
	print('MF best in %d Proteins'%(len(df_diff_full.loc[df_diff_full['method']=='MF'])))
	print('MF mean: %f'%(mf_mean))
	print('PLM best in %d Proteins\n\n\n'%(len(df_diff_full.loc[df_diff_full['method']=='PLM'])))
	print('PLM mean: %f'%(plm_mean))
	print("#----------------------------------------------------------------------#")

df_diff['num_seq_range'] = pd.cut(df_diff['num_seq'],np.arange(min(df_diff['num_seq']),max(df_diff['num_seq']),step=10000))
df_diff_full['num_seq_range'] = pd.cut(df_diff_full['num_seq'],np.arange(min(df_diff_full['num_seq']),max(df_diff_full['num_seq']),step=10000))

# get row counts for each num_seq range
num_seq_ranges = df_diff_full.num_seq_range.unique()

logging_x = True
logging_x = False
if logging_x:
	log_min =np.log(min(df_diff['num_seq'])) 
	log_max =np.log(max(df_diff['num_seq'])) 
	print('logspace in range: (%f,%f)\n'%(log_min,log_max))
	log_ranges = np.linspace(log_min,log_max,num=5)
	print(log_ranges)
	df_diff['log_num_seq_range'] = pd.cut(np.log(df_diff['num_seq']),log_ranges)
	df_diff_full['log_num_seq_range'] = pd.cut(np.log(df_diff_full['num_seq']),log_ranges)
	print(df_diff['log_num_seq_range'].unique())

#-----------------------------------------------------------------#
#----------- Plot Method Counts across Sequence Ranges -----------#
#-----------------------------------------------------------------#
show_plot = False
show_plot = True
sns.set(style="darkgrid")

# set axes
if logging_x:
	x,y,hue = 'log_num_seq_range', 'method','best_method' 
else:
	x,y,hue = 'num_seq_range', 'method','best_method' 
	x,y,hue = 'seq_len_range', 'method','best_method' 
value_y = 'AUC'
value_y = 'Score'
value_hue = 'method'

print(min(df_diff_full.loc[df_diff_full['method']=='PLM']['Score']))
print(np.mean(df_diff_full.loc[df_diff_full['method']=='PLM']['Score']))


#---------- Plotting --------------#


#---------- Group ---------#
# group sequence ranges by best_method and get pfam counts
df_seq_range_count= (df_diff[x].groupby(df_diff[hue]).value_counts().rename(y).reset_index())
df_seq_range_count_full= (df_diff_full[x].groupby(df_diff_full[hue]).value_counts().rename(y).reset_index())
#print(df_seq_range_count_full)

# For the first plot only keep sequence number ranges where there are > 10 proteins
if not logging_x:
	if x == 'num_seq_range':
		num_seq_order = df_seq_range_count_full.loc[df_seq_range_count_full['method']>10]['num_seq_range'].sort_values().unique()
		print('only plotting sequence number ranges with counts > 10: \n',num_seq_order)
	if x == 'seq_len_range':
		seq_len_order = df_seq_range_count_full.loc[df_seq_range_count_full['method']>10]['seq_len_range'].sort_values().unique()
		print('only plotting sequence number ranges with counts > 10: \n',seq_len_order)
else:
	num_seq_order = df_seq_range_count_full.loc[df_seq_range_count_full['method']>10]['log_num_seq_range'].sort_values().unique()
	print('only plotting sequence number ranges with counts > 10: \n',num_seq_order)



# get total and method count ranges
total = float(len(df_seq_range_count)) # one person per row 
full_total = float(len(df_seq_range_count_full)) # one person per row 
print('full count: ',full_total, ' vs >.5 count: ',total)
#--------------------------#

if 0:
	# --plot summary method scores and counts --#
	f, axes = plt.subplots(1, 2,sharey=True)
	plt.ylim((.8,1.1))
	sns.barplot(x=value_hue, y=value_y, hue=value_hue, data=df_diff_full,order = num_seq_order,estimator=mean,ax=axes[0])
	sns.barplot(x=value_hue,y=value_y, hue=value_hue, data=df_diff,order = num_seq_order,estimator=mean,ax=axes[1])
	plt.ylim((.8,1.1))
	for ax in axes:
		for label in ax.xaxis.get_ticklabels()[0::2]:
		    label.set_visible(False)

	axes[1].set_title('Mean (AUC >.5)')
	axes[0].set_title('Mean  (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_mean-Score_summary.pdf', dpi=500)
	if show_plot:
		plt.show()
	plt.close()
	print(df_seq_range_count_full)

	#-- plot pfam count --#
	f, axes = plt.subplots(1, 2)
	sns.barplot(y=y, hue=hue,order = num_seq_order, data=df_seq_range_count_full,ax=axes[0])
	sns.barplot(y=y, hue=hue,order = num_seq_order, data=df_seq_range_count,ax=axes[1])
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
	#---------------------#

	#-------------------------------------------#


	# --plot mean value-- #
	f, axes = plt.subplots(1, 2,sharey=True)
	plt.ylim((.8,1.1))
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_diff_full,order = num_seq_order,estimator=mean,ax=axes[0])
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_diff,order = num_seq_order,estimator=mean,ax=axes[1])
	plt.ylim((.8,1.1))
	for ax in axes:
		for label in ax.xaxis.get_ticklabels()[0::2]:
		    label.set_visible(False)

	axes[1].set_title('Mean (AUC >.5)')
	axes[0].set_title('Mean  (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_mean-Score.pdf', dpi=500)
	if show_plot:
		plt.show()
	plt.close()
	#---------------------#

#-- plot pfam count --#
f, axes = plt.subplots(1, 2)
if x =='num_seq_range':
	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_seq_range_count_full,ax=axes[0])
	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_seq_range_count_full,ax=axes[0])
if x == 'seq_len_range':
	sns.barplot(x=x, y=y, hue=hue,order = seq_len_order, data=df_seq_range_count,ax=axes[1])
	sns.barplot(x=x, y=y, hue=hue,order = seq_len_order, data=df_seq_range_count,ax=axes[1])
for ax in axes:
	ax.set_ylabel('Protein Count')
	for label in ax.xaxis.get_ticklabels()[0::2]:
		label.set_visible(False)

axes[1].set_title('Pfam Count (AUC >.5)')
axes[0].set_title('Pfam Count  (All Pfam)')

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
f.set_size_inches((11, 8.5), forward=False)
plt.savefig('bm_count.pdf',dpi=500)
if show_plot:
	plt.show()
plt.close()
#---------------------#
#----------------------------------#


zooming_in = False
if zooming_in:
	#---------- Potting --------------#

	#---------- Group ---------#
	# Zoom in on first sequence range
	print("\n\nnew sequence range ",num_seq_ranges[0])
	df_sr = df_diff.copy()
	df_sr = df_sr.loc[df_sr['num_seq_range']==num_seq_ranges[0]]
	df_sr_full = df_diff_full.copy()
	df_sr_full = df_sr_full.loc[df_sr_full['num_seq_range']==num_seq_ranges[0]]
	print('creating new ranges for num_seq from %d to %d"\n\n"'%(min(df_sr['num_seq']),max(df_sr['num_seq'])))

	df_sr['num_seq_range'] = pd.cut(df_sr['num_seq'],np.arange(min(df_sr['num_seq']),max(df_sr['num_seq']),step=1000))
	df_sr['seq_len_range'] = pd.cut(df_sr['seq_len'],np.arange(min(df_sr['seq_len']),max(df_sr['seq_len']),step=10))
	df_sr_full['num_seq_range'] = pd.cut(df_sr_full['num_seq'],np.arange(min(df_sr_full['num_seq']),max(df_sr_full['num_seq']),step=1000))
	df_sr_full['seq_len_range'] = pd.cut(df_sr_full['seq_len'],np.arange(min(df_sr_full['seq_len']),max(df_sr_full['seq_len']),step=10))

	# group sequence ranges byt best_method and get counts
	df_sr_count= (df_sr[x].groupby(df_sr[hue]).value_counts().rename(y).reset_index())
	df_sr_count_full= (df_sr_full[x].groupby(df_sr_full[hue]).value_counts().rename(y).reset_index())

	# Keep sequence number ranges where there are > 10 proteins
	num_seq_order = df_sr_count_full.loc[df_sr_count_full['method']>10]['num_seq_range'].sort_values().unique()
	print('only plotting sequence number ranges with counts > 10: \n',num_seq_order)
	num_seq_ranges = df_sr_full['num_seq_range'].sort_values().unique()
	print('sequence number ranges', num_seq_ranges)
	#--------------------------#



	# --plot mean value-- #
	f, axes = plt.subplots(1, 2,sharey=True)
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_sr_full,order = num_seq_order,estimator=mean,ax=axes[0])
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_sr,order = num_seq_order,estimator=mean,ax=axes[1])
	plt.ylim((.8,1.1))
	for ax in axes:
		for label in ax.xaxis.get_ticklabels()[::2]:
		    label.set_visible(False)

	axes[1].set_title('Mean (AUC >.5)')
	axes[0].set_title('Mean  (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_mean-Score_%d-%d.pdf'%(min(df_sr_full['num_seq']),max(df_sr_full['num_seq'])),dpi=500)
	if show_plot:
		plt.show()
	plt.close()
	#---------------------#


	#-- plot pfam count --#
	f, axes = plt.subplots(1, 2,sharey=True)
	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_sr_count_full,ax=axes[0])
	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_sr_count,ax=axes[1])

	for ax in axes:
		ax.set_ylabel('Protein Count')
		for label in ax.xaxis.get_ticklabels()[::2]:
		    label.set_visible(False)
	axes[1].set_title('Pfam Count (AUC >.5)')
	axes[0].set_title('Pfam Count  (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_count_%d-%d.pdf'%(min(df_sr_full['num_seq']),max(df_sr_full['num_seq'])),dpi=500)

	if show_plot:
		plt.show()
	plt.close()
	#---------------------#
	#----------------------------------#



	#---------- Plotting --------------#

	# Zoom in on first sequence range
	print("\n\nnew sequence range ",num_seq_ranges[0],"\n\n")
	df_sr1 = df_sr.copy()
	df_sr1 = df_sr1.loc[df_sr1['num_seq_range']==num_seq_ranges[0]]
	df_sr1_full = df_sr_full.copy()
	df_sr1_full = df_sr1_full.loc[df_sr1_full['num_seq_range']==num_seq_ranges[0]]
	#print(df_sr1['num_seq'])
	print('creating new ranges for num_seq from %d to %d"\n\n"'%(min(df_sr1['num_seq']),max(df_sr1['num_seq'])))

	df_sr1['num_seq_range'] = pd.cut(df_sr1['num_seq'],np.arange(min(df_sr1['num_seq']),max(df_sr1['num_seq']),step=100))
	df_sr1['seq_len_range'] = pd.cut(df_sr1['seq_len'],np.arange(min(df_sr1['seq_len']),max(df_sr1['seq_len']),step=10))
	df_sr1_full['num_seq_range'] = pd.cut(df_sr1_full['num_seq'],np.arange(min(df_sr1_full['num_seq']),max(df_sr1_full['num_seq']),step=100))
	df_sr1_full['seq_len_range'] = pd.cut(df_sr1_full['seq_len'],np.arange(min(df_sr1_full['seq_len']),max(df_sr1_full['seq_len']),step=10))

	num_seq_ranges = df_sr1_full.num_seq_range.unique()
	print(num_seq_ranges)

	# group sequence ranges byt best_method and get counts
	df_sr1_count= (df_sr1[x].groupby(df_sr1[hue]).value_counts().rename(y).reset_index())
	df_sr1_count_full= (df_sr1_full[x].groupby(df_sr1_full[hue]).value_counts().rename(y).reset_index())


	# Keep sequence number ranges where there are > 10 proteins
	num_seq_order = df_sr1_count_full.loc[df_sr1_count_full['method']>10]['num_seq_range'].sort_values().unique()
	print('only plotting sequence number ranges with counts > 10: \n',num_seq_order)


	# --plot mean value-- #
	f, axes = plt.subplots(1, 2,sharey=True)
	plt.ylim((.8,1.1))
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_sr1_full,order = num_seq_order,estimator=mean,ax=axes[0])
	sns.barplot(x=x, y=value_y, hue=value_hue, data=df_sr1,order = num_seq_order,estimator=mean,ax=axes[1])
	plt.ylim((.8,1.1))
	for ax in axes:
		for label in ax.xaxis.get_ticklabels()[::2]:
		    label.set_visible(False)

	axes[1].set_title('Mean (AUC >.5)')
	axes[0].set_title('Mean (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)

	plt.savefig('bm_mean-Score_%d-%d.pdf'%(min(df_sr1_full['num_seq']),max(df_sr1_full['num_seq'])),dpi=500)

	if show_plot:
		plt.show()
	plt.close()
	#---------------------#

	#-- plot pfam count --#
	f, axes = plt.subplots(1, 2)


	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_sr1_count_full,ax=axes[0])
	sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_sr1_count,ax=axes[1])

	for ax in axes:
		ax.set_ylabel('Protein Count')
		for label in ax.xaxis.get_ticklabels()[::2]:
		    label.set_visible(False)
	axes[1].set_title('Pfam Count (AUC >.5)')
	axes[0].set_title('Pfam Count (All Pfam)')
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	f.set_size_inches((11, 8.5), forward=False)
	plt.savefig('bm_count_%d-%d.pdf'%(min(df_sr1_full['num_seq']),max(df_sr1_full['num_seq'])),dpi=500)

	if show_plot:
		plt.show()
	plt.close()
	#---------------------#

	#----------------------------------#
	#-----------------------------------------------------------------#


