import sys,os
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from numpy import mean

finding_best_method = True
finding_best_method = False
	
show_plot = False
show_plot = True
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

	df_diff_full = df_AUC_diff.copy()

	#remove PLM
	df_diff_full = df_diff_full.loc[df_diff_full['method']!='PLM']

	df_diff_full['best_method'] = 'None'
	bad_guess_pfams = []

	df_diff_full['Score'] = 0.

	for protein_family in df_diff_full['Pfam']:
		#print(protein_family)


		try:
			df_protein = df_diff_full.loc[df_diff_full['Pfam']==protein_family]
			methods = df_protein['method'].unique()
			print(methods)

			for method in methods:
				AUC_method = df_protein.at[df_protein[df_protein['method'] == method].index.tolist()[0],'AUC']
				non_method_df = df_protein.loc[df_protein['method'] != method]
				AUC_other = non_method_df.at[non_method_df[non_method_df['AUC'] \
				== non_method_df['AUC'].max()].index.tolist()[0],'AUC']

				score = (AUC_method - AUC_other) 
				#score = (AUC_max - AUC_min) * np.heaviside(AUC_max,.5)  

				print(method,' Score = ',AUC_method,' - ' ,AUC_other,' = ', score)

				df_diff_full['Score'].loc[df_diff_full['Pfam']==protein_family,df_diff_full['method']==method] = score


			# get best method
			method_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'method']
			if df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'] > .5:
				df_diff_full.loc[df_diff_full['Pfam']==protein_family,'best_method'] = method_max
			else:
				#print('for pfam ',protein_family,' method ',method_max,' has has best AUC: ,',df_protein.at[df_protein[df_protein['AUC']== df_protein['AUC'].max()].index.tolist()[0],'AUC'])
				bad_guess_pfams.append(protein_family)
			df_diff_full.loc[df_diff_full['Pfam']==protein_family,'best_method'] = method_max

		except IndexError:
			print('ERROR in : ',protein_family)
			print(df_diff_full.loc[df_diff_full['Pfam']==protein_family])
		#print(df_diff_full.loc[df_diff['Pfam']==protein_family])
	np.save('bad_guess_pfams.npy',np.array(bad_guess_pfams))
	df_diff_full.to_pickle('df_ER-MF.pkl')	
else:
	df_diff_full =pd.read_pickle('df_ER-MF.pkl')
	bad_guess_pfams = np.load('bad_guess_pfams.npy')


#------------------------------------------- Seaborn bar plt -------------------------------------------#
#print("plotting with columns: ",df_AUPR_diff.columns)
df_diff = df_diff_full.copy()
df_diff = df_diff.loc[df_diff['best_method'] == df_diff['method']]
df_diff_full = df_diff_full.loc[df_diff_full['best_method'] == df_diff_full['method']]
print(df_diff.head())


df_diff['num_seq_range'] = pd.cut(df_diff['num_seq'],np.arange(min(df_diff['num_seq']),max(df_diff['num_seq']),step=10000))
df_diff['seq_len_range'] = pd.cut(df_diff['seq_len'],np.arange(min(df_diff['seq_len']),max(df_diff['seq_len']),step=100))

df_diff_full['num_seq_range'] = pd.cut(df_diff_full['num_seq'],np.arange(min(df_diff_full['num_seq']),max(df_diff_full['num_seq']),step=10000))
df_diff_full['seq_len_range'] = pd.cut(df_diff_full['seq_len'],np.arange(min(df_diff_full['seq_len']),max(df_diff_full['seq_len']),step=100))



num_seq_ranges = df_diff_full.num_seq_range.unique()
seq_len_ranges = df_diff_full.seq_len_range.unique()
num_seq_range_counts = []
seq_len_range_counts = []


# get row counts for each num_seq range
for num_seq_range in num_seq_ranges:
	num_seq_range_counts.append(df_diff_full.loc[df_diff_full['num_seq_range'] == num_seq_range].count())
for seq_len_range in seq_len_ranges:
	seq_len_range_counts.append(df_diff_full.loc[df_diff_full['seq_len_range'] == seq_len_range].count())

#print(num_seq_range_counts)	


#-----------------------------------------------------------------#
#----------- Plot Method Counts across Sequence Ranges -----------#
#-----------------------------------------------------------------#
show_plot = False
show_plot = True
sns.set(style="darkgrid")

# set axes
x,y,hue = 'num_seq_range', 'method','best_method' 
value_y = 'AUC'
value_y = 'Score'



#---------- Plotting --------------#

#---------- Group ---------#
# group sequence ranges by best_method and get pfam counts
df_seq_range_count= (df_diff[x].groupby(df_diff[hue]).value_counts().rename(y).reset_index())
df_seq_range_count_full= (df_diff_full[x].groupby(df_diff_full[hue]).value_counts().rename(y).reset_index())
#print(df_seq_range_count_full)

# For the first plot only keep sequence number ranges where there are > 10 proteins
num_seq_order = df_seq_range_count_full.loc[df_seq_range_count_full['method']>10]['num_seq_range'].sort_values().unique()
print('only plotting sequence number ranges with counts > 10: \n',num_seq_order)

# get total and method count ranges
total = float(len(df_seq_range_count)) # one person per row 
full_total = float(len(df_seq_range_count_full)) # one person per row 
print('full count: ',full_total, ' vs >.5 count: ',total)
#--------------------------#



# --plot mean value-- #
f, axes = plt.subplots(1, 2)
sns.barplot(x=x, y=value_y, hue=hue, data=df_diff_full,order = num_seq_order,estimator=mean,ci='sd',ax=axes[0])
sns.barplot(x=x, y=value_y, hue=hue, data=df_diff,order = num_seq_order,estimator=mean,ci='sd',ax=axes[1])
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
sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_seq_range_count_full,ax=axes[0])
sns.barplot(x=x, y=y, hue=hue,order = num_seq_order, data=df_seq_range_count,ax=axes[1])
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


#---------- Plotting --------------#

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
f, axes = plt.subplots(1, 2)
sns.barplot(x=x, y=value_y, hue=hue,order = num_seq_order, data=df_sr,estimator=mean,ci='sd',ax=axes[0])
sns.barplot(x=x, y=value_y, hue=hue,order = num_seq_order, data=df_sr,estimator=mean,ci='sd',ax=axes[1])
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
f, axes = plt.subplots(1, 2)
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
f, axes = plt.subplots(1, 2)
sns.barplot(x=x, y=value_y, hue=hue,order = num_seq_order, data=df_sr1,estimator=mean,ci='sd',ax=axes[0])
sns.barplot(x=x, y=value_y, hue=hue,order = num_seq_order, data=df_sr1,estimator=mean,ci='sd',ax=axes[1])
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



