from mpl_toolkits.mplot3d import Axes3D
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
from matplotlib import colors as mpl_colors
import ecc_tools as tools

xtick_params = {'xtick.labelsize':'small'}
pylab.rcParams.update(xtick_params)
print('using backend: ',matplotlib.get_backend())

finding_best_method = True
finding_best_method = False
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

	df_diff['best_method'] = 'None'
	df_diff_full['best_method'] = 'None'
	bad_guess_pfams = []

	df_diff_full['Score'] = 0.
	count = 0
	for protein_family in df_diff['Pfam']:
		#print(protein_family)
		#count = count+1

		try:
			df_protein = df_diff.loc[df_diff['Pfam']==protein_family]
			methods = df_protein['method'].unique()
			print(methods)

			for method in methods:
				AUC_method = df_protein.at[df_protein[df_protein['method'] == method].index.tolist()[0],'AUC']
				non_method_df = df_protein.loc[df_protein['method'] != method]
				AUC_other = non_method_df.at[non_method_df[non_method_df['AUC'] \
				== non_method_df['AUC'].max()].index.tolist()[0],'AUC']

				score = (AUC_method - AUC_other) 
				#score = (AUC_max - AUC_min) * np.heaviside(AUC_max,.5)  

				#print(method,' Score = ',AUC_method,' - ' ,AUC_other,' = ', score)

				df_diff_full['Score'].loc[df_diff_full['Pfam']==protein_family,df_diff_full['method']==method] = score


			# get best method
			method_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'method']
			print('best method: ',method_max,' score:',df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'])
			if df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'] > .5:
				df_diff.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
			else:
				#print('for pfam ',protein_family,' method ',method_max,' has has best AUC: ,',df_protein.at[df_protein[df_protein['AUC']== df_protein['AUC'].max()].index.tolist()[0],'AUC'])
				bad_guess_pfams.append(protein_family)
			df_diff_full.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
			#print('df_diff_full[pfam]:\n',df_diff_full.loc[df_diff['Pfam']==protein_family])
			if count>50:
				sys.exit()

		except IndexError:
			print('ERROR in : ',protein_family)
			print(df_diff.loc[df_diff['Pfam']==protein_family])
		#print(df_diff_full.loc[df_diff['Pfam']==protein_family])
	np.save('bad_guess_pfams.npy',np.array(bad_guess_pfams))
	df_diff_full.to_pickle('df_axis_method.pkl')	
else:
	df_diff_full =pd.read_pickle('df_axis_method.pkl')
	bad_guess_pfams = np.load('bad_guess_pfams.npy')

df_diff = df_diff_full.copy()
df_diff = df_diff.loc[df_diff['AUC']>.5]

#print(df_diff_full.columns)
#print(df_diff.columns)

plot_best_method_only = False
if plot_best_method_only:
	df_diff = df_diff.loc[df_diff['best_method'] == df_diff['method']]
	df_diff_full = df_diff_full.loc[df_diff_full['best_method'] == df_diff_full['method']]
#print(df_diff.head())


df_diff['num_seq_range'] = pd.cut(df_diff['num_seq'],np.arange(min(df_diff['num_seq']),max(df_diff['num_seq']),step=10000))
df_diff['seq_len_range'] = pd.cut(df_diff['seq_len'],np.arange(min(df_diff['seq_len']),max(df_diff['seq_len']),step=100))

df_diff_full['num_seq_range'] = pd.cut(df_diff_full['num_seq'],np.arange(min(df_diff_full['num_seq']),max(df_diff_full['num_seq']),step=10000))
df_diff_full['seq_len_range'] = pd.cut(df_diff_full['seq_len'],np.arange(min(df_diff_full['seq_len']),max(df_diff_full['seq_len']),step=100))

# get row counts for each num_seq range
num_seq_ranges = df_diff_full.num_seq_range.unique()




#-----------------------------------------------------------------#
#----------- Plot Method Mean Scores ----------------- -----------#

#-----------------------------------------------------------------#

# set axes
x,y,hue = 'num_seq_range', 'method','best_method' 
value_y = 'AUC'
value_y = 'Score'

df_ER, df_MF, df_PLM, rgba_colors = tools.get_score_df(df_diff_full)

# plot

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.scatter(xs=df_ER['AUC'], ys=df_MF['AUC'], zs=df_PLM['AUC'], color=rgba_colors)
ax.set_ylabel('MF')
ax.set_xlabel('ER')
ax.set_zlabel('PLM')

ax = fig.add_subplot(222)
ax.scatter(x=df_ER['AUC'], y=df_MF['AUC'], color=rgba_colors)
ax.set_ylabel('MF')
ax.set_xlabel('ER')
ax.set_title('ER vs MF')

ax = fig.add_subplot(223)
ax.scatter(x=df_ER['AUC'], y=df_PLM['AUC'], color=rgba_colors)
ax.set_ylabel('PLM')
ax.set_xlabel('ER')
ax.set_title('ER vs PLM')

ax = fig.add_subplot(224)
ax.scatter(x=df_MF['AUC'], y=df_PLM['AUC'], color=rgba_colors)
ax.set_ylabel('PLM')
ax.set_xlabel('MF')
ax.set_title('MF vs PLM')

plt.savefig('axis_score_3D.pdf')

plt.show()
plt.close()




plot_zoom = True
if plot_zoom:
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

	df_ER, df_MF, df_PLM, rgba_colors = tools.get_score_df(df_sr_full)


	fig = plt.figure()
	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(xs=df_ER['AUC'], ys=df_MF['AUC'], zs=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('MF')
	ax.set_xlabel('ER')
	ax.set_zlabel('PLM')

	ax = fig.add_subplot(222)
	ax.scatter(x=df_ER['AUC'], y=df_MF['AUC'], color=rgba_colors)
	ax.set_ylabel('MF')
	ax.set_xlabel('ER')
	ax.set_title('ER vs MF')

	ax = fig.add_subplot(223)
	ax.scatter(x=df_ER['AUC'], y=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('PLM')
	ax.set_xlabel('ER')
	ax.set_title('ER vs PLM')

	ax = fig.add_subplot(224)
	ax.scatter(x=df_MF['AUC'], y=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('PLM')
	ax.set_xlabel('MF')
	ax.set_title('MF vs PLM')

	plt.savefig('axis_score_%d-%d.pdf'%(min(df_sr_full['num_seq']),max(df_sr_full['num_seq'])))

	plt.show()
	plt.close()


	num_seq_ranges = df_sr_full['num_seq_range'].sort_values().unique()

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

	df_ER, df_MF, df_PLM, rgba_colors = tools.get_score_df(df_sr1_full)

	fig = plt.figure()
	ax = fig.add_subplot(221, projection='3d')
	ax.scatter(xs=df_ER['AUC'], ys=df_MF['AUC'], zs=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('MF')
	ax.set_xlabel('ER')
	ax.set_zlabel('PLM')

	ax = fig.add_subplot(222)
	ax.scatter(x=df_ER['AUC'], y=df_MF['AUC'], color=rgba_colors)
	ax.set_ylabel('MF')
	ax.set_xlabel('ER')
	ax.set_title('ER vs MF')

	ax = fig.add_subplot(223)
	ax.scatter(x=df_ER['AUC'], y=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('PLM')
	ax.set_xlabel('ER')
	ax.set_title('ER vs PLM')

	ax = fig.add_subplot(224)
	ax.scatter(x=df_MF['AUC'], y=df_PLM['AUC'], color=rgba_colors)
	ax.set_ylabel('PLM')
	ax.set_xlabel('MF')
	ax.set_title('MF vs PLM')

	plt.savefig('axis_score_%d-%d.pdf'%(min(df_sr1_full['num_seq']),max(df_sr1_full['num_seq'])))

	plt.show()
	plt.close()
