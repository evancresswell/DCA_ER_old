# libraries
from mpl_toolkits.mplot3d import Axes3D
import sys,os
os.chdir('/data/cresswellclayec/DCA_ER/biowulf/')
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

logging_x = True
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
	
	if logging_x:
		df_diff['log_num_seq']= np.log(df_diff['num_seq'])
		df_diff_full['log_num_seq']= np.log(df_diff_full['num_seq'])

	df_diff['num_seq_range'] = pd.cut(df_diff['num_seq'],np.arange(min(df_diff['num_seq']),max(df_diff['num_seq']),step=10000))
	df_diff['seq_len_range'] = pd.cut(df_diff['seq_len'],np.arange(min(df_diff['seq_len']),max(df_diff['seq_len']),step=100))

	df_diff_full['num_seq_range'] = pd.cut(df_diff_full['num_seq'],np.arange(min(df_diff_full['num_seq']),max(df_diff_full['num_seq']),step=10000))
	df_diff_full['seq_len_range'] = pd.cut(df_diff_full['seq_len'],np.arange(min(df_diff_full['seq_len']),max(df_diff_full['seq_len']),step=100))
	
	df_diff['best_method'] = 'None'
	df_diff_full['best_method'] = 'None'
	bad_guess_pfams = []
	for protein_family in df_diff['Pfam']:
		#print(protein_family)
		try:
			df_protein = df_diff.loc[df_diff['Pfam']==protein_family]
			method_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'method']
			if df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'] > .5:
				df_diff.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
			else:
				print('for pfam ',protein_family,' method ',method_max,' has has best AUC: ,',df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC'])
				bad_guess_pfams.append(protein_family)
			df_diff_full.loc[df_diff['Pfam']==protein_family,'best_method'] = method_max
		except IndexError:
			print('ERROR in : ',protein_family)
			print(df_diff.loc[df_diff['Pfam']==protein_family])
	#print(df_diff.loc[df_diff['Pfam']==protein_family])
	np.save('bad_guess_pfams.npy',np.array(bad_guess_pfams))
	df_diff.to_pickle('df_best_method.pkl')	
	df_diff_full.to_pickle('df_best_method_full.pkl')	

	df_diff_full['Score'] = 0.
	for protein_family in df_diff_full['Pfam']:
			df_protein = df_diff_full.loc[df_diff_full['Pfam']==protein_family]
			AUC_max = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].max()].index.tolist()[0],'AUC']
			AUC_min = df_protein.at[df_protein[df_protein['AUC'] == df_protein['AUC'].min()].index.tolist()[0],'AUC']
			score = (AUC_max - AUC_min) * np.heaviside(AUC_max,.5)  
			print(AUC_max, AUC_min, score)
			df_diff_full['Score'][df_diff_full['Pfam']==protein_family] = score

	print('Pfam count: ',len(df_diff_full['Pfam']) )
	print('Removing loser rows')
	df_diff = df_diff.loc[df_diff['best_method'] == df_diff['method']]
	df_diff_full = df_diff_full.loc[df_diff_full['best_method'] == df_diff_full['method']]
	print('Pfam count: ',len(df_diff_full['Pfam']) )
	df_diff_full.to_pickle('df_bm_full.pkl')	

	print(df_diff_full.head())
else:
	df_diff =pd.read_pickle('df_best_method.pkl')
	df_diff_full =pd.read_pickle('df_bm_full.pkl')
	bad_guess_pfams = np.load('bad_guess_pfams.npy')

print(df_diff_full.head())

# get row counts for each num_seq range
num_seq_ranges = df_diff_full.num_seq_range.unique()
num_seq_range_counts = []
for num_seq_range in num_seq_ranges:
	num_seq_range_counts.append(df_diff_full.loc[df_diff_full['num_seq_range'] == num_seq_range].count())
#print(num_seq_range_counts)	


#----------- Plot Method Counts across Sequence Ranges -----------#
# set axes
x,y,hue,z = 'log_num_seq', 'seq_len','best_method','Score' 

#print(df_diff_full['Score'])

scores = df_diff_full['Score'].values.tolist()
multiplier = 1./max(scores)
scores = [score * multiplier for score in scores]
print(max(scores))
print(min(scores))

color_dict = {'ER':'blue','PLM':'green','MF':'orange'}
colors = [ color_dict[c] for c in df_diff_full['best_method'].values.tolist() ] 
#cmap = colors.LinearSegmentedColormap.from_list('incr_alpha', [(0, (*colors.to_rgb(c),0)), (1, c)])
rgba_colors = np.zeros((len(scores),4))
rgba_colors[:,0:3] = [ mpl_colors.to_rgb(c) for c in colors ]  
rgba_colors[:,3] = scores
#print(rgba_colors)

# plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(df_diff_full[x], df_diff_full[y], df_diff_full[z], color=rgba_colors)
#plt.savefig('bm_surface_3D.pdf')
#plt.show()
#plt.close()


#print(scores)
#print(len(df_diff_full))

df_diff_full['alpha'] =  scores

alphas = df_diff_full.alpha.sort_values()
cmap_bool = True
if cmap_bool:
	sizes = df_diff_full['Score'].values.tolist()
	sizes = [s* 100 for s in sizes] 
	#print(sizes)
	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111 )
	ax.scatter(df_diff_full[x], df_diff_full[y],s=sizes, color=rgba_colors)
	plt.savefig('bm_surface_alpha.pdf')
	#plt.show()
	#plt.close()
else:
	ax = sns.scatterplot(x=x,y=y,hue=hue,s=scores,  data= df_diff_full[df_diff_full.alpha == alphas[0]] ,alpha=alphas[0], hue_order = ['ER','MF','PLM'])
	for i,alpha in enumerate(alphas[1:]):
		print(i, ' of ', len(alphas[1:]))
		sns.scatterplot(x=x,y=y,hue=hue,s=scores,  data= df_diff_full[df_diff_full.alpha == alpha] ,alpha=alpha, hue_order = ['ER','MF','PLM'],ax=ax)
	plt.savefig('bm_surface_alpha_ordered.pdf')
plt.show()
plt.close

print(max(df_diff_full['Score']))
print(min(df_diff_full['Score']))
print(np.mean(df_diff_full['Score']))

#df_diff_ordered = df_diff_full.sort_values(by=['Score','log_num_seq'])
lowest_scores = [0.,.05,.1,.15,.2,.25,.3]
f, axes = plt.subplots(1,len(lowest_scores))
for i,ax in enumerate(axes):
	print('lowest score allowed: ',lowest_scores[i])
	df_ax = df_diff_full.loc[df_diff_full['Score']>lowest_scores[i]]
	sns.scatterplot(x=x, y=y, hue=hue, s=sizes, data= df_ax, hue_order = ['ER','MF','PLM'], ax=ax)			
	ax.set_title('Score > %f'%(lowest_scores[i]))
plt.show()

plt.close()


sns.scatterplot(x=x,y=y,hue=hue,s=scores,data= df_diff_full, hue_order = ['ER','MF','PLM'])
plt.savefig('best_method_surface.pdf')
plt.show()
plt.close()
df_ER_MF = df_diff_full.loc[df_diff_full['best_method'] != 'PLM']
sns.scatterplot(x=x,y=y,hue=hue,s=scores,data= df_ER_MF, hue_order = ['ER','MF','PLM'])
plt.savefig('ER_MF_surface.pdf')
plt.show()


