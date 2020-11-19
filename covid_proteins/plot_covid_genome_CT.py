import sys,os
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
#matplotlib.rcParams['text.usetex'] = True
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

#========================================================================================
data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'
# TO RUN: 		singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python plot_covid_genome_CT.py cov_genome_DI.pickle cov_genome_DP.pickle
# FOR PROCESSED RUN: 	singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python plot_covid_genome_CT.py cov_full_genome_DI_processed.pickle cov_genome_DP.pickle


def distance_restr_sortedDI(site_pair_DI_in, s_index=None):
	#print(site_pair_DI_in[:10])
	restrained_DI= dict()
	for site_pair, score in site_pair_DI_in:
		# if s_index exists re-index sorted pair
		if s_index is not None:
			pos_0 = s_index[site_pair[0]]
			pos_1 = s_index[site_pair[1]]
			#print('s_index positions: (%d, %d)'%(pos_0,pos_1))
		else:
			pos_0 = site_pair[0]
			pos_1 = site_pair[1]
		indices = (pos_0 , pos_1)

		if abs(pos_0- pos_1)<2:
			restrained_DI[indices] = 0
		else:
			restrained_DI[indices] = score
	sorted_DI  = sorted(restrained_DI.items(), key = lambda k : k[1], reverse=True)
	print(sorted_DI[:10])
	return sorted_DI

def delete_sorted_DI_duplicates(sorted_DI):
	temp1 = []
	print(sorted_DI[:10])
	print(len(sorted_DI))
	DI_out = dict() 
	counter = 0
	for (a,b), score in sorted_DI:
		counter = counter + 1
		print('pair %d of %d '%(counter, len(sorted_DI)))
		if (a,b) not in temp1 and (b,a) not in temp1: #to check for the duplicate tuples
			temp1.append(((a,b)))
			if a>b:
				DI_out[(b,a)]= score
			else:
				DI_out[(a,b)]= score
	print('Sorting DI values')
	DI_out = sorted(DI_out.items(), key = lambda k : k[1], reverse=True)
	#DI_out.sort(key=lambda x:x[1],reverse=True) 
	print(DI_out[:10])
	return DI_out 



def closest(lst, K): 
      
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 
      

#------------------------------------------------------------------------------------------------------------------#
#------------------------------------- Post-Process and Plot Genome Contact DI ------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

pf_dict_file = root_dir + 'cov_genome_DP.pickle'
pf_dict_file = root_dir + sys.argv[2]
# Load Simulation Files
with open(pf_dict_file, 'rb') as f:
	pf_dict = pickle.load(f)
f.close()


s_index  = pf_dict['s_index']
s_ipdb  = pf_dict['s_ipdb']
cols_removed  = pf_dict['cols_removed']



post_processing = False
if post_processing:

	cov_DI_file = root_dir+'cov_genome_DI.pickle'
	cov_DI_file = root_dir+sys.argv[1]
	with open(cov_DI_file, 'rb') as f:
	    er_gen_DI = pickle.load( f)
	f.close()

	print('\n\n#--------------------------- Plotting Covid Genome Contacts ------------------------#')
	print('%d columns removed'%(len(cols_removed)))
	print('%d columns left'%(len(s_index)))
	print('#-----------------------------------------------------------------------------------#\n')

	print('\n#-----------------------------------------------------------------------------------#')
	print('Un-Processed DI')
	print(er_gen_DI[:10])

	print('\nSorting DI\nRestraining guesses Linear Distance')
	print('Sorted-DR DI')
	sorted_DI = distance_restr_sortedDI(er_gen_DI, s_index)
	print(sorted_DI[:10])

	print('\nDeleting DI Duplicates')
	#sorted_DI = delete_sorted_DI_duplicates(sorted_DI)
	print('Final DI:')
	print(sorted_DI[:10])

	import re
	cov_DI_file = re.sub('.pickle','',cov_DI_file)
	print(cov_DI_file)
	cov_DI_file += '_processed.pickle'
	print('saving  processed pickle file: ', cov_DI_file)
	with open(cov_DI_file, 'wb') as f:
	    pickle.dump(sorted_DI, f)
	f.close()


	print('#-----------------------------------------------------------------------------------#\n')
else:
	with open(root_dir+sys.argv[1], 'rb') as f:
		sorted_DI = pickle.load( f)
	f.close()
	print('\n#-----------------------------------------------------------------------------------#')
	print('Processed DI')
	print(sorted_DI[:10])
	print('#-----------------------------------------------------------------------------------#\n')




print('\n#-----------------------------------------------------------------------------------#')
print('Plotting Regular Distance Map')
print('We have predictions for %d Positions'%len(s_index))
print('#-----------------------------------------------------------------------------------#\n')
# https://www.gisaid.org/epiflu-applications/hcov-19-reference-sequence/
protein_ranges = {}				#  buffer of 265 --> [0, 264]
protein_ranges['ORF1ab']	= [266,21555] 	#  21290 	# 21289
protein_ranges['S'] 		= [21563,25384] #  3822 	# 3821
protein_ranges['ORF3a'] 	= [25393,26220] #  828 		# 827
protein_ranges['ORF3b'] 	= [25765,26220] #     	 	# 455
protein_ranges['E'] 		= [26245,26472] #  228		# 227 
protein_ranges['M'] 		= [26523,27191] #  669		# 668 
protein_ranges['ORF6']	 	= [27202,27387] #  186		# 185
protein_ranges['ORF7a'] 	= [27394,27759] #  366		# 365
protein_ranges['ORF7b'] 	= [27756,27887] #  132		# 131
protein_ranges['ORF8'] 		= [27894,28259] #  193		# 265
protein_ranges['N'] 		= [28274,29533] #  908		# 1259
protein_ranges['ORF10'] 	= [29558,29674] #  117		# 116
							#  buffer of 229 --> [29015, 29244]

linear_dist = 10
# Smaller Clades
dot_infer = .025
max_infer = .0025

# Full Genome
dot_infer = .1
max_infer = .01


strongest_DI = []
# Generate DI matrix of our predictions only    

di_predict = np.zeros((len(s_index),len(s_index)))
scores = []
indices_i = []
indices_j = []

i_predictions = []
j_predictions = []

#focus_range = (26140,26150) # Focus range for ORF3A V-determinant position  26144
focus_range = (11080,11090) # Focus range for NSP6 V-determinant position  11083 
focus_range = (0,0) # Do not Focus range
focus_ranges = [(240,240),(3036,3036),(14407,14407),(23402,23402),(25562,25562),(28880,28882),(8781,8781),(28143,28143),(11082,11082),(26143,26143)]

for coupling in sorted_DI:
	if coupling[0][0] >= focus_range[0] and coupling[0][0] <=focus_range[1] and coupling[1] > .01:
	#if coupling[0][0] > focus_range[0] and coupling[0][0] <focus_range[1]:
		print (coupling)
	if coupling[1] > dot_infer:
		if abs(coupling[0][0] - coupling[0][1]) > linear_dist:
			strongest_DI.append(coupling)
			#i_predictions.append(coupling[0][0])
			#j_predictions.append(coupling[0][1])
			i_predictions.append(np.where(s_index==coupling[0][0]))
			j_predictions.append(np.where(s_index==coupling[0][1]))
			i_predictions.append(np.where(s_index==coupling[0][1]))
			j_predictions.append(np.where(s_index==coupling[0][0]))



	di_predict[np.where(s_index==coupling[0][0]),np.where(s_index==coupling[0][1])] = coupling[1]
	di_predict[np.where(s_index==coupling[0][1]),np.where(s_index==coupling[0][0])] = coupling[1]
	if coupling[0][0] not in indices_i:
		indices_i.append(coupling[0][0])
	if coupling[0][1] not in indices_j:
		indices_j.append(coupling[0][1])
	scores.append(coupling[1])

print('Max index: (%d, %d)'%(max(indices_i),max(indices_j)))
max_index = max(max(indices_i),max(indices_j))

i_predictions_full = []
j_predictions_full = []

di_full = np.zeros((max(s_index),max(s_index)))
for i,coupling in enumerate(sorted_DI):
	if coupling[1] > dot_infer:
		if abs(coupling[0][0] - coupling[0][1]) > linear_dist:
			i_predictions_full.append(coupling[0][0])
			j_predictions_full.append(coupling[0][1])
			i_predictions_full.append(coupling[0][1])
			j_predictions_full.append(coupling[0][0])

print(i_predictions)
i_predictions = [i[0].tolist()[0] for i in i_predictions if len(i[0].tolist())>0]
j_predictions = [j[0].tolist()[0] for j in j_predictions if len(j[0].tolist())>0]
print(i_predictions)

if 0:
	fig, ax = plt.subplots()
	plt.hist(scores,range= (0.,0.001),bins = 10000)
	plt.ylim((0,2000))
	plt.show()

fig, ax = plt.subplots(figsize=(10,10))
#plt.title('Contact Map')
plt.imshow(di_predict,cmap='Greys',origin='lower')
plt.clim(0,max_infer)
plt.xlabel('i')
plt.ylabel('j')
plt.xlim((0,len(s_index)))
plt.ylim((0,len(s_index)))

tick_locs = np.arange(0,len(s_index),step=100)
tick_labels = [s_index[loc] for loc in tick_locs] 
ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_locs)
ax.set_yticklabels(tick_labels)

#plt.title('hCoV-19 Genome Interaction Map')
plt.colorbar(fraction=0.045, pad=0.05)
plt.scatter(i_predictions,j_predictions,marker=  'o',color='r',label='>%f'%dot_infer)
plt.legend(loc='upper left')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +ax.get_xticklabels() + ax.get_yticklabels()):
	item.set_fontsize(12)


plt.savefig('hCoV19_full_interactions_poster.pdf')
#plt.show()
plt.close()

zooming = True
if zooming:
	for protein_name in protein_ranges.keys():
		protein_range = protein_ranges[protein_name]
		try:
			index_start = min( i for i in  s_index if i > protein_range[0])
			index_end = max( i for i in  s_index if i < protein_range[1])
		except(ValueError):
			pass
		if index_end - index_start == 0 :
			print(protein_name,' - genome section has no variance in columns (>90% conserved)')
			continue
		index_range = ( np.where(s_index==index_start)[0], np.where(s_index==index_end)[0] )

		# Print top DI pairs for this protein range
		print('\n\n',protein_name)
		for i,coupling in enumerate(sorted_DI):
			if coupling[1] > .1: # should be same as scatter plot limit
				if coupling[0][0] > index_start and coupling[0][0] < index_end:
						print(coupling)
		print('\n\nClade Determinants in ', protein_name)
		for i,coupling in enumerate(sorted_DI):
			if coupling[1] > .1: # should be same as scatter plot limit
				if coupling[0][0] > index_start and coupling[0][0] < index_end:
					for fr in focus_ranges:
						if coupling[0][0] >= fr[0] and coupling[0][0] <= fr[1] :
							print(coupling)
		print('\n\n')	

		fig_zoom, ax_zoom = plt.subplots(figsize=(10,10))
		#plt.title('Contact Map')
		plt.imshow(di_predict,cmap='Greys',origin='lower')
		#plt.clim(0,.001)
		plt.clim(0,max_infer)
		plt.xlabel('i')
		plt.ylabel('j')


		tick_locs = [index_range[0],index_range[1]] 
		tick_labels = [s_index[index_range[0]],s_index[index_range[1]]] 
		ax_zoom.set_xticks(tick_locs)
		ax_zoom.set_xticklabels(tick_labels)
		ax_zoom.set_yticks(tick_locs)
		ax_zoom.set_yticklabels(tick_labels)
		plt.xlim(index_range)
		plt.ylim(index_range)
		#plt.title('hCoV-19 Genome Interaction Map\n%s (%d, %d)'%(protein_name,protein_range[0],protein_range[1]))
		for item in ([ax_zoom.title, ax_zoom.xaxis.label, ax_zoom.yaxis.label] +ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels()):
			item.set_fontsize(12)

		plt.colorbar(fraction=0.045, pad=0.05)
		plt.scatter(i_predictions,j_predictions,marker=  'o',color='r',label = '>%f'%dot_infer)
		plt.legend(loc='upper left')
		plt.savefig('hCoV19_%s _interactions_poster.pdf'%(protein_name))
		#plt.show()
		plt.close()


print('#-----------------------------------------------------------------------------------#\n')

















