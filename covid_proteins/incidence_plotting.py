import sys,os
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

#========================================================================================
data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'



region = sys.argv[1]
clade = sys.argv[2]
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





region_incidences = np.load('%s_%s_incidence.npy'%(clade,region))

print('Plotting %s Region'%region)

max_cols =  protein_ranges[region][1] - protein_ranges[region][0]

conserved_fracs = 	[tup[1] for tup in region_incidences]
incidences = 		[tup[0] for tup in region_incidences]
print('Region Incidences:\n',incidences)
print('Conserved Fracs:\n',conserved_fracs)

N = len(incidences)
ind = np.arange(N) 
width = 1

for indx,incidence in enumerate(incidences):
	plt.bar(ind[indx], incidence,width)

plt.xlabel('Conserved Percentage')
plt.ylabel('Number of Nucleotide Postions')
plt.title('%s Incidence'%region)

#plt.xticks(ind + width, [str(interval) for interval in intervals])
plt.xticks(ind[::2], ['%.2f'%frac for frac in conserved_fracs[::2]])
plt.legend(loc='best')
plt.ylim((0, max_cols+50))

manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.savefig('%s_%s_incidence.pdf'%(clade,region))
plt.show()

plt.close()

plot_clades = True
if plot_clades:
	#Plot G Clades
	fig, (ax_full, ax_g, ax_gr, ax_gh) = plt.subplots(1, 4, sharey=True,figsize=(20,5))
	fig.suptitle('%s Incidence'%region)
	axes = [ ax_full, ax_g, ax_gr, ax_gh ]
	clades = [ 'Full', 'G', 'GR', 'GH' ]

	for ii,ax in enumerate(axes):

		region_incidences = np.load('%s_%s_incidence.npy'%(clades[ii],region))
		conserved_fracs = 	[tup[1] for tup in region_incidences]
		incidences = 		[tup[0] for tup in region_incidences]

		for indx,incidence in enumerate(incidences):
			ax.bar(ind[indx], incidence,width)
		ax.set_title('%s'%(clades[ii]))
		ax.set_ylim((0, max_cols+50))
		ax.set_xlabel('Conserved Percentage')
		if ii==0:
			ax.set_ylabel('Number of Nucleotide Postions')
		#plt.xticks(ind + width, [str(interval) for interval in intervals])
		ax.set_xticks(ind[::10])
		ax.set_xticklabels( ['%.2f'%frac for frac in conserved_fracs[::10]])

	plt.savefig('%s_incidence.pdf'%(region))
	plt.show()
	plt.close()

	#Plot all Clades
	fig, (ax_full, ax_g, ax_gr, ax_gh, ax_s, ax_v) = plt.subplots(1, 6, sharey=True,figsize=(20,5))
	fig.suptitle('%s Incidence'%region)
	axes = [ ax_full, ax_g, ax_gr, ax_gh, ax_s, ax_v ]
	clades = [ 'Full', 'G', 'GR', 'GH', 'S', 'V' ]

	for ii,ax in enumerate(axes):
		region_incidences = np.load('%s_%s_incidence.npy'%(clades[ii],region))
		conserved_fracs = 	[tup[1] for tup in region_incidences]
		incidences = 		[tup[0] for tup in region_incidences]

		for indx,incidence in enumerate(incidences):
			ax.bar(ind[indx], incidence,width)
		ax.set_title('%s'%(clades[ii]))
		ax.set_ylim((0, max_cols+50))
		ax.set_xlabel('Conserved Percentage')
		if ii==0:
			ax.set_ylabel('Number of Nucleotide Postions')

		#plt.xticks(ind + width, [str(interval) for interval in intervals])
		ax.set_xticks(ind[::10])
		ax.set_xticklabels( ['%.2f'%frac for frac in conserved_fracs[::10]])

	plt.savefig('%s_incidence.pdf'%(region))
	plt.show()







