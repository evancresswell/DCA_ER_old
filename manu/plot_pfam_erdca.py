import os,sys
import datetime
import numpy as np
on_pc = False
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

from direct_info import sort_di

import ecc_tools as tools
import data_processing as dp

# import inference_dca for mfDCA
from inference_dca import direct_info_dca

"""
NOTES:
"""

pfam_id = sys.argv[1]


# import pydca for plmDCA
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities

data_path = '../../Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
data_path = '/home/eclay/Pfam-A.full'

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'

ref_outfile = '/home/eclay/DCA_ER/biowulf/pfam_ecc/PP_ref_%s.fa'%pfam_id
ref_outfile = '/home/eclay/DCA_ER/biowulf/pfam_ecc/PP_ref_%s_match.fa'%pfam_id
ref_outfile = '/home/eclay/DCA_ER/biowulf/pfam_ecc/PP_ref_%s_range.fa'%pfam_id


print ('Plotting Protein Famility ', pfam_id)
# Load PDB structure 
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

#---------- Pre-Process Structure Data ----------------#
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

# data processing THESE SHOULD BE CREATED DURING DATA GENERATION
ipdb = 0


#---------------------- Load DI -------------------------------------#
print("Unpickling DI pickle files for %s"%(pfam_id))
with open("%sER_DI_%s.pickle"%(er_directory,pfam_id),"rb") as f:
	DI_er = pickle.load(f)
f.close()
with open("%splm_DI_%s.pickle"%(plm_directory,pfam_id),"rb") as f:
	DI_plm = pickle.load(f)
f.close()
with open("%smf_DI_%s.pickle"%(mf_directory,pfam_id),"rb") as f:
	DI_mf = pickle.load(f)
f.close()
print(len(DI_er))
print(len(DI_plm))
print(len(DI_mf))

erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = ref_outfile,
	sorted_dca_scores = DI_er,
	linear_dist = 4,
	contact_dist = 8.0,
)
mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = ref_outfile,
	sorted_dca_scores = DI_mf[:len(DI_er)],
	linear_dist = 4,
	contact_dist = 8.0,
)

plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
	refseq_file = ref_outfile,
	sorted_dca_scores = DI_plm[:len(DI_er)],
	linear_dist = 4,
	contact_dist = 8.0,
)

slick_contact_maps = [ erdca_visualizer, mfdca_visualizer, plmdca_visualizer]
slick_titles = ['ER', 'MF', 'PLM']

slick_contact_maps = [ erdca_visualizer]
slick_titles = ['ER']



# Create subplots
#fig, axes = plt.subplots(nrows=1,ncols=len(slick_contact_maps), sharex='all',figsize=(15,5))

# Plot
for i,slick_map in enumerate(slick_contact_maps):
	fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
	contact_map_data = slick_map.plot_contact_map(ax)
	plt.show()
	#pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

for i,slick_map in enumerate(slick_contact_maps):
	fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
	tpr_data = slick_map.plot_true_positive_rates()
	plt.show()
	#pdf.savefig()  # saves the current figure into a pdf page
	plt.close()



