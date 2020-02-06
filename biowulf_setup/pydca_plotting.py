import numpy as np
# import pydca for plmDCA
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities

import pickle

data_path = '../../Pfam-A.full'
pfam_id = 'PF00011'
ref_outfile = 'pfam_ecc/ref_'+pfam_id+'.fa'

sorted_DI_mf = pickle.load(open('DI/MF/mf_DI_%s.pickle'%(pfam_id), 'rb' ) )
sorted_DI_er = pickle.load(open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'rb' ) )
sorted_DI_plm = pickle.load(open('DI/PLM/plm_DI_%s.pickle'%(pfam_id), 'rb' ) )

# Using PYDCA contact mapping module
print("Dimensions of DI Pairs:")
print("ER: ",len(sorted_DI_er))
print("PLM: ",len(sorted_DI_plm))
print("MF: ",len(sorted_DI_mf))

# Load PDB data
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

# Pre-Process Structure Data
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

ipdb = 0
print('seq:',int(pdb[ipdb,1]))

erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
    refseq_file = ref_outfile,
    sorted_dca_scores = sorted_DI_er,
    linear_dist = 4,
    contact_dist = 8.0,
)

mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
    refseq_file = ref_outfile,
    sorted_dca_scores = sorted_DI_mf,
    linear_dist = 4,
    contact_dist = 8.0,
)

plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
    refseq_file = ref_outfile,
    sorted_dca_scores = sorted_DI_plm,
    linear_dist = 4,
    contact_dist = 8.0,
)

er_contact_map_data = erdca_visualizer.plot_contact_map()
mf_contact_map_data = mfdca_visualizer.plot_contact_map()
plm_contact_map_data = plmdca_visualizer.plot_contact_map()

er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
mf_tp_rate_data = mfdca_visualizer.plot_true_positive_rates()
plm_tp_rate_data = plmdca_visualizer.plot_true_positive_rates()
