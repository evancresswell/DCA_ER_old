import sys
import numpy as np
import pickle
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.meanfield_dca import meanfield_dca
import expectation_reflection as ER
from direct_info import direct_info
from joblib import Parallel, delayed
#========================================================================================
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

msa_outfile = 'pfam_ecc/MSA_%s.fa'%(pfam_id) 

# MF instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    msa_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,
)

# Compute DCA scores 
sorted_DI_mf = mfdca_inst.compute_sorted_DI()

with open('DI/MF/mf_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_mf, f)

