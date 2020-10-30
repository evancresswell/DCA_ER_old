import os
import data_processing as dp
import numpy as np
from subtract_lists import subtract_lists
from joblib import Parallel, delayed

pfam_list = np.loadtxt('pfam_DIs.txt',dtype='str')

f = open('gen_scores.swarm','w')
for pfam in pfam_list:
	f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python gen_DI_scores.py %s\n'%(pfam))    
f.close()


