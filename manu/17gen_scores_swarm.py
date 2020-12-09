import sys,os
import data_processing as dp
import numpy as np
from subtract_lists import subtract_lists
from joblib import Parallel, delayed

file_in = sys.argv[1] 
#file_in = 'pfam_DIs.txt'
pfam_list = np.loadtxt(file_in,dtype='str')

f = open('gen_scores.swarm','w')
for pfam in pfam_list:
	f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_regularized.simg python gen_DI_scores.py %s\n'%(pfam))    
f.close()


