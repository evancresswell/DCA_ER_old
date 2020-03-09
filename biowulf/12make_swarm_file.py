import os
import numpy as np
from subtract_lists import subtract_lists

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

#s = np.vstack([s1,s2])
#s = np.vstack([s,s3])

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')
s_er = np.loadtxt('er_swarm.txt',dtype='str')
s_plm = np.loadtxt('plm_swarm.txt',dtype='str')
s_mf = np.loadtxt('mf_swarm.txt',dtype='str')

#n = s.shape[0]
#pfam_list = s[:,0]


#--------------------------------------------------------------#
#--------------------------------------------------------------#
# create swarmfiles for each method

f = open('er.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()

f = open('plm.swarm','w')
for pfam in s_plm:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    #f.write('python 1main_PLM.py %s\n'%(pfam))    
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_PLM.py %s\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()

f = open('mf.swarm','w')
for pfam in s_mf:
    #f.write('python 1main_DCA.py %s\n'%(pfam))    
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_DCA.py %s\n'%(pfam))    
f.close()
#--------------------------------------------------------------#
