import os
import numpy as np

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

#s = np.vstack([s1,s2])
#s = np.vstack([s,s3])

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')
s = np.loadtxt('pfam_list_0-590.txt',dtype='str')

#n = s.shape[0]
#pfam_list = s[:,0]
pfam_list = s


#--------------------------------------------------------------#
#--------------------------------------------------------------#
# create swarmfiles for each method

f = open('swarm_ER.txt','w')
for pfam in pfam_list:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/biowulf_setup/ /data/cresswellclayec/biowulf_setup/dca_er.simg python 1main_ER.py %s\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()

f = open('swarm_PLM.txt','w')
for pfam in pfam_list:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    #f.write('python 1main_PLM.py %s\n'%(pfam))    
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/biowulf_setup/ /data/cresswellclayec/biowulf_setup/dca_er.simg python 1main_PLM.py %s\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()

f = open('swarm_MF.txt','w')
for pfam in pfam_list:
    #f.write('python 1main_DCA.py %s\n'%(pfam))    
    f.write('module load singularity; ')    
    f.write('singularity exec -B /data/cresswellclayec/biowulf_setup/ /data/cresswellclayec/biowulf_setup/dca_er.simg python 1main_DCA.py %s\n'%(pfam))    
f.close()
#--------------------------------------------------------------#

