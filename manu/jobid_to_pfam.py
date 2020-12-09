import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool

#-----------------------------------------------------------------------------------#
#------------------------------- Create List of Pfams ------------------------------#
#-----------------------------------------------------------------------------------#

# Jobload info from text file 
filename = sys.argv[1]
file_out = sys.argv[2]

#if os.path.exists(filepath[:-4]+'.pkl'):
#	df = pd.read_pickle(filepath[:-4]+'.pkl')
#	print(df)
#else:

# Read in DataFrame
with open(filename) as f:
	swarm_jobs = f.readlines()
f.close()
swarm_jobs = [job.strip(' \n') for job in swarm_jobs]

#print(df)

# Iterate Through Jobs and add to DataFrame
swarm_pfams = []
for i,job_id in enumerate(swarm_jobs):
	with open("swarm_output/swarm_%s.o"%(job_id)) as f:
		families = re.findall(r'PF+\d+',f.read())
	swarm_pfams += families
		
print('Jobs:' , swarm_pfams)

if 0:
	f = open(file_out,'w')
	for pfam in swarm_pfams:
	    f.write('module load singularity; ')    
	    f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER_coupling.py %s\n'%(pfam))    
	f.close()
if 1:
	f = open(file_out,'w')
	for pfam in swarm_pfams:
	    f.write('module load singularity; ')    
	    f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER_cov_coupling.py %s\n'%(pfam))    
	f.close()

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


