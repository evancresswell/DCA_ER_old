import os, sys
import datetime
import pickle
import numpy as np

job_list = sys.argv[1] 

s = np.loadtxt(job_list,dtype='str')
id_method = ['PLM', 'ER', 'MF']

for i,job_id in enumerate(s):
	biowulf_output = '%s_job-%s_swarm_ouput.txt'%(id_method[i],job_id)
	biowulf_file = open(biowulf_output, "w")
	biowulf_file.write(str(job_id) + " "+ id_method[i]+"\n")	
	biowulf_file.close()
	os.system('jobhist %s >> %s'%(job_id,biowulf_output))
