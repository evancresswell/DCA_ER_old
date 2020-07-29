import os, sys
import datetime
import pickle
import numpy as np

job_id = sys.argv[1]
id_method = ['PLM', 'ER', 'MF']
id_method = 'covER' # ER method with weights initialized with inverse of covariance matrix
id_method = 'coupER' # ER method with weights initialized with inverse of coupling matrix
id_method = sys.argv[2]

biowulf_output = '%s_job-%s_swarm_ouput.txt'%(id_method,job_id)
biowulf_file = open(biowulf_output, "w")
biowulf_file.write(str(job_id) + " "+ id_method+"\n")	
biowulf_file.close()
os.system('jobhist %s >> %s'%(job_id,biowulf_output))

