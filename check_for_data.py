import os,sys
import numpy as np
#--------------------------------------------------------------#


pfam_id_list = "../../pfam_full_list.txt"

s = np.loadtxt(pfam_id_list,dtype='str')

missing_files = open("missing_er_pickle_files.txt","w")
print("Checking current directory for Protein data from %s\n",pfam_id_list)
for pfam_id in s:
	if not os.path.exists('er_DI_%s.pickle'%pfam_id):
		missing_files.write("%s\n"% pfam_id)
missing_files.close()

