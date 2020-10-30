#	RUN:
#		singularity exec -B /data/cresswellclayec/DCA_ER/ /data/cresswellclayec/DCA_ER/dca_er.simg python concat_dfs.py <JOBIDA> <METHOD>
#			- this creates dataframe for full swarm simulation: ../<METHOD>_<JOBID>_full.pkl
import pandas as pd
import os,sys
import glob


pfam_txt_files =  glob.glob('DI/PF*.txt')


testing = False
testing = True
if testing:
	# make smaller dataframes
	print("\n\nMAKING TEST SIZE DATAFRAME\n\n")
	pfam_txt_files = pfam_txt_files[:50]

pf_score = {}
pf_num_seq = {}
for i,filename in enumerate(pfam_txt_files):
	f = open(filename, "r")
	f_input = f.read().split()
	pfam_id = f_input[0]
	pf_score[pfam_id] = [float(f_input[1]),float(f_input[2]),float(f_input[3])]
	pf_score[pfam_id] = int(f_input[4])
	if testing:
		print(pfam_id) 
		print(f_input)
		print(pf_score[pfam_id])

