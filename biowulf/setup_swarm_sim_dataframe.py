import pandas as pd
import pickle
import sys,os,re
import ecc_tools as tools
import numpy as np
import data_processing as dp
from multiprocessing import Pool

# ------- Create job reference Dataframe ---------------#
# Jobload info from text file 
filepath = sys.argv[1]
#file_out = sys.argv[2]

# Read curated-Jobhist  in DataFrame
df = pd.read_csv(filepath, sep='\s+', engine='python',header=0)
#print(df)

# Polish MemReq Column
df.MemReq = df.MemReq.str.extract(r'(\d+\.+\d+)(GB/node)' ,expand=False) 
df['MemReq'] =pd.to_numeric(df['MemReq'],downcast='float')
df = df.rename(columns={'MemReq': 'GB/node'})

# Polish MemReq Column
df.MemUsed = df.MemUsed.str.extract(r'(\d+\.+\d+)(GB)' ,expand=False) 
df['MemUsed'] =pd.to_numeric(df['MemUsed'],downcast='float')

#print(df)

# Iterate Through Jobs and add to DataFrame
jobs = df.Jobid

for i,job_id in enumerate(jobs):
	try:
		#print(job_id)
		with open("swarm_output/swarm_%s.o"%(job_id)) as f:
			families = re.findall(r'PF+\d+',f.read())
		df = df.append([df.loc[df['Jobid']==job_id]]*(len(families)-1), ignore_index=True)	
		df.loc[df.Jobid == job_id,'Pfam'] = families
	except(FileNotFoundError): 
		print("No swarm output file for %s, assume there are no corresponding DI"%job_id)

print(df)

job_count = len(df.Jobid.unique())
pfam_count = len(df.Pfam.unique())
print("df dataframe has %d jobs with %d Pfams"%(job_count,pfam_count))


# Generate Swarm File
if filepath[-4:] == ".txt":
	job_string = filepath[:-4]
else:
	print("Setup input must be .txt file (curated jobhist output of DI simulation jobid.. See README")
	sys.exit()

df_filename = job_string+'_setup.pkl' 
df.to_pickle(df_filename)

print('Generating swarm file: %s.swarm'%job_string)
f = open('%s.swarm'%job_string,'w')
for job_id in df.Jobid.unique():
	f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/dca_er.simg python gen_ROC_jobID_df.py %s %s\n'%(df_filename,job_id))    
f.close()




