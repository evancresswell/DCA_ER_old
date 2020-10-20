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
			f.close()
		families = np.unique(families)

		#print('\n\nSetting up DF for families: ',families,' \nlen: %d\n'%(len(families)-1))
		#print(df.loc[df['Jobid']==job_id],'\n\n')
		df_jobid = pd.DataFrame(np.repeat(df.loc[df['Jobid']==job_id].values,int(len(families)-1),axis=0))
		df_jobid.columns = df.columns
		df = pd.concat([df,df_jobid], ignore_index=True)	

		df.loc[df.Jobid == job_id,'Pfam'] = families
		#print(df.loc[df['Jobid']==job_id],'\n\n')

		# Find MM-Seq
		muscle_error = "PPseq-MSA"
		with open("swarm_output/swarm_%s.o"%(job_id)) as f:
			for line in f:
				if line[:18] == "Calculating DI for":
					current_pfam = line[19:26]
				if line[:len(muscle_error)] == muscle_error:
					df.loc[df.Pfam == current_pfam,'ERR'] = muscle_error 
			
			f.close()
	except(FileNotFoundError): 
		print("No swarm output file for %s, assume there are no corresponding DI"%job_id)

print(df.head())
print("\ndf dataframe has %d jobs with %d Pfams\n\n"%(len(df.Jobid.unique()),len(df.Pfam.unique())))

if df['Pfam'].isnull().values.any():
	print('There are %d missing Pfam rows in dataframe\n\n'%df['Pfam'].isnull().sum())

if 0:
	# Removes all rows for some reason.. trying to remove rows with pfam lt 6	
	print("df dataframe has %d jobs with %d Pfams"%(len(df.Jobid.unique()),len(df.Pfam.unique())))
	print('\n\nThere are %d duplicate rows'%(df.duplicated(subset='Pfam').sum()))
	df = df[df['Pfam'].str.split().str.len().gt(4)]
	print('There are now %d duplicate rows\n\n'%(df.duplicated(subset='Pfam').sum()))
	print(df.head())

if df.duplicated(subset='Pfam').sum() > 0:
	print(df[df.duplicated(subset='Pfam')] )
	print('Removing duplicate rows')
	df = df[~df.duplicated(subset = 'Pfam')]

print("\n\ndf dataframe has %d jobs with %d Pfams\n\n"%(len(df.Jobid.unique()),len(df.Pfam.unique())))

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
	f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/erdca.simg python gen_ROC_jobID_df.py %s %s\n'%(df_filename,job_id))    
f.close()




