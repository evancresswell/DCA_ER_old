import pandas as pd
import os,sys
import glob

dfs = sys.argv[1:]

for df_filename in dfs:
	df = pd.read_pickle(df_filename)	

print(df.keys())
print('Error types:')
print(df['ERR'].unique())
print('\n\n')
for error in df['ERR'].unique():
	print("There are %d %s errors"%(len(df.loc[df['ERR']==error]),error))

print(df.loc[df['ERR']=='Indexing_CT' ]['Pfam'])

