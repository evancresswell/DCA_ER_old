# RUN : singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/dca_er.simg python print_df_info.py <pkl file>

import pandas as pd
import sys,os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
#pd.set_option('display.max_columns', None)


#df_796 = pd.read_pickle('job_ROC_dfs/60721904_796.pkl')
#print(df_796.loc[df_796.Pfam=='PF01032'])

filepath = sys.argv[1]
print("\n\n\nLoading DataFrame: %s \n\n\n"%filepath)
df = pd.read_pickle(filepath)

print('Dataframe Head: ',df.head())
print('DF shape : ',df.shape)
print(df.loc[df.Pfam== 'PF01032'])


print('Beginning and end of DF index: \n',df.index[:10], df.index[-10:])
print('Length of Dataframe index is %d '% len(df.index.tolist()))
print("\n\nDuplicate Rows: ")
df_duplicates = df[df.duplicated(['Pfam'])]
print("Duplicates df size: ", df_duplicates.shape)
pfam = df_duplicates['Pfam'].tolist()[0]
print('consider duplicate example for ', pfam)
print(df.loc[df['Pfam'] == pfam])
print('Dropping Duplicates..' )
df_trimmed = df.drop_duplicates(subset='Pfam')
print("Trimed df: ", df_trimmed.shape)
print(df_trimmed.loc[df_trimmed.Pfam== 'PF01032'])


