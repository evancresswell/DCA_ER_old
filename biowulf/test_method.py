import sys,os
import pandas as pd
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt

tp_file = "TP_df.pkl"
fp_file = "FP_df.pkl"
p_file = "P_df.pkl"
pr_file = "PR_df.pkl"
auc_file = "AUC_df.pkl"
aupr_file = "AUPR_df.pkl"

df_TP 	= pd.read_pickle(tp_file)
df_FP 	= pd.read_pickle(fp_file)
df_P 	= pd.read_pickle(p_file)
df_PR 	= pd.read_pickle(pr_file)
df_AUPR = pd.read_pickle(aupr_file)
df_AUC 	= pd.read_pickle(auc_file)


print('\n\n')
print('AUC and TP shapes: ',df_AUC.shape,' ', df_TP.shape)
df_AUC = df_AUC.loc[df_AUC.index.isin(df_P.index)]
print("Remove indices in AUC not in TP")
print('AUC and TP shapes: ',df_AUC.shape,' ', df_TP.shape)

#------------------------ MF > ER -------------------------------------#
print('\n\n')
print("Consider when MF does better than ER: ")
print()

mf_TP = df_TP.loc[df_AUC["MF"] > df_AUC["ER"]]
mf_FP = df_FP.loc[df_AUC["MF"] > df_AUC["ER"]]
mf_P = df_P.loc[df_AUC["MF"] > df_AUC["ER"]]
mf_AUC = df_AUC.loc[df_AUC["MF"] > df_AUC["ER"]]

print('AUC and TP shapes: ',mf_AUC.shape,' ', mf_TP.shape)

# Select only ROWs where MF > ER AND AUC > .5
mf_TP = mf_TP.loc[mf_AUC["MF"] >.5]
mf_FP = mf_FP.loc[mf_AUC["MF"] > .5]
mf_P = mf_P.loc[mf_AUC["MF"] > .5]
mf_AUC = mf_AUC.loc[mf_AUC["MF"] > .5]

print('AUC and TP shapes: (AUC > .5)',mf_AUC.shape,' ', mf_TP.shape)

for index,row in mf_FP.iterrows():
	plt.plot(row["MF"],mf_TP.loc[index,"MF"], color='r',alpha=.01)
	plt.plot(row["ER"],mf_TP.loc[index,"ER"], color='b',alpha=.01)
plt.plot(mf_FP["ER"].mean(),mf_TP["ER"].mean(), linewidth=2.5,label="ER")
plt.plot(mf_FP["MF"].mean(),mf_TP["MF"].mean(), linewidth=2.5,label="MF")
plt.legend(loc="lower right")
plt.title("  MF (red)  >  ER (blue)")
plt.savefig("MF_bt_ER_ROC.pdf")
plt.show()
plt.close()


#----------------------------------------------------------------------#


#------------------------ MF > ER -------------------------------------#
print('\n\n')
print("Consider when ER does better than MF: ")
print()

er_TP = df_TP.loc[df_AUC["ER"] > df_AUC["MF"]]
er_FP = df_FP.loc[df_AUC["ER"] > df_AUC["MF"]]
er_P = df_P.loc[df_AUC["ER"] > df_AUC["MF"]]
er_AUC = df_AUC.loc[df_AUC["ER"] > df_AUC["MF"]]

print('AUC and TP shapes: ',er_AUC.shape,' ', er_TP.shape)

# Select only ROWs where ER > ER AND AUC > .5
er_TP = er_TP.loc[er_AUC["ER"] >.5]
er_FP = er_FP.loc[er_AUC["ER"] > .5]
er_P = er_P.loc[er_AUC["ER"] > .5]
er_AUC = er_AUC.loc[er_AUC["ER"] > .5]

print('AUC and TP shapes: (AUC > .5)',er_AUC.shape,' ', er_TP.shape)


for index,row in er_FP.iterrows():
	plt.plot(row["MF"],er_TP.loc[index,"MF"], color='r',alpha=.01)
	plt.plot(row["ER"],er_TP.loc[index,"ER"], color='b',alpha=.01)
plt.plot(er_FP["ER"].mean(),er_TP["ER"].mean(),linewidth=2.5,label="ER")
plt.plot(er_FP["MF"].mean(),er_TP["MF"].mean(),linewidth=2.5,label="MF")
plt.legend(loc="lower right")
plt.title("ER (blue)  >  MF (red)")
plt.savefig("ER_bt_MF_ROC.pdf")
plt.show()
plt.close()
#----------------------------------------------------------------------#



