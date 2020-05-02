import sys,os
import pandas as pd
import numpy as np

tp_file = "TP_df.pkl"
fp_file = "FP_df.pkl"
p_file = "P_df.pkl"
df_TP = pd.read_pickle(tp_file)
df_FP = pd.read_pickle(fp_file)
df_P = pd.read_pickle(p_file)

#----------------- Precision-Recall and AUPR --------------------#
for method in ["MF","PLM","ER"]:
	PR_list = []
	AUPR_list = []

	df_TP_method = df_TP[method]
	df_FP_method = df_FP[method]
	#print(len(df_FP_method))
	for index,row in enumerate(df_TP_method):
		#print("row in df_TP ",type(row))
		#print("len of row ",len(row))
		#print("len of row ",len(df_FP_method.iloc[index]))
		#Create AU - PR
		PR_list.append( [tp / (tp + df_FP_method.iloc[index][i]) for i,tp in enumerate(row.tolist())] )
		#print(PR_list[-1])
		#print(PR_list[-1])
		AUPR_list.append(np.sum(PR_list[-1])/len(PR_list[-1]))
	# Save AU - PR Data
	print(method)
	#print(AUPR_list)
	data_AUPR[method] = AUPR_list
	data_PR[method] = PR_list


df_AUPR = pd.DataFrame.from_dict(data_AUPR)
df_PR = pd.DataFrame.from_dict(data_PR)
df_PR = df_PR.loc[df_PR[method].str.len()!=0]
#----------------------------------------------------------------#

#print(df_AUPR["ER"])
#print(df_AUPR["MF"])
#print(df_AUPR["PLM"])
plt.figure();
df_AUPR.plot.hist(bins=100,alpha = .5,density=True)
plt.ylim([0,1])
plt.legend(loc='upper left')
plt.savefig("AUPR_hist.pdf")
plt.close()



