import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

data_AUC = {}
data_AUPR = {}
data_PR = {}
data_TP = {}
data_FP = {}
data_P = {}
ROC = {}
for filepath in sys.argv[1:]:
	df = pd.read_pickle(filepath)
	#for column in df.columns:
	#	print(column)

	print("DataFrame size: ",df.shape)
	print("Mean AUC: ",df['AUC'].mean())
	#for column in df.columns:
	#	print(column)
	#df['TP'] = df['TP'].apply(lambda x: x.empty if len(x) == 0)
	if filepath[:2] =="ER":
		method = "ER"
		print(method)
		data_AUC["ER"] = df["AUC"]
		data_TP["ER"] = df["TP"]
		data_FP["ER"] = df["FP"]
		data_P["ER"] = df["P"]
	elif filepath[:3] =="PLM":
		method = "PLM"
		print(method)
		data_AUC["PLM"] = df["AUC"]
		data_TP["PLM"] = df["TP"]
		data_FP["PLM"] = df["FP"]
		data_P["PLM"] = df["P"]
	elif filepath[:2] == "MF":
		method = "MF"
		print(method)
		data_AUC["MF"] = df["AUC"]
		data_TP["MF"] = df["TP"]
		data_FP["MF"] = df["FP"]
		data_P["MF"] = df["P"]
	else: 
		print("Method not defined")

	# Trim empty lists from dfs.	
	df = df.loc[df['TP'].str.len()!=0]
	for index,row in df.iterrows():
		#print(len(row['TP']))
		#print(index, row['TP'])
		plt.plot(row['FP'],row['TP'])
	plt.plot(df['FP'].mean(),df['TP'].mean(),linewidth=2.5,linestyle='--',color='k',label='Mean')
	ROC[method] = [df['FP'].mean(),df['TP'].mean()]
	plt.title(method+" Roc")
	plt.legend(loc='lower right')
	plt.savefig(method+"_ROC.pdf")
	plt.close()

	df.to_pickle(method+".pkl")

df_P = pd.DataFrame.from_dict(data_P)
df_TP = pd.DataFrame.from_dict(data_TP)
df_FP = pd.DataFrame.from_dict(data_FP)
df_AUC = pd.DataFrame.from_dict(data_AUC)

# Remove empty Rows
for method in ["MF","PLM","ER"]:
	df_P = df_P.loc[df_P[method].str.len()!=0]
	df_TP = df_TP.loc[df_TP[method].str.len()!=0]
	df_FP = df_FP.loc[df_FP[method].str.len()!=0]
print("shapes: ",df_P.shape,df_TP.shape,df_FP.shape)


#----------------- Precision-Recall and AUPR --------------------#
for method in ["MF","PLM","ER"]:
	PR_list = []
	AUPR_list = []

	df_TP_method = df_TP[method]
	df_FP_method = df_FP[method]
	for index,row in enumerate(df_TP_method):
		print(type(row))
		print(row)
		#Create AU - PR
		PR_list.append( [tp / (tp + df_FP_method.iloc[index][i]) for i,tp in enumerate(row.tolist())] )
		#print(PR_list[-1])
		AUPR_list.append(np.sum(PR_list[-1])/len(PR_list[-1]))
	# Save AU - PR Data
	print(method)
	print(AUPR_list)
	data_AUPR[method] = AUPR_list
	data_PR[method] = PR_list


df_AUPR = pd.DataFrame.from_dict(data_AUPR)
df_PR = pd.DataFrame.from_dict(data_PR)
df_PR = df_PR.loc[df_PR[method].str.len()!=0]
#----------------------------------------------------------------#

df_P.to_pickle("P_df.pkl")
df_TP.to_pickle("TP_df.pkl")
df_FP.to_pickle("FP_df.pkl")
df_AUC.to_pickle("AUC_df.pkl")
df_AUPR.to_pickle("AUPR_df.pkl")
df_PR.to_pickle("PR_df.pkl")

p_er = df_P["ER"].mean()
p_mf = df_P["MF"].mean()
p_plm = df_P["PLM"].mean()

tp_er = df_TP["ER"].mean()
tp_mf = df_TP["MF"].mean()
tp_plm = df_TP["PLM"].mean()

fp_er = df_FP["ER"].mean()
fp_mf = df_FP["MF"].mean()
fp_plm = df_FP["PLM"].mean()

plt.plot( p_er,tp_er / (tp_er + fp_er),'b-',label='ER')
plt.plot( p_mf,tp_mf / (tp_mf + fp_mf),'r-',label='MF')
plt.plot( p_plm,tp_plm / (tp_plm + fp_plm),'g-',label='PLM')
plt.title("Mean Precision-Recall")
plt.legend(loc='upper right')
plt.savefig("MeanPrecision.pdf")
plt.close()


plt.plot( fp_er,tp_er ,'b-',label='ER')
plt.plot( fp_mf,tp_mf ,'r-',label='MF')
plt.plot( fp_plm,tp_plm ,'g-',label='PLM')
plt.title("Mean ROC")
plt.legend('upper left')
plt.savefig("MeanROC.pdf")
plt.close()

plt.figure();
df_AUC.plot.hist(bins=100,alpha=.5,density=True)
plt.legend(loc='upper left')
plt.xlim([0,1])
plt.savefig("AUC_hist.pdf")
plt.close()

print(df_AUPR["ER"])
print(df_AUPR["MF"])
print(df_AUPR["PLM"])
plt.figure();
df_AUPR.plot.hist(bins=100,alpha = .5,density=True)
plt.xlim([0,1])
plt.legend(loc='upper left')
plt.savefig("AUPR_hist.pdf")
plt.close()

