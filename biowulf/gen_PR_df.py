import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np

data_AUC = {}
data_AUPR = {}
data_PR = {}
data_TP = {}
data_FP = {}
data_P = {}
ROC = {}
generate_TPFP_df = False
if generate_TPFP_df:
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
		plotting_ROC = False
		if plotting_ROC:
			for index,row in df.iterrows():
				#print(len(row['TP']))
				#print(index, row['TP'])
				plt.plot(row['FP'],row['TP'])
			plt.plot(df['FP'].mean(),df['TP'].mean(),linewidth=2.5,linestyle='--',color='k',label='Mean')
			ROC[method] = [df['FP'].mean(),df['TP'].mean()]
			plt.title(method+" Roc")
			plt.legend(loc='lower right')
			plt.savefig(method+"_ROC_new.pdf")
			plt.close()
		print(len(df['TP']))
		df.to_pickle(method+"_summary.pkl")

	df_P = pd.DataFrame.from_dict(data_P)
	df_TP = pd.DataFrame.from_dict(data_TP)
	df_FP = pd.DataFrame.from_dict(data_FP)
	df_AUC = pd.DataFrame.from_dict(data_AUC)

	# Remove empty Rows
	for method in ["MF","PLM","ER"]:
		df_P = df_P.loc[df_P[method].str.len()!=0]
		df_P = df_P.loc[df_P[method].str.len()!=0]
		df_TP = df_TP.loc[df_TP[method].str.len()!=0]
		df_FP = df_FP.loc[df_FP[method].str.len()!=0]
		df_AUC = df_AUC.loc[df_P[method].str.len()!=0]

	print("shapes: ",df_P.shape,df_TP.shape,df_FP.shape,df_AUC.shape)

	df_P.to_pickle("P_df_summary.pkl")
	df_TP.to_pickle("TP_df_summary.pkl")
	df_FP.to_pickle("FP_df_summary.pkl")
	df_AUC.to_pickle("AUC_df_summary.pkl")

merge_dfs = False
if merge_dfs:
	for i,method in enumerate(["MF","PLM","ER"]):
		df = pd.read_pickle(method+"_summary.pkl")
		if i == 0:
			df_TP = df.loc[df['TP']]
			df_FP = df.loc[df['FP']]
		else:
			df_TP = df_TP.merge(df.loc[df['TP']])
			df_FP = df_FP.merge(df.loc[df['FP']])

#----------------- Precision-Recall and AUPR --------------------#
"""
ERROR:
MF
gen_PR_df.py:99: RuntimeWarning: invalid value encountered in double_scalars
  PR_list.append( [tp / (tp + df_FP_method.iloc[index][i]) for i,tp in enumerate(row.tolist())] )
MF
6249
PLM
gen_PR_df.py:99: RuntimeWarning: invalid value encountered in double_scalars
  PR_list.append( [tp / (tp + df_FP_method.iloc[index][i]) for i,tp in enumerate(row.tolist())] )
PLM
6249
ER
gen_PR_df.py:99: RuntimeWarning: invalid value encountered in double_scalars
  PR_list.append( [tp / (tp + df_FP_method.iloc[index][i]) for i,tp in enumerate(row.tolist())] )
ER
6237
6237 != 6249
method_summary.pkl files not good
"""

generate_PR = True
if generate_PR:
	TP_methods = {}
	FP_methods ={}
	P_methods ={}
	AUC_methods ={}
	try:
		# Create TP and FP dataframes from all three mthods
		for method in ["MF","PLM","ER"]:
			df = pd.read_pickle(method+"_summary.pkl")
			df = df.set_index('Pfam')

			df_TP_method = df['TP']
			TP_methods[method] = df_TP_method

			df_FP_method = df['FP']
			FP_methods[method] = df_FP_method

			df_P_method = df['P']
			P_methods[method] = df_P_method

			df_AUC_method = df['AUC']
			AUC_methods[method] = df_AUC_method

			# populate PR and AUPR using above dataframes
			PR_list = []
			AUPR_list = []
			for index,row in enumerate(df_TP_method):
				#Create AU - PR
				PR_list.append( [tp / (tp + df_FP_method[index][i]) for i,tp in enumerate(row.tolist())] )
				#print(PR_list[-1])
				AUPR_list.append(np.sum(PR_list[-1])/len(PR_list[-1]))
			# Save AU - PR Data
			data_AUPR[method] = pd.Series(AUPR_list,index=df_TP_method.index)
			data_PR[method] = pd.Series(PR_list,index=df_TP_method.index)

		# add num seq and seq len to dictionaries
		#print(df['num_seq'])
		#print(df['seq_len'])
		TP_methods['num_seq'] = df['num_seq']
		FP_methods['num_seq'] = df['num_seq']
		P_methods['num_seq'] = df['num_seq']
		AUC_methods['num_seq'] = df['num_seq']

		TP_methods['seq_len'] = df['seq_len']
		FP_methods['seq_len'] = df['seq_len']
		P_methods['seq_len'] = df['seq_len']
		AUC_methods['seq_len'] = df['seq_len']

		#idx_er = TP_methods['ER'].index	
		#idx_mf = TP_methods['MF'].index	
		#idx_er_diff = idx_er.difference(idx_mf)	
		#idx_mf_diff = idx_mf.difference(idx_er)	
		#print("er diff ",idx_er_diff)
		#print("mf diff ",idx_mf_diff)

		df_TP = pd.concat(TP_methods,axis=1,join='inner')
		df_FP = pd.concat(FP_methods,axis=1,join='inner')
		df_P = pd.concat(P_methods,axis=1,join='inner')
		df_AUC = pd.concat(AUC_methods,axis=1,join='inner')
		#print(df_TP.head())

		df_TP.to_pickle("TP_df_summary.pkl")
		df_FP.to_pickle("FP_df_summary.pkl")
		df_P.to_pickle("P_df_summary.pkl")
		df_AUC.to_pickle("AUC_df_summary.pkl")



		df_AUPR = pd.concat(data_AUPR,axis=1,join='inner')
		df_PR = pd.concat(data_PR,axis=1,join='inner')
		#print(df_PR['MF'].index)
		#print(df_PR['ER'].index)
		#print(df['seq_len'].index)
		#print(df_PR.head())

		df_PR = pd.concat([df_PR,df['num_seq'],df['seq_len']],axis=1,join='inner')
		df_AUPR = pd.concat([df_AUPR,df['num_seq'],df['seq_len']],axis=1,join='inner')
		#print(df_PR.index)
		#print(df_PR.head())

		for method in ["MF","PLM","ER"]:
			df_PR = df_PR.loc[df_PR[method].str.len()!=0]

		#print("PR: ,",df_PR.index)
		#print("AUPR: ,",df_AUPR.index)
		#print("AUPR:len ,",len(df_AUPR.index))

		df_AUPR = df_AUPR.loc[df_PR.index]	
		#print("AUPR: ,",df_AUPR.index)
		#print("AUPR:len ,",len(df_AUPR.index))

		print(df_AUPR.head())
		print(df_PR.head())
	
		df_AUPR.to_pickle("AUPR_df_summary.pkl")
		df_PR.to_pickle("PR_df_summary.pkl")

	except ValueError as err:
		print(err.args)	

#----------------------------------------------------------------#

plotting = False
if plotting:
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
	plt.savefig("MeanPrecision_summary.pdf")
	plt.close()


	plt.plot( fp_er,tp_er ,'b-',label='ER')
	plt.plot( fp_mf,tp_mf ,'r-',label='MF')
	plt.plot( fp_plm,tp_plm ,'g-',label='PLM')
	plt.title("Mean ROC")
	plt.legend(loc='upper left')
	plt.savefig("MeanROC_summary.pdf")
	plt.close()

	plt.figure();
	df_AUC.plot.hist(bins=100,alpha=.5,density=True)
	plt.legend(loc='upper left')
	plt.xlim([0,1])
	plt.savefig("AUC_hist_summary.pdf")
	plt.close()

	print(df_AUPR["ER"])
	print(df_AUPR["MF"])
	print(df_AUPR["PLM"])
	plt.figure();
	df_AUPR.plot.hist(bins=100,alpha = .5,density=True)
	plt.xlim([0,1])
	plt.legend(loc='upper left')
	plt.show()
	plt.savefig("AUPR_hist_summary.pdf")
	plt.close()

