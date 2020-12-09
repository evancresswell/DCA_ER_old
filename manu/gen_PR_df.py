# RUN : singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/dca_er.simg python gen_PR_df.py ER_53759610_full_test.pkl covER_60721904_full_test.pkl coupER_60311898_full_test.pkl MF_53760930_full_test.pkl PLM_53760928_full_test.pkl

import pandas as pd
import sys,os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
method_list = ["ER","coupER","covER"]
method_list = ["MF","PLM","ER","coupER","covER"]
method_list = ["ER","PLM", "MF"]
#data_AUC = {}
data_AUPR = {}
data_PR = {}
data_TP = {}
data_FP = {}
data_P = {}
data_TPR = {}
data_AUTPR = {}
ROC = {}
generate_TPFP_df = True
join_style = 'outer'
print('\n\n\nTHE FIRST PKL FILE MUST BE ER!!!!\n\n\n')
if generate_TPFP_df:
	df_TP = pd.DataFrame()
	df_FP = pd.DataFrame()
	#df_AUC = pd.DataFrame()
	df_AUTPR = pd.DataFrame()
	df_TPR = pd.DataFrame()
	df_P = pd.DataFrame()
	for ii,filepath in enumerate(sys.argv[1:]):
		print("\n\n\n\nLoading ROC DataFrame: ",filepath)
		df = pd.read_pickle(filepath)
		print("\nDataFrame size: ",df.shape)

		print(df.keys())

		# Remove duplicate rows...
		print('Removing Duplicates...')
		df = df.drop_duplicates(subset='Pfam')
		print("DataFrame size: ",df.shape)
		
		print('\n\nConverting string index to integer:')
		# Make Pfam column the index
		df = df.set_index(df['Pfam'])	
		df = df.sort_index()
		print(df.index[:10])
	
		# Convert Pfam to integer and reset column	
		pfam_index_str = df.index.tolist()
		pfam_ints = [int(pfam_str.lstrip('PF0')) for pfam_str in pfam_index_str]
		df.index = pfam_ints
		print('New index',df.index[:10])
		
		#for column in df.columns:
		#	print(column)
		#print("Mean AUC: ",df['AUC'].mean())


		print('\n\nIntial df head: \n',df.head())
		# remove empty rows
		print('Different ERRORS: ',df['ERR'].unique())
		for err in df['ERR'].unique():
			print('%s error count: %d'%(err,len(df.loc[df['ERR']==err])))
		df = df.loc[df['ERR']=='None']
		print("remove ERR rows")
		print("df head: \n",df.head())

		if filepath[0:6] =="coupER":
			method = "coupER"
			print('\n\n Adding Info for: '+method)
			
			df_TP = pd.concat((df_TP,df["TP"].rename("coupER").to_frame()),axis = 1,join=join_style)
			df_P = pd.concat((df_P,df["P"].rename("coupER").to_frame()),axis = 1,join=join_style)
			df_FP = pd.concat((df_FP,df["FP"].rename("coupER").to_frame()),axis = 1,join=join_style)
			#df_AUC = pd.concat((df_AUC,df["AUC"].rename("coupER").to_frame()),axis = 1,join=join_style)
			df_AUTPR = pd.concat((df_AUTPR,df["Score"].rename("coupER").to_frame()),axis = 1,join=join_style)
			df_TPR = pd.concat((df_TPR,df["TPR_Method"].rename("coupER").to_frame()),axis = 1,join=join_style)

			#data_AUC["coupER"] = df["AUC"]
			data_TP["coupER"] = df["TP"]
			data_FP["coupER"] = df["FP"]
			data_P["coupER"] = df["P"]
			data_TPR["coupER"] = df["TPR_Method"]
			data_AUTPR["coupER"] = df["Score"]
		elif filepath[0:5] =="covER":
			method = "covER"
			print('\n\nAdding info for method: '+method)
			df_TP = pd.concat((df_TP,df["TP"].rename("covER").to_frame()),axis = 1,join=join_style)
			df_P = pd.concat((df_P,df["P"].rename("covER").to_frame()),axis = 1,join=join_style)
			df_FP = pd.concat((df_FP,df["FP"].rename("covER").to_frame()),axis = 1,join=join_style)
			#df_AUC = pd.concat((df_AUC,df["AUC"].rename("covER").to_frame()),axis = 1,join=join_style)
			df_AUTPR = pd.concat((df_AUTPR,df["Score"].rename("covER").to_frame()),axis = 1,join=join_style)
			df_TPR = pd.concat((df_TPR,df["TPR_Method"].rename("covER").to_frame()),axis = 1,join=join_style)

			#data_AUC["covER"] = df["AUC"]
			data_TP["covER"] = df["TP"]
			data_FP["covER"] = df["FP"]
			data_P["covER"] = df["P"]
			data_TPR["covER"] = df["TPR_Method"]
			data_AUTPR["covER"] = df["Score"]

		elif filepath[:2] =="ER":
			method = "ER"
			print('\n\nAdding info for method: '+method)
			#df_TP = pd.concat((df_TP,df["TP"].rename("ER")),axis = 1,join='outer')
			#df_P = pd.concat((df_P,df["P"].rename("ER")),axis = 1,join='outer')
			#df_FP = pd.concat((df_FP,df["FP"].rename("ER")),axis = 1,join='outer')
			#df_AUC = pd.concat((df_AUC,df["AUC"].rename("ER")),axis = 1,join='outer')
			df_TP = df["TP"].rename("ER").to_frame()
			df_FP = df["FP"].rename("ER").to_frame()
			df_P = df["P"].rename("ER").to_frame()
			#df_AUC = df["AUC"].rename("ER").to_frame()
			df_AUTPR = pd.concat((df_AUTPR,df["Score"].rename("ER").to_frame()),axis = 1,join=join_style)
			df_TPR = pd.concat((df_TPR,df["TPR_Method"].rename("ER").to_frame()),axis = 1,join=join_style)

			#data_AUC["ER"] = df["AUC"]
			data_TP["ER"] = df["TP"]
			data_FP["ER"] = df["FP"]
			data_P["ER"] = df["P"]
			data_TPR["ER"] = df["TPR_Method"]
			data_AUTPR["ER"] = df["Score"]

		elif filepath[:3] =="PLM":
			method = "PLM"
			print('\n\nAdding info for method: '+method)
			df_TP = pd.concat((df_TP,df["TP"].rename("PLM").to_frame()),axis = 1,join=join_style)
			df_P = pd.concat((df_P,df["P"].rename("PLM").to_frame()),axis = 1,join=join_style)
			df_FP = pd.concat((df_FP,df["FP"].rename("PLM").to_frame()),axis = 1,join=join_style)
			#df_AUC = pd.concat((df_AUC,df["AUC"].rename("PLM").to_frame()),axis = 1,join=join_style)
			df_AUTPR = pd.concat((df_AUTPR,df["Score"].rename("PLM").to_frame()),axis = 1,join=join_style)
			df_TPR = pd.concat((df_TPR,df["TPR_Method"].rename("PLM").to_frame()),axis = 1,join=join_style)

			#data_AUC["PLM"] = df["AUC"]
			data_TP["PLM"] = df["TP"]
			data_FP["PLM"] = df["FP"]
			data_P["PLM"] = df["P"]
			data_TPR["PLM"] = df["TPR_Method"]
			data_AUTPR["PLM"] = df["Score"]

		elif filepath[:2] == "MF":
			method = "MF"
			print('\n\nAdding info for method: '+method)
			df_TP = pd.concat((df_TP,df["TP"].rename("MF").to_frame()),axis = 1,join=join_style)
			df_P = pd.concat((df_P,df["P"].rename("MF").to_frame()),axis = 1,join=join_style)
			df_FP = pd.concat((df_FP,df["FP"].rename("MF").to_frame()),axis = 1,join=join_style)
			#df_AUC = pd.concat((df_AUC,df["AUC"].rename("MF").to_frame()),axis = 1,join=join_style)
			df_AUTPR = pd.concat((df_AUTPR,df["Score"].rename("MF").to_frame()),axis = 1,join=join_style)
			df_TPR = pd.concat((df_TPR,df["TPR_Method"].rename("MF").to_frame()),axis = 1,join=join_style)

			#data_AUC["MF"] = df["AUC"]
			data_TP["MF"] = df["TP"]
			data_FP["MF"] = df["FP"]
			data_P["MF"] = df["P"]
			data_TPR["MF"] = df["TPR_Method"]
			data_AUTPR["MF"] = df["Score"]

		else: 
			print("Method not defined")
		print('\nAfter adding %s, DF stats are:'%method)	
		print('DF size: ', len(df))
		#print('df_FP index:',df_FP.index)
		print('df_TP:',df_TP.head())
		#print('\n df_AUC: \n',df_AUC.head(),'\n\n\n')
		#print('\n\n\n')

		# Trim empty lists from dfs.	
		df = df.loc[df['TPR_Method'].str.len()!=0]
		df.to_pickle(method+"_summary.pkl")


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
		df.to_pickle(method+"_summary.pkl")
	# Remove nan lines print length of dfs
	print("\n\n Removing NaN Values\n")
	print("Dimensions Before drop NaN:")
	print("AUTPR: ", len(df_AUTPR))
	print("TPR: ", len(df_TPR))
	print("TP: ", len(df_TP))
	print("FP: ", len(df_FP))
	print("P: ", len(df_P))
	#print("AUC: ", len(df_AUC))
	df_TP = df_TP.dropna()
	df_FP = df_FP.dropna()
	df_P = df_P.dropna()
	#df_AUC = df_AUC.dropna()
	print("Dimensions After drop NaN:")
	print("AUTPR: ", len(df_AUTPR))
	print("TPR: ", len(df_TPR))
	print("TP: ", len(df_TP))
	print("FP: ", len(df_FP))
	print("P: ", len(df_P))
	#print("AUC: ", len(df_AUC))
	print("\n\n\n")


	#output = open('pfam_df_auc_allMethods.pkl', 'wb')
	#pickle.dump(df_AUC, output)	
	output = open('pfam_df_tp_allMethods.pkl', 'wb')
	pickle.dump(df_TP, output)	
	output = open('pfam_df_fp_allMethods.pkl', 'wb')
	pickle.dump(df_FP, output)	
	output = open('pfam_df_p_allMethods.pkl', 'wb')
	pickle.dump(df_P, output)


	#output = open('data_auc.pkl', 'wb')
	#pickle.dump(data_AUC, output)	
	output = open('data_tp.pkl', 'wb')
	pickle.dump(data_TP, output)	
	output = open('data_fp.pkl', 'wb')
	pickle.dump(data_FP, output)	
	output = open('data_p.pkl', 'wb')
	pickle.dump(data_P, output)	
	output = open('data_autpr.pkl', 'wb')
	pickle.dump(data_AUTPR, output)	
	output = open('data_tpr.pkl', 'wb')
	pickle.dump(data_TPR, output)	

	"""
	else:
	with open('data_auc.pkl', 'rb') as f:
		data_AUC = pickle.load(f)
	with open('data_p.pkl', 'rb') as f:
		data_P = pickle.load(f)
	with open('data_tp.pkl', 'rb') as f:
		data_TP = pickle.load(f)
	with open('data_fp.pkl', 'rb') as f:
		data_FP = pickle.load(f)
	"""

	using_data_dicts =False
	if using_data_dicts:
		# remove duplicate indices
		for method in method_list:
			data_P[method] = data_P[method][~data_P[method].index.duplicated()]
			data_AUTPR[method] = data_AUTPR[method][~data_AUTPR[method].index.duplicated()]
			data_TPR[method] = data_TPR[method][~data_TPR[method].index.duplicated()]
			data_TP[method] = data_TP[method][~data_TP[method].index.duplicated()]
			data_FP[method] = data_FP[method][~data_FP[method].index.duplicated()]
			data_AUC[method] = data_AUC[method][~data_AUC[method].index.duplicated()]

		df_TPR = pd.DataFrame.from_dict(data_TPR)
		df_AUTPR = pd.DataFrame.from_dict(data_AUTPR)
		df_P = pd.DataFrame.from_dict(data_P)
		df_TP = pd.DataFrame.from_dict(data_TP)
		df_FP = pd.DataFrame.from_dict(data_FP)
		df_AUC = pd.DataFrame.from_dict(data_AUC)
		print(df_P.head())
		# not necessary
		if 0:
			# only keep union indices  for dataframes
			df_list = [df_P,df_TP,df_FP,df_AUC]
			index_list = []
			for df in df_list: 		
				index_list.append(df.index.tolist())
			index_union_list = set(index_list[0]).intersection(*index_list)
			print(index_union_list)
		
			df_P = df_P[df_P.index.isin(index_union_list)]
			df_TP = df_TP[df_TP.index.isin(index_union_list)]
			df_FP = df_FP[df_FP.index.isin(index_union_list)]
			df_AUC = df_AUC[df_AUC.index.isin(index_union_list)]

			print(len(df_AUC.index.tolist()))
			print(len(df_P.index.tolist()))
	# Remove empty Rows
	for method in method_list:
		df_P = df_P.loc[df_P[method].str.len()!=0]
		df_TP = df_TP.loc[df_TP[method].str.len()!=0]
		df_FP = df_FP.loc[df_FP[method].str.len()!=0]
		
		print('\n\n#---Indices Info---#')
		#print('AUC DF: ',df_AUC.index)
		print('TP DF: ',df_TP.index)
		print('AUTPR DF: ',df_AUTPR.index)
		print('FP DF: ',df_FP.index)
		print('P DF: ',df_P.index)
		print('#-----------------#\n\n')

	print("shapes: ",df_P.shape,df_TP.shape,df_FP.shape,df_AUTPR.shape)

	df_P.to_pickle("P_df_summary_allMethods.pkl")
	df_TP.to_pickle("TP_df_summary_allMethods.pkl")
	df_FP.to_pickle("FP_df_summary_allMethods.pkl")
	df_AUTPR.to_pickle("AUTPR_df_summary_allMethods.pkl")
	df_TPR.to_pickle("TPR_df_summary_allMethods.pkl")
	#df_AUC.to_pickle("AUC_df_summary_allMethods.pkl")

merge_dfs = False
if merge_dfs:
	for i,method in enumerate(method_list):
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

generate_PR = False
if generate_PR:
	TP_methods = {}
	FP_methods ={}
	P_methods ={}
	AUC_methods ={}
	try:
		# Create TP and FP dataframes from all three mthods
		for method in method_list:
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

		for method in method_list:
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
	p_covER = df_P["covER"].mean()
	p_coupER = df_P["coupER"].mean()

	tp_er = df_TP["ER"].mean()
	tp_mf = df_TP["MF"].mean()
	tp_plm = df_TP["PLM"].mean()
	tp_covER = df_TP["covER"].mean()
	tp_coupER = df_TP["coupER"].mean()


	fp_er = df_FP["ER"].mean()
	fp_mf = df_FP["MF"].mean()
	fp_plm = df_FP["PLM"].mean()
	fp_covER = df_FP["covER"].mean()
	fp_coupER = df_FP["coupER"].mean()



	plt.plot( p_er,tp_er / (tp_er + fp_er),'b-',label='ER')
	plt.plot( p_mf,tp_mf / (tp_mf + fp_mf),'r-',label='MF')
	plt.plot( p_plm,tp_plm / (tp_plm + fp_plm),'g-',label='PLM')
	plt.plot( p_coupER,tp_coupER / (tp_coupER+ fp_coupER),'g-',label='coupER')
	plt.plot( p_covER,tp_covER / (tp_covER+ fp_covER),'g-',label='covER')
	plt.title("Mean Precision-Recall")
	plt.legend(loc='upper right')
	plt.savefig("MeanPrecision_summary.pdf")
	plt.close()


	plt.plot( fp_er,tp_er ,'b-',label='ER')
	plt.plot( fp_mf,tp_mf ,'r-',label='MF')
	plt.plot( fp_plm,tp_plm ,'g-',label='PLM')
	plt.plot( fp_coupER,tp_coupER ,'g-',label='coupER')
	plt.plot( fp_covER,tp_covER ,'g-',label='covER')
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
	print(df_AUPR["coupER"])
	print(df_AUPR["covER"])
	plt.figure();
	df_AUPR.plot.hist(bins=100,alpha = .5,density=True)
	plt.xlim([0,1])
	plt.legend(loc='upper left')
	plt.show()
	plt.savefig("AUPR_hist_summary.pdf")
	plt.close()

