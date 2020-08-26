# RUN: 
# singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/dca_er.simg python gen_method_column_df.py 

import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
method_list = ["MF","PLM","ER","covER","coupER"]
method_list = ["MF","PLM","ER"]
data_AUC = {}
data_AUPR = {}
data_PR = {}
data_TP = {}
data_FP = {}
data_P = {}
ROC = {}

# Generate PR creates a DF where methods are considered a column value requires the <method>_summary.pkl created in gen_PR_df.py
#									---> should change so that we only generate summary pkls in there...
generate_PR = True
if generate_PR:
	TP_methods = {}
	FP_methods ={}
	P_methods ={}
	AUC_methods ={}
	PR_methods = {}
	AUPR_methods = {}
	try:
		# Create TP and FP dataframes from all three mthods
		for method in method_list:
			df = pd.read_pickle(method+"_summary.pkl")
			#df = df.set_index('Pfam')

			df_TP = pd.DataFrame(df['TP'])
			df_FP = pd.DataFrame(df['FP'])
			df_P = pd.DataFrame(df['P'])
			df_AUC = pd.DataFrame(df['AUC'])

			# Remove rows with empty vectors
			df_P = df_P.loc[df_P['P'].str.len()!=0]
			df_TP = df_TP.loc[df_TP['TP'].str.len()!=0]
			df_FP = df_FP.loc[df_FP['FP'].str.len()!=0]

			# populate PR and AUPR using above dataframes
			PR_list = []
			AUPR_list = []
			try:
				for index,row in df_TP['TP'].items():
					#Create AU - PR
					FP = df_FP['FP']
					PR_list.append( [tp / (tp + FP[index][i]) for i,tp in enumerate(row)] )
					#print(PR_list[-1])
					AUPR_list.append(np.sum(PR_list[-1])/len(PR_list[-1]))
			except KeyError as err:
				print("Key Error at !!: ", index)
				print(err)
				sys.exit()

			# Save AU - PR Data
			data_AUPR = {'AUPR':AUPR_list}
			data_PR = {'PR':PR_list}
			df_AUPR = pd.DataFrame(data_AUPR,index=df_TP.index)
			df_PR = pd.DataFrame(data_PR,index=df_TP.index)

			df_TP['method'] = method
			df_TP['Pfam'] = df['Pfam']
			df_TP['num_seq'] = df['num_seq']
			df_TP['seq_len'] = df['seq_len']
			TP_methods[method] = df_TP

			df_FP['method'] = method
			df_FP['Pfam'] = df['Pfam']
			df_FP['num_seq'] = df['num_seq']
			df_FP['seq_len'] = df['seq_len']
			FP_methods[method] = df_FP

			df_P['method'] = method
			df_P['Pfam'] = df['Pfam']
			df_P['num_seq'] = df['num_seq']
			df_P['seq_len'] = df['seq_len']
			P_methods[method] = df_P

			df_AUC['method'] = method
			df_AUC['Pfam'] = df['Pfam']
			df_AUC['num_seq'] = df['num_seq']
			df_AUC['seq_len'] = df['seq_len']
			AUC_methods[method] = df_AUC

			df_PR['method'] = method
			df_PR['Pfam'] = df['Pfam']
			df_PR['num_seq'] = df['num_seq']
			df_PR['seq_len'] = df['seq_len']
			PR_methods[method] = df_PR

			df_AUPR['method'] = method
			df_AUPR['Pfam'] = df['Pfam']
			df_AUPR['num_seq'] = df['num_seq']
			df_AUPR['seq_len'] = df['seq_len']
			AUPR_methods[method] = df_AUPR

			print(df_AUPR)

		df_TP = pd.concat(TP_methods)
		df_FP = pd.concat(FP_methods)
		df_P = pd.concat(P_methods)
		df_AUC = pd.concat(AUC_methods)
		df_AUPR = pd.concat(AUPR_methods)
		df_PR = pd.concat(PR_methods)

		print(df_AUPR)
		print(df_PR)

		df_TP.to_pickle("df_TP_method_summary.pkl")
		df_FP.to_pickle("df_FP_method_summary.pkl")
		df_P.to_pickle("df_P_method_summary.pkl")
		df_AUC.to_pickle("df_AUC_method_summary.pkl")
		df_PR.to_pickle("df_PR_method_summary.pkl")
		df_AUPR.to_pickle("df_AUPR_method_summary.pkl")

		#print("AUPR: ,",df_AUPR.index)
		#print("AUPR:len ,",len(df_AUPR.index))
	except ValueError as err:
		print(err.args)	
#----------------------------------------------------------------#
# THIS PLOTTING IS OBSOLETE 8/12/2020
plotting = True
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

