import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from itertools import permutations


#df_AUC = pd.read_pickle("AUC_df.pkl")
df_AUC = pd.read_pickle("AUC_df_summary.pkl")
print(df_AUC.shape)
df_diff = df_AUC.copy()

x_columns = []
methods = df_diff.keys()
for method_combo in permutations(methods,2):
	column = str(method_combo[0])+">"+str(method_combo[1])
	x_columns.append(column)
	df_diff[column] = 0.
	print(column)
	df_diff[column][df_diff[str(method_combo[0])] > df_diff[str(method_combo[1])]] = df_diff[str(method_combo[0])]-df_diff[str(method_combo[1])]

print(df_diff)
print(df_diff.shape)

df_plot = df_diff[x_columns]

df_sum = df_plot[df_plot > 0 ].sum(numeric_only=True)

df_sum.plot.bar()
plt.savefig("agg_AUC_bar_summary.pdf")
plt.show()
plt.close()

print(df_plot[df_plot > 0])
df_mean = df_plot[df_plot > 0].mean(numeric_only=True)
df_var = df_plot[df_plot > 0].var(numeric_only=True)
df_std = df_plot[df_plot > 0].std(numeric_only=True)

fig, ax = plt.subplots()
plot = df_mean.plot(kind='bar',yerr=df_std,ax=ax)
plt.savefig("mean_AUC_bar_summary.pdf")
plt.show()
