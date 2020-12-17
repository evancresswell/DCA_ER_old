import sys, os
import genome_data_processing as gdp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC

#indices are actual postions-1 ie 14408 (real position) --> 14407

# indices of coevolving with ORF1ab
orf1ab_index_list = 	[
		(1058, 'NSP2'),
		(1162, 'NSP2'),
		(2236, 'NSP2'),
		(3036, 'NSP3'),
		(7539, 'NSP3'),
		(14407,'NSP12'),
		(14804,'NSP12'),
		(18554,'NSP14a2'),
		(19289, 'NSP14a2'),
		(19557, 'NSP14a2'),
		(21253, 'NSP16'), 
		(21366, 'NSP16'),
		(21368, 'NSP16'),
		(22226, 'S'),
		(22362, 'S'),
		(23400, 'S'),
		(23402, 'S'),
		(23403, 'S'),
		(25562, 'ORF3a'),
		(26143, 'ORF3a'),
		]

S_index_list = 	[
		 (1162, 'NSP2'),
		 (7539, 'NSP3'),
		 (16646, 'NSP13'),
		 (18554,'NSP14a2'),
		 (22333, 'S'),
		 (22335, 'S'),
		 (22352, 'S'),
		 (22353, 'S'),
		 (22362, 'S'),
		 (22383, 'S'),
		 (22487, 'S'),
		 (22494, 'S'),
		 (22496, 'S'),
		 (22523, 'S'),
		 (22525, 'S'),
		 (22538, 'S'),
		 (22538, 'S'),
		 (22540, 'S'),
		 (23400, 'S'),
		]

orf3a_index_list = [
		 (1058, 'NSP2'),
		 (20267, 'NSP15'),
		 (22443, 'S'),
		 (22991, 'S'),
		 (25428, 'ORF3a'),
		 (25562, 'ORF3a'),
		]

orf7a_index_list = [
		(27174,'ORF7a'),		
		(27563,'ORF7a'),
		(27565,'ORF7a'),		
		(27578,'ORF7a'),
		(27580,'ORF7a'),		
		(27620,'ORF7a'),
		(27622,'ORF7a'),
		(27626,'ORF7a'),
		(27628,'ORF7a'),		
		(27630,'ORF7a'),
		(27652,'ORF7a'),
		(27687,'ORF7a'),
		(27694,'ORF7a'),
		(27696,'ORF7a'),		
		(27697,'ORF7a'),
		(27699,'ORF7a'),		
		(27712,'ORF7a'),
		(27741,'ORF7a'),
		(27750,'ORF7a'),		
		(27751,'ORF7a'),		
		(27791,'ORF7b')
		]


orf7b_index_list = [
		(27687,'ORF7a'),
		(27758,'ORF7b'),
		(27760,'ORF7b'),
		(27773,'ORF7b'),
		(27775,'ORF7b'),
		(27784,'ORF7b'),		
		(27786,'ORF7b'),
		(27787,'ORF7b'),
		(27789,'ORF7b'),
		(27791,'ORF7b'),
		(27793,'ORF7b'),
		(27794,'ORF7b'),
		(27795,'ORF7b'),
		(27796,'ORF7b'),
		(27800,'ORF7b'),
		(27802,'ORF7b'),
		(27803,'ORF7b'),
		(27804,'ORF7b'),
		(27805,'ORF7b'),
		(27806,'ORF7b'),
		(27804,'ORF7b'),
		(27807,'ORF7b')
		]

N_index_list = [
		(28880, 'N'),
		(28882, 'N')
		]
f = open('orf1ab_codon_mapping.swarm','w')
for i,(index,region) in enumerate(orf1ab_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()

f = open('s_codon_mapping.swarm','w')
for i,(index,region) in enumerate(S_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()

f = open('orf3a_codon_mapping.swarm','w')
for i,(index,region) in enumerate(orf3a_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()

f = open('orf7a_codon_mapping.swarm','w')
for i,(index,region) in enumerate(orf7a_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()

f = open('orf7b_codon_mapping.swarm','w')
for i,(index,region) in enumerate(orf7b_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()

f = open('n_codon_mapping.swarm','w')
for i,(index,region) in enumerate(N_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))
f.close()
