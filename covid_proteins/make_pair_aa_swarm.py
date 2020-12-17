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
orf1ab_pairs_list = 	[
		((3036, 'NSP3'),(14407,'NSP12')),
		((19290, 'NSP14a2'),(19558,'NSP14a2')),
		((1058, 'NSP2'),(25562, 'ORF3a')),
		((14804,'NSP12'),(26143, 'ORF3a')),
		((7539, 'NSP3'),(23400, 'S')),
		((14407,'NSP12'),(23403, 'S')),
		((3036, 'NSP3'),(23402, 'S')),
		((21253, 'NSP16'), (22226, 'S')),
		((18554,'NSP14a2'), (23400, 'S')),
		((21366, 'NSP16'), (21368, 'NSP16')),
		((1162, 'NSP2'), (23400, 'S')),
		]

# indices of coevolving with S 
s_pairs_list = 	[
		((22362, 'S'),(22383,'S')),
		((23400, 'S'),(7539,'NSP3')),
		((23400, 'S'),(18554,'NSP14a2')),
		((22496, 'S'),(22494,'S')),
		((22352, 'S'),(22354,'S')),
		((22400, 'S'),(1162,'NSP2')),
		((22400, 'S'),(16646,'NSP13')),
		((22333, 'S'),(22335,'S')),
		((22538, 'S'),(22540,'S')),
		((22525, 'S'),(22523,'S')),
		((22353, 'S'),(22487,'S')),
		]

# indices of coevolving with ORF3a
orf3a_pairs_list = [
		((25562,'ORF3a'),(1058,'NSP2')),
		((25562,'ORF3a'),(22443,'S')),
		((25562,'ORF3a'),(22991, 'S')),
		((25562,'ORF3a'),(25428,'ORF3a')),
		((25562,'ORF3a'),(20267, 'NSP15')),
		]

# indices of coevolving with ORF7a
orf7a_pairs_list = [
		((27687,'ORF7a'),(27791,'ORF7b')),		
		((27697,'ORF7a'),(27699,'ORF7a')),		
		((27578,'ORF7a'),(27580,'ORF7a')),		
		((27687,'ORF7a'),(27751,'ORF7a')),		
		((27652,'ORF7a'),(27750,'ORF7a')),		
		((27694,'ORF7a'),(27696,'ORF7a')),		
		((27630,'ORF7a'),(27628,'ORF7a')),		
		((27626,'ORF7a'),(27628,'ORF7a')),		
		((27563,'ORF7a'),(27565,'ORF7a')),		
		((27712,'ORF7a'),(27714,'ORF7a')),		
		((27620,'ORF7a'),(27622,'ORF7a')),		
		]	

# indices of coevolving with ORF7a
orf7b_pairs_list = [
		((27802,'ORF7b'),(27800,'ORF7b')),
		((27791,'ORF7b'),(27687,'ORF7a')),
		((27803,'ORF7b'),(27805,'ORF7b')),
		((27796,'ORF7b'),(27794,'ORF7b')),
		((27806,'ORF7b'),(27804,'ORF7b')),
		((27805,'ORF7b'),(27807,'ORF7b')),
		((27760,'ORF7b'),(27758,'ORF7a')),
		((27804,'ORF7b'),(27807,'ORF7b')),
		((27789,'ORF7b'),(27787,'ORF7b')),
		((27773,'ORF7b'),(27775,'ORF7b')),
		((27786,'ORF7b'),(27784,'ORF7b')),
		((27795,'ORF7b'),(27793,'ORF7b'))
		]

N_pairs_list = [
		((28882,'N'),(28880,'N'))
		]

f = open('orf1ab_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(orf1ab_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('orf1ab_pair_list.npy',orf1ab_pairs_list)



f = open('s_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(s_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('s_pair_list.npy',s_pairs_list)



f = open('orf3a_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(orf3a_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('orf3a_pair_list.npy',orf3a_pairs_list)

f = open('orf7a_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(orf7a_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('orf7a_pair_list.npy',orf7a_pairs_list)

f = open('orf7b_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(orf7b_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('orf7b_pair_list.npy',orf7b_pairs_list)

f = open('n_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(N_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))
f.close()
np.save('n_pair_list.npy',N_pairs_list)






