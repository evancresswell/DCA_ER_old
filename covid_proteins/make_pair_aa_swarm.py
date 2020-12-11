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


f = open('orf1ab_pair_aa_counts.swarm','w')
for i,((pos1,region1),(pos2,region2)) in enumerate(orf1ab_pairs_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s %d %s\n"%(pos1,region1,pos2,region2))

f.close()

np.save('orf1ab_pair_list.npy',orf1ab_pairs_list)
