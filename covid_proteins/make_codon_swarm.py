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
		(18554,'NSP15a2'),
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

index_list = 	[
		(22383, 'S'),
		(26140, 'ORF3a'),
		(28881,'N'),
			]

f = open('codon_mapping.swarm','w')
for i,(index,region) in enumerate(orf1ab_index_list):
	f.write("singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python codon_mapping.py %d %s\n"%(index,region))

f.close()
