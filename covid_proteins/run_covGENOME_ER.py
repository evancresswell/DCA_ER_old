import sys,os
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

#========================================================================================
data_path = '/home/eclay/DCA_ER/covid_proteins/'
root_dir = '/home/eclay/DCA_ER/covid_proteins/'
data_path = '/data/cresswellclayec/DCA_ER/covid_proteins/'
root_dir = '/data/cresswellclayec/DCA_ER/covid_proteins/'


cpus_per_job = int(sys.argv[1])
print("Calculating DI for Sars-Cov-2 using %d (of %d) threads"%(cpus_per_job-4,cpus_per_job))

# Swarm aligned file  
msa_file = root_dir+"covid_genome_full_aligned.fasta"
ref_file = root_dir+"wuhan_ref.fasta"

input_data_file = "cov_genome_DP.pickle"
print('\n\nPre-Processing MSA with muscle alignment\n\n')


# Clade aligned File 
if len(sys.argv) > 2:
	clade_file = sys.argv[2]
	msa_file = root_dir+clade_file
	ref_file = root_dir+"wuhan_ref.fasta"

	input_data_file = "cov_genome_clade__%s_DP.pickle"%(clade_file[:2])
	print(input_data_file)
	print('\n\nPre-Processing MSA with muscle alignment\n\n')



preprocessing = True
saving_preprocessed = True



# data processing
if preprocessing:
	# Preprocess data using ATGC
	s0,cols_removed,s_index,s_ipdb = gdp.data_processing(msa_file,0,\
    				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.95)
else:
	with open(input_data_file, 'rb') as f:
		pf_dict = pickle.load(f)
	f.close()

	s0 = pf_dict['s0']
	s_index = pf_dict['s_index']
	cols_removed = pf_dict['cols_removed']
	s_ipdb = pf_dict['s_ipdb']


    
if saving_preprocessed:
	# Save processed data
	msa_outfile, ref_outfile = gdp.write_FASTA(s0,'COV_GENOME',s_ipdb,path=data_path)	
	pf_dict = {}
	pf_dict['s0'] = s0
	pf_dict['s_index'] = s_index
	pf_dict['s_ipdb'] = s_ipdb
	pf_dict['cols_removed'] = cols_removed
	pf_dict['s_ipdb'] = s_ipdb

	with open(input_data_file, 'wb') as f:
		pickle.dump(pf_dict, f)
	f.close()



# if  data processing do not compute
if preprocessing:
	sys.exit()



nucleotide_letters_full = np.array(['A','C','G','T','N','R','Y','S','W','K','M','B','D','H','V','U','-'])

s0_letter = gdp.convert_number2letter(s0)

print('s0        shape: ',s0.shape)
print('s0 letter shape: ',s0_letter.shape)


n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)
print('s0: \n',s0)
print('s0_letter:\n',s0_letter)
s = onehot_encoder.fit_transform(s0)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)

saving_onehot = False
if saving_onehot:
	nucleotide_count = {}
	if s0_letter.shape[1] == len(s_index):
		for i in range(s0_letter.shape[1]):
			column = s0_letter[:,i]
			#print(column)
			letter_counts = []
			for letter in nucleotide_letters_full:
			
				letter_counts.append( np.count_nonzero(column == letter) )
			nucleotide_count[s_index[i]] = letter_counts
			print('nucleotide_counts[%d] : '%(s_index[i]),nucleotide_count[s_index[i]])

		print(len(nucleotide_count))	
		if len(sys.argv) > 2:
			with open('cov_genome_nucleotide_count.pickle', 'wb') as f:
				pickle.dump(nucleotide_count, f)
			f.close()
		else:
			with open('cov_genome_clade_%s_nucleotide_count.pickle'%(clade_file[:2]), 'wb') as f:
				pickle.dump(nucleotide_count, f)
			f.close()

	else:
		print('S_index and s0 shape do not match!!')
	for i0 in range(n_var):
		i1,i2 = i1i2[i0,0],i1i2[i0,1]

		x = np.hstack([s[:,:i1],s[:,i2:]])
		y = s[:,i1:i2]
		print('x:\n',x)
		print('y:\n',y)
		#for x_mini in x:
		    #print(len(x_mini)) # 3343
		print('x:\n',len(x)) # 137634 
		print('y:\n',len(y)) # 137634
		sys.exit()


#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1

#-------------------------------
# parallel
res = Parallel(n_jobs = cpus_per_job-4)(delayed(predict_w)\
        (s,i0,i1i2,niter_max=10,l2=100.0)\
        for i0 in range(n_var))

#-------------------------------
for i0 in range(n_var):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
       
    h01 = res[i0][0]
    w1 = res[i0][1]

    h0[i1:i2] = h01    
    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]

# make w to be symmetric
w = (w + w.T)/2.
di = direct_info(s0,w)

er_gen_DI = sort_di(di)

for site_pair, score in er_gen_DI[:5]:
    print(site_pair, score)

if len(sys.argv) > 2:
	with open(root_dir+ "cov_genome_clade_%s_DI.pickle"%(clade_file[:2]),'wb') as f:
	    pickle.dump(er_gen_DI, f)
	f.close()
else:
	with open(root_dir+'cov_genome_DI.pickle', 'wb') as f:
	    pickle.dump(er_gen_DI, f)
	f.close()


