# Copyright 2014 - Christoph Feinauer and Marcin J. Skwark (christophfeinauer@gmail.com, marcin@skwark.pl)
# Copyright 2012 - by Magnus Ekeberg (magnus.ekeberg@gmail.com)
# All rights reserved
# 
# Permission is granted for anyone to copy, use, or modify this
# software for any uncommercial purposes, provided this copyright 
# notice is retained, and note is made of any changes that have 
# been made. This software is distributed without any warranty, 
# express or implied. In no event shall the author or contributors be 
# liable for any damage arising out of the use of this software.
# 
# The publication of research using this software, modified or not, must include an 
# appropriate citation to:
#       C. Feinauer, M.J. Skwark, A. Pagnani, E. Aurell, Improving Contact Prediction
#       along Three Dimensions. PLoS Comput Biol 10(10): e1003847.
#
# 	M. Ekeberg, C. Lövkvist, Y. Lan, M. Weigt, E. Aurell, Improved contact
# 	prediction in proteins: Using pseudolikelihoods to infer Potts models, Phys. Rev. E 87, 012707 (2013) 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import data_processing as dp
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
#import emachine as EM
from direct_info import direct_info

import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import timeit

import matplotlib.pyplot as plt
#%matplotlib inline
import ecc_tools as tools
import itertools
import random

def gapMat(y):
	# variables 
	#int N,M,*y,m,n;

	# input 
	M,N = y.shape # not sure about this

	# output 
	leftGapMat = np.zeros(y.shape)
	rightGapMat = np.zeros(y.shape)
	#int k1; 
	#int k2;
	#int n2;
	#for(m=0; m<M; m++){
	for m in range(M):
		k1=0
		k2=0
		for n in range(N-1,0,-1):
			if y[m,n] == 0:
				k1+=1
			else:
				k1=0
			rightGapMat[m,n]=k1;
			n2=N-n-1;
			if y[m,n2]==0:
				k2+=1
			else:
				k2=0
			leftGapMat[m,n2]=k2;
	return leftGapMat, rightGapMat

def return_alignment(PfamID,leave_gaps):

	if not leave_gaps:
		# import MSA
		# Read in Protein structure
		data_path = '../Pfam-A.full'
		pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,PfamID))

		# delete 'b' in front of letters (python 2 --> python 3)
		pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
			 for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

		# Print number of pdb structures in Protein ID folder
		npdb = pdb.shape[0]
		print('number of pdb structures:',npdb)
	# Print PDB array #print(pdb)
		#print(pdb[0])

		# Create pandas dataframe for protein structure
		df = pd.DataFrame(pdb,columns = ['PF','seq','id','uniprot_start','uniprot_start',\
						 'pdb_id','chain','pdb_start','pdb_end'])
		#df.head()

		Y,removed_cols,s_index = dp.data_processing(data_path,PfamID,ipdb=0,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.8)
		
		# Matlab=> ind = align_full(1).Sequence ~= '.' & align_full(1).Sequence == upper( align_full(1).Sequence ) 
		#	-> returns the sequences without '.' and the sequences as upper case

		B = Y.shape[0] #assuming length() in matlab is supposed to be rows ie # sequences
		print(B)
		N = Y.shape[1] #sum(ind) where ind is the processed sequence data, N is the number of columns..
		print(N)

		q = max([max(y) for y in Y])  # Dimensionality of sequence data ie 21 possible states (could be less I guess)
		print(q)
	else:

		print("leaving gaps in not implemented yet ie gplmDCA")

	

	
	return [N,B,q,Y, s_index]

def gplmDCA_asymmetric(PfamID,  lambda_h, lambda_J, lambda_G, reweighting_threshold, nb_of_cores, M):
	"""
	PfamID: 
		ID of Protein family alignent to analyse
	lambda_h:
		Field-regularization strength (typical value: 0.01).

		lambda_J:
		Coupling-regularization strength (typical value: 0.01 - this is tested for N in the range 50-100 or so, and the optimal value may be different for longer domains).

		lambda_chi:
		Gap-regularization strength (typical value 0.001).

		reweighting_threshold:
		Required fraction of nonidentical AA for two sequences to be counted as independent (typical values: 0.1-0.3).
		Note that this is not the threshold 'x' as defined in the paper, but 1-x.
		
		nr_of_cores:
		The number of processors to use on the local machine. 
		If this argument is >1, the program calls functions from MATLAB's Parallel Computing Toolbox.
		M: Maximal gap length (if -1 it is set automatically
	"""
	
	#Read inputfile (removing inserts), remove duplicate sequences, and calculate weights and B_eff.
	#[N,B_with_id_seq,q,Y]=return_alignment(fastafile)
	# Get sequence alginment from Pfam ID
	[N,B_with_id_seq,q,Y,s_index] = return_alignment(PfamID)


	#Y = np.unique(Y,0) # remove duplicate sequences
	print(Y)
	"""
	[lH,rH]=gapMat(int(Y-1))
	lH=int(lH)
	rH=int(rH)
	[B,N]=Y.shape
	weights = np.ones(B,1)
	"""
