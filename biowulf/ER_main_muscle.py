import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit

# import pydca-ER module
from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.dca_utilities import dca_utilities
import numpy as np

# Import Bio data processing features 
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning

from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from direct_info import sort_di
from joblib import Parallel, delayed

np.random.seed(1)
#pfam_id = 'PF00025'
#pfam_id = 'PF03068'
#pfam_id = 'PF00186'
pfam_id = sys.argv[1]

using_PP = True
if using_PP:
    data_path = '../../../Pfam-A.full'
    data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
    data_path = '/home/eclay/Pfam-A.full'

    # Read in Reference Protein Structure
    pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))                                                                                                                   
    # convert bytes to str (python 2 to python 3)                                                                       
    pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
    ipdb = 0
    tpdb = int(pdb[ipdb,1])
    print('Ref Sequence # should be : ',tpdb-1)

    # Load Multiple Sequence Alignment
    s = dp.load_msa(data_path,pfam_id)

    # Load Polypeptide Sequence from PDB as reference sequence
    print(pdb[ipdb,:])
    pdb_id = pdb[ipdb,5]                                                                              
    pdb_chain = pdb[ipdb,6]                                                                           
    pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])                                             
    #print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        

    #print('download pdb file')                                                                       
    pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
    #pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    
    #---------------------------------------------------------------------------------------------------------------------#            
    chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
    ppb = PPBuilder().build_peptides(chain)                                                       

    #    print(pp.get_sequence())
    print('peptide build of chain produced %d elements'%(len(ppb)))                               

    found_match = True
    matching_seq_dict = {}
    poly_seq = list()
    for i,pp in enumerate(ppb):
       for char in str(pp.get_sequence()):
           poly_seq.append(char)                                     
    print('PDB Polypeptide Sequence: \n',poly_seq)
    #check that poly_seq matches up with given MSA
        
    pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False)
    # Incorporate SequenceBackmapper to see if PP sequence is in the MSA already. 
    #Or if theres a close enough match

    #just add using muscle:
    #https://www.drive5.com/muscle/manual/addtomsa.html
    #https://www.drive5.com/muscle/downloads.htmL
    muscle_msa_file = 'PP_muscle_msa_'+pfam_id+'.fa'
    os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
    print("PP sequence added to alignment via MUSCLE")


    # create MSATrimmer instance 
    trimmer = msa_trimmer.MSATrimmer(
        muscle_msa_file, biomolecule='protein', 
        refseq_file=pp_ref_file
    )
    # Adding the data_processing() curation from tools to erdca.
    preprocessed_data,s_index, cols_removed,s_ipdb,s0 = trimmer.get_preprocessed_msa(printing=True, saving = False)

    #write trimmed msa to file in FASTA format
    preprocessed_data_outfile = 'MSA_PF00186_PreProcessed.fa'
    with open(preprocessed_data_outfile, 'w') as fh:
        for seqid, seq in preprocessed_data:
           fh.write('>{}\n{}\n'.format(seqid, seq))

    print('Reference sequence (poly_seq): ',poly_seq) 
    print('Reference sequence (s0[%d])'%s_ipdb,s0[s_ipdb])
    s0 = np.asarray(s0)


    #========================================================================================
else:
    s0 = np.loadtxt('pfam_ecc/%s_s0.txt'%(pfam_id))
print(s0.shape)

n_var = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

#onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
onehot_encoder = OneHotEncoder(sparse=False)

s = onehot_encoder.fit_transform(s0)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)

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
res = Parallel(n_jobs = 16)(delayed(predict_w)\
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

sorted_DI_er = sort_di(di)

with open('DI/ER/er_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_er, f)
f.close()
