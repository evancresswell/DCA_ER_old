## 2018.12.24: replace 'Z', 'X', and gap by elements in the same columns and with probability
## 2018.12.26: separate remove gaps (first) and remove conserved positions (last)
    
import numpy as np
from scipy.stats import itemfreq

"""
seq_file = 'fasta_final.txt'

#amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
#    'T','V','W','Y']

# B: Asparagine (N) or Aspartic (D)
# Z: glutamine (Q) or glutamic (E)
# X: unknown (can be any animo acid)

def read_seq_file(seq_file):
    seq = []
    with open(seq_file,'r') as seq_file:
        for i,line in enumerate(seq_file):
            seq.append(line.rstrip('\n'))
    
    l,n = len(seq),len(seq[0])
    
    seq = np.array([seq[t][i] for t in range(l) for i in range(n)]).reshape((l,n))
    #np.savetxt('seq.txt',seq,fmt='%s',delimiter='')
    
    return seq
"""
#=========================================================================================
def remove_bad_seqs(s,fgs=0.3):
    # remove bad sequences having a gap fraction of fgs  
    l,n = s.shape
    
    frequency = [(s[t,:] == '-').sum()/float(n) for t in range(l)]
    bad_seq = [t for t in range(l) if frequency[t] > fgs]
    
    return np.delete(s,bad_seq,axis=0)

#------------------------------
def remove_bad_cols(s,fg=0.3,fc=0.9):
    # remove positions having a fraction fc of converved residues or a fraction fg of gaps 

    l,n = s.shape
    # gap positions:
    frequency = [(s[:,i] == '-').sum()/float(l) for i in range(n)]
    cols_gap = [i for i in range(n) if frequency[i] > fg]

    # conserved positions:
    frequency = [max(np.unique(s[:,i], return_counts=True)[1]) for i in range(n)]
    cols_conserved = [i for i in range(n) if frequency[i]/float(l) > fc]

    cols_remove = cols_gap + cols_conserved

    return np.delete(s,cols_remove,axis=1),cols_remove

#------------------------------
def set_z_as_q_or_e(s):
    # set z as q or e
    qe = ['Q','E']
    z_pos = (s == 'Z')
    s[z_pos] = np.random.choice(qe,size=z_pos.sum())
    
    return s

#------------------------------
def set_b_as_n_or_d(s):
    # set b as Asparagine (N) or Aspartic (D)
    nd = ['N','D']
    b_pos = (s == 'B')
    s[b_pos] = np.random.choice(nd,size=b_pos.sum())
    
    return s

#------------------------------
def set_x_as_random(s):
    # set x as a random amino acids
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
    'T','V','W','Y']
    
    x_pos = (s == 'X')
    s[x_pos] = np.random.choice(amino_acids,size=x_pos.sum())
    
    return s

#------------------------------
def set_gap_as_random(s):
    # set gap as a random amino acids
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
    'T','V','W','Y']
    gap_pos = (s == '-')
    s[gap_pos] = np.random.choice(amino_acids,size=gap_pos.sum())
    
    return s

#------------------------------
def find_bad_cols(s,fg=0.2):
# remove positions having a fraction fg of gaps
    l,n = s.shape
    # gap positions:
    frequency = [(s[:,i] == '-').sum()/float(l) for i in range(n)]
    bad_cols = [i for i in range(n) if frequency[i] > fg]

    #return np.delete(s,gap_cols,axis=1),np.array(gap_cols)
    return np.array(bad_cols)

#------------------------------
def find_conserved_cols(s,fc=0.8):
# remove positions having a fraction fc of converved residues
    l,n = s.shape

    # conserved positions:
    frequency = [max(np.unique(s[:,i], return_counts=True)[1]) for i in range(n)]
    conserved_cols = [i for i in range(n) if frequency[i]/float(l) > fc]

    #return np.delete(s,conserved_cols,axis=1),np.array(conserved_cols)
    return np.array(conserved_cols)

#------------------------------
def number_residues(s):
    # number of residues at each position
    l,n = s.shape
    mi = np.zeros(n)
    for i in range(n):
        s_unique = np.unique(s[:,i])
        mi[i] = len(s_unique)
        
    return mi  

#------------------------------
def covert_letter2number(s):
    letter2number = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8,'L':9,\
     'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15,'T':16, 'V':17, 'W':18, 'Y':19, '-':20}
     #,'B':20, 'Z':21, 'X':22}

    l,n = s.shape
    return np.array([letter2number[s[t,i]] for t in range(l) for i in range(n)]).reshape(l,n)

#=========================================================================================
#2018.12.24: replace value at a column with probility of elements in that column
def value_with_prob(a,p1):
    """ generate a value (in a) with probability
    input: a = np.array(['A','B','C','D']) and p = np.array([0.4,0.5,0.05,0.05]) 
    output: B or A (likely), C or D (unlikely)
    """
    p = p1.copy()
    # if no-specific prob --> set as uniform distribution
    if p.sum() == 0:
        p[:] = 1./a.shape[0] # uniform
    else:
        p[:] /= p.sum() # normalize

    ia = int((p.cumsum() < np.random.rand()).sum()) # cordinate

    return a[ia]    
#------------------------------
def find_and_replace(s,z,a):
    """ find positions of s having z and replace by a with a probality of elements in s column
    input: s = np.array([['A','Q','A'],['A','E','C'],['Z','Q','A'],['A','Z','-']])
           z = 'Z' , a = np.array(['Q','E'])    
    output: s = np.array([['A','Q','A'],['A','E','C'],['E','Q','A'],['A','Q','-']]           
    """  
    xy = np.argwhere(s == z)

    for it in range(xy.shape[0]):
        t,i = xy[it,0],xy[it,1]

        na = a.shape[0]
        p = np.zeros(na)    
        for ii in range(na):
            p[ii] = (s[:,i] == a[ii]).sum()

        s[t,i] = value_with_prob(a, p)
    return s 

#=========================================================================================
def replace_lower_by_higher_prob(s,p0=0.3):
    # input: s: 1D numpy array ; threshold p0
    # output: s in which element having p < p0 were placed by elements with p > p0, according to prob
    
    f = itemfreq(s)
    # element and number of occurence
    a,p = f[:,0],f[:,1].astype(float)

    # probabilities    
    p /= float(p.sum())

    # find elements having p > p0:
    iapmax = np.argwhere(p>p0).reshape((-1,))  # position
                        
    apmax = a[iapmax].reshape((-1,))           # name of aminoacid
    pmax = p[iapmax].reshape((-1,))            # probability
            
    # find elements having p < p0
    apmin = a[np.argwhere(p < p0)].reshape((-1,))

    if apmin.shape[0] > 0:
        for a in apmin:
            ia = np.argwhere(s==a).reshape((-1,))
            for iia in ia:
                s[iia] = value_with_prob(apmax,pmax)
            
    return s

#--------------------------------------
def min_res(s):
    n = s.shape[1]
    minfreq = np.zeros(n)
    for i in range(n):
        f = itemfreq(s[:,i])
        minfreq[i] = np.min(f[:,1])  
        
    return minfreq
#=========================================================================================    
    
#s = read_seq_file(seq_file)
#print(s.shape)

#print('fit with PDB structure file')
#s = s[:,:-3] # remove 3 last digits to compare with PDB structure
#print(s.shape)

#pfam_id = 'PF00186'
#ipdb=0

def data_processing(data_path,pfam_id,ipdb=0,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.8):
#def data_processing(data_path,pfam_id,ipdb=0,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004):

    # read parse_pfam data:
    #print('read original aligned pfam data')
    #s = np.load('../%s/msa.npy'%pfam_id).T
    s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
    #print(s.shape)
    
    # convert bytes to str
    s = np.array([s[t,i].decode('UTF-8') for t in range(s.shape[0]) \
         for i in range(s.shape[1])]).reshape(s.shape[0],s.shape[1])

    #print('select only column presenting as uppercase at PDB sequence')
    #pdb = np.load('../%s/pdb_refs.npy'%pfam_id)
    pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
    #ipdb = 0

    # convert bytes to str (python 2 to python 3)
    pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

    tpdb = int(pdb[ipdb,1])
    #print(tpdb)

    gap_pdb = s[tpdb] =='-'
    s = s[:,~gap_pdb]    

    #print(s.shape)
    #print(s)

    lower_cols = np.array([i for i in range(s.shape[1]) if s[tpdb,i].islower()])

    #print(lower_cols)
    
    #upper = np.array([x.isupper() for x in s[tpdb]])

    #print('select only column presenting as uppercase at the first row')
    #upper = np.array([x.isupper() for x in s[0]])
    #s = s[:,upper]
    #print(s.shape)

    #print('remove sequences containing too many gaps')
    s = remove_bad_seqs(s,gap_seqs)
    #print(s.shape)

    #print('remove bad cols')
    bad_cols = find_bad_cols(s,gap_cols)
    #print('number of bad columns removed:',gap_cols.shape[0])
    #print(s.shape)

    # 2018.12.24:
    # replace 'Z' by 'Q' or 'E' with prob
    #print('replace Z by Q or E')
    s = find_and_replace(s,'Z',np.array(['Q','E']))

    # replace 'B' by Asparagine (N) or Aspartic (D)
    #print('replace B by N or D')
    s = find_and_replace(s,'B',np.array(['N','D']))

    # replace 'X' as amino acids with prob
    #print('replace X by other aminoacids')
    amino_acids = np.array(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
    'T','V','W','Y'])
    s = find_and_replace(s,'X',amino_acids)

    # remove conserved cols
    #print('remove conserved columns')
    conserved_cols = find_conserved_cols(s,conserved_cols)
    #print(s.shape)
    #print('number of conserved columns removed:',conserved_cols.shape[0])

    removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))
    removed_cols = np.array(list(set(removed_cols) | set(lower_cols)))

    # 2019.09.17: excluse conserved cols
    #removed_cols = np.array(list(set(bad_cols) | set(lower_cols)))

    s = np.delete(s,removed_cols,axis=1)

    #print('replace gap(-) by other aminoacids')
    #s = find_and_replace(s,'-',amino_acids)

    # convert letter to number:
    s = covert_letter2number(s)
    #print(s.shape)

    # replace lower probs by higher probs 
    #print('replace lower probs by higher probs')
    for i in range(s.shape[1]):
        s[:,i] = replace_lower_by_higher_prob(s[:,i],prob_low)

    #min_res = min_res(s)
    #print(min_res)

    #remove_cols = np.hstack([gap_cols,conserved_cols])
    #remove_cols = np.hstack([remove_cols,lower_cols]) ## 2019.01.22

    #np.savetxt('s0.txt',s,fmt='%i')
    #np.savetxt('cols_remove.txt',remove_cols,fmt='%i')

    #f = open('n_pos.txt','w')
    #f.write('%i'%(s.shape[1]))
    #f.close()

    #mi = number_residues(s)
    #print(mi.mean())

    return s,removed_cols
#=========================================================================================

