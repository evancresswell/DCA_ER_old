# Import data_processing module from local directory 
from data_processing import data_processing

# Import Bio data processing features 
import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
import numpy as np

def contact_map(pdb,ipdb,cols_removed):
    pdb_id = pdb[ipdb,5]
    pdb_chain = pdb[ipdb,6]
    pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
    #print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)

    #print('download pdb file')
    pdb_file = pdb_list.retrieve_pdb_file(pdb_id,file_format='pdb')
    #pdb_file = pdb_list.retrieve_pdb_file(pdb_id)
    chain = pdb_parser.get_structure(pdb_id,pdb_file)[0][pdb_chain]
    coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
    coords = coords_all[pdb_start-1:pdb_end]
    #print('original pdb:')
    #print(coords.shape)

    coords_remain = np.delete(coords,cols_removed,axis=0)
    #print(coords_remain.shape)

    ct = distance_matrix(coords_remain,coords_remain)

    return ct

def roc_curve(ct,di,ct_thres):
    ct1 = ct.copy()
    
    ct_pos = ct1 < ct_thres           
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0],dtype=bool), k=1)
    # argsort sorts from low to high. [::-1] reverses 
    order = di[mask].argsort()[::-1]

    ct_flat = ct1[mask][order]

    tp = np.cumsum(ct_flat, dtype=float)
    fp = np.cumsum(~ct_flat.astype(int), dtype=float)

    if tp[-1] !=0:
        tp /= tp[-1]
        fp /= fp[-1]
    
    # bining (to reduce the size of tp,fp and make fp having the same values for every pfam)
    nbin = 101
    pbin = np.linspace(0,1,nbin, endpoint=True)

    #print(pbin)

    fp_size = fp.shape[0]

    fpbin = np.ones(nbin)
    tpbin = np.ones(nbin)
    for ibin in range(nbin-1):
        # find value in a range
        t1 = [(fp[t] > pbin[ibin] and fp[t] <= pbin[ibin+1]) for t in range(fp_size)]

        if len(t1)>0 :            
            fpbin[ibin] = fp[t1].mean()
            tpbin[ibin] = tp[t1].mean()
        else:
            #print(i)
            tpbin[ibin] = tpbin[ibin-1] 

    #print(fp,tp)
    #return fp,tp,pbin,fpbin,tpbin
    return pbin,tpbin,fpbin


