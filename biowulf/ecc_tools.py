# Import Bio data processing features 
import sys,os,errno
import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np


import random

def contact_map(pdb,ipdb,cols_removed,s_index):
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
    #print(coords_all.shape)
    #print(coords.shape)
    #print(s_index.shape)

    coords_remain = np.delete(coords,cols_removed,axis=0)
    print(coords_remain.shape)
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
    #print(di[mask][order][:15])
    #print(ct1[mask][order][:15])
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
            try:
                 fpbin[ibin] = fp[t1].mean()
            except RuntimeWarning:
                 #print("Empty mean slice")
                 fpbin[ibin] = 0
            try:
                 tpbin[ibin] = tp[t1].mean()
            except RuntimeWarning:
                 #print("Empty mean slice")
                 tpbin[ibin] = 0
        else:
            #print(i)
            tpbin[ibin] = tpbin[ibin-1] 
    #print(fp,tp)
    #return fp,tp,pbin,fpbin,tpbin
    return pbin,tpbin,fpbin


on_pc = False
if on_pc:
	from IPython.display import HTML
	def hide_toggle(for_next=False):
	    this_cell = """$('div.cell.code_cell.rendered.selected')"""
	    next_cell = this_cell + '.next()'

	    toggle_text = 'Toggle show/hide'  # text shown on toggle link
	    target_cell = this_cell  # target cell to control with toggle
	    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

	    if for_next:
	        target_cell = next_cell
	        toggle_text += ' next cell'
	        js_hide_current = this_cell + '.find("div.input").hide();'

	    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

	    html = """
		<script>
		    function {f_name}() {{
			{cell_selector}.find('div.input').toggle();
		    }}

		    {js_hide_current}
		</script>

		<a href="javascript:{f_name}()">{toggle_text}</a>
	    """.format(
		f_name=js_f_name,
		cell_selector=target_cell,
		js_hide_current=js_hide_current, 
		toggle_text=toggle_text
	    )

	    return HTML(html)

#=========================================================================================
def distance_restr_sortedDI(sorted_DI_in):
	sorted_DI = sorted_DI_in.copy()
	count = 0
	for site_pair, score in sorted_DI_in:
		if abs(site_pair[0] - site_pair[1])<5:
			sorted_DI[count] = site_pair,0
		count += 1
	sorted_DI.sort(key=lambda x:x[1],reverse=True)  
	return sorted_DI
#=========================================================================================
def distance_restr(di,s_index,make_large=False):
	# Hamstring DI matrix by setting all DI values st |i-j|<5 to 0
	if di.shape[0] != s_index.shape[0]:
		print("Distance restraint cannot be imposed, bad input")
		raise( FileNotFoundError( errno.ENONENT, os.sterror(errno.ENOENT), 'S and DI do not match') ) 
	di_distal = np.zeros(di.shape)
	for i in range(di.shape[0]):
		for j in range(di.shape[1]):
			if(abs(s_index[i]-s_index[j])<5):
				if make_large:
					di_distal[i][j]=35.
				else:	
					di_distal[i][j]=0.
			else:
				di_distal[i][j] = di[i][j]

	return di_distal
