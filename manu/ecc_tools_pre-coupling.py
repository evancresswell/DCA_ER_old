import pandas as pd
import numpy as np
import sys,os,errno
# Import Bio data processing features 
import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
from matplotlib import colors as mpl_colors

import random




def get_score_df(df_diff_full):
	# pass dataframes generated in gen_method_column_df.py
	# creates Dataframe for each method
	# creates rgba list for 3 methods
	printing = False
	df_ER = df_diff_full.loc[df_diff_full['method']=='ER']
	df_MF = df_diff_full.loc[df_diff_full['method']=='MF']
	df_PLM = df_diff_full.loc[df_diff_full['method']=='PLM']

	#print(df_MF.loc[df_MF['best_method']=='MF' ].loc[df_MF['AUC']<0.]['Score'])
	if printing:
		print(len(df_ER['Score']))
		print(len(df_PLM['Score']))
		print(len(df_MF['Score']))

	common = df_ER.merge(df_MF,on='Pfam')
	print(len(common))
	df_ER = df_ER[df_ER.Pfam.isin(common.Pfam)].dropna()
	df_PLM = df_PLM[df_PLM.Pfam.isin(common.Pfam)].dropna()
	df_MF = df_MF[df_MF.Pfam.isin(common.Pfam)].dropna()

	if printing:
		print(len(df_ER['Score']))
		print(len(df_PLM['Score']))
		print(len(df_MF['Score']))
	df_ER = df_ER.sort_values(by='Pfam')
	df_MF = df_MF.sort_values(by='Pfam')
	df_PLM = df_PLM.sort_values(by='Pfam')

	if printing:
		print(df_ER['Pfam'])
		print(df_MF['Pfam'])
		print(df_PLM['Pfam'])


	df_winner = df_diff_full[df_diff_full.Pfam.isin(common.Pfam)].dropna()
	df_winner = df_winner.loc[df_winner['best_method'] == df_winner['method'] ] 
	df_winner = df_winner.sort_values(by='Pfam')
	scores = df_winner['Score'].values.tolist()
	color_dict = {'ER':'blue','PLM':'green','MF':'orange'}
	colors = [ color_dict[c] for c in df_ER['best_method'].values.tolist() ] 
	print(len(scores),len(colors))
	#cmap = colors.LinearSegmentedColormap.from_list('incr_alpha', [(0, (*colors.to_rgb(c),0)), (1, c)])
	rgba_colors = np.zeros((len(scores),4))
	rgba_colors[:,0:3] = [ mpl_colors.to_rgb(c) for c in colors ]  
	rgba_colors[:,3] = scores
	#print(rgba_colors)


	return df_ER, df_MF, df_PLM,rgba_colors



def gen_DI_matrix(DI):
    n_seq = max([coupling[0][0] for coupling in DI]) 
    di = np.zeros((n_var,n_var))
    for coupling in DI:
        di[coupling[0][0],coupling[0][1]] = coupling[1]
        di[coupling[0][1],coupling[0][0]] = coupling[1]
    return di


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
		raise( FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), 'S and DI do not match') ) 
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
