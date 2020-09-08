from numba import prange as parallel_range
from pprint import pprint
import pandas as pd
import numpy as np
import sys,os,errno
# Import Bio data processing features 
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
from matplotlib import colors as mpl_colors

import random
def split_and_shift_contact_pairs(list_of_contacts):
	"""Separating contacting site pairs into two lists: One containing all
	first entries (xdata), and the other containing all second entries (ydata).

	Note that, each site pair is shifted by 1 so as to make them ready for
	visualization (since the sites have been counted starting from 0)

	Parameters
	----------
	    list_of_contacts : list/tuple
		A list or tuple containing tuples of contacting site pairs.

	Returns
	-------
	    xdata : list
		List containing shifted first entries of site pairs as obtained
		from list_of_contacts parameter.
	    ydata : list
		List containing shifted second entries of site pairs as obtained
		from list_of_contacts parameter
	"""
	xdata = list()
	ydata = list()
	for first, second in list_of_contacts:
		xdata.append(first + 1) # shift indexing by 1 for visualization (output)
		ydata.append(second + 1)

	return xdata, ydata


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


def contact_map(pdb,ipdb,cols_removed,s_index,use_old=False):
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
    if use_old:
        coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
        #---PROOF THAT THE ABOVE METHOD IS WRONG.. IT ITERATES THROUGH ATOMS! NOT RESIDUES OR PARTICULAR ATOMS IN RESIDUES----#
        #for a in chain.get_atoms():
        #    print(a.get_name())
        print(len(coords_all))
        coords = coords_all[pdb_start-1:pdb_end]
    else:
        # Correct Method
        ppb = PPBuilder().build_peptides(chain)
        #    print(pp.get_sequence())
        print('peptide build of chain produced %d elements'%(len(ppb)))
	
        for i,pp in enumerate(ppb):
            poly_seq = [char for char in str(pp.get_sequence())]
            #poly_seq = poly_seq[pdb_start-1:pdb_end]
            poly_seq = np.delete(poly_seq,cols_removed)
            print('poly_seq: \n', poly_seq)
            if(len(poly_seq)>= s_index[-1]-1):
                print('poly_seq[s_index]:\n',poly_seq[s_index],'\n\n')
            print('peptide seq len: ',len(poly_seq))
            print('s_index len: ',len(s_index))
            print('s_index largest index: ',s_index[-1])
            good_coords = []
            coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
            ca_residues = np.array([a.get_name()=='CA' for a in chain.get_atoms()])
            ca_coords = coords_all[ca_residues]
            good_coords = ca_coords[pdb_start-1:pdb_end]
            n_amino = len(good_coords)
            print("s_index and col removed len %d "%(len(s_index)+len(cols_removed)))
            print('all coords %d, all ca coords: %d , protein rangs ca coords len: %d' % (len(coords_all),len(ca_coords),len(good_coords)))
            #for i,a in enumerate(chain.get_atoms()):
            #    if a.get_name() == 'CA':
            #        good_coords.append(a.get_coord())		

            ct_full = distance_matrix(good_coords,good_coords)
        """
        for i,ca in enumerate(ppb[0].get_ca_list()):		
           #print(ca.get_coord())
           #coords_zhang.append(a.get_coord())
           good_coords.append(ca.get_coord())
        """
        coords = np.array(good_coords)
    
    #---------------------------------------------------------------------------------------------------------------------#

    #print('original pdb:')
    #print(coords_all.shape)
    #print(coords.shape)
    #print(s_index.shape)

    coords_remain = np.delete(coords,cols_removed,axis=0)
    print(coords_remain.shape)
    #print(coords_remain.shape)
    

    ct = distance_matrix(coords_remain,coords_remain)
    if use_old:
        return ct
    else:
        return ct,ct_full,n_amino

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


on_pc = True
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
def distance_restr_sortedDI(sorted_DI_in, s_index=None):
	sorted_DI = sorted_DI_in.copy()
	count = 0
	for site_pair, score in sorted_DI_in:

		# if s_index exists re-index sorted pair
		if s_index is not None:
			pos_0 = s_index[site_pair[0]]
			pos_1 = s_index[site_pair[1]]
		else:
			pos_0 = site_pair[0]
			pos_1 = site_pair[1]
			print('MAKE SURE YOUR INDEXING IS CORRECT!!')
			print('		 or pass s_index to distance_restr_sortedDI()')

		if abs(pos_0- pos_1)<5:
			sorted_DI[count] = (pos_0,pos_1),0
		else:
			sorted_DI[count] = (pos_0,pos_1),score
		count += 1
	sorted_DI.sort(key=lambda x:x[1],reverse=True)  
	return sorted_DI
#=========================================================================================
def distance_restr(di,s_index,make_large=False):
	# Hamstring DI matrix by setting all DI values st |i-j|<5 to 0
	if di.shape[0] != s_index.shape[0]:
		print("Distance restraint cannot be imposed, bad input")
		#IndexError: index 0 is out of bounds for axis 0 with size 0
		print("s_index: ",s_index.shape[0],"di shape: ",di.shape[0])
		#print('di:\n',di[0])
		raise IndexError("matrix input dimensions do not matchup with simulation n_var")
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
#=========================================================================================
def distance_restr_ct(ct,s_index,make_large=False):
	# Hamstring DI matrix by setting all DI values st |i-j|<5 to 0
	if ct.shape[0] < s_index[-1]:
		print("Distance restraint cannot be imposed, bad input")
		#IndexError: index 0 is out of bounds for axis 0 with size 0
		print("s_index max index: ",s_index[-1],"ct shape: ",ct.shape[0])
		#print('di:\n',di[0])
		raise IndexError("matrix input dimensions do not matchup with simulation n_var")
	ct_distal = np.zeros(ct.shape)
	for i in range(ct.shape[0]):
		for j in range(ct.shape[1]):
			if(abs(i-j<5)):
				if make_large:
					ct_distal[i][j]=35.
				else:	
					ct_distal[i][j]=0.
			else:
				ct_distal[i][j] = ct[i][j]

	return ct_distal


# ER coupling setup 6/20/2020
def compute_sequences_weight(alignment_data=None, seqid=None):
    """Computes weight of sequences. The weights are calculated by lumping
    together sequences whose identity is greater that a particular threshold.
    For example, if there are m similar sequences, each of them will be assigned
    a weight of 1/m. Note that the effective number of sequences is the sum of
    these weights.

    Parameters
    ----------
        alignmnet_data : np.array()
            Numpy 2d array of the alignment data, after the alignment is put in
            integer representation
        seqid : float
            Value at which beyond this sequences are considered similar. Typical
            values could be 0.7, 0.8, 0.9 and so on

    Returns
    -------
        seqs_weight : np.array()
            A 1d numpy array containing computed weights. This array has a size
            of the number of sequences in the alignment data.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    seqs_weight = np.zeros((num_seqs,), dtype=np.float64)
    #count similar sequences
    for i in parallel_range(num_seqs):
        seq_i = alignment_data[i]
        for j in range(num_seqs):
            seq_j = alignment_data[j]
            iid = np.sum(seq_i==seq_j)
            if np.float64(iid)/np.float64(seqs_len) > seqid:
                seqs_weight[i] += 1
    #compute the weight of each sequence in the alignment
    for i in range(num_seqs): seqs_weight[i] = 1.0/float(seqs_weight[i])
    return seqs_weight




def compute_single_site_freqs(alignment_data=None, seqs_weight=None,mx = None ):
    """Computes single site frequency counts for a particular aligmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array of alignment data represented in integer form.

        num_site_states : int
            An integer value fo the number of states a sequence site can have
            including a gap state. Typical value is 5 for RNAs and 21 for
            proteins.

        seqs_weight : np.array()
            A 1d numpy array of sequences weight

    Returns
    -------
        single_site_freqs : np.array()
            A 2d numpy array of of data type float64. The shape of this array is
            (seqs_len, num_site_states) where seqs_len is the length of sequences
            in the alignment data.
    """
    alignment_shape = alignment_data.shape
    #num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    if seqs_len != len(mx):
        print('sequence length = %d and mx length = %d'%(seqs_len,len(mx)))
    m_eff = np.sum(seqs_weight)
    #single_site_freqs = np.zeros(shape = (seqs_len, num_site_states),dtype = np.float64)
    single_site_freqs = [] # list form so its easier to handle varied num_site_states
    for i in range(seqs_len):
        #for a in range(1, num_site_states + 1):#we need gap states single site freqs too
        single_site_freqs.append([])
        num_site_states = mx[i] 
        #print('seq position %d has %d states'%(i,num_site_states))
        column_i = alignment_data[:,i]
        for a in np.unique(column_i):#we use varying site states (unique vals in col)
            #print('	a = ',a)
            #print(np.unique(column_i)) # what values are in column_i?
            freq_ia = np.sum((column_i==a)*seqs_weight)
            single_site_freqs[-1].append(freq_ia/m_eff)
    return single_site_freqs

def get_reg_single_site_freqs(single_site_freqs = None, seqs_len = None,
        mx = None, pseudocount = None):
    """Regularizes single site frequencies.

    Parameters
    ----------
        single_site_freqs : np.array()
            A 2d numpy array of single site frequencies of shape
            (seqs_len, num_site_states). Note that gap state frequencies are
            included in this data.
        seqs_len : int
            The length of sequences in the alignment data
        num_site_states : int
            Total number of states that a site in a sequence can accommodate. It
            includes gap states.
        pseudocount : float
            This is the value of the relative pseudo count of type float.
            theta = lambda/(meff + lambda), where meff is the effective number of
            sequences and lambda is the real pseudo count.

    Returns
    -------
        reg_single_site_freqs : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of single site
            frequencies after they are regularized.
    """
    reg_single_site_freqs = single_site_freqs
    for i in range(seqs_len):
        num_site_states = mx[i]
        theta_by_q = np.float64(pseudocount)/np.float64(num_site_states)
        for a in range(num_site_states):
            reg_single_site_freqs[i][ a] = theta_by_q + (1.0 - pseudocount)*reg_single_site_freqs[i][ a]
    return reg_single_site_freqs



# This function is replaced by the parallelized version below
def compute_pair_site_freqs_serial(alignment_data=None, mx=None,
        seqs_weight=None):
    
    """Computes pair site frequencies for an alignmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array conatining alignment data. The residues in the
            alignment are in integer representation.
        num_site_states : int
            The number of possible states including gap state that sequence
            sites can accomodate. It must be an integer
        seqs_weight:
            A 1d numpy array of sequences weight

    Returns
    -------
        pair_site_freqs : np.array()
            A 3d numpy array of shape
            (num_pairs, num_site_states, num_site_states) where num_pairs is
            the number of unique pairs we can form from sequence sites. The
            pairs are assumed to in the order (0, 1), (0, 2) (0, 3), ...(0, L-1),
            ... (L-1, L). This ordering is critical and any change must be
            documented.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    num_site_pairs = (seqs_len -1)*seqs_len/2
    num_site_pairs = np.int64(num_site_pairs)
    m_eff = np.sum(seqs_weight)
    #pair_site_freqs = np.zeros(
    #    shape=(num_site_pairs, num_site_states - 1, num_site_states - 1),
    #    dtype = np.float64)
    pair_site_freqs = [] # list form so its easier to handle varied num_site_states 
    pair_counter = 0
    for i in range(seqs_len-1):
        column_i = alignment_data[:, i]
        i_site_states = mx[i] 
        if len(np.unique(column_i))!=i_site_states:
            print('unique vals doesn\'match site states')
            sys.exit()

        for j in range(i+1, seqs_len):
            column_j = alignment_data[:, j]
            j_site_states = mx[j] 
            if len(np.unique(column_j))!=j_site_states:
                print('unique vals doesn\'match site states')
                sys.exit()
            pair_site_freqs.append([])

            for a in np.unique(column_i):
                pair_site_freqs[-1].append([])
                count_ai = column_i==a

                for b in np.unique(column_j):
                    count_bj = column_j==b
                    count_ai_bj = count_ai * count_bj
                    freq_ia_jb = np.sum(count_ai_bj*seqs_weight)
                    #pair_site_freqs[pair_counter, a-1, b-1] = freq_ia_jb/m_eff
                    pair_site_freqs[-1][-1].append(freq_ia_jb/m_eff)
            #move to the next site pair (i, j)
            pair_counter += 1
    if len(pair_site_freqs) != num_site_pairs:
        print('Not enough site pairs generated')
        sys.exit()
    return pair_site_freqs





# I think this is wahte msa_numerics uses to initialize weights..
# maybe we can use this to initialize our weights (w i think)
# what is w and what is its purpose!?!?!?!
def construct_corr_mat(reg_fi = None, reg_fij = None, seqs_len = None,
        mx = None):
    """Constructs correlation matrix from regularized frequency counts.

    Parameters
    ----------
        reg_fi : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of regularized
            single site frequncies. Note that only fi[:, 0:num_site_states-1] are
            used for construction of the correlation matrix, since values
            corresponding to fi[:, num_site_states]  are the frequncies of gap
            states.
        reg_fij : np.array()
            A 3d numpy array of shape (num_unique_pairs, num_site_states -1,
            num_site_states - 1), where num_unique_pairs is the total number of
            unique site pairs execluding self-pairings.
        seqs_len : int
            The length of sequences in the alignment
        num_site_states : int
            Total number of states a site in a sequence can accommodate.

    Returns
    -------
        corr_mat : np.array()
            A 2d numpy array of shape (N, N)
            where N = seqs_len * num_site_states -1
    """
    #corr_mat_len = seqs_len * (num_site_states - 1)
    corr_mat_len = mx.cumsum()[-1]
    print('Generating NxN correlation matrix with N=',corr_mat_len)
    corr_mat = np.zeros((corr_mat_len, corr_mat_len), dtype=np.float64)
    pair_counter = 0
    for i in range(seqs_len-1):
        if i == 0 :
            site_i = 0
        else:
            site_i = mx.cumsum()[i-1]
        for j in range(i+1, seqs_len):
            site_j = mx.cumsum()[j-1]
            for a in range(mx[i]):
                row = site_i + a
                for b in range(mx[j]):
                    col = site_j + b
                    if i==j:
                        print('Iteration through non-symmetric reg_fij list is not working ')
                        sys.exit()
                    else:
                        try:
                            corr_ij_ab = reg_fij[pair_counter][ a][ b] - reg_fi[i][ a] * reg_fi[j][ b]
                        except IndexError:
                            print('pair %d: (%d,%d)'%(pair_counter,i,j))
                            print('Indices: ', mx.cumsum())
                            print('Site Counts: ', mx)
                            print('Index out of bound')
                            print('par ranges: a= [%d,%d],b= [%d,%d]'%(site_i,site_i+range(mx[i])[-1],site_j,site_j+range(mx[j])[-1]))
                            print('pair_counter = %d of %d (%d)'%(pair_counter,len(reg_fij),len(reg_fij)))
                            print('i site state = %d of %d (%d)'%(a,mx[i],len(reg_fij[pair_counter])))
                            print(b)
                            sys.exit()
                    #print(corr_mat)
                    #print(corr_ij_ab)
                    try:
                        corr_mat[row, col] = corr_ij_ab
                        corr_mat[col, row] = corr_ij_ab
                    except IndexError:
                        print('ERROR: \n	row = %d of %d'%(row,mx.cumsum()[-1]) )
                        print('       \n	col = %d of %d'%(col,mx.cumsum()[-1]) )
                        sys.exit()

            if i != j: pair_counter += 1
    # fill in diagonal block
    for ii,site_block in enumerate(mx):
        if ii==0:
            site_block_start = 0
        else:
            site_block_start = mx.cumsum()[ii-1]
        for a in range(site_block):
            for b in range(a,site_block):
                row = site_block_start + a
                col = site_block_start + b
                #print('combo (%d,%d)'%(row,col))
                fia, fib = reg_fi[ii][ a], reg_fi[ii][ b]
                corr_ij_ab = fia*(1.0 - fia) if a == b else -1.0*fia*fib
                corr_mat[row, col] = corr_ij_ab
                corr_mat[col, row] = corr_ij_ab


    return corr_mat
def compute_couplings(corr_mat = None):
    """Computes the couplings by inverting the correlation matrix

    Parameters
    ----------
        corr_mat : np.array()
            A numpy array of shape (N, N) where N = seqs_len *(num_site_states -1)
            where seqs_len  is the length of sequences in the alignment data and
            num_site_states is the total number of states a site in a sequence
            can accommodate, including gapped states.

    Returns
    -------
        couplings : np.array()
            A 2d numpy array of the same shape as the correlation matrix. Note
            that the couplings are the negative of the inverse of the
            correlation matrix.
    """
    couplings = np.linalg.inv(corr_mat)
    couplings = -1.0*couplings
    return couplings
def slice_couplings(couplings = None, site_pair = None, mx=None):
    """Returns couplings corresponding to site pair (i, j). Note that the
    the couplings involving gaps are included, but they are set to zero.

    Parameters
    ----------
        couplings : np.array
            A 2d numpy array of couplings. It has a shape of (L(q-1), L(q-1))
            where L and q are the length of sequences in alignment data and total
            number of standard residues plus gap.
        site_pair : tuple
            A tuple of site pairs. Example (0, 1), (0, L-1), ..., (L-2, L-1).
        num_site_states : int
            The value of q.

    Returns
    -------
        couplings_ij : np.array
            A2d numpy array of shape (q, q) containing the couplings. Note that
            couplings_ij[q, :] and couplings[:, q] are set to zero.
    """
    qi = mx[site_pair[0]] 
    qj = mx[site_pair[1]] 
    couplings_ij = np.zeros((qi, qj), dtype = np.float64)
    row_begin = mx.cumsum()[site_pair[0]-1] 
    row_end = row_begin + qi 
    column_begin = mx.cumsum()[site_pair[1] -1]
    column_end = column_begin + qj
    couplings_ij[:qi-1, :qj-1] = couplings[row_begin:row_end, column_begin:column_end]
    return couplings_ij




