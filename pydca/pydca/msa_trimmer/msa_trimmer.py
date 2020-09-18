from Bio import AlignIO
from ..sequence_backmapper.sequence_backmapper import SequenceBackmapper
import logging

# Pre-Processing imports
import numpy as np

"""Trims MSA data by gap percentage or removing all gaps corresponding to best
matching sequence to a reference sequence.

Author: Mehari B. Zerihun
"""

logger = logging.getLogger(__name__)

class MSATrimmerException(Exception):
    """Raises exceptions related to MSA trimming
    """

class MSATrimmer:

    def __init__(self, msa_file, biomolecule=None,max_gap=None, refseq_file=None):
        """
        Parameters
        ----------
            self : MSATrimmer
                An instance of MSATrimmer class
            msa_file : str
                Path to the FASTA formatted MSA file
            biomolecule : str
                Type of biomolecule (protein or RNA)
        """
        self.__msa_file = msa_file
        self.__refseq_file = refseq_file
        self.__max_gap = 0.5 if max_gap is None else max_gap
        self.__s_ipdb = 0
        if self.__max_gap > 1.0 or self.__max_gap < 0.0:
            logger.error('\n\tThe value of max_gap should be between 0 and 1')
            raise MSATrimmerException
        if biomolecule is not None:
            self.__biomolecule = biomolecule.strip().upper()
        else:
            self.__biomolecule = biomolecule
        self.__alignment_data = list(AlignIO.read(self.__msa_file, 'fasta'))

        logger.info('\n\tMSA file: {0}'
            '\n\tReference sequence file: {1}'
            '\n\tbiomolecule: {2}'
            ''.format(self.__msa_file, self.__refseq_file,
                self.__biomolecule,
            )
        )
        return None


    @property
    def alignment_data(self):
        """
        """
        return self.__alignment_data


    def compute_msa_columns_gap_size(self):
        """Computes the gap size of each column in MSA

        Parameters
        ----------
            self : MSATrimmer
                Instance of MSATrimmer class

        Returns
        -------
            msa_columns_gap_size : tuple
                A tuple of column gap sizes. The column gap size is computed as
                the fraction of gaps in a particular MSA column.

        """
        logger.info('\n\tObtaining columns containing more than {}% of gaps'.format(
            self.__max_gap * 100)
        )
        seqs_len = len(self.__alignment_data[0].seq)
        num_seqs = len(self.__alignment_data)
        logger.info('\n\tTotal number of sequences read from MSA file:{}'
            '\n\tLength of the sequences:{}'.format(num_seqs, seqs_len)
        )
        msa_columns_gap_size = list()
        for i in range(seqs_len):
            num_gaps = 0
            for record in self.__alignment_data:
                state_i = record.seq[i]
                if state_i == '.' or state_i == '-': num_gaps += 1
            gap_fraction_i = float(num_gaps)/float(num_seqs)
            msa_columns_gap_size.append(gap_fraction_i)
        max_gap_size = max(msa_columns_gap_size)
        min_gap_size  = min(msa_columns_gap_size)
        logger.info('\n\tMinimum and maximum gap percentages, respectively:'
            '{0:.2f}% and {1:.2f}%'.format(max_gap_size * 100, min_gap_size * 100)
        )
        return tuple(msa_columns_gap_size)


    def msa_columns_beyond_max_gap(self):
        """Obtains the columns in MSA tha contain more than the given fraction of
        gaps treshold.

        Parameters
        ----------
            self : MSATrimmer
                An instance of MSATrimmer class

        Returns
        -------
            msa_columns_beyond_max_gap : tuple
                A tuple of MSA columns that contain fraction of gaps beyond the
                max_gap
        """
        columns_gap_size = self.compute_msa_columns_gap_size()
        seqs_len = len(self.__alignment_data[0].seq)
        msa_columns_beyond_max_gap = [
            i for i in range(seqs_len) if columns_gap_size[i] > self.__max_gap
        ]
        return tuple(msa_columns_beyond_max_gap)


    def trim_by_gap_size(self):
        """Returns a tuple of MSA columns that have beyond self.__max_gap gap
        fraction.

        Parameters
        ---------
            self : MSATrimmer
                An instance of MSATrimmer class

        Returns
        -------
            columns_to_remove : tuple
                A tuple containing columns that are going to to trimmed. These
                are MSA columns that have a gap fraction beyond self.__max_gap.
        """
        columns_to_remove = self.msa_columns_beyond_max_gap()
        return tuple(columns_to_remove)


    def trim_by_refseq(self, remove_all_gaps=False):
        """Obtains columns in MSA that contain gaps more that the gap treshold
        and do not involve residues in the best matchin sequence with reference.
        If remove_all_gaps is set True, all columns involving gaps in the matching
        sequence to reference are removed.

        Parameters
        ----------
            self : MSATrimmer
                An instance of MSATrimmer
            remove_all_gaps : bool
                If set to True, all columns with gaps in the matching sequence
                with the reference are removed.

        Returns
        -------
            columns_to_remove : tuple
                A tuple of MSA column positions. These columns are going to
                be removed from the MSA.
        """
        seqbackmapper = SequenceBackmapper(msa_file = self.__msa_file,
            refseq_file = self.__refseq_file,
            biomolecule = self.__biomolecule,
        )
        matching_seqs, matching_seqs_indx = seqbackmapper.find_matching_seqs_from_alignment()
        print('sequence indices which match ref seq: \n',matching_seqs_indx)
        print('sequences which match ref seq: \n',matching_seqs,'\n\n#------------------------------------------------#\n')
        #print('self.__alignment[%d] =  '%matching_seqs_indx[0],seqbackmabber.__alignment[matching_seqs_indx[0]],'\n\n#------------------------------------------------#\n')
        logger.info('\n\tRemoving gapped columns corresponding to best'
            ' matching sequence to the reference'
        )
        first_matching_seq = matching_seqs[0]
        first_matching_indx = matching_seqs_indx[0]
        self.__s_ipdb = first_matching_indx
        # adding one because find_matching_seqs_from_alignment indexes on pairwise comparison
        logger.info('\n\tSequence in MSA (seq num {}) that matches the reference'
            '\n\t{}'.format(first_matching_indx,first_matching_seq)
        )

        gap_symbols = ['-', '.']
        if not remove_all_gaps:
            candidate_columns_to_remove = self.msa_columns_beyond_max_gap()
            # find out MSA columns that does correspond to gaps w.r.t the sequence
            # in MSA that matches with the reference
            logger.info('\n\tNumber of columns with more than {0:.2f}% gaps:{1}'
                ''.format(self.__max_gap* 100, len(candidate_columns_to_remove))
            )
            columns_to_remove = [
                i for i in candidate_columns_to_remove if first_matching_seq[i] in gap_symbols
            ]
            logger.info('\n\tNumber of columns to remove: {}'.format(len(columns_to_remove)))
        else: # if remove all gaps
            logger.info('\n\tRemoving all columns corresponding to gaps in the matching sequence')
            seqs_len = len(self.__alignment_data[0].seq)
            columns_to_remove = [
                i for i in range(seqs_len) if first_matching_seq[i] in gap_symbols
            ]
            logger.info('\n\tNumber of columns to be removed from MSA:{}'.format(
                len(columns_to_remove))
            )

        return tuple(columns_to_remove)

    
    def get_msa_trimmed_by_refseq(self, remove_all_gaps=False):
        """
        """
        columns_to_remove = self.trim_by_refseq(remove_all_gaps=remove_all_gaps)
        trimmed_msa = list()
        for record in self.__alignment_data:
            seq, seqid = record.seq, record.id
            trimmed_seq = [seq[i] for i in range(len(seq)) if i not in columns_to_remove]
            id_seq_pair = seqid, ''.join(trimmed_seq) 
            trimmed_msa.append(id_seq_pair)
        return trimmed_msa

    #=========================================================================================#
    # Our Pre-Processing - from data_processing.py 2020.9.5
    #=========================================================================================#
    def remove_bad_seqs(self, s,fgs=0.3):
        # remove bad sequences having a gap fraction of fgs  
        l,n = s.shape

        tpdb = self.__s_ipdb
        
        frequency = [(s[t,:] == '-').sum()/float(n) for t in range(l)]
        bad_seq = [t for t in range(l) if frequency[t] > fgs]
        new_s = np.delete(s,bad_seq,axis=0)
    	# Find new sequence index of Reference sequence tpdb
        seq_index = np.arange(s.shape[0])
        seq_index = np.delete(seq_index,bad_seq)
        new_tpdb = np.where(seq_index==tpdb)
        print("tpdb is now ",new_tpdb[0][0])
        self.__s_ipdb = new_tpdb[0][0]
        
        return new_s
    
    #------------------------------
    def remove_bad_cols(self, s,fg=0.3,fc=0.9):
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
    def find_bad_cols(self, s,fg=0.2):
    # remove positions having a fraction fg of gaps
        l,n = s.shape
        # gap positions:
        frequency = [(s[:,i] == '-').sum()/float(l) for i in range(n)]
        bad_cols = [i for i in range(n) if frequency[i] > fg]
    
        #return np.delete(s,gap_cols,axis=1),np.array(gap_cols)
        return np.array(bad_cols)
    
    #------------------------------
    def find_conserved_cols(self, s,fc=0.8):
    # remove positions having a fraction fc of converved residues
        l,n = s.shape
    
        # conserved positions:
        frequency = [max(np.unique(s[:,i], return_counts=True)[1]) for i in range(n)]
        conserved_cols = [i for i in range(n) if frequency[i]/float(l) > fc]
    
        #return np.delete(s,conserved_cols,axis=1),np.array(conserved_cols)
        return np.array(conserved_cols)
    
    #------------------------------
    def number_residues(self, s):
        # number of residues at each position
        l,n = s.shape
        mi = np.zeros(n)
        for i in range(n):
            s_unique = np.unique(s[:,i])
            mi[i] = len(s_unique)
            
        return mi  
    
    #------------------------------
    def covert_letter2number(self, s):
        letter2number = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8,'L':9,\
         'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15,'T':16, 'V':17, 'W':18, 'Y':19, '-':20}
         #,'B':20, 'Z':21, 'X':22}
    
        l,n = s.shape
        return np.array([letter2number[s[t,i]] for t in range(l) for i in range(n)]).reshape(l,n)
    #------------------------------
    def convert_number2letter(self,s):
    
        number2letter = {0:'A', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L',\
    	 				10:'M', 11:'N', 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y', 20:'-'} 
        print('converting s with shape : ',s.shape)
        try:
            l,n = s.shape
            return np.array([number2letter[s[t,i]] for t in range(l) for i in range(n)]).reshape(l,n)
        except(ValueError):
            return np.array([number2letter[r] for r in s])
    
    
    #------------------------------
    #2018.12.24: replace value at a column with probility of elements in that column
    def value_with_prob(self, a,p1):
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
    def find_and_replace(self, s,z,a):
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
    
            s[t,i] = self.value_with_prob(a, p)
        return s 
    
    #------------------------------
    def replace_lower_by_higher_prob(self, s,p0=0.3):
        # input: s: 1D numpy array ; threshold p0
        # output: s in which element having p < p0 were placed by elements with p > p0, according to prob
        
        #f = itemfreq(s)  replaced by next line due to warning
        f = np.unique(s,return_counts=True) 
        # element and number of occurence
        a,p = f[0],f[1].astype(float)
    
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
                    s[iia] = self.value_with_prob(apmax,pmax)
                
        return s
    #--------------------------------------
   
    #--------------------------------------
    def min_res(self, s):
        n = s.shape[1]
        minfreq = np.zeros(n)
        for i in range(n):
            #f = itemfreq(s[:,i])
            f = "" # removing previous line due to warning
            minfreq[i] = np.min(f[:,1])  
            
        return minfreq
    def load_msa(self, data_path,pfam_id):
        printing = True
        s = np.load('%s/%s/msa.npy'%(data_path,pfam_id)).T
        if printing:
        	print("shape of s (import from msa.npy):\n",s.shape)
       
        # convert bytes to str
        try:
            s = np.array([s[t,i].decode('UTF-8') for t in range(s.shape[0]) \
                 for i in range(s.shape[1])]).reshape(s.shape[0],s.shape[1])
            if printing:
        	    print("shape of s (after UTF-8 decode):\n",s.shape)
        except:
            print("\n\nUTF not decoded, pfam_id: %s \n\n"%pfam_id,s.shape)
            print("Exception: ",sys.exc_info()[0])
            # Create list file for missing pdb structures
            if not os.path.exists('missing_MSA.txt'):
                file_missing_msa = open("missing_MSA.txt",'w')
                file_missing_msa.write("%s\n"% pfam_id)
                file_missing_msa.close()
            else:
                file_missing_msa = open("missing_MSA.txt",'a')
                file_missing_msa.write("%s\n"% pfam_id)
                file_missing_msa.close()
            return
        return s

    def preprocess_msa(self, printing = False, gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.8):
        # get msa trimmed by ref_seq || as above
        #columns_to_remove = self.trim_by_refseq(remove_all_gaps=True)
  
        s_trimmed = self.get_msa_trimmed_by_refseq(remove_all_gaps=True)

       	s = list()
        for ii,record in enumerate(s_trimmed):
            pfam, seq =record 
            #trimmed_seq = [seq[i] for i in range(len(seq)) if i not in columns_to_remove]
            trimmed_seq = list(seq)
            s.append(trimmed_seq)
            if ii == self.__s_ipdb-1:
                print('s_trimmed[%d] = '%self.__s_ipdb,trimmed_seq)
        s = np.asarray(s)
 
        if printing:
            print('MSA trimmed by internal function\n')
            print('\n\nstarting shape: ',s.shape)
            print("\n\n#-------------------------Remove Gaps--------------------------#")
            #print("s = \n",s_pydca)
  

        # no pdb_ref structure for covid proteins, ref strucutre is always s[0]
        gap_pdb = s[self.__s_ipdb] =='-' # returns True/False for gaps/no gaps
	# if reference sequence contains gaps, remove them 
        # this is already done in  trim_by_refseq() 
        if any(gap_pdb): 
            s = s[:,~gap_pdb] # removes gaps  
        s_index = np.arange(s.shape[1])
    
        if printing:
            print("s[tpdb] shape is ",s[self.__s_ipdb].shape)
            print("though s still has gaps, s[%d] does not:\n"%(self.__s_ipdb),s[self.__s_ipdb])
            print("s shape is ",s.shape)
            print("Saving indices of reference sequence s[%d](length=%d):\n"%(self.__s_ipdb,s_index.shape[0]),s_index)
            print("#--------------------------------------------------------------#\n\n")
      
        lower_cols = np.array([i for i in range(s.shape[1]) if s[self.__s_ipdb,i].islower()])
    
        if printing:
            print(s.shape)
    
        if printing:
            print("In Data Processing Reference Sequence (shape=",s[self.__s_ipdb].shape,"): \n",s[self.__s_ipdb])
        
        s = self.remove_bad_seqs(s,gap_seqs) # removes all sequences (rows) with >gap_seqs gap || updates self.__s_ipdb

        bad_cols = self.find_bad_cols(s,gap_cols)
        if printing:
            print('found bad columns :=',bad_cols)
    
        # 2018.12.24:
        # replace 'Z' by 'Q' or 'E' with prob
        s = self.find_and_replace(s,'Z',np.array(['Q','E']))
    
        # replace 'B' by Asparagine (N) or Aspartic (D)
        s = self.find_and_replace(s,'B',np.array(['N','D']))
    
        # replace 'X' as amino acids with prob
        amino_acids = np.array(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
        'T','V','W','Y'])
        s = self.find_and_replace(s,'X',amino_acids)
    
        # remove conserved cols
        conserved_cols = self.find_conserved_cols(s,conserved_cols)
        if printing:
            print("found conserved columns (80% repetition):\n",conserved_cols)
    
	# removed bad columns and conservet columns 
        removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))
        removed_cols = np.array(list(set(removed_cols) | set(lower_cols)))

        if printing:
            print("We remove conserved and bad columns with, at the following indices:\n",removed_cols)
    
        s = np.delete(s,removed_cols,axis=1)
        s_index = np.delete(s_index,removed_cols)
        if printing:
            print("Removed Columns...")
            print("s now has shape: ",s.shape)
            print(s_index)
    
        if printing:
            print("In Data Processing Reference Sequence (shape=",s[self.__s_ipdb].shape,"): \n",s[self.__s_ipdb])
    
        # convert letter to number:
        s = self.covert_letter2number(s)


        # replace lower probs by higher probs 
        for i in range(s.shape[1]):
            s[:,i] = self.replace_lower_by_higher_prob(s[:,i],prob_low)
    	
        return removed_cols,s_index,s

    def get_preprocessed_msa(self, printing,saving):
        """
        """
        cols_removed,s_index,s  = self.preprocess_msa(printing=printing)
        if saving:
            np.save("%s/removed_cols.npy"%pfam_id,removed_cols)

        trimmed_msa = list()
        for record in self.__alignment_data:
            seq, seqid = record.seq, record.id
            trimmed_seq = [seq[i] for i in range(len(seq)) if i not in cols_removed] # array of chars
            id_seq_pair = seqid, ''.join(trimmed_seq) 
            trimmed_msa.append(id_seq_pair)
        return trimmed_msa, s_index, cols_removed, self.__s_ipdb,s


    #=========================================================================================
    
   
    
