from __future__ import absolute_import, division
import sys
from . import msa_numerics
from . import dca_msa_numerics as dca_numerics
from . import dca_orig_msa_numerics as dca_orig_numerics
from . import ER_protein_msa_numerics as LADER
from pydca.fasta_reader import fasta_reader
import logging
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.fasta_reader.fasta_reader import get_alignment_from_fasta_file
from pydca.fasta_reader.fasta_reader import get_alignment_int_form
from pydca.meanfield_dca import meanfield_dca
import ecc_tools as tools

logger = logging.getLogger(__name__)

class ERDCAException(Exception):
    """
    """

class ERDCA:
    """ERDCA class. Instances of this class are used to carry out Direct
    Coupling Analysis (DCA) of residue coevolution using the mean-field DCA
    algorithm.
    """
    def __init__(self, msa_file, biomolecule,s_index = None, num_threads = None, pseudocount=None, seqid=None):
        self.__pseudocount = pseudocount  if pseudocount is not None else 0.5
        self.__seqid = seqid if seqid is not None else 0.8
        #Validate the value of pseudo count incase user provide an invalid one
        if self.__pseudocount >= 1.0 or self.__pseudocount < 0:
            logger.error('\n\tValue of relative pseudo-count must be'
            ' between 0 and 1.0. Typical value is 0.5')
            raise ValueError
        #Validate the value of sequence identity

        if self.__seqid > 1.0 or self.__seqid <= 0.0:
            logger.error('\n\tValue of sequence-identity must'
            ' not exceed 1 nor less than 0. Typical values are 0.7, 0.8., 0.9')
            raise ValueError
        biomolecule = biomolecule.strip().upper()
        self.__msa_file = msa_file
        if biomolecule=='RNA':
            self.__num_site_states = 5
        elif biomolecule=='PROTEIN':
            self.__num_site_states = 21
        else:
            logger.error(
                '\n\tUnknown biomolecule ... must be protein (PROTEIN) or rna (RNA)',
            )
            raise ValueError
        
        self.__sequences = fasta_reader.get_alignment_int_form(
            self.__msa_file,
            biomolecule=biomolecule,
        )
        print('number of sequences:  ',len(self.__sequences))
        if s_index is None:
            logger.info('S_INDEX not passed\nMake sure the indexing on DI is correct!!!\n\n')
        else: 
            self.__s_index = s_index
        self.__num_sequences = len(self.__sequences)
        self.__sequences_len = len(self.__sequences[0])
        self.__num_threads = 1 if num_threads is None else num_threads
        self.__biomolecule = biomolecule

        #--- Joblib Parallel for msa_numerics.py:computing_sequences_weight() does not yet work ---#
        print('Not accounting for sequence similarity...\n    (must get Joblib, Parallel working first)')
        self.__seqid = 1.0 # therefore you pass on below if statement
        if self.__seqid < 1.0:
            self.__sequences_weight = self.compute_seqs_weight()
        else :
            # assign each sequence a weight of one
            self.__sequences_weight = np.ones((self.__num_sequences,), dtype = np.float64)
        #------------------------------------------------------------------------------------------#

        self.__effective_num_sequences = np.sum(self.__sequences_weight)
        #sometimes users might enter the wrong biomolecule type
        #verify biomolecule type

        er_dca_info = """\n\tCreated a ERDCA object with the following attributes
        \tbiomolecule: {}
        \ttotal states at sites: {}
        \tNumber of cores: {}
        \tpseudocount: {}
        \tsequence identity: {}
        \talignment length: {}
        \ttotal number of unique sequences (excluding redundant sequences with 100 percent similarity): {}
        \teffective number of sequences (with sequence identity {}): {}
        """.format(
            biomolecule,
            self.__num_site_states,
            self.__num_threads,
            self.__pseudocount,
            self.__seqid,
            self.__sequences_len,
            self.__num_sequences,
            self.__seqid,
            self.__effective_num_sequences,
        )
        logger.info(er_dca_info)
        return None


    def __str__(self):
        """Describes the ERDCA object.

        Parameters
        ----------
            self: ERDCA
                Instance of ERDCA class

        Returns
        -------
            description : str
                A representation about objects created from
                the ERDCA class.
        """
        description = '<instance of ERDCA>'
        return description


    def __call__(self, pseudocount = 0.5 , seqid = 0.8):
        """Resets the value of pseudo count and sequence identity through
        the instance.

        Parameters
        ----------
            self : ERDCA
                ERDCA instance.
            pseudocount : float
                The value of the raltive pseudo count. It must be between
                0 and 1. Default value is 0.5.
            seqid : float
                Threshold sequence similarity for computing sequences weight.
                This parameter must be between 0 and 1. Typical values are
                0.7, 0.8, 0.9 or something in between these numbers.

        Returns
        -------
                None : None
        """

        #warn the user that paramertes are being reset
        self.__pseudocount = pseudocount
        self.__seqid = seqid
        logger.warning('\n\tYou have changed one of the parameters (pseudo count or sequence identity)'
        '\n\tfrom their default values'
        '\n\tpseudocount: {} \n\tsequence_identity: {}'.format(
            self.__pseudocount, self.__seqid,
            )
        )
        return None


    @property
    def alignment(self):
        """Alignment data getter.
        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        --------
            self.__sequences : list
                A 2d list of alignment sequences in integer representation.
        """

        return self.__sequences

    @property
    def biomolecule(self):
        """Sequence type getter

        Parameters
        ----------
            Self : ERDCA
                Instance of ERDCA class
        Returns
        -------
            self.__biomolecule : str
                Biomolecule type (protein or RNA)
        """
        return self.__biomolecule
    @property
    def sequences_len(self):
        """Sequences length getter.

        Parameters
        ---------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__sequences_len : int
                Sequences length in alignment data
        """

        return self.__sequences_len


    @property
    def num_site_states(self):
        """Get number of states for an MSA (eg. 5 for RNAs and 21 for proteins)

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__num_site_states : int
                Maximum number of states in a sequence site
        """

        return self.__num_site_states

    @property
    def num_sequences(self):
        """Getter for the number of sequences read from alignment file

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__num_sequences : int
                The total number of sequences in alignment data
        """

        return self.__num_sequences


    @property
    def sequence_identity(self):
        """Getter for the value of sequence indentity.

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__seqid : float
                Cut-off value for sequences similarity above which sequences are
                considered identical
        """

        return self.__seqid


    @property
    def pseudocount(self):
        """Getter for value of pseudo count

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__pseudocount : float
                Value of pseudo count usef for regularization
        """

        return self.__pseudocount


    @property
    def sequences_weight(self):
        """Getter for the weight of each sequences in alignment data.

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            self.__sequences_weight : np.array(dtype=np.float64)
                A 1d numpy array containing the weight of each sequences in the
                alignment.
        """

        return self.__sequences_weight


    @property
    def effective_num_sequences(self):
        """Getter for the effective number of sequences.

        Parameters
        ----------
            self : ERDCA
                Instance of ERDCA class

        Returns
        -------
            np.sum(self.__sequences_weight) : float
                The sum of each sequence's weight.
        """

        return np.sum(self.__sequences_weight)
    
    def compute_sequences_weight(self):
        """Computes the weight of each sequences in the alignment. If the
        sequences identity is one, each sequences has equal weight and this is
        the maximum weight a sequence in the alignment data can have. Whenever
        the sequence identity is set a value less than one, sequences that have
        similarity beyond the sequence identity are lumped together. If there are
        m similar sequences, their corresponding weight is the reciprocal.

        Parameters
        ----------
            self : MeanFieldDCA
                The instance

        Returns
        -------
            weights : np.array
                A 1d numpy array of size self.__num_sequences containing the
                weight of each sequence.
        """

        logger.info('\n\tComputing sequences weights')
        weights = dca_orig_numerics.compute_sequences_weight(
            alignment_data= np.array(self.__sequences, dtype=np.int32),
            seqid = self.__seqid,
        )
        return weights


    def get_single_site_freqs(self):
        """Computes single site frequency counts.

        Parameters
        ----------
            self : MeanFieldDCA
                The instance.

        Returns
        -------
            single_site_freqs : np.array
                A 2d numpy array of shape (L, q) containing the frequency
                count of residues at sequence sites. L is the length of
                sequences in the alignment, and q is the maximum possible
                states a site can accommodate. The last state (q) of each
                site represents a gap.
        """

        logger.info('\n\tComputing single site frequencies')

        single_site_freqs = dca_orig_numerics.compute_single_site_freqs(
            alignment_data = np.array(self.__sequences),
            num_site_states = self.__num_site_states,
            #unique_vals = self.__unique_states,
            seqs_weight = self.__sequences_weight,
            )
        return single_site_freqs


    def get_reg_single_site_freqs(self):
        """Regularizes single site frequencies.

        Parameters
        ----------
            self : MeanFieldDCA
                The instance

        Returns
        -------
            reg_single_site_freqs : np.array
                A 2d numpy array of shape (L, q) containing regularized single
                site frequencies. L and q are the sequences length and maximum
                number of site-states respectively.
        """

        single_site_freqs = self.get_single_site_freqs()

        logger.info('\n\tRegularizing single site frequencies')

        reg_single_site_freqs = dca_orig_numerics.get_reg_single_site_freqs(
            single_site_freqs = single_site_freqs,
            seqs_len = self.__sequences_len,
            num_site_states = self.__num_site_states,
            #mx = self.__mx,
            pseudocount = self.__pseudocount,
        )
        return reg_single_site_freqs


    def get_pair_site_freqs(self):
        """Computes pair site frequencies

        Parameters
        ----------
            self : MeanFieldDCA
                The instance.

        Returns
        -------
            pair_site_freqs : np.array
                A 2d numpy array of pair site frequncies. It has a shape of
                (N, q-1, q-1) where N is the number of unique site pairs and q
                is the maximum number of states a site can accommodate. Note
                site pairig is performed in the following order: (0, 0), (0, 1),
                ..., (0, L-1), ...(L-1, L) where L is the sequences length. This
                ordering is critical that any computation involding pair site
                frequencies must be implemented in the righ order of pairs.
        """

        logger.info('\n\tComputing pair site frequencies')
        #pair_site_freqs = dca_numerics.compute_pair_site_freqs(
        pair_site_freqs = dca_orig_numerics.compute_pair_site_freqs_serial(
        alignment_data = np.array(self.__sequences),
        num_site_states = self.__num_site_states,
        #mx = self.__mx,
        seqs_weight = self.__sequences_weight,
        )
        return pair_site_freqs


    def get_reg_pair_site_freqs(self):
        """Regularizes pair site frequencies

        Parameters
        ----------
            self : MeanFieldDCA
                The instance.

        Returns
        -------
            reg_pair_site_freqs : np.array
                A 3d numpy array of shape (N, q-1, q-1) containing regularized
                pair site frequencies. N is the number of unique site pairs and
                q is the maximum number of states in a sequence site. The
                ordering of pairs follows numbering like (unregularized) pair
                site frequencies.
        """

        pair_site_freqs = self.get_pair_site_freqs()
        logger.info('\n\tRegularizing pair site frequencies')
        reg_pair_site_freqs = dca_orig_numerics.get_reg_pair_site_freqs(
            pair_site_freqs = pair_site_freqs,
            seqs_len = self.__sequences_len,
            num_site_states = self.__num_site_states,
            #mx = self.__mx,
            pseudocount = self.__pseudocount,
        )
        return reg_pair_site_freqs


    def construct_corr_mat(self, reg_fi, reg_fij):
        """Constructs the correlation matrix from regularized frequencies.

        Parameters
        ----------
            self : MeanFieldDCA
                The instance.
            reg_fi : np.array
                Regularized single site frequencies.
            reg_fij : np.array
                Regularized pair site frequncies.

        Returns
        -------
            corr_mat : np.array
                A 2d numpy array of (N, N) where N = L*(q-1) where L and q are
                the length of sequences and number of states in a site
                respectively.
        """

        logger.info('\n\tConstructing the correlation matrix')
        corr_mat = dca_orig_numerics.construct_corr_mat(
            reg_fi = reg_fi,
            reg_fij = reg_fij,
            seqs_len = self.__sequences_len,
            num_site_states = self.__num_site_states,
            #mx = self.__mx,
        )
        return corr_mat


    def compute_couplings(self, corr_mat):
        """Computing couplings by inverting the matrix of correlations. Note that
        the couplings are the negative of the inverse of the correlation matrix.

        Parameters
        ----------
            self : MeanFieldDCA
                The instance.
            corr_mat : np.array
                The correlation matrix formed from regularized  pair site and
                single site frequencies.

        Returns
        -------
            couplings : np.array
                A 2d numpy array of the same shape as the correlation matrix.
        """

        logger.info('\n\tComputing couplings')
        try:
            couplings = dca_orig_numerics.compute_couplings(corr_mat = corr_mat)
        except Exception as e:
            logger.error('\n\tCorrelation {}\n\tYou set the pseudocount {}.'
                ' You might need to increase it.'.format(e, self.__pseudocount)
            )
            raise
        # capture couplings to avoid recomputing
        self.__couplings = couplings 
        logger.info('\n\tMaximum and minimum couplings: {}, {}'.format(
            np.max(couplings), np.min(couplings)))
        return couplings


    def shift_couplings(self, couplings_ij):
        """Shifts the couplings value.

        Parameters
        ----------
            self : ERDCA 
                An instance of ERDCA class
            couplings_ij : np.array
                1d array of couplings for site pair (i, j)
        Returns
        -------
            shifted_couplings_ij : np.array
                A 2d array of the couplings for site pair (i, j)
        """
        qm1 = self.__num_site_states - 1
        couplings_ij = np.reshape(couplings_ij, (qm1,qm1))
        avx = np.mean(couplings_ij, axis=1)
        avx = np.reshape(avx, (qm1, 1))
        avy = np.mean(couplings_ij, axis=0)
        avy = np.reshape(avy, (1, qm1))
        av = np.mean(couplings_ij)
        couplings_ij = couplings_ij -  avx - avy + av
        return couplings_ij 

   
    def  get_mapped_site_pairs_dca_scores(self, sorted_dca_scores, seqbackmapper):
        """Filters mapped site pairs with a reference sequence. 

        Parameters
        -----------
            self : ERDCA
                An instance of ERDCA class
            sorted_dca_scores : tuple of tuples
                A tuple of tuples of site-pair and DCA score sorted by DCA scores 
                in reverse order.
            seqbackmapper : SequenceBackmapper 
                An instance of SequenceBackmapper class
        
        Returns
        -------
            sorted_scores_mapped : tuple
                A tuple of tuples of site pairs and dca score
        """
        mapping_dict = seqbackmapper.map_to_reference_sequence()
        # Add attribute __reseq_mapping_dict
        self.__refseq_mapping_dict = mapping_dict 
        sorted_scores_mapped = list()
        num_mapped_pairs = 0
        for pair, score in sorted_dca_scores:
            try:
                mapped_pair = mapping_dict[pair[0]], mapping_dict[pair[1]]
            except  KeyError:
                pass 
            else:
                current_pair_score = mapped_pair, score 
                sorted_scores_mapped.append(current_pair_score)
                num_mapped_pairs += 1
        # sort mapped pairs in case they were not
        sorted_scores_mapped = sorted(sorted_scores_mapped, key = lambda k : k[1], reverse=True)
        logger.info('\n\tTotal number of mapped sites: {}'.format(num_mapped_pairs))
        return tuple(sorted_scores_mapped)

 
    def compute_seqs_weight(self):
        """Computes sequences weight

        Parameters
        ----------
            self: PlmDCA
                An instance of PlmDCA class
        
        Returns 
        -------
            seqs_weight : np.array
                A 1d numpy array containing sequences weight.
        """
        logger.info('\n\tComputing sequences weight with sequence identity {}'.format(self.__seqid))
        alignment_data = np.array(
            get_alignment_int_form(self.__msa_file, 
            biomolecule = self.__biomolecule)
        )
        seqs_weight = msa_numerics.compute_sequences_weight(
            alignment_data=alignment_data, 
            seqid = self.__seqid,
            num_threads = self.__num_threads
        )
        Meff = np.sum(seqs_weight)
        logger.info('\n\tEffective number of sequences: {}'.format(Meff))
        self.__seqs_weight = seqs_weight 
        self.__eff_num_seqs = Meff
        return seqs_weight 
  


    #========================================================================================================#
    #------------------------------------- Expextation Reflection -------------------------------------------#
    #========================================================================================================#
    # ------- Author: Evan Cresswell-Clay ---------- Date: 8/24/2020 ----------------------------------------#
    #========================================================================================================#
    #========================================================================================================#
    def compute_lader_weights(self,init_w=None):
        """Computing weights by applying Expectation Reflection with LAD inference.

        Parameters
        ----------
            self : ERDCA
                The instance.

        Returns
        -------
            couplings : np.array
                A 2d numpy array of the same shape .
        """

        # THIS IS ASSUMING sequences passed in via MSA file 
		# ---> were already preprocessed
        s0 = np.asarray(self.__sequences)
        print(s0.shape)
        
        n_var = len(s0[0])

        self.__mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
        self.__unique_states = np.array([np.unique(s0[:,i]) for i in range(n_var)])
        # use all possible states at all locations 
        #mx = np.array([self.__num_site_states]* n_var) 
     
        mx_cumsum = np.insert(self.__mx.cumsum(),0,0)

        i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
        
        onehot_encoder = OneHotEncoder(sparse=False)
        
        s = onehot_encoder.fit_transform(s0)
        
        mx_sum = mx.sum()
        my_sum = mx.sum() #!!!! my_sum = mx_sum
        
        w = np.zeros((mx_sum,my_sum))
        h0 = np.zeros(my_sum)

        # Pass computation to parallelization in ER_protein_msa_numerics.py 
        logger.info('\n\tComputing ER couplings')
        try:
            if init_w is not None: 
                res = LADER.compute_lader_weights(n_var,s,i1i2,num_threads= self.__num_threads,couplings = int_w)
            else:
                res = LADER.compute_lader_weights(n_var,s,i1i2,num_threads= self.__num_threads)
        except Exception as e:
            logger.error('\n\tCorrelation {}\n\tYou set the pseudocount {}.'
                ' You might need to increase it.'.format(e, self.__pseudocount)
            )
            raise
   
 
        # Set weight-coupling matrix
        for i0 in range(n_var):
            i1,i2 = i1i2[i0,0],i1i2[i0,1]
               
            h01 = res[i0][0]
            w1 = res[i0][1]
        
            h0[i1:i2] = h01    
            w[:i1,i1:i2] = w1[:i1,:]
            w[i2:,i1:i2] = w1[i1:,:]
        
        # make w to be symmetric
        w = (w + w.T)/2.
    
        # capture couplings (ie symmetric weights matrix) to avoid recomputing
        self.__couplings = w
        logger.info('\n\tMaximum and minimum couplings: {}, {}'.format(
            np.max(w), np.min(w)))

        return w,s


    def compute_er_weights(self,init_w = None):
        """Computing weights by applying Expectation Reflection.

        Parameters
        ----------
            self : ERDCA
                The instance.

        Returns
        -------
            couplings : np.array
                A 2d numpy array of the same shape .
        """

        # THIS IS ASSUMING sequences passed in via MSA file 
		# ---> were already preprocessed
        s0 = np.asarray(self.__sequences)
        print('in compute_er_weights\ns0 shape:',s0.shape)
        
        n_var = s0.shape[1]

        self.__mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])

        self.__unique_states = np.array([np.unique(s0[:,i]) for i in range(n_var)])
        unique_aminos = []
        for states in self.__unique_states: 
            unique_aminos.append(states[states!=21])
        print('uniqe_aminos[0]:\n',unique_aminos[0])
        # use all possible states at all locations 
        #mx = np.array([self.__num_site_states]* n_var) 
     
        mx_cumsum = np.insert(self.__mx.cumsum(),0,0)

        i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
        
        onehot_encoder = OneHotEncoder(sparse=False)
        
        s = onehot_encoder.fit_transform(s0)
        print('s shape:',s.shape)
        
        mx_sum = self.__mx.sum()
        my_sum = self.__mx.sum() #!!!! my_sum = mx_sum

    
        #========================================================================================
        # Compute ER couplings using MF initialization
        #========================================================================================
  
        if init_w is None: 
            if 1:
                print('\n\n#--------------------------------------------------------------------------------- Calulating DCA-couplings for initial weights ---------------------------------------------------------------------------------\n\n')
                reg_fi = self.get_reg_single_site_freqs()
                reg_fij = self.get_reg_pair_site_freqs()
                corr_mat = self.construct_corr_mat(reg_fi, reg_fij)
                print('corr_mat (orig DCA-couplings) shape: ', corr_mat.shape)
                couplings = self.compute_couplings(corr_mat)
                print('couplings (orig DCA-couplings) shape: ', couplings.shape)
                print('couplings first row:\n', couplings[0,:21])

                gap_state_indices = [x for x in range(couplings.shape[1] +1) if x % 20==0 and x!=0]
                print('gap_stat_indices (len=%d): '%(len(gap_state_indices)), gap_state_indices)

                er_couplings = np.insert(couplings,gap_state_indices,0 ,axis=1)
                er_couplings = np.insert(er_couplings,gap_state_indices,0,axis=0)
                print('er_couplings (orig DCA-couplings with -) shape: ', er_couplings.shape)
                print('er_couplings first row:\n', er_couplings[0,:21])


                # list of columns 
                print(mx_cumsum)
                non_states = []
                col = 0
                mx_col = 0
                for i in range(n_var):
                    #print(mx_col)
                    for j in range(1,self.__num_site_states+1):
                        if i==0:
                            print(j)
                        if j!=21:
                            if j not in unique_aminos[i]:
                                non_states.append(col)
                            else:
                                mx_col+=1
                        col += 1
                print('deleting %d colums/rows of %d, column/row count should be %d '%(len(non_states),col,s.shape[1]))
                er_couplings = np.delete(er_couplings,non_states,axis=0)
                er_couplings = np.delete(er_couplings,non_states,axis=1)

                #np.save('pfam_ecc/%s_couplings.npy'%(pfam_id),couplings)
                print('er_couplings (orig DCA-couplings trimmed non-states) shape: ', er_couplings.shape)
                print('er_couplings first row:\n', couplings[0][:21])
                print('\n\n#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n')
                w_in = er_couplings

		 
            else:
                #========================================================================================
                # ER - COV-COUPLINGS
                #========================================================================================
                
                s_av = np.mean(s,axis=0)
                ds = s - s_av
                l,n = s.shape
                
                l2 = 100.
                # calculate covariance of s (NOT DS) why not???
                s_cov = np.cov(s,rowvar=False,bias=True)
                print('s_cov (orig DCA-couplings) shape: ', s_cov.shape)
                # tai-comment: 2019.07.16:  l2 = lamda/(2L)
                s_cov += l2*np.identity(n)/(2*l)

                theta_by_qsqrd = self.__pseudocount / float(self.__num_site_states ** 2. )
                s_cov += theta_by_qsqrd*np.identity(n)

                s_inv = linalg.pinvh(s_cov)

                w_in = s_inv
                print(np.shape(s_inv))
                #sys.exit()


        else:
            print('\n\nUsing passed init_w for initial weights\n',init_w)
            w_in = init_w    
 
        w = np.zeros((mx_sum,my_sum))
        h0 = np.zeros(my_sum)



        # Pass computation to parallelization in msa_numerics.py 
        logger.info('\n\tComputing ER couplings')
        try:
            res = msa_numerics.compute_er_weights(n_var,s,i1i2,num_threads= self.__num_threads,couplings=w_in)

        except Exception as e:
            logger.error('\n\tCorrelation {}\n\tYou set the pseudocount {}.'
                ' You might need to increase it.'.format(e, self.__pseudocount)
            )
            raise
   
 
        # Set weight-coupling matrix
        for i0 in range(n_var):
            i1,i2 = i1i2[i0,0],i1i2[i0,1]
               
            h01 = res[i0][0]
            w1 = res[i0][1]
        
            h0[i1:i2] = h01    
            w[:i1,i1:i2] = w1[:i1,:]
            w[i2:,i1:i2] = w1[i1:,:]
        
        # make w to be symmetric
        w = (w + w.T)/2.
    
        # capture couplings (ie symmetric weights matrix) to avoid recomputing
        self.__couplings = w
        logger.info('\n\tMaximum and minimum couplings: {}, {}'.format(
            np.max(w), np.min(w)))

        return w,s

    # altering the following two functions (originally defined) to implement ER
    def get_site_pair_di_score(self,LAD=False,init_w=None):
        # my version
        """Obtains Expectation Reflection weights and computes direct information (DI) scores 
        and puts them a list of tuples of in (site-pair, score) form.

        Parameters
        ----------
            self : ERDCA
                The instance.

        Returns
        -------
            site_pair_di_score : list
                A list of tuples containing site pairs and DCA score, i.e., the
                list [((i, j), score), ...] for all unique ite pairs (i, j) 
                such that j > i.
        """
        if init_w is not None: 
            print('using initial w:\n',init_w)
            print(init_w.shape)
        if LAD:
            couplings, s = self.compute_lader_weights(init_w)
        else:
            couplings, s = self.compute_er_weights(init_w)
        print('ER couplings dimensions: ', couplings.shape)

        # Compute DI using local computation in msa_numerics.py
        #fields_ij = self.compute_two_site_model_fields(couplings, reg_fi)

        logger.info('\n\tComputing direct information')

        s0 = np.asarray(self.__sequences)
        di = msa_numerics.direct_info(s0,couplings)
        

        site_pair_di_score= dict()
        ind = np.unravel_index(np.argsort(di,axis=None),di.shape)    
        for i,indices in enumerate(np.transpose(ind)):
            site_pair = (indices[0] , indices[1])
            site_pair_di_score[site_pair] = di[indices[0] , indices[1]]

        return site_pair_di_score


    def compute_sorted_DI(self,LAD=False,init_w=None,seqbackmapper=None):
        # my version
        """Computes direct informations for each pair of sites and sorts them in
        descending order of DCA score.

        Parameters
        ----------
            self : ERDCA
                The instance.
            seqbackmapper : SequenceBackmapper
                An instance of SequenceBackmapper class.

        Returns
        -------
            sorted_DI : list
                A list of tuples containing site pairs and DCA score, i.e., the
                contents of sorted_DI are [((i, j), score), ...] for all unique
                site pairs (i, j) such that j > i.
        """
        print('Computing site pair DI scores')
        unsorted_DI = self.get_site_pair_di_score(LAD,init_w)
        print('Sorting DI scores')
        sorted_DI = sorted(unsorted_DI.items(), key = lambda k : k[1], reverse=True)
        if seqbackmapper is not None:
            print('Mapping site pair DCA scores')
            sorted_DI = self.get_mapped_site_pairs_dca_scores(sorted_DI, seqbackmapper)
            print(sorted_DI[:10],'\n')

        print('Imposing Distance Restraint')
        if self.__s_index is not None:
            #sorted_DI = self.distance_restr_sortedDI(sorted_DI,s_index=self.__s_index)
            #sorted_DI = self.s_index_map(sorted_DI,s_index=self.__s_index)
            print('\n\nNO COLUMNS REMOVED POST TRIMMING\ns_index should be range 0 - seq_len\ns_index: ',self.__s_index)
        else:
            print('(NO TRUE INDEXING ON PREPROCESSED DATA)')
            # no longer enforcing distance restr (linear distance handled in contact_visualizer)
            #sorted_DI = self.distance_restr_sortedDI(sorted_DI)
        print('Deleting DI duplicates')
        sorted_DI = self.delete_sorted_DI_duplicates(sorted_DI)

        return sorted_DI


    def s_index_map(self,site_pair_DI_in, s_index=None):
        print(site_pair_DI_in[:10])
        s_index_DI= dict()
        for site_pair, score in site_pair_DI_in:
            pos_0 = s_index[site_pair[0]]
            pos_1 = s_index[site_pair[1]]
    
       	    indices = (pos_0 , pos_1)
    
            s_index_DI[indices] = score

        sorted_DI  = sorted(s_index_DI.items(), key = lambda k : k[1], reverse=True)
        print(sorted_DI[:10])
        return sorted_DI
    

    def distance_restr_sortedDI(self,site_pair_DI_in, s_index=None):
        print(site_pair_DI_in[:10])
        restrained_DI= dict()
        for site_pair, score in site_pair_DI_in:
            # if s_index exists re-index sorted pair
            if s_index is not None:
                pos_0 = s_index[site_pair[0]]
                pos_1 = s_index[site_pair[1]]
            else:
    	        pos_0 = site_pair[0]
    	        pos_1 = site_pair[1]
    
       	    indices = (pos_0 , pos_1)
    
            if abs(pos_0- pos_1)<5:
                restrained_DI[indices] = 0
            else:
                restrained_DI[indices] = score
        sorted_DI  = sorted(restrained_DI.items(), key = lambda k : k[1], reverse=True)
        print(sorted_DI[:10])
        return sorted_DI
    
    def delete_sorted_DI_duplicates(self,sorted_DI):
        temp1 = []
        print(sorted_DI[:10])
        DI_out = dict() 
        for (a,b), score in sorted_DI:
             if (a,b) not in temp1 and (b,a) not in temp1: #to check for the duplicate tuples
                temp1.append(((a,b)))
                if a>b:
                    DI_out[(b,a)]= score
                else:
                    DI_out[(a,b)]= score
        DI_out = sorted(DI_out.items(), key = lambda k : k[1], reverse=True)
        #DI_out.sort(key=lambda x:x[1],reverse=True) 
        print(DI_out[:10])
        return DI_out 
    
    #========================================================================================================#
    #========================================================================================================#
    #========================================================================================================#
if __name__ == '__main__':
    """
    """
