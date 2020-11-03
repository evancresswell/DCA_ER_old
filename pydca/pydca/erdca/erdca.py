from __future__ import absolute_import, division
from . import msa_numerics
from . import ER_protein_msa_numerics as LADER
from pydca.fasta_reader import fasta_reader
import logging
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.fasta_reader.fasta_reader import get_alignment_from_fasta_file
from pydca.fasta_reader.fasta_reader import get_alignment_int_form


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
    def compute_cov_weights(self):
        """Computing weights by applying Expectation Reflection initializing with the inverse of the covariance (ie cov_coupling) .

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

        mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
        # use all possible states at all locations 
        #mx = np.array([self.__num_site_states]* n_var) 
     
        mx_cumsum = np.insert(mx.cumsum(),0,0)

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



    def compute_lader_weights(self,initialize):
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

        mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
        # use all possible states at all locations 
        #mx = np.array([self.__num_site_states]* n_var) 
     
        mx_cumsum = np.insert(mx.cumsum(),0,0)

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


    def compute_er_weights(self,initialize):
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
        print(s0.shape)
        
        n_var = len(s0[0])

        mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
        # use all possible states at all locations 
        #mx = np.array([self.__num_site_states]* n_var) 
     
        mx_cumsum = np.insert(mx.cumsum(),0,0)

        i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
        
        onehot_encoder = OneHotEncoder(sparse=False)
        
        s = onehot_encoder.fit_transform(s0)
        
        mx_sum = mx.sum()
        my_sum = mx.sum() #!!!! my_sum = mx_sum
        
        w = np.zeros((mx_sum,my_sum))
        h0 = np.zeros(my_sum)

        # Iinitialize weights using pseuod inverse.
        if initialize:
             #========================================================================================
             # ER - COV-COUPLINGS
             #========================================================================================
             
             s_av = np.mean(s,axis=0)
             ds = s - s_av
             l,n = s.shape
             
             l2 = 100.
             # calculate covariance of s (NOT DS) why not???
             s_cov = np.cov(s,rowvar=False,bias=True)
             # tai-comment: 2019.07.16:  l2 = lamda/(2L)
             s_cov += l2*np.identity(n)/(2*l)
             s_inv = linalg.pinvh(s_cov)
             print('s_inv shape: ', s_inv.shape)
             
             

        # Pass computation to parallelization in msa_numerics.py 
        logger.info('\n\tComputing ER couplings')
        try:
            if initialize:
                res = msa_numerics.compute_er_weights(n_var,s,i1i2,num_threads= self.__num_threads,couplings=s_inv)
            else:
                res = msa_numerics.compute_er_weights(n_var,s,i1i2,num_threads= self.__num_threads)

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
    def get_site_pair_di_score(self,LAD=False,initialize=False):
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

        if LAD:
            couplings, s = self.compute_lader_weights(initialize)
        else:
            couplings, s = self.compute_er_weights(initialize)
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


    def compute_sorted_DI(self,LAD=False,initialize=False,seqbackmapper=None):
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
        unsorted_DI = self.get_site_pair_di_score(LAD,initialize)
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
