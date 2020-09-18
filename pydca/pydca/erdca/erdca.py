from __future__ import absolute_import, division
from . import msa_numerics
from pydca.fasta_reader import fasta_reader
import logging
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.fasta_reader.fasta_reader import get_alignment_from_fasta_file
from pydca.fasta_reader.fasta_reader import get_alignment_int_form

"""This module implements Direc Coupling Analysis (DCA) of residue coevolution
for protein and RNA sequences using the mean-field algorithm. The final
coevolution score is computed from the direct probability. The general steps
carried out are outlined as follows

For a detailed information about Direct Coupling Analysis, one can refer to the
following articles:

    a)  Identification of direct residue contacts in protein-protein interaction
        by message-passing
        Martin Weigt, Robert A White, Hendrik Szurmant, James A Hoch, Terence Hwa
        Journal: Proceedings of the National Academy of Sciences
        Volume: 106
        Issue: 1
        Pages: 67-72
    b)  Direct-coupling analysis of residue coevolution captures native contacts
        across many protein families
        Faruck Morcos, Andrea Pagnani, Bryan Lunt, Arianna Bertolino,
        Debora S Marks, Chris Sander, Riccardo Zecchina, Jose N Onuchic,
        Terence Hwa, Martin Weigt
        Journal: Proceedings of the National Academy of Sciences
        Volume: 108
        Issue: 49
        Pages: E1293-E1301

Author(s)  Mehari B. Zerihun, Alexander Schug
"""

logger = logging.getLogger(__name__)

class ERDCAException(Exception):
    """
    """

class ERDCA:
    """ERDCA class. Instances of this class are used to carry out Direct
    Coupling Analysis (DCA) of residue coevolution using the mean-field DCA
    algorithm.
    """
    def __init__(self, msa_file, biomolecule, num_threads = None, pseudocount=None, seqid=None):
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
    def compute_er_weights(self):
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

        # THIS IS ASSUMING UNTRIMMED?? NO S_INDEX>>>>>>>>???       
        s0 = np.asarray(self.__sequences)
        print(s0.shape)
        
        n_var = s0.shape[1]

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

        # Pass computation to parallelization in msa_numerics.py 
        logger.info('\n\tComputing ER couplings')
        try:
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
    def get_site_pair_di_score(self):
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

        couplings, s = self.compute_er_weights()
        print('ER couplings dimensions: ', couplings.shape)

        # Compute DI using local computation in msa_numerics.py
        #fields_ij = self.compute_two_site_model_fields(couplings, reg_fi)

        logger.info('\n\tComputing direct information')
        if 0:
            unsorted_DI = msa_numerics.compute_direct_info(
            couplings = couplings,
            fields_ij = fields_ij,
            reg_fi = reg_fi,
            seqs_len = self.__sequences_len,
            num_site_states = self.__num_site_states,
            )

        s0 = np.asarray(self.__sequences)
        di = msa_numerics.direct_info(s0,couplings)
        

        site_pair_di_score= dict()
        pair_counter = 0
        ind = np.unravel_index(np.argsort(di,axis=None),di.shape)    
        for i,indices in enumerate(np.transpose(ind)):
            site_pair = (indices[0] , indices[1])
            site_pair_di_score[site_pair] = di[indices[0] , indices[1]]
            pair_counter += 1

        return site_pair_di_score


    def compute_sorted_DI(self,seqbackmapper=None):
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
        unsorted_DI = self.get_site_pair_di_score()
        sorted_DI = sorted(unsorted_DI.items(), key = lambda k : k[1], reverse=True)
        if seqbackmapper is not None:
            sorted_DI = self.get_mapped_site_pairs_dca_scores(sorted_DI, seqbackmapper)
        #print(sorted_DI)

        sorted_DI = self.distance_restr_sortedDI(sorted_DI)
        sorted_DI = self.delete_sorted_DI_duplicates(sorted_DI)

        return sorted_DI

    def distance_restr_sortedDI(self,site_pair_DI_in, s_index=None):
        print(site_pair_DI_in[:10])
        pair_counter = 0
        sorted_DI= dict()
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
                sorted_DI[indices] = 0
                #sorted_DI[count] = (pos_0,pos_1),0
            else:
                sorted_DI[indices] = score
                #sorted_DI[count] = (pos_0,pos_1),score
            pair_counter += 1
        sorted_DI = sorted(sorted_DI.items(),reverse=True)  
        print(sorted_DI[:10])
        return sorted_DI
    
    def delete_sorted_DI_duplicates(self,sorted_DI):
        temp1 = []
        print(sorted_DI[:10])
        DI_out = dict() 
        for (a,b), score in sorted_DI:
             if (a,b) not in temp1 and (b,a) not in temp1: #to check for the duplicate tuples
                temp1.append(((a,b)))
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
