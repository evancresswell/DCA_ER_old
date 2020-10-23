"""pydca demo

Author: Evan Cresswell-Clay
"""
import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC
import matplotlib.pyplot as plt

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
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

pfam_id = sys.argv[1]
num_threads = int(sys.argv[2])

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
processed_data_path = '/data/cresswellclayec/DCA_ER/biowulf/pfam_ecc/'

# pdb_ref should give list of
# 0) accession codes,
# 1) number of residues,
# 2) number of sequences,
# 3) and number of pdb references
# Read in Reference Protein Structure
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
# convert bytes to str (python 2 to python 3)                                                                       
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])      for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
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
pdb_range = [pdb_start-1, pdb_end]
#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    

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
print('PDB Polypeptide Sequence (len=%d): \n'%len(poly_seq),poly_seq)
poly_seq_range = poly_seq[pdb_range[0]:pdb_range[1]]
print('PDB Polypeptide Sequence (In Proteins PDB range len=%d): \n'%len(poly_seq_range),poly_seq_range)

#check that poly_seq matches up with given MSA
    
pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path=processed_data_path)

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
#Add correct range of PP sequence to MSA using muscle
#https://www.drive5.com/muscle/manual/addtomsa.html
muscling = False
muscle_msa_file = processed_data_path + 'PP_muscle_msa_'+pfam_id+'.fa'
if muscling:
        
        # Search MSA for for PP-seq (in range) match.
        slice_size = 10
        matches = np.zeros(s.shape[0])
        #print(poly_seq_range)
        
        # MSA Sequence which best matches PP Sequence
        for i in range(len(poly_seq_range)-slice_size + 1):
            poly_seq_slice = poly_seq_range[i:i+slice_size]
            print(''.join(poly_seq_slice))
            #print(s.shape)
            for row in range(s.shape[0]):
                #print(''.join(s[row,:]).replace('-','').upper())
                if ''.join(poly_seq_slice).upper() in ''.join(s[row,:]).replace('-','').upper():
                    #print('Match found on Row %d : \n'%row,''.join(s[row,:]).replace('-','').upper() )
                    matches[row]+=1    
            
        
        m = max(matches)
        if m > 0:
            best_matches_indx = [i for i,j in enumerate(matches) if j==m]
        print('PP sequence: \n',''.join(poly_seq_range).upper())
        print('best match in msa at indices: ',best_matches_indx)
        match_ref = ''.join(s[best_matches_indx[0],:]).replace('-','').upper()
        print('MSA sequence: \n',match_ref,'\n\n Finding index mapping...')
        
        import re
        largest_match = [0,-1]
        for i in range(len(poly_seq_range)-slice_size + 1):
            poly_seq_slice = poly_seq_range[i:i+slice_size]
            pp_slice_string = ''.join(poly_seq_slice).upper()
            #print(''.join(s[row,:]).replace('-','').upper())
            if pp_slice_string in match_ref:
                matching = True
                match_start = re.search(pp_slice_string,match_ref).start()
                #rint('finding length of match starting at ',match_start)
                ii = match_start+1
                while(matching):
                    ii = ii + 1
                    if ii > len(poly_seq_range)-slice_size:
                        match_end = regex_match.end()
                        break
                    poly_seq_slice = poly_seq_range[ii:ii+slice_size]
                    pp_slice_string = ''.join(poly_seq_slice).upper()
                    if pp_slice_string in match_ref:
                        regex_match = re.search(pp_slice_string,match_ref)
                        match_end = regex_match.end()
                    else:
                        if match_end - match_start > largest_match[1] - largest_match[0]:
                            largest_match = [match_start,match_end]
                            print('                      ...')
                        matching = False
                        i = match_end+1
        
        print('\n\nMSA match range', largest_match)
        
        # Find indices corresponding for MSA's matching reference and PP sequence.
        new_match_ref = match_ref[largest_match[0]:largest_match[1]]
        regex_polyseq = re.search(''.join(new_match_ref),''.join(poly_seq_range) )
        new_poly_seq = ''.join(poly_seq_range)[regex_polyseq.start(): regex_polyseq.end()] 
        
        print('PP match range [%d, %d]\n\nMatched sequences: \n' %(regex_polyseq.start(), regex_polyseq.end()))
        
        print(''.join(new_poly_seq))
        print(''.join(new_match_ref))
        
        # Write new poly peptide sequence to fasta for alignment    
        pp_msa_file, pp_ref_file = tools.write_FASTA(new_poly_seq, s, pfam_id, number_form=False,processed=False,path=processed_data_path)
        
       	sys.exit() 
    
        #just add using muscle:
        #https://www.drive5.com/muscle/manual/addtomsa.html
        #https://www.drive5.com/muscle/downloads.htmL
        os.system("./muscle -profile -in1 %s -in2 %s -out %s"%(pp_msa_file,pp_ref_file,muscle_msa_file))
        print("PP sequence added to alignment via MUSCLE")

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            


preprocessing = True

muscling = True # so we use the pp_range-MSAmatched and muscled file!!

preprocessed_data_outfile = processed_data_path + 'MSA_%s_PreProcessed.fa'%pfam_id
if preprocessing:
        try:

            # create MSATrimmer instance 
            if muscling:
                trimmer = msa_trimmer.MSATrimmer(
                    muscle_msa_file, biomolecule='PROTEIN',
                    refseq_file=pp_ref_file
                )
            else:
                trimmer = msa_trimmer.MSATrimmer(
                    pp_msa_file, biomolecule='PROTEIN',
                    refseq_file=pp_ref_file
                )  

            # Adding the data_processing() curation from tools to erdca.
            preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
            
        except(MSATrimmerException):
            try:
                # Re-Write references file as original pp sequence with pdb_refs-range
                pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False)
                trimmer = msa_trimmer.MSATrimmer(
                        pp_msa_file, biomolecule='PROTEIN',
                        refseq_file=pp_ref_file
                    )

                # Adding the data_processing() curation from tools to erdca.
                preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)


            except(MSATrimmerException):
                ERR = 'PPseq-MSA'
                print('Error with MSA trimms (%s)'%ERR)
                sys.exit()




        """
        # create MSATrimmer instance 
        if muscling:
            trimmer = msa_trimmer.MSATrimmer(
                muscle_msa_file, biomolecule='PROTEIN',
                refseq_file=pp_ref_file
            )
        else:
            trimmer = msa_trimmer.MSATrimmer(
                pp_msa_file, biomolecule='PROTEIN',
                refseq_file=pp_ref_file
            )  

        # Adding the data_processing() curation from tools to erdca.
        try:

                preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
        except(MSATrimmerException):
                ERR = 'PPseq-MSA'
                print('Error with MSA trimms (%s)'%ERR)
                sys.exit()
        """
        # Save processed data dictionary and FASTA file
        pfam_dict = {}
        pfam_dict['s0'] = s
        pfam_dict['msa'] = preprocessed_data
        pfam_dict['s_index'] = s_index
        pfam_dict['s_ipdb'] = s_ipdb
        pfam_dict['cols_removed'] = cols_removed

        input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
        with open(input_data_file,"wb") as f:
                pickle.dump(pfam_dict, f)
        f.close()

        #write trimmed msa to file in FASTA format
        with open(preprocessed_data_outfile, 'w') as fh:
            for seqid, seq in preprocessed_data:
                fh.write('>{}\n{}\n'.format(seqid, seq))
else:
        input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
        with open(input_data_file,"rb") as f:
                pfam_dict =  pickle.load(f)
        f.close()
        cols_removed = pfam_dict['cols_removed']
        s_index= pfam_dict['s_index']
        s_ipdb = pfam_dict['s_ipdb']
        preprocess_data = pfam_dict['msa']
        print('Shape of data which gives our predictions: ',np.array(preprocess_data).shape)
        preprocessed_data = []
        for seq_info in preprocess_data:
            preprocessed_data.append([char for char in seq_info[1]])
        print(np.array(preprocessed_data).shape)



#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            

computing_DI = True
if computing_DI:
        # Compute DI scores using Expectation Reflection algorithm
        erdca_inst = erdca.ERDCA(
            preprocessed_data_outfile,
            'PROTEIN',
            s_index = s_index,
            pseudocount = 0.5,
            num_threads = num_threads,
            seqid = 0.8)

        # Compute average product corrected Frobenius norm of the couplings
        start_time = timeit.default_timer()
        erdca_DI = erdca_inst.compute_sorted_DI()
        run_time = timeit.default_timer() - start_time
        print('ER run time: %f \n\n'%run_time)

        for site_pair, score in erdca_DI[:5]:
            print(site_pair, score)

        di_filename = 'DI/ER/test_er_DI_%s.pickle'%(pfam_id)
        print('\n\nSaving file as ', di_filename)
        with open(di_filename, 'wb') as f:
            pickle.dump(erdca_DI, f)
        f.close()
else:
        with open('DI/ER/tesit_er_DI_%s.pickle'%(pfam_id), 'rb') as f:
            erdca_DI = pickle.load( f)
        f.close()

#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
# Print Details of protein PDB structure Info for contact visualizeation
erdca_plotting = False
if erdca_plotting:
    
    # translate DI indices to pdb range:
    pdb_s_index = np.arange(pdb_range[0],pdb_range[1])
    pdb_s_index = np.delete(pdb_s_index,cols_removed)
    print(s_index)
    print(pdb_s_index)
    pdb_range_DI = dict()
    for (indx1,indx2), score in erdca_DI:
            if indx1 in s_index and indx2 in s_index:
                pos1 = pdb_s_index[np.where(s_index==indx1)[0][0]] - pdb_range[0]
                pos2 = pdb_s_index[np.where(s_index==indx2)[0][0]] - pdb_range[0]
                indices = (pos1,pos2)
                pdb_range_DI[indices] = score
    print(erdca_DI[:10])
    pdb_range_DI  = sorted(pdb_range_DI.items(), key = lambda k : k[1], reverse=True)
    print(pdb_range_DI[:10])

    print('Using chain ',pdb_chain)
    print('PDB ID: ', pdb_id)

    
    
    from pydca.contact_visualizer import contact_visualizer
    erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
    refseq_file = pp_ref_file,
    #---------------- DI For Visualization ----------------#                                                    
    sorted_dca_scores = erdca_DI,
    #sorted_dca_scores = pdb_range_DI,
    #------------------------------------------------------#        
    linear_dist = 5,
    contact_dist = 10.)

    
    
    
    er_contact_map_data = erdca_visualizer.plot_contact_map()
    plt.show()
    plt.savefig('contact_map_%s.pdf'%pfam_id)
    print(er_contact_map_data.keys())
    print("TP contacts: ",len(er_contact_map_data['tp']))
    print("FP contacts: ",len(er_contact_map_data['fp']))
    print("Missing Contacts: ",len(er_contact_map_data['missing']))
    print("PDB Contacts ",len(er_contact_map_data['pdb']))


    plt.close()
    er_tp_rate_data = erdca_visualizer.plot_true_positive_rates()
    plt.show()
    print(len(er_tp_rate_data))
    print(len(er_tp_rate_data[0]['dca']))
    print(len(er_tp_rate_data[0]['pdb']))
    print(er_tp_rate_data[1])

    plt.savefig('TP_rate_%s.pdf'%pfam_id)
    plt.close()

