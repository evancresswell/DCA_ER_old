from joblib import Parallel, delayed
import os
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import ecc_tools as tools
import data_processing as dp

# import inference_dca for mfDCA
from inference_dca import direct_info_dca

# import pydca for plmDCA
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities

data_path = '../../Pfam-A.full'
directory = './DI/ER/'


def gen_rem_col(filename):
		pfam_id = filename.strip('er_DI.pickle')
		print (pfam_id)
		# data processing
		ipdb = 0
		s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
						gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
#-------------------------------
# parallel
res = Parallel(n_jobs = 32)(delayed(gen_rem_col)\
        (filename)\
        for filename in os.listdir(directory) if filename.endswith(".pickle"))


