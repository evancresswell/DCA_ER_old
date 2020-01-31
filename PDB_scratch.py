{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "        <script>\n",
       "            function code_toggle_12150235286856347335() {\n",
       "                $('div.cell.code_cell.rendered.selected').find('div.input').toggle();\n",
       "            }\n",
       "\n",
       "            \n",
       "        </script>\n",
       "\n",
       "        <a href=\"javascript:code_toggle_12150235286856347335()\">Toggle show/hide</a>\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import ecc_tools as tools\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy import linalg\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "#import emachine as EM\n",
    "from direct_info import direct_info\n",
    "from ecc_tools import distance_restr\n",
    "\n",
    "import Bio.PDB, warnings\n",
    "pdb_list = Bio.PDB.PDBList()\n",
    "pdb_parser = Bio.PDB.PDBParser()\n",
    "from scipy.spatial import distance_matrix\n",
    "from Bio import BiopythonWarning\n",
    "warnings.simplefilter('ignore', BiopythonWarning)\n",
    "\n",
    "from scipy.sparse import csr_matrix\n",
    "from joblib import Parallel, delayed\n",
    "import timeit\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "import sys\n",
    "import numpy as np\n",
    "from scipy import linalg\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "import expectation_reflection as ER\n",
    "from direct_info import direct_info\n",
    "from joblib import Parallel, delayed\n",
    "\n",
    "# import pydca for plmDCA\n",
    "from pydca.plmdca import plmdca\n",
    "from pydca.meanfield_dca import meanfield_dca\n",
    "from pydca.sequence_backmapper import sequence_backmapper\n",
    "from pydca.msa_trimmer import msa_trimmer\n",
    "from pydca.contact_visualizer import contact_visualizer\n",
    "from pydca.dca_utilities import dca_utilities\n",
    "\n",
    "tools.hide_toggle()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "@> Connecting wwPDB FTP server RCSB PDB (USA).\n",
      "@> 1k2a downloaded (1k2a.pdb.gz)\n",
      "@> PDB download via FTP completed (1 downloaded, 0 failed).\n",
      "@> UniProt ID code RNAS2_HUMAN for 1k2a chain A will be used.\n",
      "@> Retrieving Pfam search results: https://pfam.xfam.org/protein/RNAS2_HUMAN?output=xml\n",
      "@> Pfam search completed in 1.78s.\n",
      "@> Query '1k2a' matched 1 Pfam families.\n",
      "@> Pfam MSA for PF00074 is written as PF00074_full.sth.\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'numpy.core._multiarray_umath'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'numpy.core._multiarray_umath'"
     ]
    },
    {
     "ename": "ImportError",
     "evalue": "numpy.core.multiarray failed to import",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-13-09eef9fcec5e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mfetchPfamMSA\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'PF00074'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m \u001b[0mmsa\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mparseMSA\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'PF00074_full.sth'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/lib/python3.7/site-packages/prody/sequence/msafile.py\u001b[0m in \u001b[0;36mparseMSA\u001b[0;34m(filename, **kwargs)\u001b[0m\n\u001b[1;32m    589\u001b[0m             \u001b[0mmsaarr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mempty\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilesize\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'|S1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    590\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mformat\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mSELEX\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0mformat\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mSTOCKHOLM\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 591\u001b[0;31m             \u001b[0;32mfrom\u001b[0m \u001b[0;34m.\u001b[0m\u001b[0mmsaio\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mparseSelex\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mparser\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    592\u001b[0m             \u001b[0mmsaarr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mempty\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilesize\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'|S1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    593\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mformat\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0mCLUSTAL\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mImportError\u001b[0m: numpy.core.multiarray failed to import"
     ]
    }
   ],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "# This code was copied from ProDy documentation.\n",
    "# Title: Evolution Analysis — ProDy\n",
    "# URL: http://prody.csb.pitt.edu/tutorials/evol_tutorial/msaanalysis.html#get-msa-data\n",
    "\n",
    "from prody import *\n",
    "from pylab import *\n",
    "Pfam_keys = searchPfam('1k2a').keys()\n",
    "\n",
    "fetchPfamMSA('PF00074')\n",
    "\n",
    "msa = parseMSA('PF00074_full.sth')\n",
    "\n",
    "\n",
    "# Read in Protein structure\n",
    "data_path = './'\n",
    "pfam_id = 'PF00186'\n",
    "msa_outfile = data_path + pfam_id + '.fa'\n",
    "ref_outfile = 'ref_'+pfam_id+'.fa'\n",
    "# create MSATrimmer instance \n",
    "trimmer = msa_trimmer.MSATrimmer(\n",
    "    msa_outfile, biomolecule='protein', \n",
    "    refseq_file=ref_outfile,\n",
    ")\n",
    "\n",
    "trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)\n",
    "\n",
    "#write trimmed msa to file in FASTA format\n",
    "trimmed_data_outfile = 'MSA_'+pfam_id+'_Trimmed.fa'\n",
    "with open(trimmed_data_outfile, 'w') as fh:\n",
    "    for seqid, seq in trimmed_data:\n",
    "        fh.write('>{}\\n{}\\n'.format(seqid, seq))\n",
    "\n",
    "tools.hide_toggle()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compute DCA scores using Pseudolikelihood maximization algorithm\n",
    "\n",
    "plmdca_inst = plmdca.PlmDCA(\n",
    "    trimmed_data_outfile,\n",
    "    'protein',\n",
    "    seqid = 0.8,\n",
    "    lambda_h = 1.0,\n",
    "    lambda_J = 20.0,\n",
    "    num_threads = 10,\n",
    "    max_iterations = 500,\n",
    ")\n",
    "#print(\"properties of plmDCA object: \\n\",dir(plmdca_inst),\"\\n\\n\")\n",
    "\n",
    "sorted_DI = plmdca_inst.compute_sorted_DI()\n",
    "\n",
    "N = plmdca_inst.sequences_len\n",
    "B = plmdca_inst.num_sequences\n",
    "print(\"sorted_DI (list) has shape: \",np.array(sorted_DI).shape,\"\\nRows: all combinations of sequence postitions ie sequence len choose 2 \")\n",
    "print(\"num sequences = %d , sequences len = %d\\n\\n\"%(B,N))\n",
    "\n",
    "print(\"Print top 10 pairs\")\n",
    "for site_pair, score in sorted_DI[:10]:\n",
    "    print(site_pair, score)\n",
    "\n",
    "\n",
    "# compute DCA scores summarized by Frobenius norm and average product corrected\n",
    "plmdca_FN_APC = plmdca_inst.compute_sorted_FN_APC()\n",
    "tools.hide_toggle()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# compute DCA scores summarized by Frobenius norm and average product corrected\n",
    "plmdca_FN_APC = plmdca_inst.compute_sorted_FN_APC()\n",
    "\n",
    "plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],\n",
    "    refseq_file = ref_outfile,\n",
    "    sorted_dca_scores = plmdca_FN_APC,\n",
    "    linear_dist = 4,\n",
    "    contact_dist = 8.0,\n",
    ")\n",
    "\n",
    "contact_map_data = plmdca_visualizer.plot_contact_map()\n",
    "tp_rate_data = plmdca_visualizer.plot_true_positive_rates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
