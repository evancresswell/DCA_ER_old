import os
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

from direct_info import sort_di

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

data_path = '../../hoangd2_data/Pfam-A.full'
data_path = '../../Pfam-A.full'

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'

s_test = np.loadtxt('test_list.txt',dtype='str')

# Get number of data files
num_files = len([name for name in os.listdir(er_directory) if name.endswith(".pickle")])
print("Plotting analysis for %d Proteins"% num_files)

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
	#for filename in os.listdir(directory):
	#	if filename.endswith(".pickle"):
	#		pfam_id = filename.strip('er_DI.pickle')
with PdfPages('multipage_pdf.pdf') as pdf:
	for pfam_id in s_test:
		print ('Plotting Protein Famility ', pfam_id)
	
		# Load PDB structure 
		pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

		#---------- Pre-Process Structure Data ----------------#
		# delete 'b' in front of letters (python 2 --> python 3)
		pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
		for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
		
		# data processing THESE SHOULD BE CREATED DURING DATA GENERATION
		ipdb = 0
		s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
						gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
		
		# number of positions
		n_var = s0.shape[1]
		
		# Save processed data
		msa_outfile, ref_outfile = dp.write_FASTA(s0,pfam_id,s_ipdb)
		#-------------------------------------------------------#


		# Plot Contact Map
		ct = tools.contact_map(pdb,ipdb,cols_removed,s_index)
		ct_distal = tools.distance_restr(ct,s_index,make_large=True)



		#---------------------- Load DI -------------------------------------#
		print("Unpickling DI pickle files for %s"%(pfam_id))
		sorted_DI_er = pickle.load(open("%ser_DI_%s.pickle"%(er_directory,pfam_id),"rb"))
		sorted_DI_mf = pickle.load(open("%smf_DI_%s.pickle"%(mf_directory,pfam_id),"rb"))
		sorted_DI_plm = pickle.load(open("%splm_DI_%s.pickle"%(plm_directory,pfam_id),"rb"))

		#sorted_DI_er = dp.delete_sorted_DI_duplicates(sorted_DI_ER_redundant)

		print("\nPrint top 10 Non-Redundant pairs")
		for x in sorted_DI_er[:10]:
			print(x)
		#--------------------------------------------------------------------#


		#---------------------- Plot Contact Map ----------------------------#

		#plt.title('Contact Map')
		plt.imshow(ct_distal,cmap='rainbow_r',origin='lower')
		plt.xlabel('i')
		plt.ylabel('j')
		plt.colorbar(fraction=0.045, pad=0.05)
		pdf.attach_note("Contact Map")  # you can add a pdf note to
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()
		#--------------------------------------------------------------------#

		#--------------------- Plot DI Color Map ----------------------------#
		distance_enforced = False
		if distance_enforced:
			for coupling in sorted_DI_er:
				di_er[coupling[0][0],coupling[0][1]] = coupling[1]
				di_er[coupling[0][1],coupling[0][0]] = coupling[1]
			for coupling in sorted_DI_mf:
				di_mf[coupling[0][0],coupling[0][1]] = coupling[1]
				di_mf[coupling[0][1],coupling[0][0]] = coupling[1]
			for coupling in sorted_DI_plm:
				di_plm[coupling[0][0],coupling[0][1]] = coupling[1]
				di_plm[coupling[0][1],coupling[0][0]] = coupling[1]
		else:
			n_seq = max([coupling[0][0] for coupling in sorted_DI_er]) 
			di_er = np.zeros((n_var,n_var))
			di_mf = np.zeros((n_var,n_var))
			di_plm = np.zeros((n_var,n_var))
			for coupling in sorted_DI_er:
				#print(coupling[1])
				di_er[coupling[0][0],coupling[0][1]] = coupling[1]
				di_er[coupling[0][1],coupling[0][0]] = coupling[1]
			for coupling in sorted_DI_mf:
				di_mf[coupling[0][0],coupling[0][1]] = coupling[1]
				di_mf[coupling[0][1],coupling[0][0]] = coupling[1]
			for coupling in sorted_DI_plm:
				di_plm[coupling[0][0],coupling[0][1]] = coupling[1]
				di_plm[coupling[0][1],coupling[0][0]] = coupling[1]

		#--------------------------------------------------------------------#

		#----------------- Generate Optimal ROC Curve -----------------------#
		# find optimal threshold of distance for both DCA and ER
		ct_thres = np.linspace(1.5,10.,18,endpoint=True)
		n = ct_thres.shape[0]

		auc_mf = np.zeros(n)
		auc_er = np.zeros(n)
		auc_plm = np.zeros(n)

		for i in range(n):
			p,tp,fp = tools.roc_curve(ct_distal,di_mf,ct_thres[i])
			auc_mf[i] = tp.sum()/tp.shape[0]
			
			################################3 need to update singularity container p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
			p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
			auc_er[i] = tp.sum()/tp.shape[0]
			
			p,tp,fp = tools.roc_curve(ct_distal,di_plm,ct_thres[i])
			auc_plm[i] = tp.sum()/tp.shape[0]

		i0_mf = np.argmax(auc_mf)
		i0_er = np.argmax(auc_er)
		i0_plm = np.argmax(auc_plm)


		p0_mf,tp0_mf,fp0_mf = tools.roc_curve(ct_distal,di_mf,ct_thres[i0_mf])
		##################################### need to update singularity container   p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
		p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
		p0_plm,tp0_plm,fp0_plm = tools.roc_curve(ct_distal,di_plm,ct_thres[i0_plm])

		#------------------ Plot ROC for optimal DCA vs optimal ER ------------------#
		#print("Optimal Contact threshold for (mf, er, plm) = (%f, %f, %f)"%(ct_thres[i0_mf],ct_thres[i0_er],ct_thres[i0_plm]))
		#print("Maximal AUC for (mf, er, plm) = (%f, %f, %f)"%(auc_mf[i0_mf], auc_er[i0_er], auc_plm[i0_plm]))


		plt.subplot2grid((1,3),(0,0))
		plt.title('ROC ')
		plt.plot(fp0_er,tp0_er,'b-',label="er")
		plt.plot(fp0_mf,tp0_mf,'r-',label="mf")
		plt.plot(fp0_plm,tp0_plm,'g-',label="plm")
		plt.plot([0,1],[0,1],'k--')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()

		# Plot AUC for DCA and ER
		plt.subplot2grid((1,3),(0,1))
		plt.title('AUC')
		plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
		plt.plot(ct_thres,auc_er,'b-',label="er")
		# Need mf and plm first              plt.plot(ct_thres,auc_mf,'r-',label="mf")
		#plt.plot(ct_thres,auc_plm,'g-',label="plm")
		#plt.ylim([min(auc_er.min(),auc_mf.min(),auc_plm.min())-0.05,max(auc_er.max(),auc_mf.max(),auc_plm.max())+0.05])
		plt.xlim([ct_thres.min(),ct_thres.max()])
		plt.xlabel('distance threshold')
		plt.ylabel('AUC')
		plt.legend()

		# Plot Precision of optimal DCA and ER
		plt.subplot2grid((1,3),(0,2))
		plt.title('Precision')
		plt.plot( p0_er,tp0_er / (tp0_er + fp0_er),'b-',label='er')
		# NEED PLM AND MF FIRST                plt.plot( p0_mf,tp0_mf / (tp0_mf + fp0_mf),'r-',label='mf')
		#plt.plot( p0_plm,tp0_plm / (tp0_plm + fp0_plm),'g-',label='plm')
		plt.plot([0,1],[0,1],'k--')
		plt.xlim([0,1])
		#plt.ylim([0,1])
		plt.ylim([.4,.8])
		plt.xlabel('Recall (Sensitivity - P)')
		plt.ylabel('Precision (PPV)')
		plt.legend()

		plt.tight_layout(h_pad=1, w_pad=1.5)
		pdf.attach_note("ROC")  # you can add a pdf note to
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close()
		#----------------------------------------------------------------------------#

		#----------------------------------------------------------------------------#
		# Using PYDCA contact mapping module
		print("Dimensions of DI Pairs:")
		print("ER: ",len(sorted_DI_er))
		##print("PLM: ",len(sorted_DI_plm))
		####print("MF: ",len(sorted_DI_mf))


		#""" NEED MF and PLM FIRST
		erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
			refseq_file = ref_outfile,
			sorted_dca_scores = sorted_DI_er,
			linear_dist = 4,
			contact_dist = 8.0,
		)
		mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
			refseq_file = ref_outfile,
			sorted_dca_scores = sorted_DI_mf,
			linear_dist = 4,
			contact_dist = 8.0,
		)

		plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
			refseq_file = ref_outfile,
			sorted_dca_scores = sorted_DI_plm,
			linear_dist = 4,
			contact_dist = 8.0,
		)

		#"""
		er_contact_map_data, er_contact_ax = erdca_visualizer.plot_contact_map()
		pdf.attach_note("ER Contact Map")  # you can add a pdf note to
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close
		mf_contact_map_data, mf_contact_ax = mfdca_visualizer.plot_contact_map()
		pdf.attach_note("MF Contact Map")  # you can add a pdf note to
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close
		plm_contact_map_data, plm_contact_ax = plmdca_visualizer.plot_contact_map()
		pdf.attach_note("PLM Contact Map")  # you can add a pdf note to
		pdf.savefig()  # saves the current figure into a pdf page
		plt.close

		#er_tp_rate_data, er_tp_ax = erdca_visualizer.plot_true_positive_rates()
		#mf_tp_rate_data, mf_tp_ax = mfdca_visualizer.plot_true_positive_rates()
		#plm_tp_rate_data, plm_tp_ax = plmdca_visualizer.plot_true_positive_rates()
		#pdf.attach_note("True Positive Rate")  # you can add a pdf note to
		#pdf.savefig()  # saves the current figure into a pdf page


		#----------------------------------------------------------------------------#
		# We can also set the file's metadata via the PdfPages object:
	d = pdf.infodict()
	d['Title'] = 'Multipage PDF Example'
	d['Author'] = 'Evan Cresswell\xe4nen'
	d['Subject'] = 'Contact inference of Pfam Proteins'
	d['Keywords'] = 'Pfam Contact Map PDB Expectation Reflection Mean-Field DCA Pseudoliklihood'
	d['CreationDate'] = datetime.datetime(2020, 2, 28)
	d['ModDate'] = datetime.datetime.today()

