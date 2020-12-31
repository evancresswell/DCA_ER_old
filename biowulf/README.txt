
#---------------------- Biowulf Simulations ----------------------# 

This directory is for the execution and analysis of simulations run
on Biowulf



#----- Simulation  -----#
- Genrate run files
	- makes list of er runs from list of Pfams: pfam_pdb_list.txt
  	RUN: python 12make_swarm.py

- Run script for ER, PLM, and MF swarm files
  	RUN: ./14submit_main_swam_biowulf.script
	- runs swarms for multiple method types (choose via # commenting) 
#-----------------------#

#---------------------------------------------------------------------#
#------------------------- Analysis Drivers --------------------------#
#---------------------------------------------------------------------#

-- The following is the flow of runs to get to plotting routines once you have completed swarm simulations for various methods
#-----------------------#
#----- WORFLOW ---------#
#----- 12/25/2020 ------#
#------- simple --------#
- collect scores by creating swarm file
	# RUN python 17gen_scores_swarm.py
- genreate score .txt files in DI directory
	# RUN ./18_gen_score.script
- concatenate and print best method from scores:
	# RUN print_AUTPR_bm.py

#-----------------------#

#---------------------------------------------------------------------------------------------------#
#---------------------------------------------- SUPPLEMENTARY FILES --------------------------------#
#---------------------------------------------------------------------------------------------------#
#---------------------- File Directory ---------------------------# 
Enty: June 8, 2020

Files to read swarm pkl files and create several different kinds of dataframes:
method_axis_plot and pfam_bar_methods created the most up-to-date figures (namely pfam_bar_method)

 Aug 15 14:00 pfam_pdb_seq_match.py - checks pdb_refs.npy against actual MSA matching the give PPB loaded PDB struture (also compares with PYDCA)
 Jun  5 16:34 ks_comparison.py - compare score/AUC histograms to discern that they are from different distributions
 Jun  5 15:20 ecc_tools.py -  added get_score_df to create method dataframes (used in method_axis_plot)
 Jun  5 15:06 method_axis_plot.py - plot method scores along xyz axis
 Jun  4 18:59 pfam_bar_method.py -  plot Score (method_auc - max AUC) and best method count for pfams in different ranges of number of sequence - uses datframes generated in gen_method_column_df.py
 Jun  4 12:06 compare_ER-MF.py - same as pfam_bar_method but only compares ER and MF
 Jun  4 09:39 pfam_best_method_surface.py - 3D axis plot score on z-axis num-seq and len-seq x-y
 May 22 20:22 gen_method_column_df.py - generated dataframes for plotting with individual methods per row
 May 22 11:51 gen_PR_df.py - generates dataframes for individual metrics (AUC,AUPR, TP, FP etc) which are used in gen_method_column_df.py
 May  7 18:32 create_sim_data_frame.py
#-----------------------------------------------------------------# 





#----------------------------------------------------------------------------------------------------#
#----------------------------------------- Create DataFrames ----------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----- WORKFLOW --------#
- Collect resulting information from swarm of simulations
  RUN: python 15sim_summary.py job_id method
	- This requires successfull swarm simulation with a swarm ID 
	- Each simulation must be of only one Method
	- Creates a file for each method 
		- <METHOD>_job-<JOB-NUM>_swarm_ouput.txt
  - Can also just use jobhist <JOB-NUM> and then curate

- Create DataFrame from txt output # SWARM RUN

  STEP 1 --> Create individual ROC dataframes
  RUN: sinteractive --mem=15g
  RUN: singularity exec -B /data/cresswellclayec/DCA_ER/ /data/cresswellclayec/DCA_ER/erdca.simg python setup_swarm_sim_dataframe.py <METHOD>_job-<JOBID>_swarm_ouput.txt
	- The text file generated with 15sim_summary.py
	- you can use jobhist <JOBID> and remove info manually (first row should be column names)
  RUN: ./16_gen_ROC_df.script <----> swarm simulation
	- uses gen_ROC_jobID_df.py to 
	- creates indidvidual swarm dfs in directory: job_ROC_dfs/
  ---> cd into job_ROC_dfs/


  RUN: sinteractive --mem=150g --cpus-per-task=4
  RUN: module load singularity
  RUN: singularity exec -B /data/cresswellclayec/DCA_ER/ /data/cresswellclayec/DCA_ER/dca_er.simg python concat_dfs.py <JOBIDA>
  	- this creates dataframe for full swarm simulation: ../<METHOD>_<JOBID>_full.pkl 
  ---> cd back up to biowulf/
  RUN: singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/Pfam-A.full/ /data/cresswellclayec/DCA_ER/dca_er.simg python gen_PR_df.py <METHOD1>_<JOBID1>_full.pkl <METHOD2>_<JOBID2>_full.pkl ...
	- this creates two dataframe types
		- a summary dataframe (used in pfam_bar_method for final plots)
		- a specialized dataframe set for TP, FP, P, AUC, AUPR, PR etc ...
			- Each df has methods as columns
#-----------------------#

#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#


 
#-----------------------------------------------------------------# 
#---------------- Plotting Drivers -------------------------------# 
#-----------------------------------------------------------------# 


#----- Plotting All Pfams --------#
- Plot BarPlot of best methods across ranges of number of sequences
  	RUN: singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/Pfam-A.full/ /data/cresswellclayec/DCA_ER/dca_er.simg python pfam_bar_method.py
- Plot AUC/Score of methods along axis (color is best method)
  	RUN: singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/Pfam-A.full/ /data/cresswellclayec/DCA_ER/dca_er.simg python method_axis_plot.py
#---------------------------------#


#----- Plotting Single Pfams --------#
- Plot Contact Map and ROC curves for Pfams
  RUN: ./pfam_plotting.script
	- runs: 
		singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/Pfam-A.full/ /data/cresswellclayec/DCA_ER/dca_er.simg python plot_single_pfam.py <PFAM>
#------------------------------------#

#-----------------------------------------------------------------# 
#-----------------------------------------------------------------# 
#-----------------------------------------------------------------# 


# THIS IS NOW DONE IN INDIVIDUAL RUNS
#------ Pre-Sim --------#
- Set up .fa and .pickle files in pfam_ecc/
	- .fa files are used for post-sim analysis
	  and MF and PLM method simulation
	- .pickle files are used for ER simulations
  RUN: ./111swarm_generate_input_data.script
	- This requires files -> [ pfam_pdb_list.swarm, ../dca_er.simg, Pfam-A.full/ ]
		- pfam_pdb_list.swarm example line (for Pfam PF00001):
			singularity exec -B /path/to/current/dir,path/to/Pfam-A.full/ path/to/dca_er.simg python gen_1PFAM_input_data.py PF00001
		- ../dca_er.simg 
			- see ../README.md section: Generate Singularity Container to Run Code
#-----------------------#


#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------#
