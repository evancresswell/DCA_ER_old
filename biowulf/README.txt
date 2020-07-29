#---------------------- Biowulf Simulations ----------------------# 

This directory is for the execution and analysis of simulations run
on Biowulf

#------ Pre-Sim --------#
- Set up .fa and .pickle files in pfam_ecc/
	- .fa files are used for post-sim analysis
	  and MF and PLM method simulation
	- .pickle files are used for ER simulations
  RUN: ./111swarm_generate_input_data.script
	- This requires files -> [ pfam_pdb_list.swarm, ../dca_er.simg, Pfam-A.full/ ]
		- pfam_pdb_list.swarm example line (for Pfam PF00001):
			singlesingularity exec -B /path/to/current/dir,path/to/Pfam-A.full/ path/to/dca_er.simg python gen_1PFAM_input_data.py PF00001
		- ../dca_er.simg 
			- see ../README.md section: Generate Singularity Container to Run Code
#-----------------------#

#----- Simulation  -----#
- Run script for ER, PLM, and MF swarm files
  RUN: ./14submit_main_swam_biowulf.script
	- This requiers files -> [er.swarm, plm.swarm, mf.swarm, dca_er.simg, 1main_ER.py, 1main_PLM.py, 1main_MF.py]
		- er.swarm exmaple line (for Pfam PF00001):
			singularity exec -B /path/to/current/dir/ /path/to/dca_er.simg python 1main_ER.py PF00001
		- plm.swarm exmaple line (for Pfam PF00001):
			singularity exec -B /path/to/current/dir/ /path/to/dca_er.simg python 1main_PLM.py PF00001
		- mf.swarm exmaple line (for Pfam PF00001):
			singularity exec -B /path/to/current/dir/ /path/to/dca_er.simg python 1main_MF.py PF00001
		- 1main_#METHOD#.py
			- Python driver to load data from pfam_ecc/ and execute method and save pickle file in DI/#METHOD#/
		- ../dca_er.simg 
			- see ../README.md section: Generate Singularity Container to Run Code
#-----------------------#


#----- Analysis --------#
- Collect resulting information from swarm of simulations
  RUN: python 15sim_summary.py job_id method
	- This requires successfull swarm simulation with a swarm ID 
	- Each simulation must be of only one Method
	- Creates a file for each method 
		- #METHOD#_job-#JOB-NUM#_swarm_ouput.txt


- Create DataFrame from txt output
  RUN: sinteractive cpus-per-task=50 --mem=100g 
  RUN: python create_sim_data_frame.py #METHOD#_job-#JOB-NUM#_swarm_ouput.txt
	- Remove info manually (first row should be column names)

  RUN: genddd...

  RUN: dkfjdkls ...
#-----------------------#

#-----------------------------------------------------------------# 
