For Covid Protein.

File used --> allprot1017.

- extract protein using: 		extract_protein.py 		(see file for singularity command)
- remove duplicate lines using: 	remove_duplicate_lines.py	(see file for singularity command)	
- Manually added zhang reference protein sequence to resulting fasta file
- align resulting fasta file:		mafft --auto --thread 16 cov_fasta_files/spike_NoDup.fasta > cov_fasta_files/spike_aligned.fasta	
- RUN ER method:			run_covPROTEIN_ER.py		(see file for singularity command)


SIMULATION NOTES:
-- SPIKE PROTEIN 	--> pre-processing finds that all columns ar 80% > conserved.. NO CHANGES in spike protein...
-- NSP1			--> pre-processing finds that all columns ar 80% > conserved.. NO CHANGES in spike protein...


Covid Genome Analysis.

Align Data
	aligned file: cov_gen_aligned.fasta
		- acquired by running: 
			- mafft --auto --keeplength --addfragments 09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa wuhan_ref.fasta > cov_gen_aligned.fasta
		-Files used:
			- 09062020-Cov19gisaid--SeqIdDesc200-aligned-FastaCikti.fa: FILE FROM---> from http://www.dilekbalik.com/SARS-CoV-2_ODOTool/
			- wuhand_ref_fasta: FILE FROM----> covid reference genome: NC_045512.2

			- Downloaded Oct 20 2020
			- https://www.epicov.org/epi3/entities/tmp/tmp_sd_2020_10_20_02_12_qigibl_4j510bafde3/sequences_2020-10-19_07-21.fasta.gz


	Alginment Process	
		- begin by breaking up un-aligned fasta files
			RUN: singularity exec -B /data/cresswellclayec/DCA_ER/biowulf,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python break_up_fasta.py #subject_genome_file#.fasta
			- Used  wuhan_ref.fasta as subject 
			 
		- Next run alignment swarm which aligns with mafft.
			RUN: ./submit_align_swarm.script 

		- finally concatenate resulting mini-alignments in cov_fasta_files/
			RUN: singularity exec -B /data/cresswellclayec/DCA_ER/biowulf,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python concat_fasta.py 


	Get Clade Alignments 
		- once you have the full genome aligned file you can get clades using get_clades.py
		- get_clades.py has subject file hard coded: 
		RUN: singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python get_clades.py





Infer Interactions using Expectation Reflection

- once you have an aligned file you can get DI using ER using run_covGENOME_ER.py
- file generates .pickle files with DI
- the different clade and full sequence run are already defined in run_cov_GENOME_ER file.
	-for all clades:
	RUN: ./submit_DI_clade_swarm.script
	-for full genome run
	- hardcoded existing full aligned file
	RUN: singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/,/data/cresswellclayec/DCA_ER/covid_proteins /data/cresswellclayec/DCA_ER/LADER.simg python run_covGENOME_ER.py 




Python Files for Post-Processing and Data Visualization
Dec 30 18:59 incidence_plotting.py
Dec 29 14:36 gen_incidence_data.py
Dec 16 16:10 make_codon_swarm.py
Dec 16 12:43 make_pair_aa_swarm.py
Dec 15 15:32 codon_mapping.py
Dec  8 14:28 print_pair_counts.py
Dec  2 11:08 covid_aa_bp_variance.py
Nov 12 00:26 plot_covid_genome_CT.py
Oct 20 09:06 ER_covid_PP.py
Oct 13 22:47 expectation_reflection.py



Result txt File list:
-rw-r----- 1 cresswellclayec cresswellclayec     1542 Dec 17 18:10 s_pairs.txt
-rw-r----- 1 cresswellclayec cresswellclayec     6746 Dec 16 18:39 orf7a_pairs.txt
-rw-r----- 1 cresswellclayec cresswellclayec    55801 Dec 16 15:49 single_site_bp_aa_freq.txt
-rw-r----- 1 cresswellclayec cresswellclayec     7442 Dec 16 12:39 orf7b_pairs.txt
-rw-r----- 1 cresswellclayec cresswellclayec     3929 Dec 16 12:38 orf3a_pairs.txt
-rw-r----- 1 cresswellclayec cresswellclayec     7075 Dec 16 12:37 orf1ab_pairs.txt
-rw-r----- 1 cresswellclayec cresswellclayec    36710 Dec 15 15:32 cov_DI_fullout.txt
-rw-r----- 1 cresswellclayec cresswellclayec     7094 Dec  8 14:29 pair_aa_counts.txt
-rw-r----- 1 cresswellclayec cresswellclayec    13681 Nov 24 14:59 ORF1ab_bp_vs_aa.txt
-rw-r----- 1 cresswellclayec cresswellclayec       49 Nov 19 11:12 codon_out.txt
-rw-r----- 1 cresswellclayec cresswellclayec    18457 Nov 12 00:33 cov_DI_G.txt
-rw-r----- 1 cresswellclayec cresswellclayec     2961 Nov 11 23:04 poster_DI.txt
-rw-r----- 1 cresswellclayec cresswellclayec    17852 Nov 11 22:53 cov_DI_S.txt
-rw-r----- 1 cresswellclayec cresswellclayec    27060 Nov 11 22:42 cov_DI_GR.txt
-rw-r----- 1 cresswellclayec cresswellclayec    19511 Nov 11 22:33 cov_DI_GH.txt
-rw-r----- 1 cresswellclayec cresswellclayec    10561 Nov 11 22:27 cov_DI_V.txt
-rw-r----- 1 cresswellclayec cresswellclayec    36658 Nov 11 15:27 cov_DI_full1.txt
-rw-r----- 1 cresswellclayec cresswellclayec    38249 Nov 11 14:51 cov_DI_full2.txt
-rw-r----- 1 cresswellclayec cresswellclayec   324630 Nov 10 17:07 cov_DI_full.txt
-rw-r--r-- 1 cresswellclayec cresswellclayec    28080 Oct 25 08:56 new_er_dis.txt
-rw-r----- 1 cresswellclayec cresswellclayec 51675344 Jul  2 17:24 covid_sim.txt
