import os
import data_processing as dp
import numpy as np
from subtract_lists import subtract_lists
from joblib import Parallel, delayed

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

#s = np.vstack([s1,s2])
#s = np.vstack([s,s3])

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')
s_er = np.loadtxt('test_list.txt',dtype='str')
s_plm = np.loadtxt('test_list.txt',dtype='str')
s_mf = np.loadtxt('test_list.txt',dtype='str')

s_er = np.loadtxt('pfam_pdb_list.txt',dtype='str')
s_plm = np.loadtxt('pfam_pdb_list.txt',dtype='str')
s_mf = np.loadtxt('pfam_pdb_list.txt',dtype='str')

#n = s.shape[0]
#pfam_list = s[:,0]
print( s_er)

#--------------------------------------------------------------#
#--------------------------------------------------------------#
# create swarmfiles for each method
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
def get_msa_size(data_path,pfam):
    s = np.load('%s/%s/msa.npy'%(data_path,pfam)).T
    return s.shape[0]

seq_num = Parallel(n_jobs = 16)(delayed(get_msa_size)\
        (data_path,s_er[i0])\
        for i0 in range(len(s_er)))

top_indices = sorted(range(len(seq_num)), key=lambda i: seq_num[i], reverse=True)
top_10p_size = [seq_num[i] for i in top_indices[:int(round(.10*len(seq_num)))]]
top_10p_pfam = [s_er[i] for i in top_indices[:int(round(.10*len(seq_num)))]]
print('Pfams for largemem (top %d): '%int(round(.10*len(seq_num))),top_10p_pfam)


f = open('er.swarm','w')
f_large = open('er_large.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    if pfam in top_10p_pfam:
        #f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    else:
        #f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()
f_large.close()

f = open('er_clean.swarm','w')
f_large = open('er_clean_large.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    if pfam in top_10p_pfam:
        #f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python er_basic_run.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    else:
        #f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python  er_basic_run.py  %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()
f_large.close()

f = open('lader_clean.swarm','w')
f_large = open('lader_clean_large.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    if pfam in top_10p_pfam:
        #f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python lader_basic_run.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    else:
        #f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_ER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
        f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_lad_reg.simg python  lader_basic_run.py  %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()
f_large.close()





f = open('lader.swarm','w')
f_large = open('lader_large.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    if pfam in top_10p_pfam:
        f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_coupling.simg python run_singlePFAM_LADER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    else:
        f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca_coupling.simg python run_singlePFAM_LADER.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()
f_large.close()


f = open('plm.swarm','w')
for pfam in s_plm:
    #f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_PLM.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/LADER.simg python run_singlePFAM_PLM.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()

f = open('mf.swarm','w')
for pfam in s_mf:
    #f.write('python 1main_DCA.py %s\n'%(pfam))    
    #f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_DCA.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
    f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/LADER.simg python run_singlePFAM_DCA.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
f.close()
#--------------------------------------------------------------#

