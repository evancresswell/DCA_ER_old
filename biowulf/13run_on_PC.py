import os
import timeit
pfam_id = 'PF04542'
pfam_id = 'PF00186'
pfam_id = 'PF00011'

start_time = timeit.default_timer()
os.system('sudo singularity exec -B ~/DCA_ER/biowulf/ ~/DCA_ER/dca_er.simg python 1main_PLM.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('plm run time:',run_time)

"""
start_time = timeit.default_timer()
os.system('sudo singularity exec -B ~/DCA_ER/biowulf/ ~/DCA_ER/dca_er-DCA.simg python 1main_ER.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('ER run time:',run_time)

start_time = timeit.default_timer()
os.system('sudo singularity exec -B ~/DCA_ER/biowulf/ ~/DCA_ER/dca_er-DCA.simg python 1main_DCA.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('mf run time:',run_time)



start_time = timeit.default_timer()
os.system('python 1main_ER.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('ER run time:',run_time)

start_time = timeit.default_timer()
os.system('python 1main_PLM.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('PLM run time:',run_time)

start_time = timeit.default_timer()
os.system('python 1main_DCA.py '+pfam_id)
run_time = timeit.default_timer() - start_time
print('MF run time:',run_time)
#"""



