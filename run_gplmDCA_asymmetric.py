import gplmDCA_asymmetric as gplm

pfam_id = 'PF00186'
lambda_h = .1
lambda_J = .1
lambda_G = .1
reweighting_threshold = .1
nb_of_cores = 4
M = -1
gplm.gplmDCA_asymmetric(pfam_id, lambda_h, lambda_J, lambda_G, reweighting_threshold, nb_of_cores, M)
