import sys
import numpy as np

pfam = 'PF00011'
#ipdb = 0
#pfam_id = sys.argv[1]
#ipdb = sys.argv[2]
#ipdb = int(ipdb)

#ext_name = '%s/%02d'%(pfam_id,ipdb)

di = np.loadtxt('%s/di.dat'%pfam)
ct = np.loadtxt('../../pfam_2_100k/%s_ct.txt'%pfam)

#try:
#    di = np.loadtxt('%s_di.dat'%ext_name)
#except:
#    pass    

#=========================================================================================
def direct_top(d,top):
    # find value of top biggest
    d1 = d.copy()
    np.fill_diagonal(d1, 0)
    #print(d1)
    
    a = d1.reshape((-1,))
    #print(a)    
    a = np.sort(a)[::-1] # descreasing sort
    #print(a)

    top_value = a[top]
    #print(top_value)
       
    # fill the top largest to be 1, other 0
    top_pos = d1 > top_value
    #print(top_pos)
    d1[top_pos] = 1.
    d1[~top_pos] = 0.
    #print(d1) 
    
    xy = np.argwhere(d1==1)   
    return xy

#=========================================================================================

tops = [20,30,40,50,60]
for top in tops:
    xy = direct_top(di,top)
    np.savetxt('%s_di_%s.dat'%(pfam,top),xy,fmt='%i')

