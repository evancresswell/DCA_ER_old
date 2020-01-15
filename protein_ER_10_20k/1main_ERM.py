import sys
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from inference import fit_additive,fit_multiplicative
from direct_info import direct_info,frequency
#========================================================================================
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

s0 = np.loadtxt('../pfam_10_100k/%s_s0.txt'%(pfam_id)).astype(int)
l,n = s0.shape
mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 

regu = 0.
nloop = 10
onehot_encoder = OneHotEncoder(sparse=False)
s = onehot_encoder.fit_transform(s0)

mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)
cost = np.zeros((n,nloop))
niter = np.zeros(n)

fa = frequency(s0,mx,i1i2)
for i0 in range(n):
    print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    #w1,h01,cost1,niter1 = fit_additive(x,y,regu,nloop)
    #w1,h01,cost1,niter1 = fit_multiplicative(x,y,nloop)
    w1,h01,cost1,niter1 = fit_multiplicative(x,y,fa[i1:i2],nloop)

    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]

    h0[i1:i2] = h01
    cost[i0,:] = cost1
    niter[i0] = niter1

#np.savetxt('%s_w.dat'%pfam_id,w,fmt='% f')
#np.savetxt('h0.dat',h0,fmt='% f')
#np.savetxt('niter.dat',niter,fmt='%i')
#np.savetxt('cost.dat',cost,fmt='% f')

di = direct_info(s0,w)
np.savetxt('%s/di.dat'%pfam_id,di,fmt='% 3.8f')

