import numpy as np
from scipy import linalg

niter_max = 10
l2 = 100

x = [[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1],[0,1,0,0,0,1],[0,1,0,1,0,0]]
x = np.asarray(x)
y = [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,0,1]]
y = np.asarray(y)
print("x = ",x)
print("y = ",y)

l,n = x.shape
m = y.shape[1]
print("l, n, m = ",l,n,m )

x_av = np.mean(x,axis=0)
dx = x - x_av
c = np.cov(dx,rowvar=False,bias=True)
print("x_av = ", x_av)
print("dx = x - x_av = ", dx)
print("yields covariance between every possibility (should be symmetric)")
print("c = ", c)

print("add the l2 identity to covariance matrix l2*I[n]/(2*l)",l2*np.identity(n)/(2*l) )
c += l2*np.identity(n)/(2*l)
print("this makes c = ", c)

c_inv = linalg.pinvh(c)
print("Inverse of covariance matrix is : ", c_inv)



