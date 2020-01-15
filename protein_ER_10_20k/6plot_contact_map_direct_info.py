import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#===================================================================================================
pfam = 'PF00011'
#ipdb = 0
#pfam_id = sys.argv[1]
#ipdb = sys.argv[2]
#ipdb = int(ipdb)

plt.figure(figsize=(14,11))

top = [20,30,40,50,60]
threshold = [2.,4.,6.,8.]

nx = len(top)
ny = len(threshold)

for j in range(ny):
    for i in range(nx):
        plt.subplot2grid((4,5),(j,i))
        
        plt.title('ct-thr = %s, di-top = %s' %(threshold[j],top[i]))
        try:
            xy1 = np.loadtxt('%s_contact_%02d.dat'%(ext_name,threshold[j]))    
            xy2 = np.loadtxt('%s_direct_%s.dat'%(ext_name,top[i]))
        except:
            pass    

        plt.scatter(xy1[:,0],xy1[:,1],marker='o',facecolors='none',edgecolors='c',label='contact',s=10)
        plt.scatter(xy2[:,0],xy2[:,1],marker='+',color='b',label='direct-info',s=10)
        
plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('%s_contact_direct.pdf'%ext_name, format='pdf', dpi=100)    
#plt.show()  
