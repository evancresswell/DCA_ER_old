# 2018.12.27: Receiver operating characteristic (ROC) curve
import numpy as np
import matplotlib.pyplot as plt
#=========================================================================================

auc_DCA = np.loadtxt('auc_av_DCA.dat')
auc_ER =  np.loadtxt('auc_av.dat')
auc_ER2 =  np.loadtxt('auc_av_20iter_100.dat')
   
plt.figure(figsize=(3.0,3.2))

#plt.errorbar(fp,tp,std)
plt.plot(auc_DCA[0],auc_DCA[1],'b^--',label='DCA')
plt.plot(auc_ER[0],auc_ER[1],'ro-',label='ER')
plt.plot(auc_ER2[0],auc_ER2[1],'r*--',label='ER-20iter')

#plt.fill_between(fp,tp-std,tp+std)
#plt.plot([0,1],[0,1],'k--')
#plt.xlim([0,1])
#plt.ylim([0,1])
plt.xlabel('cutoff')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('auc_av.pdf', format='pdf', dpi=100)
