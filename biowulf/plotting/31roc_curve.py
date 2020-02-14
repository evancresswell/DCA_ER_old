# 2018.12.27: Receiver operating characteristic (ROC) curve
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#=========================================================================================

#pfam_id = 'PF00025'
#ipdb = 1
#pfam_id = sys.argv[1]
#ipdb = sys.argv[2]
#ipdb = int(ipdb)

#ext_name = '%s/%02d'%(pfam_id,ipdb)

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
s = np.loadtxt('pfam_10_20k.txt',dtype='str')
pfam_list = s[:,0]

#pfam_list = ['PF00011']

ct_thres_list = [1.5,2.0,3.0,4.0]
#==========================================================================================
def roc_curve(ct,di,ct_thres):    
    ct1 = ct.copy()
    
    ct_pos = ct1 < ct_thres           
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0],dtype=bool), k=1)
    order = di[mask].argsort()[::-1]

    ct_flat = ct1[mask][order]

    tp = np.cumsum(ct_flat, dtype=float)
    fp = np.cumsum(~ct_flat.astype(int), dtype=float)

    if tp[-1] !=0:
        tp /= tp[-1]
        fp /= fp[-1]
    
    # bining (to reduce the size of tp,fp and make fp having the same values for every pfam)
    nbin = 101
    pbin = np.linspace(0,1,nbin, endpoint=True)

    #print(pbin)

    fp_size = fp.shape[0]

    fpbin = np.ones(nbin)
    tpbin = np.ones(nbin)
    for ibin in range(nbin-1):
        # find value in a range
        t1 = [(fp[t] > pbin[ibin] and fp[t] <= pbin[ibin+1]) for t in range(fp_size)]

        if len(t1)>0 :            
            fpbin[ibin] = fp[t1].mean()
            tpbin[ibin] = tp[t1].mean()
        else:
            #print(i)
            tpbin[ibin] = tpbin[ibin-1] 

    #print(fp,tp)
    #return fp,tp,pbin,fpbin,tpbin
    return pbin,tpbin

#=========================================================================================
for pfam_id in pfam_list:
    try:    
        ct = np.loadtxt('../DI/%s_ct.txt'%pfam_id).astype(float)

        #di = np.loadtxt('%s/di.dat'%pfam_id).astype(float)    
        di_er = np.loadtxt('../DI/ER/er_DI_%s.pickle'%pfam_id).astype(float)
        di_plm = np.loadtxt('../DI/PLM/plm_DI_%s.pickle'%pfam_id).astype(float)
        di_mf = np.loadtxt('../DI/MF/mf_DI_%s.pickle'%pfam_id).astype(float)

        # find optimal threshold of distance
        ct_thres = np.linspace(1.5,10.0,18,endpoint=True)
        n = ct_thres.shape[0]

		#-------------------- AUC Curves -------------------#
        auc_er = np.zeros(n)
        for i in range(n):
            fp_er,tp_er = roc_curve(ct,di_er,ct_thres[i])

            #print(tp.shape,fp.shape)

            auc_er[i] = tp_er.sum()/tp_er.shape[0]    
            #print(ct_thres[i],auc[i])
                   
            # nan to zero
            auc_er = np.where(np.isnan(auc_er), 0, auc_er)

        np.savetxt('DI/%s/auc_er.dat'%(pfam_id),(ct_thres,auc_er),fmt='%f')

        auc_plm = np.zeros(n)
        for i in range(n):
            fp_plm,tp_plm = roc_curve(ct,di_plm,ct_thres[i])

            #print(tp.shape,fp.shape)

            auc_plm[i] = tp_plm.sum()/tp_plm.shape[0]    
            #print(ct_thres[i],auc[i])
                   
            # nan to zero
            auc_plm = np.where(np.isnan(auc_plm), 0, auc_plm)

        np.savetxt('DI/%s/auc_plm.dat'%(pfam_id),(ct_thres,auc_plm),fmt='%f')

        auc_mf = np.zeros(n)
        for i in range(n):
            fp_mf,tp_mf = roc_curve(ct,di_mf,ct_thres[i])

            #print(tp.shape,fp.shape)

            auc_mf[i] = tp_mf.sum()/tp_mf.shape[0]    
            #print(ct_thres[i],auc[i])
                   
            # nan to zero
            auc_mf = np.where(np.isnan(auc_mf), 0, auc_mf)

        np.savetxt('DI/%s/auc_mf.dat'%(pfam_id),(ct_thres,auc_mf),fmt='%f')
		#---------------------------------------------------#

        #-----------------------------------------------------------------------------------------
        # plot at optimal threshold:
        i0_er = np.argmax(auc_er)
        i0_plm = np.argmax(auc_plm)
        i0_mf = np.argmax(auc_mf)
  
        print('auc_er max:',ct_thres[i0_er],auc_er[i0_er])
        print('auc_plm max:',ct_thres[i0_plm],auc_plm[i0_plm])
        print('auc_mf max:',ct_thres[i0_mf],auc_mf[i0_mf])

        fp0_er,tp0_er = roc_curve(ct,di_er,ct_thres[i0_er])
        fp0_plm,tp0_plm = roc_curve(ct,di_plm,ct_thres[i0_plm])
        fp0_mf,tp0_mf = roc_curve(ct,di_mf,ct_thres[i0_mf])

        iplot = [2,3,5,7]
        plt.figure(figsize=(9.0,3.2))

		### PLOTS NEED TO BE GENERATLIZED TO ALL METHODS
        plt.subplot2grid((1,3),(0,0))
        plt.title('ROC various thres')
        plt.plot([0,1],[0,1],'k--')
        for i in iplot:
            fp,tp = roc_curve(ct,di,ct_thres[i])
            if ct_thres[i] in ct_thres_list:
                np.savetxt('%s/roc_%1.1f.dat'%(pfam_id,ct_thres[i]),(fp,tp),fmt='% f')
                
            plt.plot(fp,tp,label='thres = %s'%ct_thres[i])

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()

        plt.subplot2grid((1,3),(0,1))
        plt.title('ROC at thres = %3.2f'%(ct_thres[i0]))
        plt.plot(fp0,tp0,'r-')
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.subplot2grid((1,3),(0,2))
        plt.title('AUC max = %f' %(auc[i0]))
        plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
        plt.plot(ct_thres,auc,'r-')
        plt.xlim([ct_thres.min(),ct_thres.max()])
        plt.ylim([0,auc.max()+0.1])
        plt.xlabel('distance threshold')
        plt.ylabel('AUC')

        plt.tight_layout(h_pad=1, w_pad=1.5)
        #plt.savefig('%s_roc_curve.pdf'%ext_name, format='pdf', dpi=100)
        plt.savefig('%s/roc.pdf'%(pfam_id), format='pdf', dpi=100)
        plt.close()

    except:
        pass  
#plt.show()
