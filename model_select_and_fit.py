import os
import numpy as np
import glob
from voxnet.utilities import absjoin, h5read
from scipy.io import mmread
from loss_fun import *

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

#loss=mean_sq_error_fro
loss=rel_MSE_2

print "Running model selection for run %s" % save_stem
# setup some variables
n_lambda=len(lambda_list)
fid=open(selected_fit_cmds,'w')
Lx_fn=absjoin(save_dir,save_stem+'_Lx.mtx')
Ly_ipsi_fn=absjoin(save_dir,save_stem+'_Ly_ipsi.mtx')
Ly_contra_fn=absjoin(save_dir,save_stem+'_Ly_contra.mtx')
# loop through the outer loop (validation sets)
for o_idx,outer_dir in enumerate(glob.glob(save_dir+'/cval*')):
    print 'Entering outer cross-val set ' + str(o_idx)
    # new model training will now be with the outer sets
    X_train_fn=absjoin(outer_dir,'X_train.h5')
    Y_train_ipsi_fn=absjoin(outer_dir,'Y_train_ipsi.h5')
    Y_train_contra_fn=absjoin(outer_dir,'Y_train_contra.h5')
    # inner loop 
    inner_dirs=glob.glob(outer_dir+'/cval*')
    n_inner=len(inner_dirs)
    err_contra=np.zeros((n_inner,n_lambda))
    err_ipsi=np.zeros((n_inner,n_lambda))
    for i,inner_dir in enumerate(inner_dirs):
        print '  Processing inner cross-val set ' + str(i)
        Omega_train_inner_fn=absjoin(inner_dir,'Omega_train.mtx')
        Omega_test_inner_fn=absjoin(inner_dir,'Omega_test.mtx')
        X_test_fn=absjoin(inner_dir,'X_test.h5')
        Y_test_ipsi_fn=absjoin(inner_dir,'Y_test_ipsi.h5')
        Y_test_contra_fn=absjoin(inner_dir,'Y_test_contra.h5')
        X_test=h5read(X_test_fn)
        Y_test_ipsi=h5read(Y_test_ipsi_fn)
        Y_test_contra=h5read(Y_test_contra_fn)
        Omega_test_inner=mmread(Omega_test_inner_fn)
        # for each lambda, evaluate error
        for j,lambda_val in enumerate(lambda_list):
            print '    Evaluating error for lambda=%1.4e' % lambda_val
            W_ipsi_fn=absjoin(inner_dir,"W_ipsi_%1.4e.h5" % lambda_val)
            W_contra_fn=absjoin(inner_dir,"W_contra_%1.4e.h5" % lambda_val)
            flag_err_ipsi=False
            flag_err_contra=False
            try:
                W_ipsi=h5read(W_ipsi_fn)
            except Exception:
                print "    Error reading %s, using checkpoint" % W_ipsi_fn
                try:
                    W_ipsi=h5read(W_ipsi_fn + ".CHECKPT")
                except Exception:
                    print "    Error reading checkpoint"
                    flag_err_ipsi=True
            try:
                W_contra=h5read(W_contra_fn)
            except Exception:
                print "    Error reading %s, using checkpoint" % W_contra_fn
                try:
                    W_contra=h5read(W_contra_fn + ".CHECKPT")
                except Exception:
                    print "    Error reading checkpoint"
                    flag_err_contra=True
            if flag_err_ipsi:
                err_ipsi[i,j]=np.nan
            else:
                err_ipsi[i,j]=loss(W_ipsi,X_test,Y_test_ipsi,Omega_test_inner)
            if flag_err_contra:
                err_contra[i,j]=np.nan
            else:
                err_contra[i,j]=loss(W_contra,X_test,Y_test_contra)
            print "     " + str(err_ipsi[i,j]) + ' ' + str(err_contra[i,j])
    # summarize errors by mean over inner sets
    err_ipsi_sum=np.nanmean(err_ipsi,axis=0)
    err_contra_sum=np.nanmean(err_contra,axis=0)
    err_total_sum=np.nansum(np.hstack((err_ipsi_sum,err_contra_sum)),axis=0)
    print 'ipsi err:  '+ str(err_ipsi_sum)
    print 'contra err:'+ str(err_contra_sum)
    print 'sums:      '+ str(err_total_sum)
    print 'lambdas:   '+ str(lambda_list)
    # select best lambda(s)
    if select_one_lambda:
        lambda_idx=np.argmin(err_total_sum)
        lambda_opt=lambda_list[lambda_idx]
        lambda_ipsi=lambda_opt
        lambda_contra=lambda_opt
        print 'Selected lambda (ipsi & contra)=%1.4e' % lambda_opt
        print 'Error=%1.4e' % err_total_sum[lambda_idx]
    else:
        lambda_ipsi_idx=np.argmin(err_ipsi_sum)
        lambda_contra_idx=np.argmin(err_contra_sum)
        lambda_ipsi=lambda_list[lambda_ipsi_idx]
        lambda_contra=lambda_list[lambda_contra_idx]
        print 'Selected lambda_ipsi=%1.4e' % lambda_ipsi
        print 'Selected lambda_contra=%1.4e' % lambda_contra
        print 'Error ipsi=%1.4e' % err_ipsi_sum[lambda_ipsi_idx]
        print 'Error contra=%1.4e' % err_contra_sum[lambda_contra_idx]
    # set up new fit
    print 'Setting up fit using all data...'
    # use last cval set as initial guess
    W_ipsi_fn=absjoin(inner_dir,"W_ipsi_%1.4e.h5" % lambda_ipsi)
    W_contra_fn=absjoin(inner_dir,"W_contra_%1.4e.h5" % lambda_contra)
    Omega_train_fn=absjoin(outer_dir,'Omega_train.mtx')
    Omega_test_fn=absjoin(outer_dir,'Omega_test.mtx')
    output_ipsi=absjoin(outer_dir,"W_ipsi_opt_%1.4e.h5" % lambda_ipsi)
    output_contra=absjoin(outer_dir,"W_contra_opt_%1.4e.h5" % lambda_contra)
    cmd_ipsi=' '.join([solver,W_ipsi_fn,Omega_train_fn,X_train_fn,
                       Y_train_ipsi_fn,Lx_fn,Ly_ipsi_fn,
                       str(lambda_ipsi),output_ipsi])
    print cmd_ipsi
    fid.write(cmd_ipsi+'\n')
    cmd_contra=' '.join([solver,W_contra_fn,X_train_fn,Y_train_contra_fn,
                         Lx_fn,Ly_contra_fn,str(lambda_contra),output_contra])
    print cmd_contra
    fid.write(cmd_contra+'\n')
    fid_l=open(absjoin(outer_dir,lambda_fn),'w')
    fid_l.write(str(lambda_ipsi)+'\n')
    fid_l.write(str(lambda_contra)+'\n')
    fid_l.close()
fid.close()
