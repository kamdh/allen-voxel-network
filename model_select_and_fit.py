import os
import numpy as np
import glob
from kam_interface.utilities import absjoin
from scipy.io import mmread
from scipy.linalg import norm

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

def sq_error_fro(W,X,Y):
    return eval_error(W,X,Y)**2

def eval_error(W,X,Y):
    return norm(np.asarray(np.dot(W,X)-Y).ravel())

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
    X_train_fn=absjoin(outer_dir,'X_train.mtx')
    Y_train_ipsi_fn=absjoin(outer_dir,'Y_train_ipsi.mtx')
    Y_train_contra_fn=absjoin(outer_dir,'Y_train_contra.mtx')
    # inner loop 
    inner_dirs=glob.glob(outer_dir+'/cval*')
    n_inner=len(inner_dirs)
    err_contra=np.zeros((n_inner,n_lambda))
    err_ipsi=np.zeros((n_inner,n_lambda))
    for i,inner_dir in enumerate(inner_dirs):
        print '  Processing inner cross-val set ' + str(i)
        X_test_fn=absjoin(inner_dir,'X_test.mtx')
        Y_test_ipsi_fn=absjoin(inner_dir,'Y_test_ipsi.mtx')
        Y_test_contra_fn=absjoin(inner_dir,'Y_test_contra.mtx')
        X_test=mmread(X_test_fn)
        Y_test_ipsi=mmread(Y_test_ipsi_fn)
        Y_test_contra=mmread(Y_test_contra_fn)
        # for each lambda, evaluate error
        for j,lambda_val in enumerate(lambda_list):
            print '    Evaluating error for lambda=%1.4e'%lambda_val
            W_ipsi_fn=absjoin(inner_dir,"W_ipsi_%1.4e.mtx"%lambda_val)
            W_contra_fn=absjoin(inner_dir,"W_contra_%1.4e.mtx"%lambda_val)
            W_ipsi=mmread(W_ipsi_fn)
            W_contra=mmread(W_contra_fn)
            err_ipsi[i,j]=sq_error_fro(W_ipsi,X_test,Y_test_ipsi)
            err_contra[i,j]=sq_error_fro(W_contra,X_test,Y_test_contra)
            print str(err_ipsi[i,j]) + ' ' + str(err_contra[i,j])
    # summarize errors by mean over inner sets
    err_ipsi_sum=np.mean(err_ipsi,axis=0)
    err_contra_sum=np.mean(err_contra,axis=0)
    err_total_sum=err_ipsi_sum + err_contra_sum
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
    print 'Setting up fit using all data...'
    output_ipsi=absjoin(outer_dir,"W_ipsi_opt_%1.4e.mtx"%lambda_ipsi)
    output_contra=absjoin(outer_dir,"W_contra_opt_%1.4e.mtx"%lambda_contra)
    cmd=' '.join([solver,X_train_fn,Y_train_ipsi_fn,
                  Lx_fn,Ly_ipsi_fn,str(lambda_ipsi),output_ipsi])
    print cmd
    fid.write(cmd+'\n')
    cmd=' '.join([solver,X_train_fn,Y_train_contra_fn,
                  Lx_fn,Ly_contra_fn,str(lambda_contra),output_contra])
    print cmd
    fid.write(cmd+'\n')
    fid_l=open(absjoin(outer_dir,lambda_fn),'w')
    fid_l.write(str(lambda_ipsi)+'\n')
    fid_l.write(str(lambda_contra)+'\n')
    fid_l.close()
fid.close()
