import os
import numpy as np
import glob
from kam_interface.utilities import absjoin
from scipy.io import mmread
from scipy.linalg import norm
from scipy.io import savemat
import pandas as pd

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

err_fun=error_MSE

# def sq_error_fro(resid):
#     ncol=resid.shape[1]
#     if ncol > 1:
#         err=0
#         for col in range(ncol):
#             err=err+eval_error(resid[:,col])
#         err=err/ncol
#         return err
#     else:
#         return eval_error(resid)

# def eval_error(resid):
#     # use vector 2-norm for speed to calculate fro norm
#     return norm(np.asarray(resid).ravel())**2

def error_MSE(resid):
    """Computes mean squared error

    Parameters
    ----------
        resid : ndarray, size (M x N)
            M=# data points (voxels, regions), N=# samples
    """
    if resid.ndim==2:
        return (norm(np.asarray(resid).ravel())**2)/resid.shape[1]
    elif resid.ndim==1:
        return (norm(np.asarray(resid).ravel())**2)
    else:
        raise Exception("array passed to error_MSE has incorrect shape")

def error_RMSE(resid):
    return np.sqrt(error_MSE(resid))

# setup some variables
if select_one_lambda:
    lambda_fn='lambda_opt'
else:
    lambda_fn='lambda_ipsi_contra_opt'
n_lambda=len(lambda_list)
fid=open(selected_fit_cmds,'w')
Lx_fn=absjoin(save_dir,save_stem+'_Lx.mtx')
Ly_ipsi_fn=absjoin(save_dir,save_stem+'_Ly_ipsi.mtx')
Ly_contra_fn=absjoin(save_dir,save_stem+'_Ly_contra.mtx')
# loop through the outer loop (validation sets)
outer_dir_list=glob.glob(save_dir+'/cval*')
n_cval=len(outer_dir_list)
err_ipsi=np.zeros((n_cval,))
err_contra=np.zeros((n_cval,))
err_reg_ipsi=np.zeros((n_cval,))
err_reg_contra=np.zeros((n_cval,))
rel_err_ipsi=np.zeros((n_cval,))
rel_err_contra=np.zeros((n_cval,))
rel_err_reg_ipsi=np.zeros((n_cval,))
rel_err_reg_contra=np.zeros((n_cval,))
for o_idx,outer_dir in enumerate(outer_dir_list):
    print 'Entering outer cross-val set ' + str(o_idx)
    # new model training will now be with the outer sets
    X_test_fn=absjoin(outer_dir,'X_test.mtx')
    Y_test_ipsi_fn=absjoin(outer_dir,'Y_test_ipsi.mtx')
    Y_test_contra_fn=absjoin(outer_dir,'Y_test_contra.mtx')
    W_ipsi_fn=glob.glob(absjoin(outer_dir,'W_ipsi_opt_*.mtx'))
    if len(W_ipsi_fn) > 1:
        raise Exception('More than one W_ipsi_opt_*.mtx')
    else:
        W_ipsi_fn=W_ipsi_fn[0]
    W_contra_fn=glob.glob(absjoin(outer_dir,'W_contra_opt_*.mtx'))
    if len(W_contra_fn) > 1:
        raise Exception('More than one W_contra_opt_*.mtx')
    else:
        W_contra_fn=W_contra_fn[0]
    X_test=mmread(X_test_fn)
    Y_test_ipsi=mmread(Y_test_ipsi_fn)
    Y_test_contra=mmread(Y_test_contra_fn)
    W_ipsi=mmread(W_ipsi_fn)
    W_contra=mmread(W_contra_fn)
    Y_pred_ipsi=W_ipsi.dot(X_test)
    r_ipsi=Y_pred_ipsi-Y_test_ipsi
    Y_pred_contra=W_contra.dot(X_test)
    r_contra=Y_pred_contra-Y_test_contra
    err_ipsi[o_idx]=err_fun(r_ipsi)
    err_reg_ipsi[o_idx]=err_fun(P_Y_ipsi.dot(r_ipsi))
    err_contra[o_idx]=err_fun(r_contra)
    err_reg_contra[o_idx]=err_fun(P_Y_contra.dot(r_contra))
    rel_err_ipsi[o_idx]=err_ipsi[o_idx]/\
      (err_fun(Y_test_ipsi)+err_fun(Y_pred_ipsi))
    rel_err_reg_ipsi[o_idx]=err_reg_ipsi[o_idx]/\
      (err_fun(P_Y_ipsi.dot(Y_test_ipsi))+err_fun(P_Y_ipsi.dot(Y_pred_ipsi)))
    rel_err_contra[o_idx]=err_contra[o_idx]/\
      (err_fun(Y_test_contra)+err_fun(Y_pred_contra))
    rel_err_reg_contra[o_idx]=err_reg_contra[o_idx]/\
      (err_fun(P_Y_contra.dot(Y_test_contra))+err_fun(P_Y_contra.dot(Y_pred_contra)))
print "Ipsi  \n======\n"
print "Errors: " + str(err_ipsi)
print "MSE:    " + str(np.mean(err_ipsi))
print "Contra\n======\n"
print "Errors: " + str(err_contra)
print "MSE:    " + str(np.mean(err_contra))
err_dict={}
err_dict['err_ipsi']=err_ipsi
err_dict['err_contra']=err_contra
save_file_name=os.path.join(save_dir,save_stem + '_cval_errors.mat')
savemat(save_file_name,err_dict,oned_as='column',do_compression=True)
errs_vox=pd.DataFrame(np.vstack((err_reg_ipsi,err_ipsi,
                                 rel_err_reg_ipsi,rel_err_ipsi,
                                 err_reg_contra,err_contra,
                                 rel_err_reg_contra,rel_err_contra)).T,
                      columns=['err_vox_reg_ipsi','err_vox_ipsi',
                               'rel_err_vox_reg_ipsi','rel_err_vox_ipsi',
                               'err_vox_reg_contra','err_vox_contra',
                               'rel_err_vox_reg_contra','rel_err_vox_contra'])
all_errs=pd.concat([errs,errs_vox],axis=1)
