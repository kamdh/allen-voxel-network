import os
import numpy as np
import scipy.optimize as sopt
from scipy.linalg import norm,pinv,kron
from scipy.io import loadmat,savemat,mmread
from sklearn import cross_validation, metrics
import pandas as pd
import glob
from voxnet.utilities import absjoin,h5read
from voxnet.lossfun import *
from scipy.sparse import find as spfind

# relative error type
# rel_type = 1 # normalize by |Y_true|
rel_type = 2 # normalize by 0.5*(|Y_true| + |Y_pred|)
proj_errors = False

# setup the run
param_fn = 'run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

save_file_name = os.path.join(save_dir,save_stem + '.mat')
mat = loadmat(save_file_name)
locals().update(mat) # load into locals namespace (MATLAB-like)

def proj_Omega(Y, Omega):
    assert np.all(Omega.shape == Y.shape), "Omega shape incompatible with Y"
    Ynew = Y.copy()
    I, J, V = spfind(Omega)
    Ynew[I, J] = 0.0
    return Ynew

def fit_linear_model_proj(X,Y,P_Y_dag,P_X,Omega):
    A = kron(X.T.dot(P_X.T),P_Y_dag)
    for ii in range(A.shape[1]):
        col = A[:,ii]
        col_proj = proj_Omega(col.reshape(Y.shape,order = 'F'),Omega)
        A[:,ii] = col_proj.flatten(order = 'F')
    b = proj_Omega(Y,Omega).flatten(order = 'F')
    W = sopt.nnls(A,b)[0]
    return W.reshape((P_Y_dag.shape[1],P_X.shape[0]), order = 'F')

def fit_linear_model(X, Y, col_wise = False):
    """Y = WX convention unless col_wise is True"""
    if not col_wise:
        X = X.T
        Y = Y.T
    W = np.zeros((np.shape(X)[1], np.shape(Y)[1]))
    for jj, col in enumerate(Y.T):
        b = col.T
        W[:,jj] = sopt.nnls(X, b)[0]
    if not col_wise:
        W = W.T
    return W

def construct_proj_op(label_list):
    unique_labels = np.unique(label_list)
    n_vox = len(label_list)
    n_reg = len(unique_labels)
    proj = np.zeros((n_reg,n_vox))
    proj_pinv = np.zeros((n_vox,n_reg))
    for i,r in enumerate(unique_labels):
        r_idx = np.where(label_list == r)[0]
        proj[i,r_idx] = 1.
    proj_pinv = pinv(proj)
    return (proj, proj_pinv)

def error_MSE(resid):
    """
    Computes mean squared error

    Parameters
    ----------
        resid : ndarray, size (M x N)
            M = # data points (voxels, regions), N = # samples
    """
    if resid.ndim == 2:
        return (norm(np.asarray(resid).ravel())**2)/float(resid.shape[1])
    elif resid.ndim == 1:
        return (norm(np.asarray(resid).ravel())**2)
    else:
        raise Exception("array passed to error_MSE has incorrect shape")

def error_RMSE(resid):
    return np.sqrt(error_MSE(resid))

def region_CV_fits_and_errors(X, Y, P_X, P_Y, P_Y_dag, err_fun,
                              Omega = None, rel_type = 2):
    n_inj = X.shape[1]
    outer_sets = cross_validation.LeaveOneOut(n_inj)
    err_reg = np.zeros((len(outer_sets),))
    err_homog = np.zeros((len(outer_sets),))
    rel_err_reg = np.zeros((len(outer_sets),))
    rel_err_homog = np.zeros((len(outer_sets),))
    GOF_reg = np.zeros((len(outer_sets),))
    GOF_homog = np.zeros((len(outer_sets),))
    for i, (train,test) in enumerate(outer_sets):
        # compare models in outer sets only, same as eventual test errors in the
        # nested cross-validation procedure
        X_train = X[:,train]
        X_test = X[:,test]
        Y_train = Y[:,train]
        Y_test = Y[:,test]
        if Omega is not None:
            Omega_train = Omega[:,train]
            Omega_test = Omega[:,test]
            # W = fit_linear_model_proj(X_train, Y_train, P_Y_dag, P_X,
            #                           Omega_train)
            W = fit_linear_model(P_X.dot(X_train), P_Y.dot(Y_train))
        else:
            W = fit_linear_model(P_X.dot(X_train), P_Y.dot(Y_train))
        # W = fit_linear_model(P_X.dot(X_train), P_Y.dot(Y_train))
        Y_pred = W.dot(P_X.dot(X_test))
        Y_pred_homog = P_Y_dag.dot(Y_pred)
        Y_test_reg = P_Y.dot(Y_test)
        resid_reg = Y_pred - Y_test_reg # regional matrix
        resid_homog = Y_pred_homog - Y_test # voxel-homogeneous matrix
        err_reg[i] = err_fun(resid_reg)
        if Omega is not None:
            err_homog[i] = err_fun(proj_Omega(resid_homog, Omega_test))
        else:
            err_homog[i] = err_fun(resid_homog)
        if rel_type == 1:
            rel_err_reg[i] = err_reg[i]/err_fun(Y_test_reg)
            rel_err_homog[i] = err_homog[i]/err_fun(Y_test)
            GOF_reg[i] = err_fun(W.dot(P_X.dot(X_train))-P_Y.dot(Y_train))/\
              err_fun(P_Y.dot(Y_train))
            GOF_homog[i] = err_fun(P_Y_dag.dot(W.dot(P_X.dot(X_train)))-Y_train)/\
              err_fun(Y_train)
        elif rel_type == 2:
            rel_err_reg[i] = \
              2*err_reg[i] / (err_fun(Y_test_reg) + err_fun(Y_pred))
            GOF_reg[i] = 2*err_fun(W.dot(P_X.dot(X_train))-P_Y.dot(Y_train))/\
              (err_fun(P_Y.dot(Y_train))+err_fun(W.dot(P_X.dot(X_train))))
            if (Omega is not None) and proj_errors:
                rel_err_homog[i] = 2*err_homog[i] / \
                  (err_fun(proj_Omega(Y_test, Omega_test))+
                   err_fun(proj_Omega(Y_pred_homog, Omega_test)))
                GOF_homog[i] = \
                  2*err_fun(P_Y_dag.dot(W.dot(P_X.dot(X_train)))-Y_train) / \
                  (err_fun(proj_Omega(Y_train, Omega_train)) +
                   err_fun(proj_Omega(P_Y_dag.dot(W.dot(P_X.dot(X_train))),
                                      Omega_train)))
            else:
                rel_err_homog[i] = 2*err_homog[i]/\
                  (err_fun(Y_test) + err_fun(Y_pred_homog))
                GOF_homog[i] = \
                  2*err_fun(P_Y_dag.dot(W.dot(P_X.dot(X_train)))-Y_train)/\
                  (err_fun(Y_train) +
                   err_fun(P_Y_dag.dot(W.dot(P_X.dot(X_train)))))
        #if i == 2:
        #    import pdb
        #    pdb.set_trace()
    return (err_reg,err_homog,rel_err_reg,rel_err_homog,
            GOF_reg,GOF_homog)

## setup error function
err_fun = error_MSE

n_inj = experiment_source_matrix.shape[0]
n_x = experiment_source_matrix.shape[1]
n_y_ipsi = experiment_target_matrix_ipsi.shape[1]
n_y_contra = experiment_target_matrix_contra.shape[1]
R_x = len(source_acronyms)
R_y = len(target_acronyms)
P_X, P_X_dag = construct_proj_op(col_label_list_source)
P_Y_ipsi, P_Y_ipsi_dag = construct_proj_op(col_label_list_target_ipsi)
P_Y_contra, P_Y_contra_dag = construct_proj_op(col_label_list_target_contra)

Omega = Omega.T
X = experiment_source_matrix.T
Y_ipsi = experiment_target_matrix_ipsi.T
Y_contra = experiment_target_matrix_contra.T
#err = pd.DataFrame(np.nan((len(outer_sets),2), columns = ['err_reg','err_vox_proj'])

## compute regional model errors via LOO cv
errs_ipsi = region_CV_fits_and_errors(X, Y_ipsi, P_X, P_Y_ipsi, P_Y_ipsi_dag,
                                      err_fun, Omega=Omega, rel_type=rel_type)
errs_contra = region_CV_fits_and_errors(X, Y_contra, P_X, P_Y_contra,
                                        P_Y_contra_dag,
                                        err_fun, rel_type=rel_type)

errs_reg = pd.DataFrame(np.vstack((errs_ipsi,errs_contra)).T,
                        columns = ['err_reg_ipsi','err_vox_ipsi',
                                'rel_reg_ipsi','rel_vox_ipsi',
                                'rel_train_reg_ipsi','rel_train_vox_ipsi',
                                'err_reg_contra','err_vox_contra',
                                'rel_reg_contra','rel_vox_contra',
                                'rel_train_reg_contra','rel_train_vox_contra'])
errs_reg['model'] = 'regional'

## evaluate voxel errors
n_lambda = len(lambda_list)
fid = open(selected_fit_cmds,'w')
Lx_fn = absjoin(save_dir,save_stem+'_Lx.mtx')
Ly_ipsi_fn = absjoin(save_dir,save_stem+'_Ly_ipsi.mtx')
Ly_contra_fn = absjoin(save_dir,save_stem+'_Ly_contra.mtx')
# loop through the outer loop (validation sets)
outer_dir_list = glob.glob(save_dir+'/cval*')
n_cval = len(outer_dir_list)

# arrays to hold error values
err_ipsi = np.zeros((n_cval,))
err_contra = np.zeros((n_cval,))
err_reg_ipsi = np.zeros((n_cval,))
err_reg_contra = np.zeros((n_cval,))
rel_vox_ipsi = np.zeros((n_cval,))
rel_vox_contra = np.zeros((n_cval,))
rel_reg_ipsi = np.zeros((n_cval,))
rel_reg_contra = np.zeros((n_cval,))
rel_train_reg_ipsi = np.zeros((n_cval,))
rel_train_reg_contra = np.zeros((n_cval,))
rel_train_vox_ipsi = np.zeros((n_cval,))
rel_train_vox_contra = np.zeros((n_cval,))
for o_idx,outer_dir in enumerate(outer_dir_list):
    print 'Entering outer cross-val set ' + str(o_idx)
    # new model training will now be with the outer sets
    X_test_fn = absjoin(outer_dir,'X_test.h5')
    Y_test_ipsi_fn = absjoin(outer_dir,'Y_test_ipsi.h5')
    Y_test_contra_fn = absjoin(outer_dir,'Y_test_contra.h5')
    X_train_fn = absjoin(outer_dir,'X_train.h5')
    Y_train_ipsi_fn = absjoin(outer_dir,'Y_train_ipsi.h5')
    Y_train_contra_fn = absjoin(outer_dir,'Y_train_contra.h5')
    W_ipsi_fn = glob.glob(absjoin(outer_dir,'W_ipsi_opt_*.h5'))
    Omega_test_fn = absjoin(outer_dir,'Omega_test.mtx')
    Omega_train_fn = absjoin(outer_dir,'Omega_train.mtx')
    if len(W_ipsi_fn) > 1:
        raise Exception('More than one W_ipsi_opt_*.h5')
    elif len(W_ipsi_fn) == 0:
        raise Exception('No W_ipsi_opt_*.h5 found')
    else:
        W_ipsi_fn = W_ipsi_fn[0]
    W_contra_fn = glob.glob(absjoin(outer_dir,'W_contra_opt_*.h5'))
    if len(W_contra_fn) > 1:
        raise Exception('More than one W_contra_opt_*.h5')
    else:
        W_contra_fn = W_contra_fn[0]
    # load data
    X_train = h5read(X_train_fn)
    Y_train_ipsi = h5read(Y_train_ipsi_fn)
    Y_train_contra = h5read(Y_train_contra_fn)
    X_test = h5read(X_test_fn)
    Y_test_ipsi = h5read(Y_test_ipsi_fn)
    Omega = mmread(Omega_test_fn)
    Omega_train = mmread(Omega_train_fn)
    Y_test_contra = h5read(Y_test_contra_fn)
    W_ipsi = h5read(W_ipsi_fn)
    W_contra = h5read(W_contra_fn)
    # predict w/ linear model
    Y_pred_ipsi = W_ipsi.dot(X_test)
    Y_pred_contra = W_contra.dot(X_test)
    Y_pred_ipsi_train = W_ipsi.dot(X_train)
    Y_pred_contra_train = W_contra.dot(X_train)
    # residuals
    r_ipsi = proj_Omega(Y_pred_ipsi-Y_test_ipsi, Omega)
    r_contra = Y_pred_contra-Y_test_contra
    r_ipsi_train = proj_Omega(Y_pred_ipsi_train - Y_train_ipsi, Omega_train)
    r_contra_train = Y_pred_contra_train-Y_train_contra
    # compute errors
    err_ipsi[o_idx] = err_fun(r_ipsi)
    err_reg_ipsi[o_idx] = err_fun(P_Y_ipsi.dot(r_ipsi))
    err_contra[o_idx] = err_fun(r_contra)
    err_reg_contra[o_idx] = err_fun(P_Y_contra.dot(r_contra))
    if rel_type == 1:
        # testing
        rel_vox_ipsi[o_idx] = err_ipsi[o_idx] / \
          err_fun(proj_Omega(Y_test_ipsi,Omega))
        rel_reg_ipsi[o_idx] = err_reg_ipsi[o_idx] / \
          err_fun(P_Y_ipsi.dot(proj_Omega(Y_test_ipsi,Omega)))
        rel_vox_contra[o_idx] = err_contra[o_idx] / \
          err_fun(Y_test_contra)
        rel_reg_contra[o_idx] = err_reg_contra[o_idx] / \
          err_fun(P_Y_contra.dot(Y_test_contra))
        # training
        rel_train_vox_ipsi[o_idx] = err_fun(r_ipsi_train) / \
          err_fun(proj_Omega(Y_train_ipsi,Omega_train))
        rel_train_reg_ipsi[o_idx] = err_fun(P_Y_ipsi.dot(r_ipsi_train)) / \
          err_fun(P_Y_ipsi.dot(proj_Omega(Y_train_ipsi,Omega_train)))
        rel_train_vox_contra[o_idx] = err_fun(r_contra_train) / \
          err_fun(Y_train_contra)
        rel_train_reg_contra[o_idx] = err_fun(P_Y_contra.dot(r_contra_train))/ \
          err_fun(P_Y_contra.dot(Y_train_contra))
    elif rel_type == 2:
        # testing
        if (Omega is not None) and proj_errors:
            rel_vox_ipsi[o_idx] = 2*err_ipsi[o_idx]/\
              (err_fun(proj_Omega(Y_test_ipsi,Omega))+
               err_fun(proj_Omega(Y_pred_ipsi,Omega)))
            rel_reg_ipsi[o_idx] = 2*err_reg_ipsi[o_idx]/\
              (err_fun(P_Y_ipsi.dot(proj_Omega(Y_test_ipsi,Omega)))+
               err_fun(P_Y_ipsi.dot(proj_Omega(Y_pred_ipsi,Omega))))
        else:
            rel_vox_ipsi[o_idx] = 2*err_ipsi[o_idx]/\
              ( err_fun(Y_test_ipsi) + 
                err_fun(Y_pred_ipsi) )
            rel_reg_ipsi[o_idx] = 2*err_reg_ipsi[o_idx]/\
              ( err_fun(P_Y_ipsi.dot(Y_test_ipsi))+
                err_fun(P_Y_ipsi.dot(Y_pred_ipsi)) )
        rel_vox_contra[o_idx] = 2*err_contra[o_idx]/\
          (err_fun(Y_test_contra)+err_fun(Y_pred_contra))
        rel_reg_contra[o_idx] = 2*err_reg_contra[o_idx]/\
          (err_fun(P_Y_contra.dot(Y_test_contra))+
           err_fun(P_Y_contra.dot(Y_pred_contra)))
        # training
        if (Omega is not None) and proj_errors:
            rel_train_vox_ipsi[o_idx] = 2*err_fun(r_ipsi_train) / \
              (err_fun(proj_Omega(Y_train_ipsi,Omega_train)) +
               err_fun(proj_Omega(Y_pred_ipsi_train,Omega_train)))
            rel_train_reg_ipsi[o_idx] = 2*err_fun(P_Y_ipsi.dot(r_ipsi_train)) /\
              (err_fun(P_Y_ipsi.dot(proj_Omega(Y_train_ipsi,Omega_train))) +
               err_fun(P_Y_ipsi.dot(proj_Omega(Y_pred_ipsi_train,Omega_train))))
        else:
            rel_train_vox_ipsi[o_idx] = 2*err_fun(r_ipsi_train) / \
              ( err_fun(Y_train_ipsi) +
                err_fun(Y_pred_ipsi_train) )
            rel_train_reg_ipsi[o_idx] = 2*err_fun(P_Y_ipsi.dot(r_ipsi_train)) /\
              ( err_fun(P_Y_ipsi.dot(Y_train_ipsi)) +
                err_fun(P_Y_ipsi.dot(Y_pred_ipsi_train)) )
        rel_train_vox_contra[o_idx] = 2*err_fun(r_contra_train) / \
          (err_fun(Y_train_contra) + err_fun(Y_pred_contra_train))
        rel_train_reg_contra[o_idx] = \
          2*err_fun(P_Y_contra.dot(r_contra_train)) / \
          (err_fun(P_Y_contra.dot(Y_train_contra)) +
           err_fun(P_Y_contra.dot(Y_pred_contra_train)))

print "Voxel errors:"
print "Ipsi  \n======\n"
print "Errors: " + str(err_ipsi)
print "MSE:    " + str(np.mean(err_ipsi))
print "Contra\n =====\n"
print "Errors: " + str(err_contra)
print "MSE:    " + str(np.mean(err_contra))
err_dict = {}
err_dict['err_ipsi'] = err_ipsi
err_dict['err_contra'] = err_contra
save_file_name = os.path.join(save_dir,save_stem + '_cval_errors.mat')
savemat(save_file_name,err_dict,oned_as = 'column',do_compression = True)
errs_vox = pd.DataFrame(np.vstack((err_reg_ipsi, err_ipsi,
                                   rel_reg_ipsi, rel_vox_ipsi,
                                   err_reg_contra, err_contra,
                                   rel_reg_contra, rel_vox_contra,
                                   rel_train_reg_ipsi, rel_train_reg_contra,
                                   rel_train_vox_ipsi, rel_train_vox_contra)).T,
                      columns = ['err_reg_ipsi','err_vox_ipsi',
                               'rel_reg_ipsi','rel_vox_ipsi',
                               'err_reg_contra','err_vox_contra',
                               'rel_reg_contra','rel_vox_contra',
                               'rel_train_reg_ipsi', 'rel_train_reg_contra',
                               'rel_train_vox_ipsi', 'rel_train_vox_contra'])
errs_vox['model'] = 'voxel'
all_errs = errs_reg.append(errs_vox)
#all_errs = pd.concat([errs_reg,errs_vox],axis = 1)
all_errs.to_csv(os.path.join(save_dir,save_stem + "_all_errors.csv"))
gp = all_errs.groupby('model')
print gp.mean()
