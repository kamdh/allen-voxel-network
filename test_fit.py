import numpy as np
from scipy.io import loadmat, mmread
#from sklearn.covariance import EmpiricalCovariance
import sys, os
sys.path.append(os.path.abspath("../"))
from smoothness_py import *
import scipy.sparse as sp

load_test=True


if load_test:
    X=mmread('../smoothness_c/test_X.mtx')
    Y_ipsi=mmread('../smoothness_c/test_Y_ipsi.mtx')
    Y_contra=mmread('../smoothness_c/test_Y_contra.mtx')
    nx=X.shape[0]
    ny_i=Y_ipsi.shape[0]
    ny_c=Y_contra.shape[0]
    Lx=sp.diags([1,-2,1],[-1,0,1],shape=(nx,nx))
    Ly_ipsi=sp.diags([1,-2,1],[-1,0,1],shape=(ny_i,ny_i))
    Ly_contra=sp.diags([1,-2,1],[-1,0,1],shape=(ny_c,ny_c))
    X=X.T
    Y_ipsi=Y_ipsi.T
    Y_contra=Y_contra.T
else:
    # setup the run
    param_fn='run_setup.py'
    with open(param_fn) as f:
        code = compile(f.read(), param_fn, 'exec')
        exec(code)

    save_file_name=os.path.join(save_dir,save_stem + '.mat')
    experiment_dict=loadmat(save_file_name)
    X=experiment_dict['experiment_source_matrix'].T
    Y_ipsi=experiment_dict['experiment_target_matrix_ipsi'].T
    Y_contra=experiment_dict['experiment_target_matrix_contra'].T
    Lx=experiment_dict['Lx'].T
    Ly_ipsi=experiment_dict['Ly_ipsi'].T
    Ly_contra=experiment_dict['Ly_contra'].T
# estimate covariances
var_i=np.var(Y_ipsi.ravel())
var_c=np.var(Y_contra.ravel())
a=var_i/var_c
Lambda=100.
est=SmoothLinBiRegression(Lambda, a, disp=1)
est.fit(X.T, np.vstack((Y_ipsi,Y_contra)).T, W0=None,
        Lx=Lx, Ly_i=Ly_ipsi, Ly_c=Ly_contra)
est.fit(X.T, np.vstack((Y_ipsi,Y_contra)).T, W0=est.W_,
        Lx=Lx, Ly_i=Ly_ipsi, Ly_c=Ly_contra)
