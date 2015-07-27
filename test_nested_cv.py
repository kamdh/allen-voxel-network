import numpy as np
from scipy.io import loadmat, mmread
#from sklearn.covariance import EmpiricalCovariance
import sys, os
sys.path.append(os.path.abspath("../"))
from smoothness_py import *
import scipy.sparse as sp
from grid_search import NestedGridSearchCV
from sklearn.cross_validation import LeaveOneOut

load_test=True
if load_test:
    X=mmread('../smoothness_c/test_X.mtx')
    Y_ipsi=mmread('../smoothness_c/test_Y_ipsi.mtx')
    Y_contra=mmread('../smoothness_c/test_Y_contra.mtx')
    nx=X.shape[0]
    n_inj=X.shape[1]
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
a_est=var_i/var_c # estimate of variance ratio
# setup estimators
param_grid= {'a': a_est*(1+np.linspace(-0.5,0.5,5)),
             'L': np.logspace(-1,2,5)**2}
est=SmoothLinBiRegression(disp=0, n_iter=300)
#loo_cv=LeaveOneOut(n_inj)
fit_params={'Lx': Lx, 'Ly_i': Ly_ipsi, 'Ly_c': Ly_contra}
nested_cv=NestedGridSearchCV(est, param_grid, 'mean_squared_error',
                             cv=4, #loo_cv,
                             inner_cv=3, #lambda _x, _y: LeaveOneOut(n_inj-1),
                             fit_params=fit_params)
nested_cv.fit(X,np.hstack((Y_ipsi,Y_contra)))

from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
    for i, scores in enumerate(nested_cv.grid_scores_):
        scores.to_csv('grid-scores-%d.csv' % (i + 1), index=False)
    print("______________")
    print(nested_cv.best_params_)
