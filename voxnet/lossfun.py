import numpy as np
from scipy.linalg import norm
from scipy.sparse import find as spfind
 
def sq_error_fro(W,X,Y,Omega=None):
    return eval_error(W,X,Y,Omega)**2
 
def eval_error(W,X,Y,Omega=None):
    r = np.dot(W,X)-Y
    if Omega is not None:
        assert np.all(Omega.shape == Y.shape), \
          "Omega shape incompatible with Y"
        Omega_idx=spfind(Omega)
        r[Omega_idx[0],Omega_idx[1]] = 0.0
    return norm(np.ravel(r),2)
 
def mean_sq_error_fro(W,X,Y,Omega=None):
    ninj=Y.shape[1]
    return sq_error_fro(W,X,Y,Omega)/ninj
 
def rel_MSE(W,X,Y,Omega=None):
    return mean_sq_error_fro(W,X,Y,Omega) / mean_sq_error_fro(0.0,0.0,Y,Omega)
 
def rel_MSE_2(W,X,Y,Omega=None):
    return 2.*mean_sq_error_fro(W,X,Y,Omega) / \
      ( mean_sq_error_fro(0.0,0.0,Y,Omega) + \
        mean_sq_error_fro(0.0,0.0,np.dot(W,X),Omega) )
