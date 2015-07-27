import numpy as np
from kam_interface.linear_model import LinearModel as LM
from kam_interface.linear_model import OldLinearModel as LMold
import kam_interface.utilities as utilities
from scipy.io import loadmat
    
def reorder_matrix(lm):
    row_keys=[lm.ontology.id_acronym_dict[key] for key in lm.row_labels]
    row_idx=np.argsort(row_keys)
    Prows=permutation(row_idx)
    col_keys=[lm.ontology.id_acronym_dict[key] for key in lm.col_labels]
    col_idx=np.argsort(col_keys)
    Pcols=permutation(col_idx)
    lm.W=Pcols*lm.W*Prows
    lm.P=Pcols*lm.P*Prows
    lm.row_labels=list(np.array(lm.row_labels)[row_idx])
    lm.col_labels=list(np.array(lm.col_labels)[col_idx])
    
def permutation(indices):
    import scipy.sparse
    n=len(indices)
    P=scipy.sparse.lil_matrix((n,n), dtype=np.int8)
    for i,j in enumerate(indices):
        P[i,j]=1
    return P

def load_ex_mat(fn):
    import h5py
    import os.path
    if os.path.isfile(fn):
        f=h5py.File(fn,'r')
        d=utilities.read_dictionary_from_group(f)
        f.close()
        return d
    else:
        raise Exception('Filename %s does not exist' % fn)
    
ex_new_fn='results/experiment_matrices.hdf5'
ex_old_fn='../../mesoscale_connectivity_linear_model/full_matrix/results/experiment_matrices.hdf5'
W_ipsi_old_fn='../../mesoscale_connectivity_linear_model/full_matrix/results/W_ipsi.hdf5'
W_contra_old_fn='../../mesoscale_connectivity_linear_model/full_matrix/results/W_contra.hdf5'
W_ipsi_new_fn='results/W_ipsi.hdf5'
W_contra_new_fn='results/W_contra.hdf5'

## check experiment matrices
ex_new=load_ex_mat(ex_new_fn)
ex_old=load_ex_mat(ex_old_fn)
assert all(ex_new['row_label_list'] == ex_old['row_label_list'])
assert all(ex_new['col_label_list_source'] == ex_old['col_label_list_source'])
assert all(ex_new['col_label_list_target'] == ex_old['col_label_list_target'])
try:
    assert np.allclose(ex_new['experiment_target_matrix_ipsi'], 
                       ex_old['experiment_target_matrix_ipsi'])
except AssertionError:
    print 'target ipsi unequal'
try:
    assert np.allclose(ex_new['experiment_source_matrix'], 
                       ex_old['experiment_source_matrix'])
except AssertionError:
    print 'source unequal'
try:
    assert np.allclose(ex_new['experiment_target_matrix_contra'], 
                       ex_old['experiment_target_matrix_contra'])
except AssertionError:
    print 'target contra unequal'


## check fit matrices
W_ipsi_old=LMold.load_from_hdf5(W_ipsi_old_fn)
W_contra_old=LMold.load_from_hdf5(W_contra_old_fn)
W_ipsi_new=LM.load_from_hdf5(W_ipsi_new_fn)
W_contra_new=LM.load_from_hdf5(W_contra_new_fn)

# no need to reorder now
#reorder_matrix(W_ipsi_new)
#reorder_matrix(W_contra_new)
## check that row and column labels now match
assert W_ipsi_old.row_labels==W_ipsi_new.row_labels, 'row labels ipsi unequal'
assert W_ipsi_old.col_labels==W_ipsi_new.col_labels, 'col labels ipsi unequal'
assert W_contra_old.row_labels==W_contra_new.row_labels, 'row labels contra unequal'
assert W_contra_old.col_labels==W_contra_new.col_labels, 'col labels contra unequal'
np.testing.assert_allclose(W_contra_old.W, W_contra_new.W, rtol=1e-5, 
                           atol=1e-6, err_msg='W contra not close')
np.testing.assert_allclose(W_ipsi_old.W, W_ipsi_new.W, rtol=1e-5, 
                           atol=1e-6, err_msg='W ipsi not close')
# without setting insignificant values to inf, we won't get close results
# this is because some of the very close to 1 values are inf in the other
W_contra_old.P[W_contra_old.P > 0.99] = np.inf
W_ipsi_old.P[W_ipsi_old.P > 0.99] = np.inf
W_contra_new.P[W_contra_new.P > 0.99] = np.inf
W_ipsi_new.P[W_ipsi_new.P > 0.99] = np.inf
np.testing.assert_allclose(W_contra_old.P, W_contra_new.P, rtol=1e-5, 
                           atol=1e-3, err_msg='P contra not close')
np.testing.assert_allclose(W_ipsi_old.P, W_ipsi_new.P, rtol=1e-5, 
                           atol=1e-3, err_msg='P ipsi not close')

print 'All tests completed successfully'
