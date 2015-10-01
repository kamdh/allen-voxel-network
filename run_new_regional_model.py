import numpy as np
import os
import h5py
from kam_interface import utilities
from kam_interface.linear_model import LinearModel as LM
import scipy.optimize as sopt

# Settings:
data_dir='../../friday-harbor/data_all'
load_file_name = 'experiment_matrices.hdf5'
rel_data_dir = '../../data/regional_model/results'
save_dir = rel_data_dir
save_file_name_ipsi = 'W_ipsi.hdf5'
save_file_name_contra = 'W_contra.hdf5'

# Initializations:
path_to_this_file = os.path.dirname(os.path.realpath(__file__))

# Load data:
f = h5py.File(os.path.join(path_to_this_file, rel_data_dir, load_file_name), 'r')
experiment_dict = utilities.read_dictionary_from_group(f)
f.close()

def fit_linear_model(A, B, col_labels, row_labels):
    X = np.zeros((np.shape(A)[1], np.shape(B)[1]))
    for jj, col in enumerate(B.T):
        b = col.T
        X[:,jj] = sopt.nnls(A, b)[0]
    return LM(X, col_labels, row_labels, data_dir=data_dir)

# Ipsilateral fit:
print 'running ipsi'
A = experiment_dict['experiment_source_matrix']
B = experiment_dict['experiment_target_matrix_ipsi']
col_labels = experiment_dict['col_label_list_target']
row_labels = experiment_dict['col_label_list_source']
ipsi_LM = fit_linear_model(A, B, col_labels, row_labels)
ipsi_LM.run_regression(A, B, col_labels, row_labels)
ipsi_LM.save_to_hdf5(os.path.join(save_dir, save_file_name_ipsi))

# Contralateral fit:
print 'running contra'
A = experiment_dict['experiment_source_matrix']
B = experiment_dict['experiment_target_matrix_contra']
col_labels = experiment_dict['col_label_list_target']
row_labels = experiment_dict['col_label_list_source']
contra_LM = fit_linear_model(A, B, col_labels, row_labels)
contra_LM.run_regression(A, B, col_labels, row_labels)
contra_LM.save_to_hdf5(os.path.join(save_dir, save_file_name_contra))

