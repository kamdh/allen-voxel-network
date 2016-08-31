import h5py
import numpy as np
from friday_harbor.structure import Ontology
import utilities
import numpy.testing as nptest

class LinearModel(object):
    def __init__(self, W, col_labels, row_labels, data_dir='.', P=[]):
        self.data_dir=data_dir
        self.W = W
        self.col_labels = list(col_labels)
        self.row_labels = list(row_labels)
        self.P = P
        self.ontology=Ontology(data_dir=data_dir)

    def export_to_dictionary(self):
        return {'W':self.W,
                'col_labels':self.col_labels,
                'row_labels':self.row_labels,
                'P':self.P,
                'data_dir':self.data_dir}

    def save_to_hdf5(self, file_name):
        f = h5py.File(file_name, 'w')
        utilities.write_dictionary_to_group(f, self.export_to_dictionary())
        f.close()
        
    @staticmethod
    def load_from_hdf5(file_name):
        f = h5py.File(file_name, 'r')
        D = utilities.read_dictionary_from_group(f)
        f.close()        
        if 'P' in D.keys():
            return LinearModel(D['W'], D['col_labels'], D['row_labels'], 
                               data_dir=D['data_dir'], P=D['P'])
        else:
            return LinearModel(D['W'], D['col_labels'], D['row_labels'],
                               data_dir=D['data_dir'])
    
    def get_w_val(self, row_val, col_val):
        if isinstance(row_val, str):
            row_val = self.ontology.acronym_id_dict[row_val]
        if isinstance(col_val, str):
            col_val = self.ontology.acronym_id_dict[col_val]
        col_ind = self.col_labels.index(col_val)
        row_ind = self.row_labels.index(row_val)
        
        return self.W[row_ind, col_ind]
    
    def get_p_val(self, row_val, col_val):
        if isinstance(row_val, str):
            row_val = self.ontology.acronym_id_dict[row_val]
        if isinstance(col_val, str):
            col_val = self.ontology.acronym_id_dict[col_val]
        col_ind = self.col_labels.index(col_val)
        row_ind = self.row_labels.index(row_val)
        return self.P[row_ind, col_ind]
    
    def run_regression(self, A, B, col_labels, row_labels, 
                       default_p_value=np.Inf):
        import statsmodels.api as sm
        
        if self.P != []:
            raise Exception
        else:
            double_check_w_dict = {}
            p_value_dict = {}
            for jj, col_and_curr_col_label in enumerate(zip(B.T, col_labels)):
                col, curr_col_label = col_and_curr_col_label
                b = col.T
                
                # For curr col, determine which rows should be kept for 
                # regression
                nonzero_ind_list = []
                nonzero_label_list = []
                for ii, curr_row_label in enumerate(row_labels):  
                    if self.get_w_val(curr_row_label, curr_col_label) != 0.0:
                        nonzero_ind_list.append(ii)
                        nonzero_label_list.append(curr_row_label)
                res = sm.OLS(b,A[:,nonzero_ind_list]).fit()
                
                # Create double-check dictionary:
                for kk, curr_w in enumerate(res.params):
                    double_check_w_dict[nonzero_label_list[kk], curr_col_label] = curr_w
                
                # Create p-value dictionary:
                for kk, curr_p in enumerate(res.pvalues):
                    p_value_dict[nonzero_label_list[kk], curr_col_label] = curr_p
                
            # Assign p-values to matrix:
            P = np.zeros((np.shape(A)[1], np.shape(B)[1]))
            for ii, curr_row_label in enumerate(self.row_labels):
                for jj, curr_col_label in enumerate(self.col_labels):     
                    if (curr_row_label, curr_col_label) in p_value_dict.keys():
                        P[ii, jj] = p_value_dict[curr_row_label, curr_col_label]
                    else:
                        P[ii, jj] = default_p_value
                    # Double-check coefficient:
                    if self.get_w_val(curr_row_label, curr_col_label) != 0.0:
                        w_opt = self.get_w_val(curr_row_label, curr_col_label)
                        w_reg = double_check_w_dict[curr_row_label, 
                                                    curr_col_label]
                        nptest.assert_almost_equal(w_opt, w_reg, 7, err_msg='Optimization does not match regression')
            self.P = P
#     
#     # Store results in a dictionary:
#     STD_dict = {}
#     P_dict = {}
#     for ii, from_id in enumerate(from_id_list_regression): 
#         STD_dict[from_id] = res.bse[ii]
#         P_dict[from_id] = res.pvalues[ii]
# 
#     # For debugging, to ensure that regression discovers same value as optimization:
#     for ii, from_id in enumerate(from_id_list_regression):
#         abs_error = np.abs((res.params[ii]- optimization_dict[from_id][0]))
#         if abs_error > .00001:
#             print 'WARNING: %s, %s' % (res.params[ii], optimization_dict[from_id][0])
#     
#     # Store away final values in matrix:
#     fill_val = np.Inf
#     target_ind = ind_acronym_dict[curr_target_node.acronym]
#     for source_id in from_id_list:
#         source_ind = ind_acronym_dict[id_acronym_dict[source_id]]
#         
#         if source_id in from_id_list_regression:
#             curr_W_value = optimization_dict[source_id][0]
#             curr_P_value = P_dict[source_id]
#             curr_STD_value = STD_dict[source_id]
#         else:
#             curr_W_value = fill_val
#             curr_P_value = fill_val
#             curr_STD_value = fill_val
#             
#         W[source_ind, target_ind] = curr_W_value
#         P[source_ind, target_ind] = curr_P_value
#         STD[source_ind, target_ind] = curr_STD_value
        
class OldLinearModel(object):

    def __init__(self, W, col_labels, row_labels, P=[]):

        self.W = W
        self.col_labels = list(col_labels)
        self.row_labels = list(row_labels)
        self.P = P

    @staticmethod
    def load_from_hdf5(file_name):
        import h5py
        import utilities
        f = h5py.File(file_name, 'r')
        D = utilities.read_dictionary_from_group(f)
        f.close()        
        if 'P' in D.keys(): 
            return OldLinearModel(D['W'], D['col_labels'], D['row_labels'], 
                                  P=D['P'])
        else:
            return OldLinearModel(D['W'], D['col_labels'], D['row_labels'])
