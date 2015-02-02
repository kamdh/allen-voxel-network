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
