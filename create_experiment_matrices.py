import os
from kam_interface.utilities import *
from kam_interface.matrices import generate_region_matrices
import h5py

# Settings:
min_vox=50
data_dir='../../friday-harbor/data_all'
save_data_dir='./results/'
save_file_name='experiment_matrices.hdf5'
experiments_fn='../../mesoscale_connectivity_linear_model/data/src/LIMS_id_list.p'
structures_fn='../../mesoscale_connectivity_linear_model/data/src/structure_id_list.p'

# Initializations:
path_to_this_file=os.path.dirname(os.path.realpath(__file__))
LIMS_id_list=unpickle(experiments_fn)

# Load data:
source_id_list=unpickle(structures_fn)
target_id_list=unpickle(structures_fn)

# Create experiment:
experiment_dict=generate_region_matrices(data_dir, source_id_list, 
                                         target_id_list, 
                                         LIMS_id_list=LIMS_id_list,
                                         min_voxels_per_injection=min_vox)
 
# Save to file:
f=h5py.File(os.path.join(path_to_this_file, save_data_dir, save_file_name), 'w')
write_dictionary_to_group(f, experiment_dict)
f.close()
