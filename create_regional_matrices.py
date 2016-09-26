from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import os
from voxnet.utilities import *
from voxnet.matrices import generate_region_matrices
import h5py

# Settings:
min_vox = 50
#data_dir=os.path.abspath('../../friday_harbor/data_all')
data_dir = '../../data/sdk_new_100'
resolution = 100
save_data_dir=os.path.abspath('../../data/regional_model/')
save_file_name='experiment_matrices.hdf5'
experiments_fn='../../mesoscale_connectivity_linear_model/data/src/LIMS_id_list.p'
structures_fn='../../mesoscale_connectivity_linear_model/data/src/structure_id_list.p'

# Initializations:
path_to_this_file=os.path.dirname(os.path.realpath(__file__))
LIMS_id_list=unpickle(experiments_fn)

# Load data:
source_id_list=unpickle(structures_fn)
target_id_list=unpickle(structures_fn)

# Setup cache and ontology
manifest_file=os.path.join(data_dir,'manifest.json')
mcc = MouseConnectivityCache(manifest_file=manifest_file,
                             resolution=resolution)
ontology = mcc.get_ontology()
sources = ontology[source_id_list]
targets = ontology[target_id_list]


# Create experiment:
experiment_dict=generate_region_matrices(mcc,
                                         source_id_list, 
                                         target_id_list, 
                                         LIMS_id_list = LIMS_id_list,
                                         min_voxels_per_injection = min_vox,
                                         verbose = True)

# Save to file:
f = h5py.File(os.path.join(path_to_this_file, save_data_dir, save_file_name), 'w')
write_dictionary_to_group(f, experiment_dict)
f.close()
