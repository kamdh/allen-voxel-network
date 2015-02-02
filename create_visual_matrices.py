from friday_harbor.structure import Ontology
from kam_interface.utilities import *
import h5py
import os
from IPython import embed
from scipy.io import savemat

min_vox=50
#save_file_name='visual_output.hdf5'
save_file_name='visual_output.mat'
data_dir='../../friday_harbor/data_all'
experiments_fn='../../mesoscale_connectivity_linear_model/data/src/LIMS_id_list.p'
# source_acronyms=['VISp','LGd']
# target_acronyms=['VISal', 'VISpm', 'VISam', 'VISpl']
source_acronyms=['VISp','VISal','VISam','VISpm','VISpl']
target_acronyms=['VISp','VISal','VISam','VISpm','VISpl']


ontology=Ontology(data_dir=data_dir)

def acro_list_to_id_list(acronyms):
    return [ontology.acronym_id_dict[name] for name in acronyms]

sources=acro_list_to_id_list(source_acronyms)
targets=acro_list_to_id_list(target_acronyms)
LIMS_id_list=unpickle(experiments_fn)

# Create experiment:
experiment_dict=generate_voxel_matrices(data_dir, sources, targets,
                                        LIMS_id_list=LIMS_id_list,
                                        min_voxels_per_injection=min_vox,
                                        laplacian=True,
                                        verbose=True,
                                        source_shell=True,
                                        source_coverage=0.8)

# Save file
# f=h5py.File(os.path.join(save_file_name), 'w')
# write_dictionary_to_group(f, experiment_dict)
# f.close()
savemat(save_file_name,experiment_dict,oned_as='column')
