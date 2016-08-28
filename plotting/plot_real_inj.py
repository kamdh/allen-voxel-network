from scipy.io import loadmat,savemat
import numpy as np
from voxlib import *
# fn_matrices='../../data/connectivities/allvis_sdk_test/allvis_sdk_test.mat'
#fout_ipsi_real='injections_sdks_ipsi_test.vti'
#fout_contra_real='injections_sdks_contra_test.vti'
# fn_matrices='../../data/connectivities/allvis_test/allvis_test.mat'
# fout_ipsi_real='injections_ipsi_test.vti'
# fout_contra_real='injections_contra_test.vti'
# fn_matrices='../../data/connectivities/allvis_sdk_new/allvis_sdk_new.mat'
# fout_ipsi_real='injections_ipsi_sdk_new.vti'
fn_matrices='../../data/connectivities/visp_sdk_new/visp_sdk_new.mat'
fout_ipsi_real='injections_ipsi_visp_sdk_new.vti'


mat=loadmat(fn_matrices)
locals().update(mat)

X=experiment_source_matrix.T
Y_ipsi=experiment_target_matrix_ipsi.T
Y_contra=experiment_target_matrix_contra.T
# log transform
# Y_ipsi=np.log10(Y_ipsi+1)
# Y_contra=np.log10(Y_contra+1)
# square root transform
X=np.sqrt(X)
Y_ipsi=np.sqrt(Y_ipsi)
Y_contra=np.sqrt(Y_contra)

Xreal_grid=map_to_regular_grid(X,voxel_coords_source)
Yreal_ipsi_grid=map_to_regular_grid(Y_ipsi,voxel_coords_target_ipsi)
Yreal_contra_grid=map_to_regular_grid(Y_contra,voxel_coords_target_contra)

# does not work:
save_as_vtk_old(fout_ipsi_real,Xreal_grid,Yreal_ipsi_grid,
                voxel_coords_source,voxel_coords_target_ipsi)
#save_as_vtk_old(fout_contra_real,Xreal_grid,Yreal_contra_grid,
#                voxel_coords_source,voxel_coords_target_contra)
save_as_vtk(fout_ipsi_real,Yreal_ipsi_grid,voxel_coords_target_ipsi)
save_as_vtk('injections_source_visp_sdk_new.vti',Xreal_grid,voxel_coords_source)
