import os
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import h5py
import time
import nrrd
from scipy.io import mmwrite
from voxnet.conn2d import *

drive_path = os.path.join(os.getenv('HOME'), 'work/allen/data/sdk_new_100')
output_dir = os.path.join(os.getenv('HOME'), 'work/allen/data/2d_test')

# When downloading 3D connectivity data volumes, what resolution do you want (in microns)?  
# Options are: 10, 25, 50, 100
resolution_um=10

# Downsampling factor
stride = 4

# Omega threshold
Omega_thresh = 0.5

# Volume error thresh, in percent
volume_fraction = 20

# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
manifest_file = os.path.join(drive_path, "manifest.json")

# Start processing data
mcc = MouseConnectivityCache(manifest_file = manifest_file,
                             resolution = resolution_um)
ontology = mcc.get_ontology()
# Injection structure of interest
isocortex = ontology['Isocortex']

# open up a pandas dataframe of all of the experiments
experiments = mcc.get_experiments(dataframe = True, 
                                  injection_structure_ids = [isocortex['id'].values[0]], 
                                  cre = False)
print "%d total experiments" % len(experiments)



## Laplacians
view_paths_fn = os.path.join(os.getenv('HOME'), 'work/allen/data/TopView/top_view_paths_10.h5')
view_paths_file = h5py.File(view_paths_fn, 'r')
view_lut = view_paths_file['view lookup'][:]
view_paths = view_paths_file['paths'][:]
view_paths_file.close()

## Compute size of each path to convert path averages to sums
norm_lut = np.zeros(view_lut.shape, dtype=int)
ind = np.where(view_lut != -1)
ind = zip(ind[0], ind[1])
for curr_ind in ind:
    curr_path_id = view_lut[curr_ind]
    curr_path = view_paths[curr_path_id, :]
    norm_lut[curr_ind] = np.sum(curr_path > 0)

view_lut = downsample(view_lut, stride)
data_mask = np.where(view_lut != -1)
# Right indices
right = np.zeros(view_lut.shape, dtype=bool)
right[:, int(view_lut.shape[1]/2):] = True
# Right hemisphere data
hemi_R_mask = np.where(np.logical_and(view_lut != -1, right))
# Left hemisphere data
hemi_L_mask = np.where(np.logical_and(view_lut != -1, np.logical_not(right)))

nx = len(hemi_R_mask[0])
ny = len(data_mask[0])
Lx = laplacian_2d(hemi_R_mask)
Ly = laplacian_2d(data_mask)
mmwrite(os.path.join(output_dir, "Lx.mtx"), Lx)
mmwrite(os.path.join(output_dir, "Ly.mtx"), Ly)

X = np.zeros((nx, len(experiments)))
Y = np.zeros((ny, len(experiments)))
Omega = np.zeros((ny, len(experiments)))
expt_drop_list = []
t0 = time.time()
#eid = experiments.iloc[5].id
#row = experiments.iloc[5]
index = 0
for eid, row in experiments.iterrows():
    print "\nRow %d\nProcessing experiment %d" % (index,eid)
    print row
    data_dir = os.path.join(os.getenv('HOME'),
                            "work/allen/data/sdk_new_100/experiment_%d/" % eid)
    # get and remap injection data
    in_fn = data_dir + "injection_density_top_view_%d.nrrd" % int(resolution_um)
    print "reading " + in_fn
    in_d_s_full = nrrd.read(in_fn)[0]
    flat_vol = np.nansum(in_d_s_full * norm_lut) * (10./1000.)**3
    expt_union = mcc.get_experiment_structure_unionizes(eid, hemisphere_ids = [3], is_injection = True,
                                                        structure_ids = [ontology['grey']['id'].values[0]])
    full_vol = float(expt_union['projection_volume'])
    in_d_s = downsample(in_d_s_full, stride)
    # get and remap projection data
    pr_fn = data_dir + "projection_density_top_view_%d.nrrd" % int(resolution_um)
    print "reading " + pr_fn
    pr_d_s = downsample(nrrd.read(pr_fn)[0], stride)
    # fill matrices
    X[:, index] = in_d_s[hemi_R_mask]
    Y[:, index] = pr_d_s[data_mask]
    this_Omega = (in_d_s[data_mask] > Omega_thresh).astype(int)
    Omega[:, index] = this_Omega
    # drop experiments without much injection volume
    if np.abs(flat_vol - full_vol) / full_vol * 100 > volume_fraction:
        print "warning, dropping experiment"
        print "flat_vol = %f\nfull_vol = %f" % (flat_vol, full_vol)
        expt_drop_list.append(index)
    index += 1
t1 = time.time()
total = t1-t0
print "%0.1f minutes elapsed" % (total/60.)

