import os
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import h5py
import time
import nrrd
from voxnet.conn2d import map_to_surface

drive_path = os.path.join(os.getenv('HOME'), 'work/allen/data/sdk_new_100')
output_dir = os.path.join(os.getenv('HOME'), 'work/allen/data/2d_test')

# When downloading 3D connectivity data volumes, what resolution do you want
# (in microns)?  
# Options are: 10, 25, 50, 100
resolution_um = 10

# Drop list criterion, in percent difference
volume_fraction = 20

# The manifest file is a simple JSON file that keeps track of all of
# the data that has already been downloaded onto the hard drives.
# If you supply a relative path, it is assumed to be relative to your
# current working directory.
manifest_file = os.path.join(drive_path, "manifest.json")

# Start processing data
mcc = MouseConnectivityCache(manifest_file=manifest_file,
                             resolution=resolution_um)
ontology = mcc.get_ontology()
# Injection structure of interest
isocortex = ontology['Isocortex']

# open up a pandas dataframe of all of the experiments
experiments = mcc.get_experiments(dataframe=True, 
                                  injection_structure_ids=\
                                      [isocortex['id'].values[0]], 
                                  cre=False)
print "%d total experiments" % len(experiments)

view_paths_fn = os.path.join(os.getenv('HOME'),
                             'work/allen/data/TopView/top_view_paths_10.h5')
view_paths_file = h5py.File(view_paths_fn, 'r')
view_lut = view_paths_file['view lookup'][:]
view_paths = view_paths_file['paths'][:]
view_paths_file.close()

# Compute size of each path to convert path averages to sums
norm_lut = np.zeros(view_lut.shape, dtype=int)
ind = np.where(view_lut != -1)
ind = zip(ind[0], ind[1])
for curr_ind in ind:
    curr_path_id = view_lut[curr_ind]
    curr_path = view_paths[curr_path_id, :]
    norm_lut[curr_ind] = np.sum(curr_path > 0)

t0 = time.time()
expt_drop_list = []
full_vols = []
flat_vols = []
#eid = experiments.iloc[5].id
#row = experiments.iloc[5]
for eid, row in experiments.iterrows():
    print "\nProcessing experiment %d" % eid
    print row
    data_dir = os.path.join(os.getenv('HOME'),
                            "work/allen/data/sdk_new_100/experiment_%d/" % eid)
    # get and remap injection data
    print "getting injection density"
    in_d, in_info = mcc.get_injection_density(eid)
    print "mapping to surface"
    in_d_s = map_to_surface(in_d, view_lut, view_paths,
                            scale=resolution_um/10., fun=np.mean)
    flat_vol = np.nansum(in_d_s * norm_lut) * (10./1000.)**3
    flat_vols.append(flat_vol)
    full_vol = np.nansum(in_d) * (10./1000.)**3
    full_vols.append(full_vol)
    print "flat_vol = %f\nfull_vol = %f" % (flat_vol, full_vol)
    # drop experiments without much injection volume
    if np.abs(flat_vol - full_vol) / full_vol * 100 > volume_fraction:
        print "warning, placing experiment in drop list"
        expt_drop_list.append(eid)
    in_fn = data_dir + "injection_density_top_view_%d.nrrd" % int(resolution_um)
    print "writing " + in_fn
    nrrd.write(in_fn, in_d_s)
    # get and remap projection data
    print "getting projection density"
    pr_d, pr_info = mcc.get_projection_density(eid)
    print "mapping to surface"
    pr_d_s = map_to_surface(pr_d, view_lut, view_paths,
                            scale=resolution_um/10., fun=np.mean)
    pr_fn = data_dir + "projection_density_top_view_%d.nrrd" % int(resolution_um)
    print "writing " + pr_fn
    nrrd.write(pr_fn, pr_d_s)

t1 = time.time()
total = t1 - t0
print "%0.1f minutes elapsed" % (total/60.)
print "flat vols: " + str(flat_vols)
print "full vols: " + str(full_vols)
print "drop list: " + str(expt_drop_list)

savemat(os.path.join(output_dir, 'volumes.mat'),
        {'flat_vols': flat_vols, 'full_vols': full_vols,
         'drop_list': expt_drop_list,
         'experiment_ids': np.array(experiments['id'])},
        oned_as='column', do_compression=True))


# import matplotlib.pyplot as plt
# plt.ion()

# fig = plt.figure(figsize = (10,10))
# ax = fig.add_subplot(121)
# h = ax.imshow(in_d_s)
# #fig.colorbar(h)

# #fig2 = plt.figure(figsize = (10,10))
# ax2 = fig.add_subplot(122)
# h2 = ax2.imshow(pr_d_s)
# #fig2.colorbar(h2)
