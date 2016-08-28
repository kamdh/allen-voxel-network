import os
from scipy.io import loadmat,savemat,mmread
import numpy as np
from voxlib import *
import matplotlib.pyplot as plt
from scipy.stats import mode
from matplotlib.colors import LinearSegmentedColormap,LogNorm
import h5py

## some parameters to set
inj_site='VISp' # for virtual injections
inj_radius=1 # units of voxels
inj_stride=2
int_axis=1
save_stem='allvis_sdk_free_noshell'
contour_list=[425,533,402]
lambda_str = '1.0000e+05'
output_dir='integrated_gaussian_retro_%s' % lambda_str
do_int_plots=True


print "Making plots for " + save_stem

# fn_connectivity='../connectivities/test_python_0.90_shell2_gaussian_lambda100.mat'
# fn_matrices='../data/visual_output_0.90_shell2_gaussian.mat'
# fout_virt='virt_0.90_gaussian_lambda100.vti'
# fout_real='real_0.90_gaussian_lambda100.vti'
# do_int_plots=True
# int_plot_dir="integrated_gaussian"

# mat=loadmat(fn_connectivity)
# loadmat(fn_matrices,mdict=mat)
# # mat=loadmat('../connectivities/test_python_ipsi.mat')
# # loadmat('../data/visual_output_0.90.mat',mdict=mat)
#         # variable_names=['source_acro', 'source_ids',
#         #                 'target_acro', 'target_ids',
#         #                 'voxel_coords_source',
#         #                 'voxel_coords_target_ipsi',
#         #                 'voxel_coords_target_contra'])
# locals().update(mat)
# X=experiment_source_matrix.T
# Y_ipsi=experiment_target_matrix_ipsi.T
# Y_contra=experiment_target_matrix_contra.T

def h5read(fn):
    return h5py.File(fn,'r')['dataset'][:]

base_dir=os.path.join('../connectivities',save_stem)
fn_matrices=os.path.join(base_dir, save_stem + '.mat')
fig_dir=os.path.join(base_dir, "figures")
int_plot_dir=os.path.join(fig_dir,output_dir)
try:
    os.makedirs(int_plot_dir)
except OSError: 
    pass
fout_virt=os.path.join(fig_dir,'virt_0.90_gaussian_lambda100.vti')
fout_real=os.path.join(fig_dir,'real_0.90_gaussian_lambda100.vti')
mat=loadmat(fn_matrices)
locals().update(mat)
#X=mmread(os.path.join(base_dir, save_stem + '_X.mtx'))
X=h5read(os.path.join(base_dir, save_stem + '_X.h5'))
#Y_ipsi=mmread(os.path.join(base_dir, save_stem + '_Y_ipsi.mtx'))
Y_ipsi=h5read(os.path.join(base_dir, save_stem + '_Y_ipsi.h5'))
#W_ipsi=h5read(os.path.join(base_dir, 'test2.h5'))
#W_ipsi=h5read(os.path.join(base_dir, 'W_ipsi_4.6416e+02.h5'))
#W_ipsi=h5read(os.path.join(base_dir, 'W_ipsi_1.0000e+04.h5'))
W_ipsi=h5read(os.path.join(base_dir, 'W_ipsi_all_%s.h5' % lambda_str)).T
#W_ipsi=h5read(os.path.join(base_dir, 'W_lowrank_res.h5'));
#W_ipsi=h5py.File(os.path.join(base_dir, 'test.h5'),'r')['dataset'][:].T
print "W dims: %d x %d" % (W_ipsi.shape[0], W_ipsi.shape[1])
print "Data all loaded"

## computation
inj_id=source_ids[np.where(source_acro==inj_site)]
coord_vox_map_source=index_lookup_map(voxel_coords_source)
coord_vox_map_target_contra=index_lookup_map(voxel_coords_target_ipsi)
coord_vox_map_target_ipsi=index_lookup_map(voxel_coords_target_contra)
Xvirt,inj_centers=build_injection_vectors(voxel_coords_source,
                              coord_vox_map_source,
                              col_label_list_source,
                              inj_id,
                              inj_radius,
                              inj_stride)
num_virt=Xvirt.shape[1]
if num_virt < 1:
    raise Exception("No virtual injections fit!")
Yvirt_ipsi=np.dot(W_ipsi,Xvirt)

# x=voxel_coords_target_ipsi[:,0]
# y=voxel_coords_target_ipsi[:,1]
# z=voxel_coords_target_ipsi[:,2]

# from matplotlib import pyplot as plt
# plt.plot(Xvirt[:,0:5])
# plt.show()

## align to grid
Xvirt_grid=map_to_regular_grid(Xvirt,voxel_coords_source)
Yvirt_ipsi_grid=map_to_regular_grid(Yvirt_ipsi,voxel_coords_target_ipsi)
Xreal_grid=map_to_regular_grid(X,voxel_coords_source)
Yreal_ipsi_grid=map_to_regular_grid(Y_ipsi,voxel_coords_target_ipsi)
Xvirt_int_grid=np.sum(Xvirt_grid,axis=int_axis)
Yvirt_ipsi_int_grid=np.sum(Yvirt_ipsi_grid,axis=int_axis)


## Save VTKs
print "Saving VTKs"
save_as_vtk_old(fout_virt,Xvirt_grid,Yvirt_ipsi_grid,
                voxel_coords_source,voxel_coords_target_ipsi)
save_as_vtk_old(fout_real,Xreal_grid,Yreal_ipsi_grid,
                voxel_coords_source,voxel_coords_target_ipsi)
print "VTKs saved."

#rearrange=lambda(arr): np.fliplr(arr)
#rearrange=lambda(arr): np.swapaxes(arr,0,1)
rearrange=lambda(arr): arr


## Plot region projections
label_grid=map_to_regular_grid(col_label_list_source,voxel_coords_source).squeeze()
label_grid[label_grid==0]=np.nan
label_mode=mode(label_grid, axis=int_axis)[0].squeeze()
label_mode[label_mode==0]=np.nan
label_unique=np.unique(col_label_list_source)
label_mode=rearrange(label_mode)
# # label 0, 1, ...
# newlab=0
# for lab in label_unique:
#     label_mode[label_mode==lab]=newlab
#     newlab=newlab+1
# label_lookup={x: label_unique[x] for x in range(len(label_unique))}

def centroid_of_region(label_mode,region):
    x,y=np.where(label_mode==region)
    return (np.mean(x),np.mean(y))

def annotate_regions():
    for label in label_unique:
        x,y=centroid_of_region(label_mode,label)
        # region_name=source_acro[source_ids==label_lookup[newlab]][0][0]
        region_name=source_acro[source_ids==label][0][0]
        # print "%s centroid at (%d, %d)" % (region_name,x,y)
        plt.annotate(region_name ,xy=(y,x))

fig,ax=plt.subplots()
ax.imshow(label_mode,
           cmap=plt.get_cmap('Accent'),interpolation='none')
plt.hold(True)
annotate_regions()
plt.tick_params(axis='both', which='both', bottom='off',
                top='off', labelbottom='off', right='off',
                left='off', labelleft='off')
plt.xlabel('center - right', fontsize=24)
plt.ylabel('posterior - anterior', fontsize=24)
plt.savefig("%s/region_names.png" % int_plot_dir)
plt.close()

# label_remapped=label_mode
# contour_list_remapped=np.array(contour_list)
# for i,label in enumerate(label_unique):
#     label_remapped[label_remapped == label] = i
#     contour_list_remapped[contour_list_remapped==label] = i
# from skimage.filters import threshold_adaptive
# from skimage.filters.rank import enhance_contrast
# from skimage.morphology import disk
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
# edges = threshold_adaptive(label_remapped/float(len(label_unique)-1),3)
# edges = enhance_contrast(edges,disk(5))
# from scipy.ndimage.interpolation import zoom
# edges = zoom(edges,4)
# edges = enhance_contrast(edges,disk(11))

# fig,ax=plt.subplots()
# ax.imshow(edges, cmap='gray',interpolation='none')
# contours=find_contours(edges,254.)
# for n, contour in enumerate(contours):
#     contour = subdivide_polygon(contour, degree=4, preserve_ends=False)
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
contours = find_contours(label_mode, 385.5)
def plot_contours():
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, c='gray')


## Plot virtual injections

def plot_integrated(fig,ax,inj,proj_cname,inj_cname,Xgrid,Ygrid):
    cax=ax.imshow(rearrange(Ygrid[:,:,inj]),
                  cmap=plt.get_cmap(proj_cname),
                  clim=(0.0,0.03),
                  #clim=(-0.003,0.003),
                  #norm=LogNorm(vmin=1e-3),
                  interpolation='none')
    cbar = fig.colorbar(cax)
    tmp=rearrange(Xgrid[:,:,inj])
    masked_tmp=np.ma.masked_where(tmp==0.0,tmp)
    ax.imshow(masked_tmp, cmap=plt.get_cmap(inj_cname),
              clim=(0.0,0.3), interpolation='none')
    return 

# easy_plot=lambda(inj): plot_integrated(fig,ax,inj,"Greens","Greens",
#                                        Xvirt_int_grid,Yvirt_ipsi_int_grid)

if do_int_plots:
    for inj in range(num_virt):
        y_inj=inj_centers[1,inj]
        fig,ax=plt.subplots()
        plot_integrated(fig,ax,inj,'Reds','Blues',
                        Xvirt_int_grid,Yvirt_ipsi_int_grid)
        # plot_integrated(fig,ax,inj,'PuOr','Blues',
        #                 Xvirt_int_grid,Yvirt_ipsi_int_grid)
        plt.tick_params(axis='both', which='both', bottom='off',
                        top='off', labelbottom='off', right='off',
                        left='off', labelleft='off')
        plt.xlabel('center - right', fontsize=24)
        plt.ylabel('posterior - anterior', fontsize=24)
        plt.title('depth y = %d' % y_inj)
        plt.hold(True)
        plot_contours()
        annotate_regions()
        # ax.imshow(edges,
        #    cmap=plt.get_cmap('gray_r'),interpolation='none',
        #    alpha=0.1)
        fig_file="%s/int_virt_inj%d.png" % (int_plot_dir,inj)
        plt.savefig(fig_file)
        plt.close()
        print fig_file

# def mymode(x,ignore=np.nan):
#     from scipy.stats import mode
#     x=np.array(x)
#     if np.all(x==ignore) or (np.isnan(ignore) and np.all(np.isnan(x))):
#         return ignore
#     else:
#         if np.isnan(ignore):
#             return mode(x[np.logical_not(np.isnan(x))], axis=None)
#         else:
#             return mode(x[x != ignore],axis=None)
# label_mode=np.apply_along_axis(lambda x: float(mymode(x,0)[0]),
#                                int_axis,label_grid)

        
## plot special injections
cdictred={'red': [(0., 0., 0.),
                  (1., 1., 1.)],
          'green': [(0., 0., 0.),
                    (1., 0., 0.)],
          'blue': [(0., 0., 0.),
                   (1., 0., 0.)]}
cdictgreen={'red': [(0., 0., 0.),
                    (1., 0., 0.)],
            'green': [(0., 0., 0.),
                      (1., 1., 1.)],
            'blue': [(0., 0., 0.),
                     (1., 0., 0.)]}
cdictblue={'red': [(0., 0., 0.),
                   (1., 0., 0.)],
           'green': [(0., 0., 0.),
                     (1., 0., 0.)],
           'blue':  [(0., 0., 0.),
                     (1., 1., 1.)]}
red=LinearSegmentedColormap('myred',cdictred)
plt.register_cmap(cmap=red)
green=LinearSegmentedColormap('mygreen',cdictgreen)
plt.register_cmap(cmap=green)
blue=LinearSegmentedColormap('myblue',cdictblue)
plt.register_cmap(cmap=blue)

#select_injections=[15, 76, 105]
select_injections=[74, 76, 89, 234, 236, 238]
select_colors=["myred", "mygreen", "myblue", "myred", "mygreen","myblue"]
#select_injections=[31, 39, 41]
 
i=0
for inj in select_injections:
    fig,ax=plt.subplots()
    color=select_colors[i]
    plot_integrated(fig,ax,inj,color,color,
                    Xvirt_int_grid,Yvirt_ipsi_int_grid)
    plt.tick_params(axis='both', which='both', bottom='off',
                    top='off', labelbottom='off', right='off',
                    left='off', labelleft='off')
    plt.xlabel('center - right', fontsize=24)
    plt.ylabel('posterior - anterior', fontsize=24)
    fig_file="%s/select_virt_inj%d.png" % (int_plot_dir,i)
    plt.savefig(fig_file)
    plt.close()
    i=i+1

# save_as_csv('test_new.csv',Xvirt_grid,Yvirt_ipsi_grid,
#             voxel_coords_source,voxel_coords_target_ipsi)


## save to matlab
# savemat('test_python_output.mat',
#         {'Yvirt_ipsi_grid':Yvirt_ipsi_grid,
#          'Xvirt_grid':Xvirt_grid})
