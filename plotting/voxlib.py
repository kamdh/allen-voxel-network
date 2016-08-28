import numpy as np
from IPython import embed

def coords2str(x):
    return " ".join(("%d" % n for n in x))

def str2coords(s):
    return np.fromstring(s, dtype=int, sep=" ")

def index_lookup_map(x):
    return {coords2str(x): i for i,x in enumerate(x)}

def bounding_box(voxels):
    mins=np.min(voxels,axis=0)
    maxs=np.max(voxels,axis=0)
    return (mins,maxs)

def gaussian_injection(center,radius):
    from scipy.ndimage.filters import gaussian_filter
    n=radius*2+1
    nhalf=radius
    data=np.zeros([n,n,n])
    data[nhalf,nhalf,nhalf]=1.0
    filt=gaussian_filter(data,radius/3.0,mode='constant',cval=0.0,truncate=3.0)
    filt=filt/np.sum(filt)
    vox_filt={}
    for index, v in np.ndenumerate(filt):
        voxel=np.array(center+index-[nhalf,nhalf,nhalf],dtype=int)
        key=coords2str(voxel)
        vox_filt[key]=v
    return vox_filt

def point_injection(center):
    key=coords2str(center)
    vox_filt={}
    vox_filt[key]=1.0
    return vox_filt

def build_injection_vectors(voxel_coords,coord_vox_map,
                            region_ids,inj_id,radius,stride):
    '''
    Tiles the injection site with virtual injections of a given radius.

    Generates the virtual injections.

    Parameters
    ----------
    voxel_coords : ndarray (N x 3)
        Coordinates x,y,z of each voxel
    coord_vox_map : dict
        Keys are coords2str([x,y,z]), values give index of that voxel
    region_ids : ndarray (N x 1)
        Regions assigned to each voxel
    inj_id : int
        Id of region to target
    radius : int
        Radius of each injection (units: voxels)
    stride : int
        How many voxels to stride when placing centers

    Returns
    -------
    Xvirt : ndarray (N x num_inj)
        Array representing the virtual injections
    inj_center : ndarray (3 x num_inj)
        Centers of the virtual injections
    '''

    index_in_source=(region_ids==inj_id)
    min_bnd,max_bnd=bounding_box(voxel_coords[np.where(index_in_source)[0],])
    N=voxel_coords.shape[0]
    num_est=np.round(np.prod(max_bnd-min_bnd)/radius**3.0)
    num=0
    Xvirt=np.zeros((N,num_est))
    inj_center=np.zeros((3,num_est))
    # y changes slowest since it is approximately depth
    for y in np.arange(min_bnd[1],max_bnd[1],stride,dtype=np.int):
        for z in np.arange(min_bnd[2],max_bnd[2],stride,dtype=np.int):
            for x in np.arange(min_bnd[0],max_bnd[0],stride,dtype=np.int):
                this_center=np.array([x,y,z],dtype=int)
                #this_inj=gaussian_injection(this_center,radius)
                this_inj=point_injection(this_center)
                these_vox=this_inj.keys()
                keep_this=True
                for v in these_vox:
                    try:
                        index=coord_vox_map[v]
                        if not index_in_source[index]:
                            keep_this=False
                            break
                    except KeyError:
                        keep_this=False
                        break
                if keep_this:
                    for v in these_vox:
                        index=coord_vox_map[v]
                        Xvirt[index,num]=this_inj[v]
                        inj_center[:,num]=this_center
                    num+=1
    Xvirt=Xvirt[:,0:num]
    inj_center=inj_center[:,0:num]
    return Xvirt,inj_center

def map_to_regular_grid(x,voxel_coords):
    '''
    Map a voxel vector into a regular grid in the bounding box.

    Parameters
    ----------
    x : ndarray (N x 1)
    voxel_coords : ndarray (N x 3)

    Returns
    -------
    Y 
    '''
    assert voxel_coords.shape[0] == x.shape[0], \
      "x and voxel_coords should have same first dimension"
    assert voxel_coords.shape[1] == 3,\
      "voxel_coords should be (N x 3)"
    min_box,max_box=bounding_box(voxel_coords)
    base_shape=shape_regular_grid(voxel_coords)
    if x.ndim==2:
        dims=list(base_shape)
        num_virt=x.shape[1]
        dims.append(num_virt)
        Y=np.zeros(dims)
        for inj in range(num_virt):
            Y[:,:,:,inj]=map_to_regular_grid(x[:,inj],voxel_coords)
    elif x.ndim==1:
        Y=np.zeros(base_shape)
        for index,value in enumerate(x):
            new_index=voxel_coords[index]-min_box
            Y[new_index[0],new_index[1],new_index[2]]=value
    else:
        raise Exception('can only map 1 or 2 dimensional arrays to a grid')
    return(Y)
        
def shape_regular_grid(voxel_coords):
    min_box,max_box=bounding_box(voxel_coords)
    dims=max_box-min_box+1
    return dims

def save_as_csv(fn,Xvirt_grid,Yvirt_grid,
                voxel_coords_source,
                voxel_coords_target):
    '''
    Save 4d arrays in CSV

    Parameters
    ----------
    fn : string
      Filename
    Xvirt_grid : ndarray
      4d array of injections aligned to grid
    Yvirt_grid : ndarray
      4d array of projections aligned to grid
    voxel_coords_source : ndarray
      num_voxel x 3 array of x,y,z coordinates
    voxel_coords_target : ndarray
      num_voxel x 3 array of x,y,z coordinates
    '''
    assert np.all(voxel_coords_source==voxel_coords_target),\
      "source and target voxel coordinates should be equal"
    assert Xvirt_grid.shape == Yvirt_grid.shape,\
      "Xvirt_grid and Yvirt_grid should have same shape"
    if Xvirt_grid.ndim == 4:
        grid_shape=Xvirt_grid[:,:,:,0].shape
        num_rows=np.prod(grid_shape)
        num_virt=Xvirt_grid.shape[3]
    else:
        raise Exception("need 4d arrays for Xvirt_grid, Yvirt_grid")
    num_cols=3+2*num_virt
    csv_data=np.zeros((np.prod(Xvirt_grid[:,:,:,0].shape),num_cols))
    base_pt=np.min(voxel_coords_source,axis=0)
    row=0
    for index in np.ndindex(grid_shape):
        x,y,z=base_pt+np.array(index)
        row_data=np.hstack((x,y,z,Xvirt_grid[index],Yvirt_grid[index]))
        csv_data[row,:]=row_data
        #print x,y,z
        row+=1
    header="X coord,Y coord,Z coord,"+\
      ','.join(["X%04d" % n for n in range(num_virt)])+','+\
      ','.join(["Y%04d" % n for n in range(num_virt)])
    np.savetxt(fn,csv_data,delimiter=',',header=header,comments='')

def save_as_vtk(fn,X_grid,
                voxel_coords):
    '''
    Save 4d arrays in VTK

    Parameters
    ----------
    fn : string
      Filename
    X_grid : ndarray
      4d array of injections aligned to grid
    voxel_coords : ndarray
      num_voxel x 3 array of x,y,z coordinates
    '''
    from tvtk.api import tvtk, write_data
    if X_grid.ndim == 4:
        grid_shape=X_grid[:,:,:,0].shape
        num_rows=np.prod(grid_shape)
        num_virt=X_grid.shape[3]
    else:
        raise Exception("need 4d arrays for X_grid")
    VTK=tvtk.ImageData(spacing=(1,1,1),
                       origin=np.min(voxel_coords,axis=0),
                       dimensions=grid_shape)
    VTK.point_data.scalars= np.arange(0,num_rows)
    VTK.point_data.scalars.name='voxel number'
    for n in range(num_virt):
        a=VTK.point_data.add_array(X_grid[:,:,:,n].ravel(order='F'))
        VTK.point_data.get_array(a).name="X%04d" % n
        VTK.point_data.update()
        del a
    write_data(VTK,fn)

def save_as_vtk_old(fn,Xvirt_grid,Yvirt_grid,
                voxel_coords_source,
                voxel_coords_target):
    '''
    Save 4d arrays in VTK

    Parameters
    ----------
    fn : string
      Filename
    Xvirt_grid : ndarray
      4d array of injections aligned to grid
    Yvirt_grid : ndarray
      4d array of projections aligned to grid
    voxel_coords_source : ndarray
      num_voxel x 3 array of x,y,z coordinates
    voxel_coords_target : ndarray
      num_voxel x 3 array of x,y,z coordinates
    '''
    from tvtk.api import tvtk, write_data
    assert np.all(voxel_coords_source==voxel_coords_target),\
      "source and target voxel coordinates should be equal"
    assert Xvirt_grid.shape == Yvirt_grid.shape,\
      "Xvirt_grid and Yvirt_grid should have same shape"
    if Xvirt_grid.ndim == 4:
        grid_shape=Xvirt_grid[:,:,:,0].shape
        num_rows=np.prod(grid_shape)
        num_virt=Xvirt_grid.shape[3]
    else:
        raise Exception("need 4d arrays for Xvirt_grid, Yvirt_grid")
    VTK=tvtk.ImageData(spacing=(1,1,1),
                       origin=np.min(voxel_coords_source,axis=0),
                       dimensions=grid_shape)
    VTK.point_data.scalars= np.arange(0,num_rows)
    VTK.point_data.scalars.name='voxel number'
    #arr_num=1
    for n in range(num_virt):
        a=VTK.point_data.add_array(Xvirt_grid[:,:,:,n].ravel(order='F'))
        VTK.point_data.get_array(a).name="%04d_inj" % n
        VTK.point_data.update()
        del a
        #arr_num+=1
        a=VTK.point_data.add_array(Yvirt_grid[:,:,:,n].ravel(order='F'))
        VTK.point_data.get_array(a).name="%04d_proj" % n
        VTK.point_data.update()
        del a
        #arr_num+=1
    write_data(VTK,fn)
