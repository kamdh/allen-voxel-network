import numpy as np

def map_to_surface(im, lut, paths, scale = 1, fun = np.max, set_nan = True):
    '''
    maps a gridded voxel image onto the cortical surface
    '''
    scale = float(scale)
    old_dims = im.shape
    new_dims = (1320, 800, 1140) # hard-coded
    for i, dim in enumerate(new_dims):
        assert np.floor(old_dims[i] * scale).astype(int) == dim, \
            "dimension mismatch"
    # deal with scaling through re-indexing
    def remap_coord(c):
        #new_dims = tuple(np.round(np.array(old_dims) * scale).astype(int))
        (I,J,K) = np.unravel_index(c, new_dims)
        I = np.round(I / scale).astype(int)
        J = np.round(J / scale).astype(int)
        K = np.round(K / scale).astype(int)
        return np.ravel_multi_index((I,J,K), old_dims)
    # calculate output array
    output_pd = np.zeros(lut.shape, dtype=im.dtype)    
    # all pixels in surface view with a stream line
    ind = np.where(lut > -1)
    ind = zip(ind[0], ind[1])
    for curr_ind in ind:
        curr_path_id = lut[curr_ind]
        curr_path = paths[curr_path_id, :]
        curr_path = curr_path[curr_path != 0] # need to ignore zeros
        if scale != 1:
            curr_path_rescale = remap_coord(curr_path)
        else:
            curr_path_rescale = curr_path
        #(I,J,K) = remap_coord(curr_path, old_dims, scale)
        # image along path
        #curr_pd_line = im[I,J,K]
        curr_pd_line = im.flat[curr_path_rescale]
        value = fun(curr_pd_line)
        output_pd[curr_ind] = value
        #if np.any(np.nonzero(curr_pd_line)):
        #    print curr_ind
        #    print curr_path
        #    #print (I,J,K)
        #    print curr_pd_line
        #    print value
        #    break
    if set_nan:
        output_pd[lut == -1] = np.nan
    return output_pd

def downsample(arr, stride):
    '''
    Downsample a 2d array
    '''
    assert type(stride) is int, "stride should be integer"
    return arr[0::stride, 0::stride]

def laplacian_2d(mask):
    '''
    Generate the laplacian matrix for a given region's voxels. This is the 
    graph laplacian of the neighborhood graph.

    Parameters
    ----------
      mask

    Returns
    -------
      L: num voxel x num voxel laplacian csc_matrix in same order as voxels
        in the region mask
    '''
    def possible_neighbors(vox):
        neighbors = np.tile(vox, (4,1))
        neighbors += [[1,0],
                      [0,1],
                      [-1,0],
                      [0,-1]]
        return neighbors

    import scipy.sparse as sp
    voxels = np.array(mask).T
    num_vox = len(mask[0])
    # num_vox=voxels.shape[0]
    vox_lookup = {tuple(vox): idx for idx, vox in enumerate(voxels)}
    L = sp.lil_matrix((num_vox, num_vox))
    for idx, vox in enumerate(voxels):
        candidates = possible_neighbors(vox)
        deg = 0
        for nei in candidates:
            try:
                idx_nei = vox_lookup[ tuple(nei) ]
                deg += 1
                L[idx, idx_nei] = 1
            except KeyError:
                continue
        L[idx, idx] = -deg
    return L.tocsc()
