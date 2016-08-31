import numpy as np

def mask_union(*masks):
    ''' Find the union of all of the nonzero voxels given an input list of masks. '''
    masks = [ m for m in masks if len(m[0]) > 0 ]

    if len(masks) == 1:
        return masks[0]

    mask_bounds = [ [ np.max(coords) for coords in m ] for m in masks ]

    if len(mask_bounds) == 0:
        return ((np.array([]), np.array([]), np.array([])))

    max_bounds = np.max(mask_bounds, 0)

    bg = np.zeros(max_bounds+1, dtype=np.uint8)

    for mask in masks:
        bg[mask] = 1

    return np.where(bg > 0)

def mask_intersection(*input_masks):
    ''' Find the intersection of all of the nonzero voxels given an input list of masks. '''
    masks = [ m for m in input_masks if len(m[0]) > 0 ]

    # if there are zero-length masks, the intersection is necessarily empty.
    if len(masks) < len(input_masks):
        return (np.array([]), np.array([]), np.array([]))

    # find the smallest box that will contain all of the mask coordinates and
    # initialize an array with those dimensions
    mask_bounds = [ [ np.max(coords) for coords in m ] for m in masks ]
    max_bounds = np.max(mask_bounds, 0)

    # intialize some images with that size
    images = [ np.zeros(max_bounds+1, dtype=np.bool) for m in masks ]

    # initialize them
    for i, mask in enumerate(masks):
        images[i][mask] = True

    # logical_and them all together
    intersection_image = np.logical_and.reduce(images)

    return np.where(intersection_image)

def mask_difference(A, B):
    ''' Find the difference between this mask and another. '''
    if len(A[0]) == 0:
        xx, yy, zz = np.array([]), np.array([]), np.array([])
    else:
        A_set = set(zip(*A))
        other_set = set(zip(*B))

        new_set = set()
        for x in A_set:
            if not x in other_set:
                new_set.add(x)

        if len(new_set) == 0:
            xx, yy, zz = np.array([]), np.array([]), np.array([]) 
        else:
            xx, yy, zz = zip(*list(new_set))


    return (np.array(xx), np.array(yy), np.array(zz))

def shell_mask(mask,radius=1):
    def unique_rows(a):
        '''
        http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
        '''
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    neighborhood_size=27
    assert isinstance(radius, int), "radius should be type int"
    voxels=np.array(mask,dtype=int).T
    candidates=np.zeros((voxels.shape[0]*neighborhood_size,3),dtype=int)
    for idx,vox in enumerate(voxels):
        neighbors=possible_neighbors(vox,size=neighborhood_size-1)
        candidates[neighborhood_size*idx:neighborhood_size*(idx+1),:]=\
          np.vstack((neighbors,vox.reshape((1,3))))
    shell_voxels=unique_rows(candidates)
    #new_mask=np.array(shell_voxels) #[:,0],shell_voxels[:,1],shell_voxels[:,2])
    new_mask=tuple([col for col in shell_voxels.T])
    if radius==1:
        return new_mask
    else:
        return shell_mask(new_mask,radius=radius-1)

def possible_neighbors(vox,size=6):
    '''
    Parameters
    ----------
      vox: 1x3 numpy array

    Returns
    -------
      neighbors: size x3 numpy array, neighborhood voxel coordinates
      size : int
        6 or 26, size of neighborhood to return
    '''
    if size==6:
        neighbors=np.tile(vox,(6,1))
        neighbors+=[[1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [-1,0,0],
                    [0,-1,0],
                    [0,0,-1]]
    elif size==26:
        neighbors=np.tile(vox,(26,1))
        idx=0
        for dx in range(-1,1):
            for dy in range(-1,1):
                for dz in range(-1,1):
                    if not(dx==0 and dy==0 and dz==0):
                        neighbors[idx,:]+=[dx,dy,dz]
    return neighbors
