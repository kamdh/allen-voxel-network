# Kameron Decker Harris
# modified from code by Nicholas Cain

def pickle(data, file_name):
    import pickle as pkl    
    f=open(file_name, "wb")
    pkl.dump(data, f)
    f.close()

def unpickle(file_name):
    import pickle as pkl
    f=open(file_name, "rb")
    data=pkl.load(f)
    f.close()
    return data

def write_dictionary_to_group(group, dictionary, create_name = None):
    if create_name != None:
        group = group.create_group(create_name)
    for key, val in dictionary.items():
        group[str(key)] = val
    return

def read_dictionary_from_group(group):
    dictionary = {}
    for name in group:
        dictionary[str(name)] = group[name].value
    return dictionary

def h5write(fn,mat):
    import h5py
    with h5py.File(fn, 'w') as f:
        f.create_dataset('dataset', data=mat)
        f.close()

def h5read(fn):
    import h5py
    with h5py.File(fn, 'r') as f:
        data=f['dataset'][()]
        f.close()
        return data

def absjoin(path,*paths):
    import os
    return os.path.abspath(os.path.join(path,*paths))

def integrate_in_mask(data, query_mask):
    '''
    Integrate a data within a certain query mask.
    Deals with error codes appropriately.

    Parameters
    ----------
    data : 3-array of values
    query_mask : mask over which to integrate (xs, ys, zs)

    Returns
    -------
    sum : integral of data within query_mask
    '''
    def safe_sum(values):
        sum = 0.0
        for val in values:
            if val == -1:
                val=0.0
            elif val == -2:
                warn("data error -2, missing tile in LIMS experiment %d" \
                     % curr_LIMS_id)
                val = 0.0
            elif val == -3:
                warn("data error -3, no data in LIMS experiment %d" \
                     % curr_LIMS_id)
                val = 0.0
            sum += val
        return sum
    if mask_len(query_mask) > 0:
        curr_sum = safe_sum(data[query_mask])
    else:
        curr_sum = 0.0
    return curr_sum

def data_in_mask_and_region(data, query_mask, region_mask):
    '''
    Returns the data within a given mask and region.
    Maps the voxel data from a 3d array into a vector.
    Deals with error codes appropriately.

    Parameters
    ----------
    data : 3-array of values
    query_mask : mask over which to integrate (xs, ys, zs)

    '''
    import numpy as np
    nvox = mask_len(region_mask)
    if mask_len(query_mask)>0:
        data_in_mask = data[region_mask]
        irrelevant_indices = np.ones((nvox,))
        list_relevant = zip(*query_mask)
        list_region = zip(*region_mask)
        for ii,el in enumerate(list_region):
            if el in list_relevant:
                irrelevant_indices[ii] = 0
        data_in_mask[irrelevant_indices==1] = 0.0
        errmask = (data_in_mask == -1)
        if np.count_nonzero(errmask) > 0:
            data_in_mask[errmask] = 0.0
        errmask = (data_in_mask == -2)
        if np.count_nonzero(errmask) > 0: 
            warn("data error -2, missing tile in LIMS experiment %d" \
                  % curr_LIMS_id)
            data_in_mask[errmask] = 0.0
        errmask = (data_in_mask == -3)
        if np.count_nonzero(errmask) > 0: 
            warn("data error -3, no data in LIMS experiment %d" \
                 % curr_LIMS_id)
            data_in_mask[errmask] = 0.0
    else:
        data_in_mask = np.zeros((nvox,))
    return data_in_mask


def get_structure_mask_nz(mcc, structure_id, ipsi=False, contra=False):
    '''
    Returns the structure mask associated with a given structure id,
    in a sparse format.

    Parameters
    ----------
    structure_id : int
      Id of query structure

    ipsi : bool (default = False)
      Whether to return ipsilateral structure coordinates

    contra : bool (default = False)
      Whether to return contralateral structure coordinates

    If both ipsi == contra == True or False, then return both (default).

    Returns
    -------
    mask_nz : tuple
      Tuple of (x,y,z) coordinates belonging to structure_id
    '''
    import numpy as np
    if (ipsi and contra) or (not ipsi and not contra):
        return np.where(mcc.get_structure_mask(structure_id)[0])
    elif ipsi and not contra:
        mask = mcc.get_structure_mask(structure_id)
        midline_coord = mask[1]['sizes'][2]/2
        mask_nz = np.where(mask[0])
        idx = (mask_nz[2] >= midline_coord)
        return (mask_nz[0][idx], mask_nz[1][idx], mask_nz[2][idx])
    elif not ipsi and contra:
        mask = mcc.get_structure_mask(structure_id)
        midline_coord = mask[1]['sizes'][2]/2
        mask_nz = np.where(mask[0])
        idx = (mask_nz[2] < midline_coord)
        return (mask_nz[0][idx], mask_nz[1][idx], mask_nz[2][idx])

def get_injection_mask_nz(mcc, expt_id, threshold=0.,
                          valid=True, shell=None):
    '''
    Custom method to get injection masks. Can ensure that the data are
    valid, as well as build a shell around injection site.

    Parameters
    ----------
    expt_id : int
      Experiment id

    threshold : float (default = 0)
      Threshold for injection fraction to be included

    valid : bool (default = True)
      Only return the valid data region

    shell : int (default = None)
      Expand the mask by a shell of this many voxels

    Returns
    -------
    mask_nz : tuple
      Tuple of (x,y,z) coordinates belonging to injection site
    '''
    import numpy as np
    from mask import shell_mask
    inj_frac = mcc.get_injection_fraction(expt_id)
    if valid:
        data_mask = mcc.get_data_mask(expt_id)
        mask_nz = np.where(
            np.logical_and(inj_frac[0] > threshold,
                           data_mask[0])
            )
    else:
        mask_nz = np.where(inj_frac[0] > threshold)
    if shell is not None:
        mask_nz = shell_mask(mask_nz, radius=shell)
    return mask_nz

def mask_len(mask):
    return len(mask[0])
