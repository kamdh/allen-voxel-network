from voxnet.mask import *
import numpy as np

grid = np.arange(27).reshape((3,3,3))

mask1 = ([0,0,1,2],[0,1,0,1],[0,2,1,1])
mask2 = ([0,0,2,2],[0,1,0,1],[0,2,1,0])
