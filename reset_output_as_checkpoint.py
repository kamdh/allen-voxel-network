#!/usr/bin/env python
import os
import glob
from voxnet.utilities import absjoin

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

for o_idx,outer_dir in enumerate(glob.glob(save_dir+'/cval*')):
    print 'Entering outer cross-val set ' + str(o_idx)
    inner_dirs=glob.glob(outer_dir+'/cval*')
    for i,inner_dir in enumerate(inner_dirs):
        print '  Processing inner cross-val set ' + str(i)
        for j,lambda_val in enumerate(lambda_list):
            W_ipsi_fn=absjoin(inner_dir,"W_ipsi_%1.4e.mtx" % lambda_val)
            W_contra_fn=absjoin(inner_dir,"W_contra_%1.4e.mtx" % lambda_val)
            for fn in [W_ipsi_fn, W_contra_fn]:
                if os.path.exists(fn):
                    fn_new=fn + ".CHECKPT"
                    os.rename(fn,fn_new)
                    #print "mv " + fn + " " + fn_new
                else:
                    print fn + " does not exist"
