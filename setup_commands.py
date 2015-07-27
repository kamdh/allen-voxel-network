import os
import glob
import numpy as np

# setup the run
save_stem='visual_output_0.80_shell_1'
save_dir=os.path.join('connectivities',save_stem)
solver=os.path.abspath('smoothness_c/solve')
lambdas=10.**np.arange(-2,4)
# build commands
Lx_fn=os.path.abspath(os.path.join(save_dir,save_stem+'_Lx.mtx'))
Ly_ipsi_fn=os.path.abspath(os.path.join(save_dir,save_stem+'_Ly_ipsi.mtx'))
Ly_contra_fn=os.path.abspath(os.path.join(save_dir,save_stem+'_Ly_contra.mtx'))
for cval_dir in glob.iglob(save_dir+'/cval*'):
    X_train_fn=os.path.abspath(os.path.join(cval_dir,'X_train.mtx'))
    Y_train_ipsi_fn=os.path.abspath(os.path.join(cval_dir,'Y_train_ipsi.mtx'))
    Y_train_contra_fn=os.path.abspath(
        os.path.join(cval_dir,'Y_train_contra.mtx'))
    for l in lambdas:
        output_ipsi=os.path.abspath(
            os.path.join(cval_dir,"W_ipsi_%f.mtx"%l))
        output_contra=os.path.abspath(
            os.path.join(cval_dir,"W_contra_%f.mtx"%l))
        cmd=' '.join([solver,X_train_fn,Y_train_ipsi_fn,Lx_fn,Ly_ipsi_fn,
                      str(l),output_ipsi])
        print cmd
        cmd=' '.join([solver,X_train_fn,Y_train_contra_fn,Lx_fn,Ly_contra_fn,
                      str(l),output_contra])
        print cmd
        
