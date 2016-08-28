import os
import numpy as np

save_stem='allvis_sdk_free_noshell'
data_dir='../../data/sdk_new_100'
resolution=100
cre=False
source_acronyms=['VISal','VISam','VISl','VISp','VISpl','VISpm']
lambda_list=np.logspace(1,10,10)
scale_lambda=True
min_vox=10
# save_file_name='visual_output.hdf5'
source_coverage=0.90
#source_shell=1
source_shell=None
save_dir=os.path.join('../../data/connectivities',save_stem)
experiments_fn=None
target_acronyms=source_acronyms
solver=os.path.abspath('../smoothness_c/solve')
cmdfile=os.path.join(save_dir,'model_fitting_cmds')
selected_fit_cmds=os.path.join(save_dir,'model_fitting_after_selection_cmds')
save_mtx=True
cross_val_matrices=True
cross_val=5
fit_gaussian=False
select_one_lambda=False
if select_one_lambda:
    lambda_fn='lambda_opt'
else:
    lambda_fn='lambda_ipsi_contra_opt'
laplacian='free'
shuffle_seed=666
