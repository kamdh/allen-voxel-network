from friday_harbor.structure import Ontology
from kam_interface.utilities import *
from kam_interface.matrices import generate_voxel_matrices
import h5py
import os
from IPython import embed
from scipy.io import savemat
import numpy as np

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

ontology=Ontology(data_dir=data_dir)
sources=acro_list_to_id_list(ontology,source_acronyms)
targets=acro_list_to_id_list(ontology,target_acronyms)
if experiments_fn is not None:
    LIMS_id_list=unpickle(experiments_fn)
else:
    LIMS_id_list=None
experiment_dict=generate_voxel_matrices(data_dir, sources, targets,
                                        LIMS_id_list=LIMS_id_list,
                                        min_voxels_per_injection=min_vox,
                                        laplacian=True,
                                        verbose=True,
                                        source_shell=source_shell,
                                        source_coverage=source_coverage,
                                        fit_gaussian=fit_gaussian)
experiment_dict['source_acro']=np.array(source_acronyms,dtype=np.object)
experiment_dict['source_ids']=sources
experiment_dict['target_acro']=np.array(target_acronyms,dtype=np.object)
experiment_dict['target_ids']=targets
# f=h5py.File(os.path.join(save_file_name), 'w')
# write_dictionary_to_group(f, experiment_dict)
# f.close()
try:
    os.mkdir(save_dir)
except OSError:
    pass
save_file_name=os.path.join(save_dir,save_stem + '.mat')
savemat(save_file_name,experiment_dict,oned_as='column',do_compression=True)
if save_mtx:
    # only save X, Y, Lx, Ly
    from scipy.io import mmwrite
    X=experiment_dict['experiment_source_matrix'].T
    Y_ipsi=experiment_dict['experiment_target_matrix_ipsi'].T
    Y_contra=experiment_dict['experiment_target_matrix_contra'].T
    Lx=experiment_dict['Lx'].T
    Ly_ipsi=experiment_dict['Ly_ipsi'].T
    Ly_contra=experiment_dict['Ly_contra'].T
    mmwrite(os.path.join(save_dir,save_stem+'_X.mtx'),X)
    mmwrite(os.path.join(save_dir,save_stem+'_Y_ipsi.mtx'),Y_ipsi)
    mmwrite(os.path.join(save_dir,save_stem+'_Y_contra.mtx'),Y_contra)
    mmwrite(os.path.join(save_dir,save_stem+'_Lx.mtx'),Lx)
    mmwrite(os.path.join(save_dir,save_stem+'_Ly_ipsi.mtx'),Ly_ipsi)
    mmwrite(os.path.join(save_dir,save_stem+'_Ly_contra.mtx'),Ly_contra)
    if cross_val_matrices:
        from sklearn import cross_validation
        fid=open(cmdfile,'w')
        n_inj=X.shape[1]
        # Sets up nested outer/inner cross-validation. The inner loop is for
        # model selection (validation), the outer for testing.
        if cross_val=='LOO':
            outer_sets=cross_validation.LeaveOneOut(n_inj)
        else:
            outer_sets=cross_validation.KFold(n_inj,n_folds=cross_val)
        for i,(train,test) in enumerate(outer_sets):
            X_train=X[:,train]
            X_test=X[:,test]
            Y_train_ipsi=Y_ipsi[:,train]
            Y_test_ipsi=Y_ipsi[:,test]
            Y_train_contra=Y_contra[:,train]
            Y_test_contra=Y_contra[:,test]
            # setup some directories
            outer_dir=os.path.join(save_dir,'cval%d'%i)
            Lx_fn=absjoin(save_dir,save_stem+'_Lx.mtx')
            Ly_ipsi_fn=absjoin(save_dir,save_stem+'_Ly_ipsi.mtx')
            Ly_contra_fn=absjoin(save_dir,save_stem+'_Ly_contra.mtx')
            try:
                os.mkdir(outer_dir)
            except OSError:
                pass
            if cross_val=='LOO':
                inner_sets=cross_validation.LeaveOneOut(len(train))
            else:
                inner_sets=cross_validation.KFold(len(train),n_folds=cross_val)
            for j,(train_inner,test_inner) in enumerate(inner_sets):
                inner_dir=os.path.join(outer_dir,'cval%d'%j)
                try:
                    os.mkdir(inner_dir)
                except OSError:
                    pass
                # pull all inner training/testing sets from outer training sets
                X_train_inner=X_train[:,train_inner]
                X_test_inner=X_train[:,test_inner]
                Y_train_ipsi_inner=Y_train_ipsi[:,train_inner]
                Y_test_ipsi_inner=Y_train_ipsi[:,test_inner]
                Y_train_contra_inner=Y_train_contra[:,train_inner]
                Y_test_contra_inner=Y_train_contra[:,test_inner]
                # filenames
                X_train_fn=absjoin(inner_dir,'X_train.mtx')
                X_test_fn=absjoin(inner_dir,'X_test.mtx')
                Y_train_ipsi_fn=absjoin(inner_dir,'Y_train_ipsi.mtx')
                Y_train_contra_fn=absjoin(inner_dir,'Y_train_contra.mtx')
                Y_test_ipsi_fn=absjoin(inner_dir,'Y_test_ipsi.mtx')
                Y_test_contra_fn=absjoin(inner_dir,'Y_test_contra.mtx')
                # save matrices
                mmwrite(X_train_fn,X_train_inner)
                mmwrite(X_test_fn,X_test_inner)
                mmwrite(Y_train_ipsi_fn,Y_train_ipsi_inner)
                mmwrite(Y_train_contra_fn,Y_train_contra_inner)
                mmwrite(Y_test_ipsi_fn,Y_test_ipsi_inner)
                mmwrite(Y_test_contra_fn,Y_test_contra_inner)
                # setup commands to run for model selection
                for k,lambda_val in enumerate(lambda_list):
                    output_ipsi=absjoin(inner_dir,"W_ipsi_%1.4e.mtx"%lambda_val)
                    output_contra=absjoin(inner_dir,
                                          "W_contra_%1.4e.mtx"%lambda_val)
                    lambda_str="%1.4e" % lambda_val
                    cmd=' '.join([solver,X_train_fn,Y_train_ipsi_fn,
                                  Lx_fn,Ly_ipsi_fn,
                                  lambda_str,output_ipsi])
                    print cmd
                    fid.write(cmd+'\n')
                    cmd=' '.join([solver,X_train_fn,Y_train_contra_fn,
                                  Lx_fn,Ly_contra_fn,
                                  lambda_str,output_contra])
                    print cmd
                    fid.write(cmd+'\n')
            # We will need these outer cross-validation sets and fit the
            # final model (using optimal lambda found across all inner
            # cross-val runs) to 'train' data. Then, we will test on 'test'.
            X_train_fn=absjoin(outer_dir,'X_train.mtx')
            X_test_fn=absjoin(outer_dir,'X_test.mtx')
            Y_train_ipsi_fn=absjoin(outer_dir,'Y_train_ipsi.mtx')
            Y_train_contra_fn=absjoin(outer_dir,'Y_train_contra.mtx')
            Y_test_ipsi_fn=absjoin(outer_dir,'Y_test_ipsi.mtx')
            Y_test_contra_fn=absjoin(outer_dir,'Y_test_contra.mtx')
            mmwrite(X_train_fn,X_train)
            mmwrite(X_test_fn,X_test)
            mmwrite(Y_train_ipsi_fn,Y_train_ipsi)
            mmwrite(Y_train_contra_fn,Y_train_contra)
            mmwrite(Y_test_ipsi_fn,Y_test_ipsi)
            mmwrite(Y_test_contra_fn,Y_test_contra)
        fid.close()
