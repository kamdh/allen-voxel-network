from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import os
import nrrd
from kam_interface.matrices import generate_voxel_matrices
from kam_interface.utilities import *
from scipy.io import savemat

# setup the run
param_fn='run_setup.py'
with open(param_fn) as f:
    code = compile(f.read(), param_fn, 'exec')
    exec(code)

manifest_file=os.path.join(data_dir,'manifest.json')
mcc = MouseConnectivityCache(manifest_file, resolution=resolution)
ontology = mcc.get_ontology()
sources = ontology[source_acronyms]
targets = ontology[target_acronyms]

if experiments_fn is not None:
    LIMS_id_list=unpickle(experiments_fn)
else:
    LIMS_id_list=None

experiment_dict=generate_voxel_matrices(mcc, sources, targets,
                                        LIMS_id_list=LIMS_id_list,
                                        min_voxels_per_injection=min_vox,
                                        laplacian=laplacian,
                                        verbose=True,
                                        source_shell=source_shell,
                                        source_coverage=source_coverage,
                                        fit_gaussian=fit_gaussian,
                                        cre=cre)
experiment_dict['source_acro']=np.array(source_acronyms,dtype=np.object)
experiment_dict['source_ids']=np.array(sources.id)
experiment_dict['target_acro']=np.array(target_acronyms,dtype=np.object)
experiment_dict['target_ids']=np.array(targets.id)

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
    Omega=experiment_dict['Omega'].T
    h5write(os.path.join(save_dir,save_stem+'_X.h5'),X)
    h5write(os.path.join(save_dir,save_stem+'_Y_ipsi.h5'),Y_ipsi)
    h5write(os.path.join(save_dir,save_stem+'_Y_contra.h5'),Y_contra)
    Lx_fn=absjoin(save_dir,save_stem+'_Lx.mtx')
    Ly_ipsi_fn=absjoin(save_dir,save_stem+'_Ly_ipsi.mtx')
    Ly_contra_fn=absjoin(save_dir,save_stem+'_Ly_contra.mtx')
    mmwrite(Lx_fn,Lx)
    mmwrite(Ly_ipsi_fn,Ly_ipsi)
    mmwrite(Ly_contra_fn,Ly_contra)
    mmwrite(os.path.join(save_dir,save_stem+'_Omega.mtx'),Omega)
    # h5write(os.path.join(save_dir,save_stem+'_W0_ipsi.h5'),
    #         np.zeros((Y_ipsi.shape[0],X.shape[0])))
    # h5write(os.path.join(save_dir,save_stem+'_W0_contra.h5'),
    #         np.zeros((Y_contra.shape[0],X.shape[0])))
    if cross_val_matrices:
        from sklearn import cross_validation
        fid=open(cmdfile,'w')
        n_inj=X.shape[1]
        # Sets up nested outer/inner cross-validation. The inner loop is for
        # model selection (validation), the outer for testing.
        if cross_val=='LOO':
            outer_sets=cross_validation.LeaveOneOut(n_inj)
        else:
            outer_sets=cross_validation.KFold(n_inj,
                                              n_folds=cross_val,
                                              shuffle=True,
                                              random_state=shuffle_seed)
        for i,(train,test) in enumerate(outer_sets):
            X_train=X[:,train]
            X_test=X[:,test]
            Y_train_ipsi=Y_ipsi[:,train]
            Y_test_ipsi=Y_ipsi[:,test]
            Omega_train = Omega[:,train]
            Omega_test = Omega[:,test]
            Y_train_contra=Y_contra[:,train]
            Y_test_contra=Y_contra[:,test]
            # setup some directories
            outer_dir=os.path.join(save_dir,'cval%d'%i)
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
                Omega_train_inner = Omega_train[:,train_inner]
                Omega_test_inner = Omega_train[:,test_inner]
                # filenames
                X_train_fn=absjoin(inner_dir,'X_train.h5')
                X_test_fn=absjoin(inner_dir,'X_test.h5')
                Y_train_ipsi_fn=absjoin(inner_dir,'Y_train_ipsi.h5')
                Y_train_contra_fn=absjoin(inner_dir,'Y_train_contra.h5')
                Y_test_ipsi_fn=absjoin(inner_dir,'Y_test_ipsi.h5')
                Y_test_contra_fn=absjoin(inner_dir,'Y_test_contra.h5')
                Omega_train_inner_fn=absjoin(inner_dir,'Omega_train.mtx')
                Omega_test_inner_fn=absjoin(inner_dir,'Omega_test.mtx')
                # save matrices
                h5write(X_train_fn,X_train_inner)
                h5write(X_test_fn,X_test_inner)
                h5write(Y_train_ipsi_fn,Y_train_ipsi_inner)
                h5write(Y_train_contra_fn,Y_train_contra_inner)
                h5write(Y_test_ipsi_fn,Y_test_ipsi_inner)
                h5write(Y_test_contra_fn,Y_test_contra_inner)
                mmwrite(Omega_train_inner_fn,Omega_train_inner)
                mmwrite(Omega_test_inner_fn,Omega_test_inner)
                # setup commands to run for model selection
                for k,lambda_val in enumerate(lambda_list):
                    output_ipsi=absjoin(inner_dir,"W_ipsi_%1.4e.h5"%lambda_val)
                    output_contra=absjoin(inner_dir,
                                          "W_contra_%1.4e.h5"%lambda_val)
                    lambda_str="%1.4e" % lambda_val
                    cmd_ipsi=' '.join([solver,'--W0_init',
                                       Omega_train_inner_fn,
                                       X_train_fn,Y_train_ipsi_fn,
                                       Lx_fn,Ly_ipsi_fn,
                                       lambda_str,output_ipsi])
                    print cmd_ipsi
                    fid.write(cmd_ipsi+'\n')
                    cmd_contra=' '.join([solver,'--W0_init',
                                         X_train_fn,Y_train_contra_fn,
                                         Lx_fn,Ly_contra_fn,
                                         lambda_str,output_contra])
                    print cmd_contra
                    fid.write(cmd_contra+'\n')
            # We will need these outer cross-validation sets and fit the
            # final model (using optimal lambda found across all inner
            # cross-val runs) to 'train' data. Then, we will test on 'test'.
            X_train_fn=absjoin(outer_dir,'X_train.h5')
            X_test_fn=absjoin(outer_dir,'X_test.h5')
            Y_train_ipsi_fn=absjoin(outer_dir,'Y_train_ipsi.h5')
            Y_train_contra_fn=absjoin(outer_dir,'Y_train_contra.h5')
            Y_test_ipsi_fn=absjoin(outer_dir,'Y_test_ipsi.h5')
            Y_test_contra_fn=absjoin(outer_dir,'Y_test_contra.h5')
            Omega_train_fn=absjoin(outer_dir,'Omega_train.mtx')
            Omega_test_fn=absjoin(outer_dir,'Omega_test.mtx')
            h5write(X_train_fn,X_train)
            h5write(X_test_fn,X_test)
            h5write(Y_train_ipsi_fn,Y_train_ipsi)
            h5write(Y_train_contra_fn,Y_train_contra)
            h5write(Y_test_ipsi_fn,Y_test_ipsi)
            h5write(Y_test_contra_fn,Y_test_contra)
            mmwrite(Omega_train_fn,Omega_train)
            mmwrite(Omega_test_fn,Omega_test)
        fid.close()
