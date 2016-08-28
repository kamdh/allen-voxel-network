conn_dir='../../data/connectivities/'

A=load([conn_dir 'allvis_sdk_test/allvis_sdk_test.mat']);
B=load([conn_dir 'allvis_test/allvis_test.mat']);

figure
subplot(2,1,1)
imagesc(A.experiment_source_matrix)     
colormap('gray')
colorbar
subplot(2,1,2)
imagesc(B.experiment_source_matrix)
colorbar

figure
subplot(2,1,1)
imagesc(A.experiment_target_matrix_ipsi)
colormap('gray')
colorbar
subplot(2,1,2)
imagesc(B.experiment_target_matrix_ipsi)
colorbar

figure
subplot(2,1,1)
imagesc(A.experiment_target_matrix_contra)
colormap('gray')
colorbar
subplot(2,1,2)
imagesc(B.experiment_target_matrix_contra)
colorbar
