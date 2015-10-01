allen-voxel-network
===================

Tools for working with Allen Institute for Brain Science voxel-scale 
connectivity data.

Generating a voxel model
========================

1. Edit `run_setup.py`. This sets which structures will be
   included, the values of the regularization parameter, etc.
2. `python create_visual_matrices.py`. This will create a hierarchy of 
   directories for model fitting with nested cross-validation.
3. Run the commands in `model_fitting_cmds` (located in the project directory) 
   to perform the model fits.
4. Run `python model_select_and_fit.py`. In the inner cross-validation loop,
   evaluate the errors and perform model selection.
5. Run the commands in `model_fitting_after_selection_cmds`. This will fit the
   selected models.
6. Run `python regional_model_fits_and_errors.py`. This will both evaluate
   the errors of the final models as well as fit regional models and compare
   their errors to the voxel models.