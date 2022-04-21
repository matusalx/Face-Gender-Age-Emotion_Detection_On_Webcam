----------------------------------------------------------------
For training script to run the following terminal (windows) command is needed:

training_all_models.py -utkface_data_dir "full_path_to_utkface" -ckplus_data_dir "full_path_to_ckplus" -n_epochs 1

-utkface_data_dir -- the full path to utkface_data folder
-ckplus_data_dir  -- the full path to ckplus_data folder
-n_epochs --number of epochs for each model ( for testing n_epochs=1 )



----------------------------------------------------------------

For testing the models in web_cam , run: 
test.py -camera_number 0 
( if using second camera use -camera_number 1 )

----------------------------------------------------------------


_________________________________________________________________

for face detection  --  facenet_pytorch.MTCNN ( pretrained )
for gender model  --  resnet18  ( train only final layer )
for age model  --  resnext50_32x4d ( train only final layer )
for emotions model  --  resnext50_32x4d ( train only final layer )


