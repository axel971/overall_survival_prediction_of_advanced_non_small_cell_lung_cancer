import numpy as np
import os
import pandas
import sys
sys.path.append('/cbica/home/largenta/dev/LANSCLC_project/sources/')
from utils import get_nii_data, get_voxel_spacing


# Get the patient IDs from the xlsx file
file_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/patientIDs.xls"
data = pandas.read_excel(file_dir, dtype = np.str_)
data_length = data.shape

subject_names = np.array(data['subject_name'], dtype = np.str_)
 
        
# Load images from the patient names using their IDs and output their min max average dimensions 
img_dir = '/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/images'
images = []
voxelSize = []
for subject_name in subject_names: 
         voxelSize.append(get_voxel_spacing(img_dir + '/' + subject_name + '.nii.gz')[0])
         
 	 
print(np.amax(voxelSize))
print(np.amin(voxelSize))
print(np.mean(voxelSize))
print(np.std(voxelSize))
print(np.median(voxelSize))

#print(shapes)
