import numpy as np
import os
import pandas
import sys
sys.path.append('/cbica/home/largenta/dev/LANSCLC_project/sources/')
from utils import get_nii_data, get_voxel_spacing
import SimpleITK as sitk

# Get the patient IDs from the xlsx file
file_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/patientIDs.csv"
output_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/tumor_volumes/volumes.csv"

data = pandas.read_csv(file_dir, dtype = np.str_)
data_length = data.shape

subject_names = np.array(data['subject_name'], dtype = np.str_)
 
        
# Load images from the patient names using their IDs and output their min max average dimensions 
img_dir = '/cbica/home/largenta/dev/LANSCLC_project/data/preprocessing/delineations/clean_delineation'

volumes = []
for subject_name in subject_names: 
         img = sitk.ReadImage(img_dir + '/' + subject_name + '_' + 'delineation.nii.gz')
         voxel_spacing = img.GetSpacing()
         voxelSize = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
         np_array = sitk.GetArrayFromImage(img)
         volumes.append( np.sum(np_array) * voxelSize) # NB: We assumes that delineation is a binary image
 
columns = {'subject_name': subject_names, 'volume': volumes}	 
data = pandas.DataFrame(columns)
data.to_csv(output_dir)
