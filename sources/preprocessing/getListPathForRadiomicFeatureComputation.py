import numpy as np
import os
import pandas
import sys
sys.path.append('/cbica/home/largenta/dev/LANSCLC_project/sources/')
from utils import get_nii_data, get_voxel_spacing


# Get the patient IDs from the xlsx file
file_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/patientIDs.xls"
output_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/imageAndMaskPathsForRadiomicFeatureComputation.csv"

data = pandas.read_excel(file_dir, dtype = np.str_)
data_length = data.shape

subject_names = np.array(data['subject_name'], dtype = np.str_)
 
        
# Load images from the patient names using their IDs and output their min max average dimensions 
img_dir = '/cbica/home/largenta/dev/LANSCLC_project/data/preprocessing/images/resampling'
delineation_dir = '/cbica/home/largenta/dev/LANSCLC_project/data/preprocessing/delineations/resampling'

image_paths = []
delineation_paths= []

for subject_name in subject_names: 
         image_paths.append(img_dir + '/' + subject_name + '_resampled_boundingBox.nii.gz')
         delineation_paths.append(delineation_dir + '/' + subject_name + '_resampled_boundingBox_delineation.nii.gz')

columns = {'Image': image_paths, 'Mask': delineation_paths}	 
data = pandas.DataFrame(columns)
data.to_csv(output_dir, index = False)
