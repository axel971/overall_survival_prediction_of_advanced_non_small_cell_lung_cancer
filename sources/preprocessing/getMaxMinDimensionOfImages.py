import numpy as np
import os
import pandas
import sys
sys.path.append('/cbica/home/largenta/dev/LANSCLC_project/sources/')
from utils import get_nii_data


# Get the patient IDs from the xlsx file
file_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/patientIDs.xls"
data = pandas.read_excel(file_dir, dtype = np.str_)
data_length = data.shape

subject_names = np.array(data['subject_name'], dtype = np.str_)

        
# Load images from the patient names using their IDs and output their min max average dimensions 
img_dir = '/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/images'
images = []
shapes = []
for subject_name in subject_names: 
      shapes.append(get_nii_data(img_dir + '/' + subject_name + '.nii.gz').shape[0])


 	 
print(np.amax(shapes))
print(np.amin(shapes))
print(np.mean(shapes))
print(np.median(shapes))
print(subject_names[np.argmax(shapes)])
print(subject_names[np.argmin(shapes)])
#print(shapes)
