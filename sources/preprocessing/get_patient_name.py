import sys
sys.path.append('..')

import numpy as np
import os
import pandas
import openpyxl


img_input_dir = "/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/Organized_Data_Final/"
listName_output_file ="/cbica/home/largenta/dev/LANSCLC_project/data/raw_data/listNameFiles/all_patientIDs.xls"
subject_name = []

for path_patient in os.listdir(img_input_dir):
	if(path_patient[0] != '.'):
		subject_name.append(os.path.splitext(os.path.splitext(path_patient)[0])[0])

col = np.array(subject_name).reshape(len(subject_name), 1)


outputFile = pandas.DataFrame(col, columns=['subject_name'])
outputFile.to_excel(listName_output_file)
