import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas
from sklearn.model_selection import train_test_split

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 784
    indices = np.arange(n_subject)
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    training_data_dir = str(ROOT_DIR / '../data/raw_data/clinics/training_clinics.xls')
    testing_data_dir = str(ROOT_DIR / '../data/raw_data/clinics/testing_clinics.xls')

    training_patientIDs_dir = str(ROOT_DIR / '../data/raw_data/listNameFiles/training_patientIDs.xls')
    testing_patientIDs_dir = str(ROOT_DIR / '../data/raw_data/listNameFiles/testing_patientIDs.xls')

    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    events = np.array(data['OSCensor'], dtype = np.float32)
    treatmentRT = np.array(data['ProtonvsIMRTvs3D'], dtype = np.str_) 

    #shuffle the cohort
    train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = 1, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))
    #train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = 0, stratify = events)

    training_subject_name = subject_names[train_id]
    testing_subject_name = subject_names[test_id]
    

    training_dataFrame = pandas.DataFrame(training_subject_name, columns=['subject_name'])
    training_dataFrame.to_excel(training_patientIDs_dir)
    training_data = data.loc[train_id, :]
    training_data.to_excel(training_data_dir)

    testing_dataFrame = pandas.DataFrame(testing_subject_name, columns=['subject_name'])
    testing_dataFrame.to_excel(testing_patientIDs_dir)
    testing_data = data.loc[test_id, :]
    testing_data.to_excel(testing_data_dir)

if __name__ == '__main__':
    main()
