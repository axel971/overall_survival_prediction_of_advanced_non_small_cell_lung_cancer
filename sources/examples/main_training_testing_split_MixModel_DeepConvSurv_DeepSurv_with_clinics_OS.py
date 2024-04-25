import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_with_clinics
from model.model import DeepConvSurv_DeepSurv_with_clinics, DeepConvSurv_DeepSurv_testing_with_clinics
from loss.loss import negative_log_likelihood

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 333
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 128]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['OS'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    risk_score_dir_1 = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    risk_score_dir_2 = str(ROOT_DIR / '../data/output/OS/DeepSurv/risk_scores')

   
    c_index_testing_all_simulations = []
    nSimulation = 5
 
    for iSimulation in range(nSimulation):
        indices = np.array(range(n_subject))
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events)    
        
        pred_risk_score_1 = np.load(risk_score_dir_1 + "/risk_scores_" + str(iSimulation) +  ".npy")
        pred_risk_score_2 = np.load(risk_score_dir_2 + "/risk_scores_" + str(iSimulation) +  ".npy")
 
        pred_risk_score = (pred_risk_score_1 + pred_risk_score_2) / 2.
        
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], pred_risk_score)
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
     
         

    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))

if __name__ == '__main__':
    main()
