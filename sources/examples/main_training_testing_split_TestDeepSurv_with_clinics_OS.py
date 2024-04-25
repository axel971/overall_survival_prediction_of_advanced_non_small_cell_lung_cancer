import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator_clinics
from model.model import DeepSurv, DeepSurv_testing
from loss.loss import negative_log_likelihood

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.metrics import concordance_index
from sklearn.model_selection import train_test_split

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 333
    subject_list = np.arange(n_subject)
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['PFS'], dtype = np.float32)
    events = np.array(data['PFScensor'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/normalize')
    risk_score_dir = str(ROOT_DIR / '../data/output/progression_free_survival/risk_score')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
    
    #extract clinical data
    data["Sex"] = data.Sex.map(dict(M = 1, F = 0))
    clinics = np.array(data[['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'ConcurrentAgentCoded','ConsolidationIO', 'Age', 'Sex', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'CHD', 'CAD', 'CHF', 'AtrialFibFlutter', 'COPD', 'Diabetes', 'HTN', 'HLD', 'EGFR', 'ALK', 'KRAS', "N", "Hosp90Days"]], dtype = np.float32)     

    pred_log_risk = np.zeros(n_subject, dtype = np.float32)
    
    indices = np.array(range(n_subject))
    train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = 1, stratify = events)
    print(np.sum(events[train_id]))
    print(np.sum(events[test_id]))
    
    structure = [{'activation': 'ReLU', 'num_units' : 64, 'auto_scaler': False}, {'activation': 'ReLU', 'num_units' : 32, 'auto_scaler' : False},{'activation': 'ReLU', 'num_units' : 16, 'auto_scaler': False}]
    model = NonLinearCoxPHModel(structure = structure, auto_scaler = True)
        
    start_training_time = time.time()
    model.fit(X = clinics[train_id], T = times[train_id], E = events[train_id], init_method = 'glorot_uniform', l2_reg = 1e-3, dropout = 0.2, num_epochs = 1000, lr = 0.0001, batch_normalization = False, bn_and_dropout = False)
    end_training_time = time.time()	
    print('training time: ' + str(end_training_time - start_training_time))
        
    #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        
        
    train_pred_log_risk = model.predict_risk(clinics[train_id])         
    c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], train_pred_log_risk) 
    print(c_index_training)

    ##### Prediction ####
        
    start_execution_time = time.time()
    testing_pred_log_risk = model.predict_risk(clinics[test_id]) 
    c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], testing_pred_log_risk)
    print(c_index_testing)
   
    end_execution_time = time.time()
    print('executation time:' + str((end_execution_time - start_execution_time)/(len(test_id))))
                       
    #np.save(risk_score_dir + "/risk_scores.npy",  pred_log_risk)
  
if __name__ == '__main__':
    main()
