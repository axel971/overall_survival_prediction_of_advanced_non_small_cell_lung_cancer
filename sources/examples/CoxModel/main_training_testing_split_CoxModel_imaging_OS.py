import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))
import sys
sys.path.append(str(ROOT_DIR))

import numpy as np

import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.utils.metrics import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 784
    indices = np.arange(n_subject)
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['OS'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)
    treatmentRT = np.array(data['ProtonvsIMRTvs3D'], dtype = np.str_)
   

    file_radiomicFeature_dir = str(ROOT_DIR / '../data/radiomic_features/radiomicFeatures.csv')
    radiomic_features = pandas.read_csv(file_radiomicFeature_dir, dtype = np.str_)
    radiomic_features = radiomic_features.loc[:, ~radiomic_features.columns.str.startswith('diagnostic')]
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/normalize')
    risk_score_dir = str(ROOT_DIR / '../data/output/OS/CoxModel/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')

    scaler = StandardScaler()

    nSimulation = 4
    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []

    for iSimulation in range(nSimulation):    
 
        #train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = events)
        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))

        #print(np.sum(events[train_id]))
        #print(np.sum(events[test_id]))
       
        model = CoxPHModel()
        scaler.fit(radiomic_features.iloc[train_id, :])
        
        start_training_time = time.time()
        model.fit(X = scaler.transform(radiomic_features.loc[train_id, :]), T = times[train_id], E = events[train_id], init_method='zeros', l2_reg = 1e-3, lr = 0.0001, tol = 1e-4, max_iter = 100, verbose = 0)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        
        
        train_pred_log_risk = model.predict_risk(scaler.transform(radiomic_features.loc[train_id,:]))         
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], train_pred_log_risk) 
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])
        ##### Prediction ####
        
        start_execution_time = time.time()
        testing_pred_log_risk = model.predict_risk(scaler.transform(radiomic_features.loc[test_id, :])) 
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], testing_pred_log_risk)    
        print(c_index_testing)
   
        end_execution_time = time.time()
        print('executation time:' + str((end_execution_time - start_execution_time)/(len(test_id))))
        
        c_index_testing_all_simulations.append(c_index_testing[0])              
        np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy", testing_pred_log_risk)

    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations)) 
    print((np.std(c_index_testing_all_simulations)))

if __name__ == '__main__':
    main()
