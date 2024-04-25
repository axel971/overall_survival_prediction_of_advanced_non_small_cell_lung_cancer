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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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
    
    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]

    img_dir = str(ROOT_DIR / '../data/preprocessing/images/normalize')
    risk_score_dir = str(ROOT_DIR / '../data/output/PFS/DeepSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
    
    #Extract clinical data
    data["Sex"] = data.Sex.map(dict(M = 1, F = 0))
    data = pandas.concat([data, pandas.get_dummies(data["Race"], prefix = "Race")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["MaritalStatus"], prefix = "MaritalStatus")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["TumorLocation"], prefix = "TumorLocation")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Histology"], prefix = "Histology")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Laterality"], prefix = "Laterality")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["PDL1"], prefix = "PDL1")], axis = 1)

    #extract clinical data
    clinics = data[['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'ConcurrentAgentCoded','ConsolidationIO', 'Age', 'Sex', 'Race_White','Race_Black or African American', 'Race_Asian','Race_Other/Unknown', 'IsHispanic', 'MaritalStatus_Single', 'MaritalStatus_Married', 'MaritalStatus_Divorced', 'MaritalStatus_Widowed', 'MaritalStatus_Other', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'CHD', 'CAD', 'CHF', 'AtrialFibFlutter', 'COPD', 'Diabetes', 'HTN', 'HLD', 'TumorLocation_Lung, upper lobe', 'TumorLocation_Lung, middle lobe', 'TumorLocation_Lung, lower lobe', 'TumorLocation_Mediastinum', 'Histology_SCC', 'Histology_ACA', 'Histology_other', 'Laterality_1','Laterality_2','PDL1_<1%', 'PDL1_>=1%','PDL1_unknown', 'EGFR', 'ALK', 'KRAS','Tgrouped' ,"N", "Hosp90Days", 'Pneumonitis', 'Esophagitis']]

    clinics  = clinics.join(volumes)

    scaler = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'volume'] )], remainder = 'passthrough')

    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    nSimulation = 5

    for iSimulation in range(nSimulation):
        indices = np.array(range(n_subject))
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events)   
          
        model = DeepSurv((clinics.shape[1]), (1))
        optimizer = optimizers.Adam(learning_rate=1e-5)
        
        
        scaler.fit(clinics.loc[train_id,:])     
        training_generator = Generator_clinics(np.array(scaler.transform(clinics.loc[train_id, :]), dtype = np.float32), times[train_id], events[train_id], batch_size = 10)
    
        model.compile(optimizer=optimizer, loss = None)

        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 100, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        model_testing = DeepSurv_testing((clinics.shape[1]))
        model_testing.set_weights(model.get_weights())
         
        train_pred_log_risk = model_testing.predict(np.array(scaler.transform(clinics.loc[train_id, :]), dtype = np.float32), verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred_log_risk)
       
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])

        ##### Prediction ####
        
        start_execution_time = time.time()
        testing_pred_log_risk = model_testing.predict(np.array(scaler.transform(clinics.loc[test_id, :]), dtype = np.float32), verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred_log_risk) 
        
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
        np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) +  ".npy",  np.exp(testing_pred_log_risk))

    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))
if __name__ == '__main__':
    main()
