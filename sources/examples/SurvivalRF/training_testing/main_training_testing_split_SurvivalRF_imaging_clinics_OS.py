import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '../..')))
import sys
sys.path.append(str(ROOT_DIR))

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
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
    times = np.array(data['OS'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)

    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]


    file_radiomicFeature_dir = str(ROOT_DIR / '../data/radiomic_features/radiomicFeatures.csv')
    radiomic_features = pandas.read_csv(file_radiomicFeature_dir, dtype = np.str_)
    radiomic_features = radiomic_features.loc[:, ~radiomic_features.columns.str.startswith('diagnostic')]

    img_dir = str(ROOT_DIR / '../data/preprocessing/images/normalize')
    risk_score_dir = str(ROOT_DIR / '../data/output/PFS/SurvivalRF/risk_scores')
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

    scaler_clinics = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'volume'] )], remainder = 'passthrough')

    scaler_radiomicFeatures = StandardScaler()

    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    nSimulation = 5    

    for iSimulation in range(nSimulation):

        indices = np.array(range(n_subject))        
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events) 
 
        model = RandomSurvivalForestModel(num_trees = 2000)

        scaler_clinics.fit(clinics.loc[train_id, :])
        scaler_radiomicFeatures.fit(radiomic_features.loc[train_id, :])

        start_training_time = time.time()
        print(scaler_clinics.transform(clinics.loc[train_id, :]).shape)
        print(scaler_radiomicFeatures.transform(radiomic_features.loc[train_id, :]).shape)
        training_data = np.concatenate((scaler_clinics.transform(clinics.loc[train_id, :]), scaler_radiomicFeatures.transform(radiomic_features.loc[train_id, :])), axis =1)
        print(training_data.shape)
        model.fit(X = training_data, T = times[train_id], E = events[train_id], max_features = 60, max_depth = 10, min_node_size = 10)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        
        
        train_pred_log_risk = model.predict_risk(training_data)         
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], train_pred_log_risk) 
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])

        ##### Prediction ####
        
        start_execution_time = time.time()
        testing_data = np.concatenate((scaler_clinics.transform(clinics.loc[test_id, :]), scaler_radiomicFeatures.transform(radiomic_features.loc[test_id, :])), axis =1)
        testing_pred_log_risk = model.predict_risk(testing_data) 
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], testing_pred_log_risk)
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
 
        end_execution_time = time.time()
        print('executation time:' + str((end_execution_time - start_execution_time)/(len(test_id))))
        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  testing_pred_log_risk)
   
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))
 
if __name__ == '__main__':
    main()
