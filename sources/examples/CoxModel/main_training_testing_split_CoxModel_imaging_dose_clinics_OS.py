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
from sklearn.decomposition import PCA

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
   
    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]


    file_img_radiomicFeature_dir = str(ROOT_DIR / '../data/radiomic_features/radiomicFeatures.csv')
    radiomic_img_features = pandas.read_csv(file_img_radiomicFeature_dir, dtype = np.str_)
    radiomic_img_features = radiomic_img_features.loc[:, ~radiomic_img_features.columns.str.startswith('diagnostic')]

    file_dose_radiomicFeature_dir = str(ROOT_DIR / '../data/radiomic_features/radiomicFeatures_dose.csv')
    radiomic_dose_features = pandas.read_csv(file_dose_radiomicFeature_dir, dtype = np.str_)
    radiomic_dose_features = radiomic_dose_features.loc[:, ~radiomic_dose_features.columns.str.startswith('diagnostic')]


    #Extract clinical data
    data["Sex"] = data.Sex.map(dict(M = 1, F = 0))

    data = pandas.concat([data, pandas.get_dummies(data["PracticeGroup"], prefix = "PracticeGroup")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["MachineTypes"], prefix = "MachineTypes")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["ECOGPriorRtStart"], prefix = "ECOGPriorRtStart")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Race"], prefix = "Race")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["PrimarySiteDesc"], prefix = "PrimarySiteDesc")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["HistologyCoded"], prefix = "HistologyCoded")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Laterality"], prefix = "Laterality")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["PDL1_Grouped2"], prefix = "PDL1_Grouped2")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Tgrouped"], prefix = "Tgrouped")], axis = 1)
    #extract clinical data
    clinics = data[['PracticeGroup_Cherry Hill', 'PracticeGroup_Chester County', 'PracticeGroup_PAH', 'PracticeGroup_PCAM', 'PracticeGroup_PPMC', 'PracticeGroup_Radnor', 'PracticeGroup_Valley Forge', 'TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'MachineTypes_Halcyon',  'MachineTypes_Halcyon, Proton', 'MachineTypes_Linac', 'MachineTypes_Linac, Proton','MachineTypes_Proton', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'ConcurrentAgentCoded', 'ConsolidationIOReceipt', 'Age', 'Sex', 'Race_White','Race_Black', 'Race_Other', 'IsHispanic', 'Partner', 'PackYrs','BMI', 'ECOGPriorRtStart_0', 'ECOGPriorRtStart_1', 'ECOGPriorRtStart_2', 'CCI', 'CHD', 'CAD', 'CHF', 'AtrialFibFlutter','Pulm', 'COPD', 'Diabetes', 'HTN', 'HLD', 'PrimarySiteDesc_Lung, upper lobe', 'PrimarySiteDesc_Lung, middle lobe', 'PrimarySiteDesc_Lung, lower lobe', 'PrimarySiteDesc_Mediastinum', 'HistologyCoded_SCC', 'HistologyCoded_ACA', 'HistologyCoded_other', 'Laterality_1','Laterality_2','PDL1_Grouped2_<1%', 'PDL1_Grouped2_>=1%','PDL1_Grouped2_unknown','Tgrouped_3-4', 'Tgrouped_0-2', "N", "Hosp90Days", 'Pneumonitis', 'Esophagitis']]

    clinics  = clinics.join(volumes)

    scaler_clinics = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI','CCI', 'volume'] )], remainder = 'passthrough')

    scaler_img_radiomicFeatures = StandardScaler()
    scaler_dose_radiomicFeatures = StandardScaler()


    nSimulation = 4
    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []

    for iSimulation in range(nSimulation):    
 
        #train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = events)
        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))

        scaler_clinics.fit(clinics.loc[train_id, :])
        scaler_img_radiomicFeatures.fit(radiomic_img_features.loc[train_id, :])
        scaler_dose_radiomicFeatures.fit(radiomic_dose_features.loc[train_id, :])
        

        pca_img_training = PCA(n_components = clinics.shape[1])
        pca_dose_training = PCA(n_components = clinics.shape[1])
        pca_img_training.fit(scaler_img_radiomicFeatures.transform(radiomic_img_features.loc[train_id, :])) 
        pca_dose_training.fit(scaler_dose_radiomicFeatures.transform(radiomic_dose_features.loc[train_id, :]))



        training_data = np.concatenate((scaler_clinics.transform(clinics.loc[train_id, :]), pca_img_training.transform(scaler_img_radiomicFeatures.transform(radiomic_img_features.loc[train_id, :])),pca_dose_training.transform(scaler_dose_radiomicFeatures.transform(radiomic_dose_features.loc[train_id, :])) ), axis =1)
 
        model = CoxPHModel()     
        model.fit(X = training_data, T = times[train_id], E = events[train_id], init_method='zeros', l2_reg = 1e-3, lr  = 1e-5, tol = 1e-4, max_iter = 1000, verbose = 0)
              
        train_pred_log_risk = model.predict_risk(training_data)         
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], train_pred_log_risk) 
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])

        ##### Prediction ####
        testing_data = np.concatenate((scaler_clinics.transform(clinics.loc[test_id, :]), pca_img_training.transform(scaler_img_radiomicFeatures.transform(radiomic_img_features.loc[test_id, :])), pca_dose_training.transform(scaler_dose_radiomicFeatures.transform(radiomic_dose_features.loc[test_id, :])) ), axis =1)        
        testing_pred_log_risk = model.predict_risk(testing_data) 
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], testing_pred_log_risk)    
        print(c_index_testing)
   
        c_index_testing_all_simulations.append(c_index_testing[0])              
       

    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations)) 
    print((np.std(c_index_testing_all_simulations)))

if __name__ == '__main__':
    main()
