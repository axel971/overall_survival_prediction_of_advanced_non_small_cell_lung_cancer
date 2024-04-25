import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_oneClassificationTask_imaging_clinics
from model.model_DeepConvSurv_classifier import DeepConvSurv_oneClassificationTask_imaging_clinics, DeepConvSurv_oneClassificationTask_imaging_clinics_testing

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 784
    indices = np.arange(n_subject)
    image_size = [128, 128, 64]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['OS'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)
    treatmentRT = np.array(data['ProtonvsIMRTvs3D'], dtype = np.str_)
    pneumonitis = np.array(data['Pneumonitis'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/doses/resampling')
    risk_score_dir = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')

    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]


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
    clinics = data[['PracticeGroup_Cherry Hill', 'PracticeGroup_Chester County', 'PracticeGroup_PAH', 'PracticeGroup_PCAM', 'PracticeGroup_PPMC', 'PracticeGroup_Radnor', 'PracticeGroup_Valley Forge', 'TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'MachineTypes_Halcyon',  'MachineTypes_Halcyon, Proton', 'MachineTypes_Linac', 'MachineTypes_Linac, Proton','MachineTypes_Proton', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'ConcurrentAgentCoded', 'ConsolidationIOReceipt', 'Age', 'Sex', 'Race_White','Race_Black', 'Race_Other', 'IsHispanic', 'Partner', 'PackYrs','BMI', 'ECOGPriorRtStart_0', 'ECOGPriorRtStart_1', 'ECOGPriorRtStart_2', 'CCI', 'CHD', 'CAD', 'CHF', 'AtrialFibFlutter','Pulm', 'COPD', 'Diabetes', 'HTN', 'HLD', 'PrimarySiteDesc_Lung, upper lobe', 'PrimarySiteDesc_Lung, middle lobe', 'PrimarySiteDesc_Lung, lower lobe', 'PrimarySiteDesc_Mediastinum', 'HistologyCoded_SCC', 'HistologyCoded_ACA', 'HistologyCoded_other', 'Laterality_1','Laterality_2','PDL1_Grouped2_<1%', 'PDL1_Grouped2_>=1%','PDL1_Grouped2_unknown','Tgrouped_3-4', 'Tgrouped_0-2', "N", "Hosp90Days", 'Esophagitis']]

    clinics  = clinics.join(volumes)

    scaler = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI','CCI', 'volume'] )], remainder = 'passthrough')


    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_resampled_boundingBox_doses.nii.gz')
   
    print('..data size:' + str(X.shape))
    print('..loading data finished')

    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    auc_pneumonitis_testing_all_simulations = []
    auc_pneumonitis_training_all_simulations = []

    nSimulation = 4

    for iSimulation in range(nSimulation):
       
        print("### Fold " + str(iSimulation))

        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))

        scaler.fit(clinics.loc[train_id,:])
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])

        training_data = (X[train_id] - mean_training) / sd_training
        training_clinics = np.array(scaler.transform(clinics.loc[train_id,:]), dtype = np.float32)     
        training_generator = Generator3D_oneClassificationTask_imaging_clinics(training_data, training_clinics, times[train_id], events[train_id], pneumonitis[train_id], batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
                
        model = DeepConvSurv_oneClassificationTask_imaging_clinics(image_size+[1], (training_clinics.shape[1]), (1), (1))
        optimizer = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss = None)
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        
       
        model_testing = DeepConvSurv_oneClassificationTask_imaging_clinics_testing(image_size + [1], (training_clinics.shape[1]))
        model_testing.set_weights(model.get_weights())
          
        train_pred = model_testing.predict([training_data, training_clinics], verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred[0])
        train_pred_pneumonitis = np.concatenate(train_pred[1])
        
               
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        auc_pneumonitis_training = roc_auc_score(pneumonitis[train_id], train_pred_pneumonitis)

        print("Training results for current fold") 
        print(c_index_training)
        print(auc_pneumonitis_training)
        c_index_training_all_simulations.append(c_index_training[0])
        auc_pneumonitis_training_all_simulations.append(auc_pneumonitis_training)


        ##### Prediction ####
        testing_data = (X[test_id] - mean_training) / sd_training
        testing_clinics = np.array(scaler.transform(clinics.loc[test_id, :]), dtype = np.float32)

        testing_pred = model_testing.predict([testing_data, testing_clinics], verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred[0])
        testing_pred_pneumonitis = np.concatenate(testing_pred[1])
 
        
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        auc_pneumonitis_testing = roc_auc_score(pneumonitis[test_id], testing_pred_pneumonitis)

        print("Testing results for current fold")
        print(c_index_testing)
        print(auc_pneumonitis_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
        auc_pneumonitis_testing_all_simulations.append(auc_pneumonitis_testing)
         
        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))
        del model
        del model_testing
        del training_generator  
        del training_data
        del training_clinics
        del testing_data
        del testing_clinics
        del train_pred_log_risk
        del testing_pred_log_risk
        del optimizer
        K.clear_session()
        gc.collect()
 
    print("##### Final Results #####")    
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    auc_pneumonitis_testing_all_simulations = np.array(auc_pneumonitis_testing_all_simulations)
    auc_pneumonitis_training_all_simulations = np.array(auc_pneumonitis_training_all_simulations)

    print(" OS Results (C-index)")
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))

    print("Pneumonitis results (AUC)")
    print(np.mean(auc_pneumonitis_training_all_simulations))
    print(np.std(auc_pneumonitis_training_all_simulations))
    print(np.mean(auc_pneumonitis_testing_all_simulations))
    print(np.std(auc_pneumonitis_testing_all_simulations))

if __name__ == '__main__':
    main()
