import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_imaging_dose_clinics
from model.model_DeepConvSurv_autoEncoder import DeepConvSurv_autoEncoder_imaging_dose_clinics, DeepConvSurv_autoEncoder_imaging_dose_clinics_testing

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
    
    ## Declare directories, file paths +  image and delineation information
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
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    dose_dir = str(ROOT_DIR / '../data/preprocessing/doses/resampling')
   
    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]

    ## Load the data
    print('.. loading the data')
    # Load clinical data 
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

    scaler = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI','CCI', 'volume'] )], remainder = 'passthrough')


    # Load imaging data
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    doses = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
        doses[n]  =  get_nii_data(dose_dir + '/' + subject_names[n] + '_resampled_boundingBox_doses.nii.gz')
   

    print('..loading data finished')

    ## Declare ouput arrays
    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    training_mae = []
    testing_mae = []

    ## Start the N repeated ramdom splits (training/testing) of the cohorte
    nSimulation = 4
    for iSimulation in range(nSimulation):
       
        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))
    
        # Compute mean and sd of the training data (imaging and clinics)
        scaler.fit(clinics.loc[train_id,:])
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])

        # Normalize training data (z-score)         
        training_data = (X[train_id] - mean_training)/ sd_training
        training_clinics = np.array(scaler.transform(clinics.loc[train_id,:]), dtype = np.float32)

        # Delare Generator
        training_generator = Generator3D_imaging_dose_clinics(training_data, doses[train_id], training_clinics, times[train_id], events[train_id], batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        # Declare training model
        model = DeepConvSurv_autoEncoder_imaging_dose_clinics(image_size+[1], image_size+[1] , (training_clinics.shape[1]), (1))
        optimizer = optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss = None)

        # Start model training
        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')

        # Declare testing model 
        model_testing = DeepConvSurv_autoEncoder_imaging_dose_clinics_testing(image_size + [1], image_size+[1], (training_clinics.shape[1]))
        model_testing.set_weights(model.get_weights())
       
        # Apply testing model on training data                
        train_pred = model_testing.predict([training_data, doses[train_id], training_clinics], verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred[0])
        train_pred_reconstruction = train_pred[1][:,:,:,:,0]
       
        # Save endpoint results for each fold (training data)
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])

        for iMae  in range(len(train_id)):
            training_mae.append(metrics.MAE(train_pred_reconstruction[iMae], training_data[iMae]))

                           
        # Normalize testing data (z-score)
        testing_data = (X[test_id] - mean_training) / sd_training
        testing_clinics = np.array(scaler.transform(clinics.loc[test_id, :]), dtype = np.float32)

        # Apply testing model on testing data
        testing_pred = model_testing.predict([testing_data, doses[test_id], testing_clinics], verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred[0]) 
        testing_pred_reconstruction = testing_pred[1][:, :, :, :, 0]
        
        # Save endpoint results for each fold (testing data)
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
        
        for iMae  in range(len(test_id)):
            testing_mae.append(metrics.MAE(testing_pred_reconstruction[iMae], testing_data[iMae]))
 
        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))
         
        # Clean memory at the end of each fold
        del model
        del model_testing
        del training_data
        del training_clinics
        del training_generator
        del train_pred
        del train_pred_log_risk
        del train_pred_reconstruction
        del testing_data
        del testing_clinics
        del testing_pred
        del testing_pred_log_risk
        del testing_pred_reconstruction
        K.clear_session()
        gc.collect()


    print("#### Final results #####")
    
    print("C-index")
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))


    print("Reconstruction Results")
    print(np.mean(training_mae))
    print(np.std(training_mae))

    print(np.mean(testing_mae))
    print(np.std(testing_mae))


if __name__ == '__main__':
    main()
