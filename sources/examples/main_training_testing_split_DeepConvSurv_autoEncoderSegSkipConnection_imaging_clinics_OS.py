import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_seg_imaging_clinics
from model.model_DeepConvSurv_autoEncoder import DeepConvSurv_autoEncoder_segSkipConnection_imaging_clinics, DeepConvSurv_autoEncoder_segSkipConnection_imaging_clinics_testing
from model.one_hot_label import  restore_labels_array
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
    n_subject = 333
    labels = [0, 1]
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 128]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['OS'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    delineation_dir = str(ROOT_DIR / '../data/preprocessing/delineations/resampling/')


    file_volume_dir = str(ROOT_DIR / '../data/raw_data/tumor_volumes/volumes.xls')
    data_volumes = pandas.read_excel(file_volume_dir, dtype = np.str_)
    volumes = data_volumes[['volume']]

    #risk_score_dir = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
    
    ## Loading the data  
    print('.. loading the data')
    # Load clinical data
    data["Sex"] = data.Sex.map(dict(M = 1, F = 0))
    data = pandas.concat([data, pandas.get_dummies(data["Race"], prefix = "Race")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["MaritalStatus"], prefix = "MaritalStatus")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["TumorLocation"], prefix = "TumorLocation")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Histology"], prefix = "Histology")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["Laterality"], prefix = "Laterality")], axis = 1)
    data = pandas.concat([data, pandas.get_dummies(data["PDL1"], prefix = "PDL1")], axis = 1)

    clinics = data[['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'ConcurrentAgentCoded','ConsolidationIO', 'Age', 'Sex', 'Race_White','Race_Black or African American', 'Race_Asian','Race_Other/Unknown', 'IsHispanic', 'MaritalStatus_Single', 'MaritalStatus_Married', 'MaritalStatus_Divorced', 'MaritalStatus_Widowed', 'MaritalStatus_Other', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'CHD', 'CAD', 'CHF', 'AtrialFibFlutter', 'COPD', 'Diabetes', 'HTN', 'HLD', 'TumorLocation_Lung, upper lobe', 'TumorLocation_Lung, middle lobe', 'TumorLocation_Lung, lower lobe', 'TumorLocation_Mediastinum', 'Histology_SCC', 'Histology_ACA', 'Histology_other', 'Laterality_1','Laterality_2','PDL1_<1%', 'PDL1_>=1%','PDL1_unknown', 'EGFR', 'ALK', 'KRAS','Tgrouped' ,"N", "Hosp90Days", 'Pneumonitis', 'Esophagitis']]

    clinics  = clinics.join(volumes)

    scaler = ColumnTransformer([('zscore', StandardScaler(), ['TotalElapsedDays', 'TotalTreatmentsDelivered', 'SrtTreatmentsDelivered', 'ProtonTreatmentsDelivered', 'ImrtTreatmentsDelivered', 'ConventionalTreatmentsDelivered', 'PrescribedDoseCgy','PrescribedFractions', 'DeliveredDoseCgy', 'Age', 'PackYrs','BMI', 'EcogPriorRtStart','CCI', 'volume'] )], remainder = 'passthrough')

    # Load imaging data
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    delineations = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)

    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
        delineations[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_resampled_boundingBox_delineation.nii.gz')

    print('..data size:' + str(X.shape))
    print('..loading data finished')


    ## Declare ouput arrays
    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    training_mae = []
    testing_mae = []
    training_dice = []
    testing_dice = []

    ## Start the N repeated ramdom splits (training/testing) of the cohorte
    nSimulation = 5
    for iSimulation in range(nSimulation):
        
        indices = np.array(range(n_subject))
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events)

        # Compute mean and sd of the training data (imaging and clinics)
        scaler.fit(clinics.loc[train_id,:])
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])

        # Normalize training data (z-score)
        training_data = (X[train_id] - mean_training)/ sd_training
        training_clinics = np.array(scaler.transform(clinics.loc[train_id,:]), dtype = np.float32)
        
        # Declare geerator
        training_generator = Generator3D_seg_imaging_clinics(training_data, delineations[train_id], training_clinics, times[train_id], events[train_id], labels, batch_size = 10)    
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        
        # Declare training model
        model = DeepConvSurv_autoEncoder_segSkipConnection_imaging_clinics(image_size+[1], (1), image_size+ [len(labels)], (training_clinics.shape[1]), len(labels))
        optimizer = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss = None)

        # Start model training
        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')

        # Declare testing model
        model_testing = DeepConvSurv_autoEncoder_segSkipConnection_imaging_clinics_testing(image_size + [1], (training_clinics.shape[1]), len(labels))
        model_testing.set_weights(model.get_weights())
      
        # Apply testing model on training data          
        train_pred  = model_testing.predict([training_data, training_clinics], verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred[0])
        train_pred_reconstruction = train_pred[1]
        train_pred_delineation = restore_labels_array(train_pred[2], labels)
      
        # Save endpoint results for each fold (training data) 
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])
     
        for iMae  in range(len(train_id)):
            training_mae.append(metrics.MAE(train_pred_reconstruction[iMae], training_data[iMae]))

        for iDice  in range(len(train_id)):
            training_dice.append(metrics.dice_multi_array(train_pred_delineation[iDice], delineations[train_id[iDice]], labels))

        # Normalize testing data (z-score)
        testing_data = (X[test_id] - mean_training) / sd_training
        testing_clinics = np.array(scaler.transform(clinics.loc[test_id, :]), dtype = np.float32)

        # Apply testing model on testing data
        testing_pred = model_testing.predict([testing_data, testing_clinics], verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred[0]) 
        testing_pred_reconstruction = testing_pred[1]
        testing_pred_delineation = restore_labels_array(testing_pred[2], labels)

        # Save endpoint results for each fold (testing data) 
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])

        for iMae  in range(len(test_id)):
            testing_mae.append(metrics.MAE(testing_pred_reconstruction[iMae], testing_data[iMae]))

        for iDice  in range(len(test_id)):
            testing_dice.append(metrics.dice_multi_array(testing_pred_delineation[iDice], delineations[test_id[iDice]], labels))

        print(np.mean(testing_dice, axis = 0))         

        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))

        # Clean memory at the end of each fold   
        del model
        del model_testing
        del training_data
        del training_clinics 
        del training_generator
        del testing_data
        del testing_clinics 
        del testing_pred
        del testing_pred_log_risk 
        del testing_pred_reconstruction
        del testing_pred_delineation
        del c_index_testing       
        del train_pred
        del train_pred_log_risk
        del train_pred_reconstruction
        del train_pred_delineation
        del c_index_training 
        del optimizer

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

    print("Segmentation Results")
    print(np.mean(training_dice, axis = 0))
    print(np.std(training_dice, axis = 0))

    print(np.mean(testing_dice, axis = 0))
    print(np.std(testing_dice, axis = 0))

if __name__ == '__main__':
    main()
