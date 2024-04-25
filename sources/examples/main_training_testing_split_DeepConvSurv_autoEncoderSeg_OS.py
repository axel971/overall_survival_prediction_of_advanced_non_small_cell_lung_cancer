import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_seg
from model.model_DeepConvSurv_autoEncoder import DeepConvSurv_autoEncoder_seg, DeepConvSurv_autoEncoder_seg_testing
from model.one_hot_label import  restore_labels_array

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split

def main():
    
    # Instantiate directories and file paths + image and delineation information
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

    #risk_score_dir = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
   
    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    delineations = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)

    affines = np.zeros((n_subject, 4, 4), dtype=np.float32)
    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n], affines[n] =  get_nii_data_and_affine_matrix(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
        delineations[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_resampled_boundingBox_delineation.nii.gz')

    print('..data size:' + str(X.shape))
    print('..loading data finished')

    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    training_mae = []
    testing_mae = []
    training_dice = []
    testing_dice = []

    nSimulation = 5

    for iSimulation in range(nSimulation):
        #shuffle the cohort
        indices = np.array(range(n_subject))
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events)

        print(image_size+ [len(labels)])
        model = DeepConvSurv_autoEncoder_seg(image_size+[1], (1), image_size+ [len(labels)], len(labels))
        optimizer = optimizers.Adam(learning_rate=1e-4)
    
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])
        training_data = (X[train_id] - mean_training)/ sd_training
 
        training_generator = Generator3D_seg(training_data, delineations[train_id], times[train_id], events[train_id], labels, batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        
        model.compile(optimizer=optimizer, loss = None)

        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        model_testing = DeepConvSurv_autoEncoder_seg_testing(image_size + [1], len(labels))
        model_testing.set_weights(model.get_weights())
                
        train_pred  = model_testing.predict(training_data, verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred[0])
        train_pred_reconstruction = train_pred[1]
        train_pred_delineation = restore_labels_array(train_pred[2], labels)
       
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        print(c_index_training)
        c_index_training_all_simulations.append(c_index_training[0])
     
        for iMae  in range(len(train_id)):
            training_mae.append(metrics.MAE(train_pred_reconstruction[iMae], training_data[iMae]))

        for iDice  in range(len(train_id)):
            training_dice.append(metrics.dice_multi_array(train_pred_delineation[iDice], delineations[train_id[iDice]], labels))

        ##### Prediction ####
        
        start_execution_time = time.time()

        testing_data = (X[test_id] - mean_training) / sd_training
        testing_pred = model_testing.predict(testing_data, verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred[0]) 
        testing_pred_reconstruction = testing_pred[1]
        testing_pred_delineation = restore_labels_array(testing_pred[2], labels)

        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        print(c_index_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])

        for iMae  in range(len(test_id)):
            testing_mae.append(metrics.MAE(testing_pred_reconstruction[iMae], testing_data[iMae]))

        for iDice  in range(len(test_id)):
            testing_dice.append(metrics.dice_multi_array(testing_pred_delineation[iDice], delineations[test_id[iDice]], labels))

        print(np.mean(testing_dice, axis = 0))         

        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))

        del model
        del model_testing
        del training_data
        del training_generator
        del testing_data
        del testing_pred
        del testing_pred_log_risk
        del testing_pred_reconstruction
        del testing_pred_delineation        
        del train_pred
        del train_pred_log_risk
        del train_pred_reconstruction
        del train_pred_delineation
        K.clear_session()
 
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
