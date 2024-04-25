import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_OS_oneClassificationTask
from model.model_DeepConvSurv import DeepConvSurv_OS_oneClassificationTask, DeepConvSurv_OS_oneClassificationTask_testing
from loss.loss import negative_log_likelihood

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    diseaseFailure = np.array(data['Failure'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    risk_score_dir = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
   
    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    affines = np.zeros((n_subject, 4, 4), dtype=np.float32)
    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n], affines[n] =  get_nii_data_and_affine_matrix(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
   
    print('..data size:' + str(X.shape))
    print('..loading data finished')

    c_index_testing_all_simulations = []
    c_index_training_all_simulations = []
    auc_diseaseFailure_testing_all_simulations = []
    auc_diseaseFailure_training_all_simulations = []

    nSimulation = 5

    for iSimulation in range(nSimulation):
        #shuffle the cohort
        print("### Fold " + str(iSimulation))

        indices = np.array(range(n_subject))
        train_id, test_id = train_test_split(indices, test_size = 0.5, random_state = iSimulation, stratify = events)

        model = DeepConvSurv_OS_oneClassificationTask(image_size+[1], (1), (1))
        optimizer = optimizers.Adam(learning_rate=1e-4)
    
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])

        training_generator = Generator3D_OS_oneClassificationTask((X[train_id] - mean_training)/ sd_training , times[train_id], events[train_id], diseaseFailure[train_id], batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        
        model.compile(optimizer=optimizer, loss = None)

        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        model_testing = DeepConvSurv_OS_oneClassificationTask_testing(image_size + [1])
        model_testing.set_weights(model.get_weights())
          
        train_pred = model_testing.predict((X[train_id] - mean_training) / sd_training, verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred[0])
        train_pred_diseaseFailure = np.concatenate(train_pred[1])
        
               
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        auc_diseaseFailure_training = roc_auc_score(diseaseFailure[train_id], train_pred_diseaseFailure)

        print("Training results for current fold") 
        print(c_index_training)
        print(auc_diseaseFailure_training)
        c_index_training_all_simulations.append(c_index_training[0])
        auc_diseaseFailure_training_all_simulations.append(auc_diseaseFailure_training)


        ##### Prediction ####
        
        start_execution_time = time.time()
        testing_pred = model_testing.predict((X[test_id] - mean_training) / sd_training, verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred[0])
        testing_pred_diseaseFailure = np.concatenate(testing_pred[1])
 
        
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        auc_diseaseFailure_testing = roc_auc_score(diseaseFailure[test_id], testing_pred_diseaseFailure)

        print("Testing results for current fold")
        print(c_index_testing)
        print(auc_diseaseFailure_testing)
        c_index_testing_all_simulations.append(c_index_testing[0])
        auc_diseaseFailure_testing_all_simulations.append(auc_diseaseFailure_testing)
         
        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))

        del model
        del model_testing
    
    print("##### Final Results #####")    
    c_index_testing_all_simulations = np.array(c_index_testing_all_simulations)
    c_index_training_all_simulations = np.array(c_index_training_all_simulations)
    auc_diseaseFailure_testing_all_simulations = np.array(auc_diseaseFailure_testing_all_simulations)
    auc_diseaseFailure_training_all_simulations = np.array(auc_diseaseFailure_training_all_simulations)

    print("C-Index Results")
    print(np.mean(c_index_training_all_simulations))
    print(np.std(c_index_training_all_simulations))
    print(np.mean(c_index_testing_all_simulations))
    print(np.std(c_index_testing_all_simulations))

    print("Pneumonitis results")
    print(np.mean(auc_diseaseFailure_training_all_simulations))
    print(np.std(auc_diseaseFailure_training_all_simulations))
    print(np.mean(auc_diseaseFailure_testing_all_simulations))
    print(np.std(auc_diseaseFailure_testing_all_simulations))

if __name__ == '__main__':
    main()
