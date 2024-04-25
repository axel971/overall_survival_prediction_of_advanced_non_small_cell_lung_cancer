import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D
from model.model import CNN_3D, CNN_3D_testing
from loss.loss import negative_log_likelihood

import numpy as np
import metrics.metrics as metrics
import time
import pandas
import gc
from utils import get_nii_data, get_nii_data_and_affine_matrix, get_nii_affine, save_image
from sksurv.metrics import concordance_index_censored

def main():
    
    # Instantiate directories and file paths + image and delineation information
    n_subject = 333
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 128]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    times = np.array(data['PFS'], dtype = np.float32)
    events = np.array(data['PFScensor'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/normalize')
    risk_score_dir = str(ROOT_DIR / '../data/output/progression_free_survival/risk_score')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
   
    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    affines = np.zeros((n_subject, 4, 4), dtype=np.float32)
    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n], affines[n] =  get_nii_data_and_affine_matrix(img_dir + '/' + subject_names[n] + '_resampled_normalized_boundingBox.nii.gz' )
   
    print('..data size:' + str(X.shape))
    print('..loading data finished')

    #shuffle the cohort
    indices = np.array(range(n_subject))
    #np.random.seed(0)
    np.random.shuffle(indices)
    X = X[indices]
    affines = affines[indices]
    times = times[indices]
    events = events[indices]
    
    # Cross-validation process
    nFold = 3
    foldSize = int(n_subject/nFold)
    
    for iFold in range(nFold):
        # Split the whole cohort on training and validation data sets for the current fold
        if iFold < nFold - 1 :
            test_id = np.array(range(iFold * foldSize, (iFold + 1) * foldSize))
            train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
            train_id = train_id[0:((nFold - 1) * foldSize)]
        else:
            test_id = np.array(range(iFold * foldSize, n_subject)) # Manage the case where the number of subject is not a multiple of the number of fold
            train_id = np.setdiff1d(np.array(range(n_subject)), test_id)
       
        print('---------fold--'+str(iFold + 1)+'---------')
        print('training set: ', train_id)
        print('training set size: ', len(train_id))
        print('testing set: ', test_id)
        print('testing set size: ', len(test_id))
        
        model = CNN_3D(image_size+[1], (1,))
        optimizer = optimizers.Adam(learning_rate=1e-4)
        training_generator = Generator3D(X[train_id], times[train_id], events[train_id], batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        
        model.compile(optimizer=optimizer, loss = None)

        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 30, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        model_testing = CNN_3D_testing(image_size + [1])
        model_testing.set_weights(model.get_weights())
        
        
        
        train_pred_log_risk = model_testing.predict(X[train_id], verbose = 0)
        train_pred_log_risk = np.concatenate(train_pred_log_risk)
       
        c_index_training = concordance_index_censored(events[train_id].astype(bool), times[train_id], np.exp(np.array(train_pred_log_risk)) )
        print(c_index_training)

        ##### Prediction ####
        
        start_execution_time = time.time()
        testing_pred_log_risk = model_testing.predict(X[test_id], verbose = 0)
        testing_pred_log_risk = np.concatenate(testing_pred_log_risk) 
        
        c_index_testing = concordance_index_censored(events[test_id].astype(bool), times[test_id], np.exp(np.array(testing_pred_log_risk)))
        print(c_index_testing)
 
        for i in range(len(test_id)):
            pred_log_risk[test_id[i]] = testing_pred_log_risk[i]
     
        end_execution_time = time.time()
        print('executation time:' + str((end_execution_time - start_execution_time)/(len(test_id))))
        
           
        del model
        del model_testing
        del training_generator
        del train_pred_log_risk
        del testing_pred_log_risk

        K.clear_session()
        gc.collect()

    c_index = concordance_index_censored(events.astype(bool), times, np.exp(pred_log_risk))
    print(c_index)         
    np.save(risk_score_dir + "/risk_scores.npy",  np.exp(pred_log_risk))

if __name__ == '__main__':
    main()
