import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator_CNN3DModel
from model.model_CNN3DClassifier import CNN3DModel
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
    n_subject = 784
    subject_list = np.arange(n_subject)
    image_size = [128, 128, 64]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    diseaseFailure = np.array(data['Failure'], dtype = np.float32)
    events = np.array(data['OSCensor'], dtype = np.float32)
    treatmentRT = np.array(data['ProtonvsIMRTvs3D'], dtype = np.str_)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
   
    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
   
    for n in range(n_subject):
        X[n] =  get_nii_data(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
   
    print('..data size:' + str(X.shape))
    print('..loading data finished')

    auc_diseaseFailure_testing_all_simulations = []
    auc_diseaseFailure_training_all_simulations = []
   
    nSimulation = 4

    for iSimulation in range(nSimulation):
        #shuffle the cohort
        print("### Fold " + str(iSimulation))

        indices = np.array(range(n_subject))
        #train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = events)
        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = pandas.concat((pandas.DataFrame(events), pandas.DataFrame(treatmentRT)), axis = 1))


        model = CNN3DModel(image_size+[1])
        optimizer = optimizers.Adam(learning_rate=1e-4)
    
        #mean_training = np.mean(X[train_id])
        #sd_training = np.std(X[train_id])

        training_generator = Generator_CNN3DModel(X[train_id], diseaseFailure[train_id], batch_size = 10)
        
        print('..generator_len: ' + str(training_generator.__len__()), flush=True)
        print('..training_n_subject: ' + str(training_generator.n_subject), flush=True)
        
        
        model.compile(optimizer=optimizer, loss = "BinaryCrossentropy")

        start_training_time = time.time()
        model.fit(x=training_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        train_pred = model.predict(X[train_id], verbose = 0)
               
        auc_diseaseFailure_training = roc_auc_score(diseaseFailure[train_id], train_pred)
        
        print("Training results for current fold") 
        print(auc_diseaseFailure_training)
      
        auc_diseaseFailure_training_all_simulations.append(auc_diseaseFailure_training)

        ##### Prediction ####    
        start_execution_time = time.time()
        testing_pred = model.predict(X[test_id], verbose = 0)
 
        auc_diseaseFailure_testing = roc_auc_score(diseaseFailure[test_id], testing_pred)

        print("Testing results for current fold")
        print(auc_diseaseFailure_testing)
        auc_diseaseFailure_testing_all_simulations.append(auc_diseaseFailure_testing)
         
        del model
    
    
    print("##### Final Results #####")    
    auc_diseaseFailure_testing_all_simulations = np.array(auc_diseaseFailure_testing_all_simulations)
    auc_diseaseFailure_training_all_simulations = np.array(auc_diseaseFailure_training_all_simulations)
    
    print("Disease Failure Results")
    print(np.mean(auc_diseaseFailure_training_all_simulations))
    print(np.std(auc_diseaseFailure_training_all_simulations))
    print(np.mean(auc_diseaseFailure_testing_all_simulations))
    print(np.std(auc_diseaseFailure_testing_all_simulations))

if __name__ == '__main__':
    main()
