import os
from pathlib import Path
ROOT_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.append(str(ROOT_DIR))

from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
# from keras.callbacks import ModelCheckpoint
from model.generator_array import Generator3D_segOnly, Generator3D_segOnly_withoutDataAugmentation
from model.model_seg import UNet3D
from model.one_hot_label import  restore_labels_array, restore_labels
from loss.loss import dice_oneVOI_loss
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
    n_subject = 784
    labels = [0, 1]
    indices = np.arange(n_subject)
    image_size = [128, 128, 64]
 
    file_dir = str(ROOT_DIR / '../data/raw_data/clinics/clinical_data.xls')
    data = pandas.read_excel(file_dir, dtype = np.str_)
    data_length = data.shape
    subject_names = np.array(data['HupMrn'], dtype = np.str_)
    events = np.array(data['OSCensor'], dtype = np.float32)
    img_dir = str(ROOT_DIR / '../data/preprocessing/images/resampling')
    delineation_dir = str(ROOT_DIR / '../data/preprocessing/delineations/resampling/')

    #risk_score_dir = str(ROOT_DIR / '../data/output/OS/DeepConvSurv/risk_scores')
    #model_dir = str(ROOT_DIR / '../data/output/3DU-Net_nonContrast_CT/models')
   
    print('.. loading the data')
    X = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)
    delineations = np.zeros((n_subject, image_size[0], image_size[1], image_size[2]), dtype = np.float32)

    pred_log_risk = np.zeros(n_subject, dtype = np.float32)

    for n in range(n_subject):
        X[n]  =  get_nii_data(img_dir + '/' + subject_names[n] + '_resampled_boundingBox.nii.gz')
        delineations[n] =  get_nii_data(delineation_dir + '/' + subject_names[n] + '_resampled_boundingBox_delineation.nii.gz')

    print('..data size:' + str(X.shape))
    print('..loading data finished')

    # Normalize the images
   # for n in range(n_subject):
   #     X[n] = (X[n] - np.mean(X[n]))/ np.std(X[n])

    training_dice_segMap = []
    training_dice = []

    testing_dice = []
    testing_dice_segMap = []

    nSimulation = 2

    for iSimulation in range(nSimulation):
        #shuffle the cohort
        train_id, test_id = train_test_split(indices, test_size = 0.3, random_state = iSimulation, stratify = events)

        print(image_size+ [len(labels)])
        model = UNet3D(image_size+[1], len(labels))
        optimizer = optimizers.Adam(learning_rate=1e-3)
    
        mean_training = np.mean(X[train_id])
        sd_training = np.std(X[train_id])
        training_data = X[train_id] #(X[train_id] - mean_training)/ sd_training
 
        training_generator = Generator3D_segOnly(training_data, delineations[train_id], labels, batch_size = 10)
        

        testing_data = X[test_id] #(X[test_id] - mean_training) / sd_training
        validation_generator = Generator3D_segOnly_withoutDataAugmentation(testing_data, delineations[test_id], labels, batch_size = 10)
       
        model.compile(optimizer=optimizer, loss = dice_oneVOI_loss, metrics = metrics.dice_oneVOI)

        start_training_time = time.time()
        model.fit(x=training_generator, validation_data = validation_generator, epochs = 60, verbose = 2)
        end_training_time = time.time()	
        print('training time: ' + str(end_training_time - start_training_time))
        
        #model.save_weights(model_dir + '/model_' + str(iFold) + '.h5')
        train_pred  = model.predict(training_data, verbose = 0)
        #train_pred_delineation = restore_labels_array(train_pred, labels)
       
        for iDice  in range(len(train_id)):
            training_dice.append(metrics.dice(restore_labels(train_pred[iDice], labels), delineations[train_id[iDice]]))  
            training_dice_segMap.append(metrics.dice(train_pred[iDice, :, :,:,1], delineations[train_id[iDice]]))
        
        print(np.mean(training_dice))
        print(np.std(training_dice))
        print(np.mean(training_dice_segMap))
        print(np.std(training_dice_segMap))

        ##### Prediction ####
        
        start_execution_time = time.time()

        testing_pred = model.predict(testing_data, verbose = 0)
        #testing_pred_delineation = restore_labels_array(testing_pred, labels)

        for iDice  in range(len(test_id)):
            current_dice =  metrics.dice(restore_labels(testing_pred[iDice], labels), delineations[test_id[iDice]])
            testing_dice.append(current_dice)
            testing_dice_segMap.append(metrics.dice(testing_pred[iDice, :,:,:,1], delineations[test_id[iDice]]))
  
            if(current_dice <= 0.40):
               print(subject_names[test_id[iDice]])
               print(current_dice)
        
        print("Result for current fold: ")
        print(np.mean(testing_dice))
        print(np.std(testing_dice))         
        print(np.mean(testing_dice_segMap))
        print(np.std(testing_dice_segMap))
        #np.save(risk_score_dir + "/risk_scores_" + str(iSimulation) + ".npy",  np.exp(testing_pred_log_risk))

        del model
        del training_data
        del training_generator
        del validation_generator
        del testing_data
        del testing_pred
        #del testing_pred_delineation        
        del train_pred
        #del train_pred_delineation
        del optimizer

        K.clear_session()
        gc.collect()
 
    print("#### Final: results #####")    
    print("Segmentation Results")
    print(np.mean(training_dice, axis = 0))
    print(np.std(training_dice, axis = 0))

    print(np.mean(testing_dice, axis = 0))
    print(np.std(testing_dice, axis = 0))

if __name__ == '__main__':
    main()
