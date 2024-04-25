

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import _ni_support
from metrics.metrics2 import hd95, specificity, recall

def multi_hausdorff(y_true, y_pred, labels):
    n_labels = len(labels)
    value = np.zeros(n_labels, dtype=np.float32)
    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        value[i] = max(directed_hausdorff(yi_true, yi_pred),directed_hausdorff(yi_pred, yi_true))
    return value


def calc_multi_distance(y_true, y_pred, labels):
    n_labels = len(labels)
    dice_value = np.zeros(n_labels, dtype=np.float32)
    dh95_value = np.zeros(n_labels, dtype=np.float32)
    specificity_value = np.zeros(n_labels, dtype=np.float32)
    recall_value = np.zeros(n_labels, dtype=np.float32)
    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        dice_value[i] = dice_arrary(yi_pred, yi_true)
        dh95_value[i] = hd95(yi_pred, yi_true)
        specificity_value[i] = specificity(yi_pred, yi_true)
        recall_value[i] = recall(yi_pred, yi_true)
    return dice_value, dh95_value, specificity_value, recall_value


def dice(y_true, y_pred):

    y_true = np.array(y_true, dtype = np.float32)
    y_pred = np.array(y_pred, dtype = np.float32)
    epsilon = K.epsilon()
    
    numerator = np.sum(y_true * y_pred)
    denominator = np.sum(y_true + y_pred) 
    return (2. * numerator) / (denominator + epsilon)


def dice_multi_array(y_true, y_pred, labels):
    n_labels = len(labels)
    dice_value = np.zeros(n_labels, dtype=np.float32)

    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        dice_value[i] = dice(yi_true, yi_pred)

    return dice_value
    
def dice_oneVOI(y_true, y_pred):

    eps = 1e-6

    y_VOI_true = y_true[:, :, :, :, 1]
    y_VOI_pred = y_pred[:, :, :, :, 1]

    numerator = tf.reduce_sum(y_VOI_true * y_VOI_pred)
    denominator = tf.reduce_sum(y_VOI_true + y_VOI_pred)

    dice =  2. * numerator / denominator
    dice = tf.where(tf.math.is_finite(dice), dice, tf.zeros_like(dice))

    return dice


def weighted_cross_entropy(y_true, y_pred):
	
	nLabel = y_pred.get_shape().as_list()[-1]
	ndim = np.ndim(y_pred.get_shape().as_list())
	
	cross_entropy_aux = []
	w = 0.0
	
	for iLabel in range(nLabel):

		yi_true = tf.cast(y_true[:, :, :, :, iLabel], dtype = tf.float32)
		yi_pred = tf.cast(y_pred[:, :, :, :, iLabel], dtype = tf.float32)
		
		yi_true_sum = tf.reduce_sum(yi_true)
		
		wi = 1. / (yi_true_sum + K.epsilon())
		w += wi
		tmp = tf.multiply(yi_true, tf.math.log(yi_pred + K.epsilon()))
		tmp = tf.multiply(wi,tmp)
		tmp = tf.multiply(-1.,tmp)
		
		cross_entropy_aux.append(tmp)
			
	return tf.reduce_mean(tf.reduce_sum(cross_entropy_aux, ndim - 1)) / w 
	
	
def focal_loss(gamma=2.):

	gamma = float(gamma)
	
	def focal_loss_aux(y_true, y_pred):
		eps = 1.e-9
		y_true = tf.convert_to_tensor(y_true, tf.float32)
		y_pred = tf.convert_to_tensor(y_pred, tf.float32)
	
		model_out = tf.add(y_pred, eps)
		ce = tf.multiply(y_true, -tf.log(model_out))
		weight = tf.pow(tf.subtract(1., model_out), gamma)
		fl = tf.multiply(weight, ce)
		reduced_fl = tf.reduce_max(fl, axis=-1)
		return tf.reduce_mean(reduced_fl)
	
	return focal_loss_aux

def jaccard_distance_loss(y_true, y_pred, smooth=1e-5):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# def weighted_focal_loss(gamma=2., class_weight):
# 
# 	gamma = float(gamma)
# 	
# 	def focal_loss_aux(y_true, y_pred):
# 		eps = 1.e-9
# 		y_true = tf.convert_to_tensor(y_true, tf.float32)
# 		y_pred = tf.convert_to_tensor(y_pred, tf.float32)
# 	
# 		model_out = tf.add(y_pred, eps)
# 		ce = tf.multiply(y_true, -tf.log(model_out))
# 		weight = tf.pow(tf.subtract(1., model_out), gamma)
# 		fl = tf.multiply(weight, ce)
# 		reduced_fl = tf.reduce_max(fl, axis=-1)
# 		
# 		index_fl = tf.argmax(fl, axis = -1)
# 
# 		return tf.reduce_mean(reduced_fl)
# 	
# 	return focal_loss_aux	

def volumeAbsoluteError_multi_array(y_true, y_pred, labels, voxelspacing):

    n_labels = len(labels)
    
    value = np.zeros(n_labels, dtype=np.float32)
    
    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        
        #Compute the volume for the ith segmentation
        voxelVolume = 1.
        for iSpacing in range(len(voxelspacing)):
        	voxelVolume *= voxelspacing[iSpacing]
        	
        volume1 = np.sum(yi_true) * voxelVolume
        volume2 = np.sum(yi_pred) * voxelVolume
        
        value[i] = abs(volume1 - volume2)

		        
         
    return value
    
def volumeRelativeError_multi_array(y_true, y_pred, labels, voxelspacing):

    n_labels = len(labels)
    
    value = np.zeros(n_labels, dtype=np.float32)
    
    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        
        #Compute the volume for the ith segmentation
        voxelVolume = 1.
        for iSpacing in range(len(voxelspacing)):
        	voxelVolume *= voxelspacing[iSpacing]
        	
        volume1 = np.sum(yi_true) * voxelVolume
        volume2 = np.sum(yi_pred) * voxelVolume
        
        value[i] = abs(volume1 - volume2) / volume1

		        
         
    return value
    

def volume_multi_array(y, labels, voxelspacing):

    n_labels = len(labels)
    
    value = np.zeros(n_labels, dtype=np.float32)
    
    for i in range(n_labels):
        yi = (y == labels[i])
         
        #Compute the volume for the ith segmentation
        voxelVolume = 1.
        for iSpacing in range(len(voxelspacing)):
        	voxelVolume *= voxelspacing[iSpacing]
        	
        volume = np.sum(yi) * voxelVolume
        
        value[i] = volume
	        
         
    return value


def MAE(y_pred, y_true):

        return np.mean(np.absolute(y_pred - y_true))

def ME(y_pred, y_true):

        return np.mean(y_pred - y_true)


