
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError as mse
from tensorflow.keras.losses import BinaryCrossentropy as binaryCrossEntropy
from tensorflow.keras.losses import CategoricalCrossentropy as categoricalCrossEntropy

def negative_log_likelihood(y_pred, events):
 
    #y_pred = K.print_tensor(y_pred) 
    log_risk =  tf.math.log(tf.math.cumsum(tf.math.exp(y_pred)))
    #log_risk = K.print_tensor(log_risk)
    uncensored_likelihood = y_pred - log_risk
    censored_likelihood = uncensored_likelihood * events
    neg_likelihood = -tf.math.reduce_sum(censored_likelihood)/ tf.math.reduce_sum(events)

    return neg_likelihood


def negative_log_likelihood_2(y_pred_1, events_1, y_pred_2, events_2):
 
    neg_log_likelihood_1 = negative_log_likelihood(y_pred_1, events_1)
    neg_log_likelihood_2 = negative_log_likelihood(y_pred_2, events_2)

    return neg_log_likelihood_1 + neg_log_likelihood_2


def l2_loss_and_negative_log_likelihood(image, reconstruction, survival_pred, events):
 
    neg_log_likelihood = negative_log_likelihood(survival_pred, events)
    mse_loss = mse()
    return mse_loss(image, reconstruction) + neg_log_likelihood



def negativeLogLikelihood_2BinaryCrossEntropy(survival_pred, events, diseaseFailure, pred_diseaseFailure, pneumonitis, pred_pneumonitis):

    neg_log_likelihood = negative_log_likelihood(survival_pred, events)
    binary_cross_entropy_loss_1 = binaryCrossEntropy(name='binary_crossentropy_1')
    binary_cross_entropy_loss_2 = binaryCrossEntropy(name='binary_crossentropy_2')
    return binary_cross_entropy_loss_1(diseaseFailure, pred_diseaseFailure) +  binary_cross_entropy_loss_2(pneumonitis, pred_pneumonitis) + neg_log_likelihood


def negativeLogLikelihood_binaryCrossEntropy(survival_pred, events, pneumonitis, pred_pneumonitis):

    neg_log_likelihood = negative_log_likelihood(survival_pred, events)
    binary_cross_entropy_loss = binaryCrossEntropy()

    return binary_cross_entropy_loss(pneumonitis, pred_pneumonitis) + neg_log_likelihood


def generalized_dice_loss(y_true, y_pred):

    eps = 1e-6
   
    counts = tf.reduce_sum(y_true, axis = -1)
    weights = 1./(counts**2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis = -1)
    summed = tf.reduce_sum(y_true + y_pred, axis = -1)
    
    numerators = tf.reduce_sum(weights*multed)
    denominators = tf.reduce_sum(weights*summed)
    
    dices =  2. * numerators / denominators
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))

    return 1 - dices


def dice_oneVOI_loss(y_true, y_pred):

    eps = 1e-6
   
    y_VOI_true = y_true[:, :, :, :, 1]
    y_VOI_pred = y_pred[:, :, :, :, 1]

    numerator = tf.reduce_sum(y_VOI_true * y_VOI_pred)
    denominator = tf.reduce_sum(y_VOI_true + y_VOI_pred)
    
    dice =  2. * numerator / denominator
    dice = tf.where(tf.math.is_finite(dice), dice, tf.zeros_like(dice))

    return 1 - dice

def dice_oneVOI_categoricalCrossEntropy_loss(y_true, y_pred):
    
    dice = dice_oneVOI_loss(y_true, y_pred)
    cce_loss = categoricalCrossEntropy() 
    
    return dice + cce_loss(y_true, y_pred)

def negativeLogLikelihood_l2Loss_DiceLoss(image, reconstruction, survival_pred, events, segmentation, seg_pred):

    neg_log_likelihood = negative_log_likelihood(survival_pred, events)
    mse_loss = mse()
    dice = dice_oneVOI_loss(segmentation, seg_pred)

    return mse_loss(image, reconstruction) + neg_log_likelihood + dice

def negativeLogLikelihood_DiceLoss(image, survival_pred, events, segmentation, seg_pred):

    neg_log_likelihood = negative_log_likelihood(survival_pred, events)
    dice = dice_oneVOI_loss(segmentation, seg_pred)

    return  neg_log_likelihood + dice


