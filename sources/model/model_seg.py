
from tensorflow.keras.layers import Conv3D, Conv1D, MaxPool3D, concatenate, Input, Dropout, PReLU, ReLU,  Conv3DTranspose, BatchNormalization, SpatialDropout3D, SpatialDropout2D, Flatten, Dense, Conv2D, MaxPooling2D, MaxPooling3D, Add, GlobalAveragePooling2D, GlobalAveragePooling3D, Lambda, GlobalMaxPooling2D, GlobalMaxPooling3D, Activation, Multiply, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras import regularizers
# from keras_contrib.layers import CRF
import tensorflow.keras as K
import tensorflow as tf
from tensorflow_addons.activations import mish
from loss.loss import negative_log_likelihood, negative_log_likelihood_2, l2_loss_and_negative_log_likelihood, negativeLogLikelihood_l2Loss_DiceLoss, negativeLogLikelihood_DiceLoss


def unet_core(x, filter_size=8, kernel_size=(3, 3, 3)):

    x = Conv3D(filters=filter_size, kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)

    x = Conv3D(filters=filter_size,kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)
    
    return x

def UNet3D(input_shape_1, n_label):

    images = Input(shape = input_shape_1)


    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    #e11 = Conv3D(filters=16, kernel_size=(3,3,3), strides = (2,2,2), padding ='same')(e1)
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    #e21 = Conv3D(filters=32, kernel_size=(3,3,3), strides = (2,2,2), padding ='same')(e2)
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size = 64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    #e31 = Conv3D(filters=64, kernel_size=(3,3,3), strides = (2,2,2), padding ='same')(e3)
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    #e41 = Conv3D(filters=128, kernel_size=(3,3,3), strides = (2,2,2), padding ='same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)
    
    e = unet_core(e41, filter_size = 256, kernel_size=(3, 3, 3))
   
    # Segmentation part 
    s = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    s = concatenate([s, e4], axis=-1)
    s = unet_core(s, filter_size=128, kernel_size=(3, 3, 3))

    s = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(s)
    s = concatenate([s, e3], axis=-1)
    s = unet_core(s, filter_size=64, kernel_size=(3, 3, 3))

    s = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(s)
    s = concatenate([s, e2], axis=-1)
    s = unet_core(s, filter_size=32, kernel_size=(3, 3, 3))

    s = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(s)
    s = concatenate([s, e1], axis=-1)
    s = unet_core(s, filter_size=16, kernel_size=(3, 3, 3))

    output_segmentation = Conv3D(filters = n_label, kernel_size=(1, 1, 1), activation = 'softmax')(s)

    model = Model(inputs = images, outputs = output_segmentation, name = '3D_U-Net')

    return model


