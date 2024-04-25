
from tensorflow.keras.layers import Conv3D, Conv1D, MaxPool3D, concatenate, Input, Dropout, PReLU, ReLU,  Conv3DTranspose, BatchNormalization, SpatialDropout3D, SpatialDropout2D, Flatten, Dense, Conv2D, MaxPooling2D, MaxPooling3D, Add, GlobalAveragePooling2D, GlobalAveragePooling3D, Lambda, GlobalMaxPooling2D, GlobalMaxPooling3D, Activation, Multiply, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras import regularizers
# from keras_contrib.layers import CRF
import tensorflow.keras as K
import tensorflow as tf
from tensorflow_addons.activations import mish
from tensorflow_addons.layers import GroupNormalization
from loss.loss import negative_log_likelihood, negativeLogLikelihood_2BinaryCrossEntropy, negativeLogLikelihood_binaryCrossEntropy

 
def conv_block(x, filter_size=8, kernel_size=(3, 3, 3)):

    x = Conv3D(filters=filter_size, kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)

    x = Conv3D(filters=filter_size,kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)

    return x



def CNN3DModel(input_shape):

	images = Input(shape = input_shape)
	
	x = conv_block(images, filter_size = 16, kernel_size = (3, 3, 3)) #128
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = conv_block(x, filter_size = 32, kernel_size = (3, 3, 3)) #64
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = conv_block(x, filter_size = 64, kernel_size = (3, 3, 3)) #32
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = conv_block(x, filter_size = 128, kernel_size = (3, 3, 3)) #16
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = conv_block(x, filter_size = 256, kernel_size = (3, 3, 3)) #8

	x = GlobalAveragePooling3D()(x)

	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)
 
	x = Dense(32, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)

	output = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform')(x)

	model = Model(inputs = images, outputs = output, name = 'Conv3DModel')
	
	return model

