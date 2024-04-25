
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

def DeepConvSurv(input_shape1, input_shape2):

	images = Input(shape = input_shape1)
	events = Input(shape = input_shape2)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	#x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	#x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
	#x = BatchNormalization()(x)
	#x = mish(x)
    
#	x = Flatten()(x)
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model

	
def DeepConvSurv_testing(input_shape):

	images = Input(shape = input_shape)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	#x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)
	
	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	#x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	#x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x) #16
	#x = BatchNormalization()(x)
	#x = mish(x)
    
	#x = Flatten()(x)
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')
	
	return model




def DeepResConvSurv2(input_shape1, input_shape2):

	images = Input(shape = input_shape1)
	events = Input(shape = input_shape2)

	x = ResBlock3D(images, filter_size = 8, kernel_size = (3, 3, 3), groupNumber = 4, dropout_ratio = 0.2)
	gap1 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 16, kernel_size = (3, 3, 3), groupNumber = 4, dropout_ratio = 0.2) #64
	gap2 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 32, kernel_size = (3, 3, 3), groupNumber = 8, dropout_ratio = 0.2) #32
	gap3 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 64, kernel_size = (3, 3, 3), groupNumber = 16, dropout_ratio = 0.3) #16
	gap4 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 128, kernel_size = (3, 3, 3), groupNumber = 32, dropout_ratio = 0.)  #8
	gap5 = GlobalAveragePooling3D()(x)
	
	x = Concatenate(axis = 1)([gap1, gap2, gap3, gap4, gap5])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model


def DeepResConvSurv2_testing(input_shape1):

	images = Input(shape = input_shape1)
	
	x = ResBlock3D(images, filter_size = 8, kernel_size = (3, 3, 3), groupNumber = 4, dropout_ratio = 0.2)
	gap1 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 16, kernel_size = (3, 3, 3), groupNumber = 4, dropout_ratio = 0.2) #64
	gap2 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 32, kernel_size = (3, 3, 3), groupNumber = 8, dropout_ratio = 0.2) #32
	gap3 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 64, kernel_size = (3, 3, 3), groupNumber = 16, dropout_ratio = 0.3) #16
	gap4 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)

	x = ResBlock3D(x, filter_size = 128, kernel_size = (3, 3, 3), groupNumber = 32, dropout_ratio = 0.)  #8
	gap5 = GlobalAveragePooling3D()(x)

	
	x = Concatenate(axis = 1)([gap1, gap2, gap3, gap4, gap5])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')

	return model

# To Do: Retrun the Global Average Pooling Layer for each block

def ResBlock3D(x, filter_size = 8, kernel_size = (3, 3, 3), groupNumber = 32, dropout_ratio = 0.3):

    x1 = Conv3D(filters = filter_size, kernel_size = kernel_size, padding ='same')(x)
    x1 = GroupNormalization(groupNumber)(x1)
    x1 = mish(x1)
    x1 = SpatialDropout3D(dropout_ratio)(x1)
    
    x1 = Conv3D(filters = filter_size, kernel_size = kernel_size, padding='same')(x1)
    x1 = GroupNormalization(groupNumber)(x1)
    x1 = mish(x1)


    # Skip connection
    x = Conv3D(filters = filter_size, kernel_size = (1, 1, 1), padding='same')(x)
    x = GroupNormalization(groupNumber)(x)
    x = mish(x)

    x = Add()([x1, x])

    return x

