
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
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)

	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model

	
def DeepConvSurv_testing(input_shape):

	images = Input(shape = input_shape)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)
	
	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x) #16

	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')
	
	return model


def DeepConvSurv_imaging_dose(input_shape1, input_shape2, input_shape3):

	images = Input(shape = input_shape1)
	doses = Input(shape = input_shape2)
	events = Input(shape = input_shape3)

	# Compute imaging features
	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)

	# Compute dosimetric features
	d = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #64
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #32
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #16
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #8
	d = BatchNormalization()(d)
	d = mish(d)

	d = GlobalAveragePooling3D()(d)

        # Fuse imaging and dosimetric features
	x = Concatenate(axis = 1)([x, d])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, doses, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model

def DeepConvSurv_imaging_dose_testing(input_shape1, input_shape2):

	images = Input(shape = input_shape1)
	doses = Input(shape = input_shape2)
	
	# Compute imaging features
	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)

	# Compute dosimetric features
	d = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #64
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #32
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #16
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #8
	d = BatchNormalization()(d)
	d = mish(d)

	d = GlobalAveragePooling3D()(d)

        # Fuse imaging and dosimetric features
	x = Concatenate(axis = 1)([x, d])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, doses], outputs = output, name = 'DeepConvSurv')
	
	return model


def DeepConvSurv_imaging_dose_clinics(input_shape1, input_shape2, input_shape3, input_shape4):

	images = Input(shape = input_shape1)
	doses = Input(shape = input_shape2)
	clinics = Input(shape = input_shape3)
	events = Input(shape = input_shape4)

	# Compute imaging features
	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
	x = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)



	# Compute dosimetric features
	d = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #64
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #32
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #16
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #8
	d = BatchNormalization()(d)
	d = mish(d)

	d = GlobalAveragePooling3D()(d)
	d = Dense(input_shape3, bias_initializer = 'glorot_uniform')(d)
	d = mish(d)


        # Fuse imaging and dosimetric features
	x = Concatenate(axis = 1)([x, d, clinics])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)
	x = Dropout(0.3)(x)

	x = Dense(32, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, doses, clinics, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model


def DeepConvSurv_imaging_dose_clinics_testing(input_shape1, input_shape2, input_shape3):

	images = Input(shape = input_shape1)
	doses = Input(shape = input_shape2)
	clinics = Input(shape = input_shape3)

	# Compute imaging features
	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
	x = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	# Compute dosimetric features
	d = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #64
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #32
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #16
	d = BatchNormalization()(d)
	d = mish(d)
	d = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(d)
	d = SpatialDropout3D(0.3)(d)

	d = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(d) #8
	d = BatchNormalization()(d)
	d = mish(d)

	d = GlobalAveragePooling3D()(d)
	d = Dense(input_shape3, bias_initializer = 'glorot_uniform')(d)
	d = mish(d)

        # Fuse imaging and dosimetric features
	x = Concatenate(axis = 1)([x, d, clinics])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)
	x = Dropout(0.3)(x)

	x = Dense(32, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, doses, clinics], outputs = output, name = 'DeepConvSurv')
	
	return model



def DeepConvSurv2(input_shape1, input_shape2):

	images = Input(shape = input_shape1)
	events = Input(shape = input_shape2)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = GroupNormalization(2)(x)
	x = mish(x)	
	gap1 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)


	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = GroupNormalization(4)(x)
	x = mish(x)
	gap2 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = GroupNormalization(8)(x)
	x = mish(x)
	gap3 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = GroupNormalization(16)(x)
	x = mish(x)
	gap4 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = GroupNormalization(32)(x)
	x = mish(x)
	gap5 = GlobalAveragePooling3D()(x)

	
	x = Concatenate(axis = 1)([gap1, gap2, gap3, gap4, gap5])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model


def DeepConvSurv2_testing(input_shape1):

	images = Input(shape = input_shape1)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = GroupNormalization(2)(x)
	x = mish(x)
	gap1 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)


	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = GroupNormalization(4)(x)
	x = mish(x)
	gap2 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = GroupNormalization(8)(x)
	x = mish(x)
	gap3 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = GroupNormalization(16)(x)
	x = mish(x)
	gap4 = GlobalAveragePooling3D()(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = GroupNormalization(32)(x)
	x = mish(x)
	gap5 = GlobalAveragePooling3D()(x)

	
	x = Concatenate(axis = 1)([gap1, gap2, gap3, gap4, gap5])
	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')

	return model


def DeepConvSurv_imaging_clinics(input_shape1, input_shape2, input_shape3):

	images = Input(shape = input_shape1)
	clinics = Input(shape = input_shape2)
	events = Input(shape = input_shape3)
        
	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D(keepdims = False)(x)
	x = Dense(input_shape2, bias_initializer = 'glorot_uniform')(x)
	x = mish(x) 
	
	# Fusion of DL features and clinical data
	x = Concatenate()([x, clinics])

	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)
	x = Dropout(0.3)(x)

	x = Dense(32, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, clinics, events], outputs = output, name = 'DeepConvSurv_imaging_clinics')
	model.add_loss(negative_log_likelihood(output, events))
	
	return model


def DeepConvSurv_imaging_clinics_testing(input_shape1, input_shape2):

	images = Input(shape = input_shape1)
	clinics = Input(shape = input_shape2)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	x = BatchNormalization()(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	x = BatchNormalization()(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	x = BatchNormalization()(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
	x = Dense(input_shape2, bias_initializer = 'glorot_uniform')(x)
	x = mish(x)

	# Fusion of DL features and clinical data
	x = Concatenate()([x, clinics])

	x = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)
	x = Dropout(0.3)(x)

	x = Dense(32, bias_initializer = 'glorot_uniform')(x)
	x = BatchNormalization()(x)
	x = mish(x)

	output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

	model = Model(inputs = [images, clinics], outputs = output, name = 'DeepConvSurv_imaging_clinics')

	return model

       
def DeepConvSurv_OS_diseaseFailure_pneumonitis(input_shape1, input_shape2, input_shape3, input_shape4):

	images = Input(shape = input_shape1)
	events = Input(shape = input_shape2)
	diseaseFailure = Input(shape = input_shape3)
	pneumonitis = Input(shape = input_shape4)

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
    
	x1 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x1 = mish(x1)
	output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x1)

	x2 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x2 = mish(x2)
	output_diseaseFailure = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x2)

	x3 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x3 = mish(x3)
	output_pneumonitis = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x3)


	model = Model(inputs = [images, events, diseaseFailure, pneumonitis], outputs = [output_OS, output_diseaseFailure, output_pneumonitis], name = 'DeepConvSurv_multi_task')
	model.add_loss(negativeLogLikelihood_2BinaryCrossEntropy(output_OS, events, diseaseFailure, output_diseaseFailure, pneumonitis, output_pneumonitis))
	
	return model


def DeepConvSurv_OS_diseaseFailure_pneumonitis_testing(input_shape1):

	images = Input(shape = input_shape1)

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
    
	x1 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x1 = mish(x1)
	output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x1)


	x2 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x2 = mish(x2)
	output_diseaseFailure = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x2)

	x3 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x3 = mish(x3)
	output_pneumonitis = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x3)


	model = Model(inputs = images, outputs = [output_OS, output_diseaseFailure, output_pneumonitis], name = 'DeepConvSurv_multi_task')

	return model


def DeepConvSurv_OS_oneClassificationTask(input_shape1, input_shape2, input_shape3):

	images = Input(shape = input_shape1)
	events = Input(shape = input_shape2)
	classificationTask = Input(shape = input_shape3)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	#x = GroupNormalization(4)(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	#x = GroupNormalization(8)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	#x = GroupNormalization(16)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	#x = GroupNormalization(32)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	#x = GroupNormalization(64)(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
    
	x1 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x1 = mish(x1)
	output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x1)


	x2 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x2 = mish(x2)
	output_classificationTask = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x2)


	model = Model(inputs = [images, events, classificationTask], outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_multi_task')
	model.add_loss(negativeLogLikelihood_binaryCrossEntropy(output_OS, events, classificationTask, output_classificationTask))
	
	return model


def DeepConvSurv_OS_oneClassificationTask_testing(input_shape1):

	images = Input(shape = input_shape1)

	x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
	#x = GroupNormalization(4)(x)
	x = mish(x)	
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
	#x = GroupNormalization(8)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
	#x = GroupNormalization(16)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.2)(x)

	x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
	#x = GroupNormalization(32)(x)
	x = mish(x)
	x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
	x = SpatialDropout3D(0.3)(x)

	x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
	#x = GroupNormalization(64)(x)
	x = mish(x)

	x = GlobalAveragePooling3D()(x)
    
	x1 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x1 = mish(x1)
	output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x1)


	x2 = Dense(64, bias_initializer = 'glorot_uniform')(x)
	x2 = mish(x2)
	output_classificationTask = Dense(1, bias_initializer = 'glorot_uniform', activation = 'sigmoid', kernel_regularizer = regularizers.L2(0.001))(x2)


	model = Model(inputs = images, outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_multi_task')

	return model


