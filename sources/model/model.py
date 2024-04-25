
from tensorflow.keras.layers import Conv3D, Conv1D, MaxPool3D, concatenate, Input, Dropout, PReLU, ReLU,  Conv3DTranspose, BatchNormalization, SpatialDropout3D, SpatialDropout2D, Flatten, Dense, Conv2D, MaxPooling2D, MaxPooling3D, Add, GlobalAveragePooling2D, GlobalAveragePooling3D, Lambda, GlobalMaxPooling2D, GlobalMaxPooling3D, Activation, Multiply, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras import regularizers
# from keras_contrib.layers import CRF
import tensorflow.keras as K
import tensorflow as tf
from tensorflow_addons.activations import mish
from loss.loss import negative_log_likelihood, negative_log_likelihood_2, l2_loss_and_negative_log_likelihood, negativeLogLikelihood_l2Loss_DiceLoss

def share_model():
        
	model = K.Sequential()
	model.add(Dense(64, bias_initializer = 'glorot_uniform', activation = mish))
	model.add(Dropout(0.3))

	model.add(Dense(32, bias_initializer = 'glorot_uniform', activation = mish))
	model.add(Dropout(0.3))

	#model.add(Dense(16, bias_initializer = 'glorot_uniform', activation = mish))
	#model.add(Dropout(0.3))
	
	return model


def DeepSurv_OS_PFS(input_shape1, input_shape2, input_shape3, input_shape4):

	clinics_OS = Input(shape = input_shape1)
	events_OS = Input(shape = input_shape2)
	clinics_PFS = Input(shape = input_shape3)
	events_PFS = Input(shape = input_shape4) 
	
	### Shared model feature computation
	my_share_model = share_model()
	x1 = my_share_model(clinics_OS)
	x2 = my_share_model(clinics_PFS)
 	
	# OS Independant feature computation
	x1 = Dense(16, bias_initializer = 'glorot_uniform' ) (x1)
	x1 = mish(x1)
	x1 = Dropout(0.3)(x1)
	
	#x1 = Dense(8, bias_initializer = 'glorot_uniform' ) (x1)
	#x1 = mish(x1)


	output_OS = Dense(1, activation = 'linear',  bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(1e-3))(x1)
 
	# PFS Independant feature computation
	x2 = Dense(16, bias_initializer = 'glorot_uniform' ) (x2)
	x2 = mish(x2)
	x2 = Dropout(0.3)(x2)

	#x2 = Dense(8, bias_initializer = 'glorot_uniform' ) (x2)
	#x2 = mish(x2)

	output_PFS = Dense(1, activation = 'linear',  bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(1e-3))(x2)

	model = Model(inputs = [clinics_OS, events_OS, clinics_PFS, events_PFS], outputs = [output_OS, output_PFS], name = 'DeepSurv_OS_PFS')
	model.add_loss(negative_log_likelihood_2(output_OS, events_OS, output_PFS, events_PFS))
	
	return model


def DeepSurv_OS_PFS_testing(input_shape1):

	clinics = Input(shape = input_shape1)
	
	### Shared model feature computation
	my_share_model = share_model()
	x = my_share_model(clinics)
	#x2 = my_share_model(clinics)
 	
	# OS Independant feature computation
	x1 = Dense(16, bias_initializer = 'glorot_uniform' ) (x)
	x1 = mish(x1)
	x1 = Dropout(0.3)(x1)

	#x1 = Dense(8, bias_initializer = 'glorot_uniform' ) (x1)
	#x1 = mish(x1)

	output_OS = Dense(1, activation = 'linear',  bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(1e-3))(x1)
 
	# PFS Independant feature computation
	x2 = Dense(16, bias_initializer = 'glorot_uniform' ) (x)
	x2 = mish(x2)
	x2 = Dropout(0.3)(x2)

	#x2 = Dense(8, bias_initializer = 'glorot_uniform' ) (x2)
	#x2 = mish(x2)

	output_PFS = Dense(1, activation = 'linear',  bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(1e-3))(x2)

	model = Model(inputs = clinics, outputs = [output_OS, output_PFS], name = 'DeepSurv_OS_PFS')

	return model


def DeepConvSurv_Attention(input_shape1, input_shape2):

        images = Input(shape = input_shape1)
        events = Input(shape = input_shape2)

        x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "0", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "1", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "2", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "3", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.3)(x)

        x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
        #x = spatialAttention3D(x, "4", (7, 7, 7))
        x = mish(x)

        x = GlobalAveragePooling3D()(x)
        #x = BatchNormalization()(x)
        #x = mish(x)

#       x = Flatten()(x)
        x = Dense(64, bias_initializer = 'glorot_uniform')(x)
        x = mish(x)

        output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

        model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
        model.add_loss(negative_log_likelihood(output, events))
        
        return model


def DeepConvSurv_Attention_testing(input_shape1):

        images = Input(shape = input_shape1)

        x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "0", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "1", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "2", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
        #x = BatchNormalization()(x)
        x = spatialAttention3D(x, "3", (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.3)(x)

        x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
        #x = spatialAttention3D(x, "4", (7, 7, 7))
        x = mish(x)

        x = GlobalAveragePooling3D()(x)
        #x = BatchNormalization()(x)
        #x = mish(x)

#       x = Flatten()(x)
        x = Dense(64, bias_initializer = 'glorot_uniform')(x)
        x = mish(x)

        output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

        model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')

        return model



def DeepConvSurv_CBAM(input_shape1, input_shape2):

        images = Input(shape = input_shape1)
        events = Input(shape = input_shape2)

        x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
        #x = BatchNormalization()(x)
        x = cbam3D(x, "0", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
        #x = BatchNormalization()(x)
        x = cbam3D(x, "1", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
        #x = BatchNormalization()(x)
        x = cbam3D(x, "2", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
        #x = BatchNormalization()(x)
        x = cbam3D(x, "3", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.3)(x)

        x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
        #x = spatialAttention3D(x, "4", (7, 7, 7))
        x = mish(x)

        x = GlobalAveragePooling3D()(x)
        #x = BatchNormalization()(x)
        #x = mish(x)

#       x = Flatten()(x)
        x = Dense(64, bias_initializer = 'glorot_uniform')(x)
        x = mish(x)

        output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

        model = Model(inputs = [images, events], outputs = output, name = 'DeepConvSurv')
        model.add_loss(negative_log_likelihood(output, events))
        
        return model


def DeepConvSurv_CBAM_testing(input_shape1):

        images = Input(shape = input_shape1)

        x = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images) #128
        #x = BatchNormalization()(x)
        x = cbam3D(x, "0", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #64
        #x = BatchNormalization()(x)
        x = cbam3D(x, "1", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #32
        #x = BatchNormalization()(x)
        x = cbam3D(x, "2", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.2)(x)

        x = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #16
        #x = BatchNormalization()(x)
        x = cbam3D(x, "3", 2, (7, 7, 7))
        x = mish(x)
        x = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x)
        x = SpatialDropout3D(0.3)(x)

        x = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x) #8
        #x = spatialAttention3D(x, "4", (7, 7, 7))
        x = mish(x)

        x = GlobalAveragePooling3D()(x)
        #x = BatchNormalization()(x)
        #x = mish(x)

#       x = Flatten()(x)
        x = Dense(64, bias_initializer = 'glorot_uniform')(x)
        x = mish(x)

        output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

        model = Model(inputs = images, outputs = output, name = 'DeepConvSurv')

        return model



#Attention module 	
def channelAttention3D(x, ratio):

	#Get the size of the channel dimension for the input layer
	channel = x.shape[-1]
	
	#Compute average pooling in the channel dimension
	avgPool = GlobalAveragePooling3D(keepdims = "True")(x)
	
	#Compute max pooling in the channel dimension
	maxPool = GlobalMaxPooling3D(keepdims = "True")(x)
	
	#Build the shared multi-layer perceptron
	shared_MLP1 = Dense(channel // ratio)
	shared_MLP2 = mish
	shared_MLP3 = Dense(channel)
	
	#Apply the shared multi-layer perceptron in the avgPool and maxPool
	avgPool = shared_MLP3(shared_MLP2(shared_MLP1(avgPool)))
# 	avgPool = BatchNormalization()(avgPool)
	
	maxPool = shared_MLP3(shared_MLP2(shared_MLP1(maxPool)))
# 	maxPool = BatchNormalization()(maxPool)
	
	feature = Add()([avgPool, maxPool])
	feature = Activation("sigmoid")(feature)
	
	return Multiply()([feature, x])
	
	
def ECA(x, k_size = 3):
	
	#Compute average pooling in the channel dimension
	avgPool = GlobalAveragePooling2D(keepdims = "True")(x)
	
	feature = Conv1D(filters = 1, kernel_size = k_size, strides = 1, padding = "same", use_bias = False)(avgPool)
	feature = Activation("sigmoid")(feature)
	
	return Multiply()([feature, x])
	
def spatialAttention3D(x, nModule, kernel_size_spatialAttention):
	
	#Compute the average pooling in the channel dimension
	avgPool = tf.reduce_mean(x, axis = -1, keepdims = True)
	
	#Compute the max pooling in the channel dimension
	maxPool = tf.reduce_max(x, axis = -1, keepdims = True)
	
	concat = Concatenate(axis = -1)([avgPool, maxPool])
	
	feature = Conv3D(filters = 1, kernel_size = kernel_size_spatialAttention, strides = 1, padding = "same", use_bias=False)(concat)
	feature = BatchNormalization()(feature)
	feature = Activation("sigmoid", name = "Attention_map_" + str(nModule))(feature)
	
	return Multiply()([feature, x])
	
	
def cbam3D(x, nModule = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):

	x = channelAttention3D(x, ratio)
	x = spatialAttention3D(x, nModule, kernel_size_spatialAttention)
	
	return x
	
def cbam_parallel_block(x, nModule = 0, ratio = 2, kernel_size_spatialAttention = (7,7)):
	
	#### Compute channel attention ######
	
	#Get the size of the channel dimension for the input layer
	channel = x.shape[-1]
	
	#Compute average pooling in the channel dimension
	avgPool_channelAttention = GlobalAveragePooling2D(keepdims = "True")(x)
	
	#Compute max pooling in the channel dimension
	maxPool_channelAttention = GlobalMaxPooling2D(keepdims = "True")(x)
	
	#Build the shared multi-layer perceptron
	shared_MLP1 = Dense(channel // ratio)
	shared_MLP2 = mish
	shared_MLP3 = Dense(channel)
	
	#Apply the shared multi-layer perceptron in the avgPool and maxPool
	avgPool_channelAttention = shared_MLP3(shared_MLP2(shared_MLP1(avgPool_channelAttention)))
# 	avgPool = BatchNormalization()(avgPool)
	
	maxPool_channelAttention = shared_MLP3(shared_MLP2(shared_MLP1(maxPool_channelAttention)))
# 	maxPool = BatchNormalization()(maxPool)
	
	feature_channelAttention = Add()([avgPool_channelAttention, maxPool_channelAttention])
	feature_channelAttention = Activation("sigmoid")(feature_channelAttention)
	
	
	### Compute the attention mechanism ###
	
	#Compute the average pooling in the channel dimension
	avgPool_attentionlAttention = tf.reduce_mean(x, axis = -1, keepdims = True)
	
	#Compute the max pooling in the channel dimension
	maxPool_attentionlAttention = tf.reduce_max(x, axis = -1, keepdims = True)
	
	concat = Concatenate(axis = -1)([avgPool_attentionlAttention, maxPool_attentionlAttention])
	
	feature_attentionlAttention = Conv2D(filters = 1, kernel_size = kernel_size_spatialAttention, strides = 1, padding = "same", use_bias=False)(concat)
	feature_attentionlAttention = BatchNormalization()(feature_attentionlAttention)
	feature_attentionlAttention = Activation("sigmoid", name = "Attention_map_" + str(nModule))(feature_attentionlAttention)

	### Apply the channel and attention mechanism ####
	
	x = Multiply()([feature_channelAttention, x])
	x = Multiply()([feature_attentionlAttention, x])
	
	return x
	

    
 
def res_block_DenseLayer(x, filter_size):

    x1 = Dense(filter_size, bias_initializer = 'glorot_uniform' )(x)
    #x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    x1 = Dense(filter_size, bias_initializer = 'glorot_uniform' ) (x1)
    #x1 = BatchNormalization()(x1)
    x1 = mish(x1)

    #Skip connection
    x  = Dense(filter_size, bias_initializer = 'glorot_uniform' )(x)
    x = Add()([x1, x])

    return x



