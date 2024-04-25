
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

    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)

    x = Conv3D(filters=filter_size,kernel_size=kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = mish(x)

    return x


def DeepConvSurv_autoEncoder(input_shape_1, input_shape_2):

    images = Input(shape = input_shape_1)
    events = Input(shape = input_shape_2)

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, events], outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(l2_loss_and_negative_log_likelihood(images, output_reconstruction, output_survival, events))


    return model

def DeepConvSurv_autoEncoder_testing(input_shape_1):

    images = Input(shape = input_shape_1)

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = images, outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')

    return model



def DeepConvSurv_autoEncoder_seg(input_shape_1, input_shape_2, input_shape_3, n_label):

    images = Input(shape = input_shape_1)
    events = Input(shape = input_shape_2)
    segmentations = Input(shape = input_shape_3)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)

    #Segmentation part
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


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, events, segmentations], outputs = [output_survival, output_reconstruction, output_segmentation], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(negativeLogLikelihood_l2Loss_DiceLoss(images, output_reconstruction, output_survival, events, segmentations, output_segmentation))


    return model


def DeepConvSurv_autoEncoder_seg_testing(input_shape_1, n_label):

    images = Input(shape = input_shape_1)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)

    #Segmentation part
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


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = images, outputs = [output_survival, output_reconstruction, output_segmentation], name = 'DeepConvSurv_autoEncoder')
  
    return model


def DeepConvSurv_seg(input_shape_1, input_shape_2, input_shape_3, n_label):

    images = Input(shape = input_shape_1)
    events = Input(shape = input_shape_2)
    segmentations = Input(shape = input_shape_3)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = unet_core(e41, filter_size = 256, kernel_size = (3, 3, 3)) #8
    
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

    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, events, segmentations], outputs = [output_survival, output_segmentation], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(negativeLogLikelihood_DiceLoss(images, output_survival, events, segmentations, output_segmentation))

    return model


def DeepConvSurv_seg_testing(input_shape_1, n_label):

    images = Input(shape = input_shape_1)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = unet_core(e41, filter_size = 256, kernel_size = (3, 3, 3)) #8

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

    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = images, outputs = [output_survival, output_segmentation], name = 'DeepConvSurv_autoEncoder')

    return model


def DeepConvSurv_seg_imaging_clinics(input_shape_1, input_shape_2, input_shape_3, input_shape_4, n_label):

    images = Input(shape = input_shape_1)
    events = Input(shape = input_shape_2)
    segmentations = Input(shape = input_shape_3)
    clinics = Input(shape = input_shape_4)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)

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

    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(input_shape_4, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, events, segmentations, clinics], outputs = [output_survival, output_segmentation], name = 'DeepConvSurv_Seg')
    model.add_loss(negativeLogLikelihood_DiceLoss(images, output_survival, events, segmentations, output_segmentation))

    return model


def DeepConvSurv_seg_imaging_clinics_testing(input_shape_1, input_shape_2, n_label):

    images = Input(shape = input_shape_1)
    clinics = Input(shape = input_shape_2)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)

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

    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(input_shape_2, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, clinics], outputs = [output_survival, output_segmentation], name = 'DeepConvSurv_Seg')

    return model


def DeepConvSurv_autoEncoder_imaging_clinics(input_shape_1, input_shape_2, input_shape_3):

    images = Input(shape = input_shape_1)
    clinics = Input(shape = input_shape_2)
    events = Input(shape = input_shape_3)

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Survival prediction part
    x = GlobalAveragePooling3D(keepdims = False)(e)
    x = Dense(input_shape_2, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, clinics, events], outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(l2_loss_and_negative_log_likelihood(images, output_reconstruction, output_survival, events))


    return model


def DeepConvSurv_autoEncoder_imaging_clinics_testing(input_shape_1, input_shape_2):

    images = Input(shape = input_shape_1)
    clinics = Input(shape = input_shape_2)

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Survival prediction part
    x = GlobalAveragePooling3D(keepdims = False)(e)
    x = Dense(input_shape_2, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, clinics], outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')

    return model


def DeepConvSurv_autoEncoder_seg_imaging_clinics(input_shape_1, input_shape_2, input_shape_3, input_shape_4, n_label):

    images = Input(shape = input_shape_1)
    events = Input(shape = input_shape_2)
    segmentations = Input(shape = input_shape_3)
    clinics = Input(shape = input_shape_4)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)

    #Segmentation part
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


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(input_shape_4, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)


    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, events, segmentations, clinics], outputs = [output_survival, output_reconstruction, output_segmentation], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(negativeLogLikelihood_l2Loss_DiceLoss(images, output_reconstruction, output_survival, events, segmentations, output_segmentation))


    return model


def DeepConvSurv_autoEncoder_seg_imaging_clinics_testing(input_shape_1, input_shape_2, n_label):

    images = Input(shape = input_shape_1)
    clinics = Input(shape = input_shape_2)

    # Encoder part
    e1 = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e11 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e1) #64
    e11 = SpatialDropout3D(0.2)(e11)

    e2 = unet_core(e11, filter_size= 32, kernel_size=(3, 3, 3))
    e21 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e2) #32
    e21 = SpatialDropout3D(0.2)(e21)

    e3 = unet_core(e21, filter_size=64, kernel_size=(3, 3, 3))
    e31 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e3) #16
    e31 = SpatialDropout3D(0.2)(e31)

    e4 = unet_core(e31, filter_size = 128, kernel_size=(3, 3, 3))
    e41 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e4)
    e41 = SpatialDropout3D(0.3)(e41)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e41) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)

    #Segmentation part
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


    # Survival prediction part
    x = GlobalAveragePooling3D()(e)
    x = Dense(input_shape_4, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, clinics], outputs = [output_survival, output_reconstruction, output_segmentation], name = 'DeepConvSurv_autoEncoder')

    return model



def DeepConvSurv_autoEncoder_imaging_dose_clinics(input_shape_1, input_shape_2, input_shape_3, input_shape_4):

    images = Input(shape = input_shape_1)
    doses =  Input(shape = input_shape_2)
    clinics = Input(shape = input_shape_3)
    events = Input(shape = input_shape_4)

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Compute dosimetric features
    do = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #64
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #32
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #16
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #8
    do = BatchNormalization()(do)
    do = mish(do)

    do = GlobalAveragePooling3D()(do)
    do = Dense(input_shape_3, bias_initializer = 'glorot_uniform')(do)
    do = mish(do)


    # Survival prediction part
    x = GlobalAveragePooling3D(keepdims = False)(e)
    x = Dense(input_shape_3, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, do, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, doses, clinics, events], outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')
    model.add_loss(l2_loss_and_negative_log_likelihood(images, output_reconstruction, output_survival, events))


    return model




def DeepConvSurv_autoEncoder_imaging_dose_clinics_testing(input_shape_1, input_shape_2, input_shape_3):

    images = Input(shape = input_shape_1)
    doses =  Input(shape = input_shape_2)
    clinics = Input(shape = input_shape_3)
 

    # Encoder part
    e = unet_core(images, filter_size= 16, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #64
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size= 32, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #32
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size=64, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e) #16
    e = SpatialDropout3D(0.2)(e)

    e = unet_core(e, filter_size = 128, kernel_size=(3, 3, 3))
    e = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(e)
    e = SpatialDropout3D(0.3)(e)

    e = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(e) #8
    e = mish(e)
   #e = GlobalAveragePooling3D(keepdims = True)(e)

    # Decoder part
    d = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3),  padding='same', strides = 2)(e)
    d = unet_core(d, filter_size=128, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=64, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=32, kernel_size=(3, 3, 3))

    d = Conv3DTranspose(filters=16, kernel_size=(3, 3, 3),  padding='same', strides=2)(d)
    d = unet_core(d, filter_size=16, kernel_size=(3, 3, 3))

    output_reconstruction = Conv3D(filters = 1, kernel_size=(1, 1, 1))(d)


    # Compute dosimetric features
    do = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(doses) #128
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #64
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #32
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #16
    do = BatchNormalization()(do)
    do = mish(do)
    do = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(do)
    do = SpatialDropout3D(0.3)(do)

    do = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(do) #8
    do = BatchNormalization()(do)
    do = mish(do)

    do = GlobalAveragePooling3D()(do)
    do = Dense(input_shape_3, bias_initializer = 'glorot_uniform')(do)
    do = mish(do)


    # Survival prediction part
    x = GlobalAveragePooling3D(keepdims = False)(e)
    x = Dense(input_shape_3, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    x = Concatenate()([x, do, clinics])

    x = Dense(64, bias_initializer = 'glorot_uniform')(x)
    x = mish(x)

    output_survival = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(x)

    model = Model(inputs = [images, doses, clinics], outputs = [output_survival, output_reconstruction], name = 'DeepConvSurv_autoEncoder')


    return model


