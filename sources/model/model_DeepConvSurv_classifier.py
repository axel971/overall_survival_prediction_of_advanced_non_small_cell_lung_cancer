
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


def DeepConvSurv_oneClassificationTask_imaging_clinics(input_shape1, input_shape2, input_shape3, input_shape4):

        images = Input(shape = input_shape1)
        clinics = Input(shape = input_shape2)
        events = Input(shape = input_shape3)
        true_output_classificationTask = Input(shape = input_shape4)

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

        os = Dense(64, bias_initializer = 'glorot_uniform')(x)
        os = BatchNormalization()(os)
        os = mish(os)
        os = Dropout(0.3)(os)

        os = Dense(32, bias_initializer = 'glorot_uniform')(os)
        os = BatchNormalization()(os)
        os = mish(os)

        output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(os)


        c = Dense(64, bias_initializer = 'glorot_uniform')(x)
        c = BatchNormalization()(c)
        c = mish(c)
        c = Dropout(0.3)(c)

        c = Dense(32, bias_initializer = 'glorot_uniform')(c)
        c = BatchNormalization()(c)
        c = mish(c)

        output_classificationTask = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform',  kernel_regularizer = regularizers.L2(0.001))(c)


        model = Model(inputs = [images, clinics, events, true_output_classificationTask], outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_imaging_clinics')
        model.add_loss(negativeLogLikelihood_binaryCrossEntropy(output_OS, events, true_output_classificationTask, output_classificationTask))

        return model


def DeepConvSurv_oneClassificationTask_imaging_clinics_testing(input_shape1, input_shape2):

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

        x = GlobalAveragePooling3D(keepdims = False)(x)
        x = Dense(input_shape2, bias_initializer = 'glorot_uniform')(x)
        x = mish(x)

        # Fusion of DL features and clinical data
        x = Concatenate()([x, clinics])

        os = Dense(64, bias_initializer = 'glorot_uniform')(x)
        os = BatchNormalization()(os)
        os = mish(os)
        os = Dropout(0.3)(os)

        os = Dense(32, bias_initializer = 'glorot_uniform')(os)
        os = BatchNormalization()(os)
        os = mish(os)

        output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(os)


        c = Dense(64, bias_initializer = 'glorot_uniform')(x)
        c = BatchNormalization()(c)
        c = mish(c)
        c = Dropout(0.3)(c)

        c = Dense(32, bias_initializer = 'glorot_uniform')(c)
        c = BatchNormalization()(c)
        c = mish(c)

        output_classificationTask = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(c)

        model = Model(inputs = [images, clinics], outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_imaging_clinics')
 
        return model


def DeepConvSurv_oneClassificationTask_2imaging_clinics(input_shape1, input_shape2, input_shape3, input_shape4, input_shape5):

        images1 = Input(shape = input_shape1)
        images2 = Input(shape = input_shape2)
        clinics = Input(shape = input_shape3)
        events = Input(shape = input_shape4)
        true_output_classificationTask = Input(shape = input_shape5)

        # Extract feature imaging 1
        x1 = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images1) #128
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #64
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #32
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #16
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #8
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)

        x1 = GlobalAveragePooling3D(keepdims = False)(x1)
        x1 = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x1)
        x1 = mish(x1)

        # Extract feature imaging 2
        x2 = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images2) #128
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #64
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #32
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #16
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #8
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)

        x2 = GlobalAveragePooling3D(keepdims = False)(x2)
        x2 = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x2)
        x2 = mish(x2)

        # Fusion of DL features and clinical data
        x = Concatenate()([x1, x2, clinics])

        os = Dense(64, bias_initializer = 'glorot_uniform')(x)
        os = BatchNormalization()(os)
        os = mish(os)
        os = Dropout(0.3)(os)

        os = Dense(32, bias_initializer = 'glorot_uniform')(os)
        os = BatchNormalization()(os)
        os = mish(os)

        output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(os)


        c = Dense(64, bias_initializer = 'glorot_uniform')(x)
        c = BatchNormalization()(c)
        c = mish(c)
        c = Dropout(0.3)(c)

        c = Dense(32, bias_initializer = 'glorot_uniform')(c)
        c = BatchNormalization()(c)
        c = mish(c)

        output_classificationTask = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform')(c)


        model = Model(inputs = [images1, images2, clinics, events, true_output_classificationTask], outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_imaging_clinics')
        model.add_loss(negativeLogLikelihood_binaryCrossEntropy(output_OS, events, true_output_classificationTask, output_classificationTask))

        return model



def DeepConvSurv_oneClassificationTask_2imaging_clinics_testing(input_shape1, input_shape2, input_shape3):

        images1 = Input(shape = input_shape1)
        images2 = Input(shape = input_shape2)
        clinics = Input(shape = input_shape3)
     
        # Extract feature imaging 1
        x1 = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images1) #128
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #64
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #32
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #16
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)
        x1 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x1)
        x1 = SpatialDropout3D(0.3)(x1)

        x1 = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x1) #8
        x1 = BatchNormalization()(x1)
        x1 = mish(x1)

        x1 = GlobalAveragePooling3D(keepdims = False)(x1)
        x1 = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x1)
        x1 = mish(x1)

        # Extract feature imaging 2
        x2 = Conv3D(16, kernel_size = (3, 3, 3),  padding = 'same', bias_initializer = 'glorot_uniform')(images2) #128
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #64
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #32
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(128, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #16
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)
        x2 = MaxPooling3D(pool_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(x2)
        x2 = SpatialDropout3D(0.3)(x2)

        x2 = Conv3D(256, kernel_size = (3, 3, 3), padding = 'same', bias_initializer = 'glorot_uniform')(x2) #8
        x2 = BatchNormalization()(x2)
        x2 = mish(x2)

        x2 = GlobalAveragePooling3D(keepdims = False)(x2)
        x2 = Dense(input_shape3, bias_initializer = 'glorot_uniform')(x2)
        x2 = mish(x2)

        # Fusion of DL features and clinical data
        x = Concatenate()([x1, x2, clinics])

        os = Dense(64, bias_initializer = 'glorot_uniform')(x)
        os = BatchNormalization()(os)
        os = mish(os)
        os = Dropout(0.3)(os)

        os = Dense(32, bias_initializer = 'glorot_uniform')(os)
        os = BatchNormalization()(os)
        os = mish(os)

        output_OS = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform', kernel_regularizer = regularizers.L2(0.001))(os)


        c = Dense(64, bias_initializer = 'glorot_uniform')(x)
        c = BatchNormalization()(c)
        c = mish(c)
        c = Dropout(0.3)(c)

        c = Dense(32, bias_initializer = 'glorot_uniform')(c)
        c = BatchNormalization()(c)
        c = mish(c)

        output_classificationTask = Dense(1, activation = 'sigmoid', bias_initializer = 'glorot_uniform')(c)


        model = Model(inputs = [images1, images2, clinics], outputs = [output_OS, output_classificationTask], name = 'DeepConvSurv_imaging_clinics')
        
        return model


