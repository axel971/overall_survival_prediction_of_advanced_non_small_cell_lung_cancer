
from tensorflow.keras.layers import Conv3D, Conv1D, MaxPool3D, concatenate, Input, Dropout, PReLU, ReLU,  Conv3DTranspose, BatchNormalization, SpatialDropout3D, SpatialDropout2D, Flatten, Dense, Conv2D, MaxPooling2D, MaxPooling3D, Add, GlobalAveragePooling2D, GlobalAveragePooling3D, Lambda, GlobalMaxPooling2D, GlobalMaxPooling3D, Activation, Multiply, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Zeros
from tensorflow.keras import regularizers
# from keras_contrib.layers import CRF
import tensorflow.keras as K
import tensorflow as tf
from tensorflow_addons.activations import mish
from tensorflow_addons.layers import GroupNormalization
from loss.loss import negative_log_likelihood, negative_log_likelihood_2, l2_loss_and_negative_log_likelihood, negativeLogLikelihood_l2Loss_DiceLoss

def DeepSurv(input_shape1, input_shape2):

        clinics = Input(shape = input_shape1)
        events = Input(shape = input_shape2)

        x = Dense(64, bias_initializer = 'glorot_uniform')(clinics)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(32, bias_initializer = 'glorot_uniform')(clinics)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(16, bias_initializer = 'glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(8, bias_initializer = 'glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = mish(x)
        #x = Dropout(0.2)(x)

        output = Dense(1, activation = 'linear',  bias_initializer = 'glorot_uniform',  kernel_regularizer = regularizers.L2(1e-3))(x)

        model = Model(inputs = [clinics, events], outputs = output, name = 'DeepSurv')
        model.add_loss(negative_log_likelihood(output, events))

        return model


def DeepSurv_testing(input_shape1):

        clinics = Input(shape = input_shape1)

        x = Dense(64, bias_initializer = 'glorot_uniform')(clinics)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(32, bias_initializer = 'glorot_uniform')(clinics)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(16, bias_initializer = 'glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = mish(x)
        x = Dropout(0.3)(x)

        x = Dense(8, bias_initializer = 'glorot_uniform')(x)
        x = BatchNormalization()(x)
        x = mish(x)
        #x = Dropout(0.2)(x)

        output = Dense(1, activation = 'linear', bias_initializer = 'glorot_uniform',  kernel_regularizer = regularizers.L2(1e-3) )(x)

        model = Model(inputs = clinics, outputs = output, name = 'DeepSurv')

        return model

