import six
from keras.layers import (
   Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    Reshape, 
    multiply, 
    add, 
    Permute
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling3D,
    AveragePooling2D,
    Conv3D,
    Conv2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import add
import tensorflow as tf


def _3Dbn_relu_spc(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _2Dbn_relu_spc(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS-1)(input)
    return Activation("relu")(norm)

def Squeeze_excitation_3Dlayer(input, ratio=2): #spectral dimention 0.9755
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]#64
    se_shape = (input._keras_shape[CONV_DIM1], input._keras_shape[CONV_DIM2], input._keras_shape[CONV_DIM3]*filters)
    se_shape2 = (input._keras_shape[CONV_DIM3],filters)
    se = Reshape(se_shape)(init)
    se = GlobalAveragePooling2D()(se)
    se = Reshape(se_shape2)(se)
    se = Permute((2, 1))(se)
    se = Dense(input._keras_shape[CONV_DIM3] // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se) #(?,64,5)
    se = Dense(input._keras_shape[CONV_DIM3], activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)#(?,64,20)
    se = Permute((2, 1))(se)
    se = se[:,tf.newaxis,tf.newaxis,:,:]
    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)
    scale = multiply([init, se])
    return scale

def Squeeze_excitation_2Dlayer(input):
    se = Conv2D(kernel_initializer="he_normal", strides=(1, 1), kernel_regularizer=regularizers.l2(0.0001),
                filters=1, kernel_size=(1,1), padding='same')(input)
    se = tf.nn.sigmoid(se)
    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)
    scale = multiply([input, se])
    return scale

def _3Dconv_bn_relu_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        conv = Conv3D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2,kernel_dim3), padding='same')(input)
        return _3Dbn_relu_spc(conv)
    return f

def _2Dconv_bn_relu_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    
    subsample = conv_params.setdefault("subsample", (1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        conv = Conv2D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2))(input)
        return _2Dbn_relu_spc(conv)
    return f

def _bn_relu_conv_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        activation = _3Dbn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)
    return f

def _2Dbn_relu_conv_spc(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    subsample = conv_params.setdefault("subsample", (1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))
    def f(input):
        activation = _2Dbn_relu_spc(input)
        return Conv2D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2), padding=border_mode)(activation)
    return f

def _shortcut_spc(input, residual):
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]
    shortcut = input
    print("input shape:", input._keras_shape)
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)
    return add([shortcut, residual])

def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input
    return f

def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 5), padding='same')(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=5, subsample=init_subsample)(input)
        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=3)(conv1)
        scale = Squeeze_excitation_3Dlayer(residual,ratio=2)
        return _shortcut_spc(input, scale)
    return f

def _bn_relu_conv(**conv_params):
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        activation = _3Dbn_relu_spc(input)
        return  Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)
    return f

def _shortcut(input, residual):
    stride_dim1 = (input._keras_shape[CONV_DIM1]+1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2]+1) // residual._keras_shape[CONV_DIM2]
    equal_channels = residual._keras_shape[CONV_DIM3] == input._keras_shape[CONV_DIM3]
    shortcut = input
    print("input shape:", input._keras_shape)
    if stride_dim1 > 1 or stride_dim2 > 1 or not equal_channels:
        shortcut = Conv2D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2), kernel_regularizer=regularizers.l2(0.0001),
                          filters=residual._keras_shape[3], kernel_size=(1, 1), padding='valid')(input)
    return add([shortcut, residual])

def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input
    return f

def basic_block(nb_filter, init_subsample=(1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv2D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3), padding='same')(input)
        else:
            conv1 = _2Dbn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, subsample=init_subsample)(input)
        residual = _2Dbn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3)(conv1)
        scale = Squeeze_excitation_2Dlayer(residual)
        return _shortcut(input, scale)
    return f

def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

def res4_model_ss(input, repetitions1, repetitions2):
    # 3D feature learning
    _handle_dim_ordering()
    conv1_spc = _3Dconv_bn_relu_spc(nb_filter=64, kernel_dim1=3, kernel_dim2=3, kernel_dim3=7, subsample=(1, 1, 1))(input)
    block_spc = conv1_spc #input of the RBAM
    nb_filter = 64
    for i, r in enumerate(repetitions1):
        block_spc = _residual_block_spc(basic_block_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block_spc)
        nb_filter *= 2
    # 2D feature learning
    block_spc = _3Dbn_relu_spc(block_spc)
    block_spc_shape = block_spc._keras_shape
    conv_spc_results = Reshape((block_spc_shape[1], block_spc_shape[2], block_spc_shape[3]*block_spc_shape[4]))(block_spc)
    print("conv_spc_result shape:", conv_spc_results._keras_shape)
    conv1 = _2Dconv_bn_relu_spc(nb_filter=128, kernel_dim1=3, kernel_dim2=3, subsample=(1, 1))(conv_spc_results)
    print("conv1 shape:", conv1._keras_shape)
    block = conv1   #input of the RSAM block
    nb_filter = 64
    for i, r in enumerate(repetitions2):
        block_spa = _residual_block(basic_block, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
        nb_filter *= 2
    # 1D feature learning
    block_output = _2Dbn_relu_spc(block_spa)
    pool2 = AveragePooling2D(pool_size=(block._keras_shape[CONV_DIM1],
                                        block._keras_shape[CONV_DIM2]),
                                        strides=(1, 1))(block_output)
    flatten1 = Flatten()(pool2)
    drop1 = Dropout(1)(flatten1)
    dense_layer2 = Dense(units=64, activation='relu')(drop1)
    return dense_layer2
