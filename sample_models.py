from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, pooling)
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling1D

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization

    tmp_data = input_data

    for idx in range(recur_layers):
        layer_name = "rnn_{}".format(idx)
        tmp_data = GRU(units, return_sequences=True,
                    implementation=2, name=layer_name)(tmp_data)
        tmp_data = BatchNormalization()(tmp_data)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_distrebuted = TimeDistributed(Dense(output_dim))(tmp_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_distrebuted)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # TODO: Add bidirectional recurrent layer

    # model = Sequential()
    # model.add(Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-1'), input_shape=(None, input_dim)))
    # model.add(Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-2')))
    # model.add(TimeDistributed(Dense(output_dim)))
    # model.add(Activation('softmax'))
    # model.output_length = lambda x: x

    # tmp_data = GRU(units, return_sequences=True, implementation=2, name='gru-1')(input_data)
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-1'))(input_data)
    # bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-1'))(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x

    print(model.summary())
    return model

# extend cnn_rnn_model model
def candidate_1_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # TODO: Specify the layers in your network

    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    # max pooling if i use
    # pooling_1 = pooling.MaxPooling1D()(conv_1d)

    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # bidirectional rnn 1
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-11'))(input_data)
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-12'))(bidir_rnn)

    # bidirectional rnn 22
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-21'))(bidir_rnn)
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-22'))(bidir_rnn)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)

    print(model.summary())
    return model


def dilated_conv(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add convolutional layer: 1
    conv_1d_1 = Conv1D(filters, kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=2,
                     name='conv1d_1')(input_data)
    # Add batch normalization
    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)

    # Add convolutional layer: 2
    conv_1d_2 = Conv1D(filters, kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=4,
                     name='conv1d_2')(bn_cnn_1)

    # Add batch normalization
    bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)

    # dropout here
    dropout_1 = Dropout(0.8)(bn_cnn_2)

    # Add convolutional layer: 3
    conv_1d_3 = Conv1D(filters, kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=8,
                     name='conv1d_3')(dropout_1)

    # Add batch normalization
    bn_cnn_3 = BatchNormalization(name='bn_conv_1d_3')(conv_1d_3)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn_3)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: x

    print(model.summary())
    return model


def cnn_rnn_model_2(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)


    # bidirectional rnn 1
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-11', dropout=0.1, recurrent_dropout=0.1))(bn_cnn)
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-12', dropout=0.1, recurrent_dropout=0.1))(bidir_rnn)

    # # Add a recurrent layer
    # simp_rnn = SimpleRNN(units, activation='relu',
    #     return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

#
#def dilated_conv_2(input_dim, filters, kernel_size, conv_stride,
#    conv_border_mode, units, output_dim=29):
#    """ Build a recurrent + convolutional network for speech
#    """
#    # Main acoustic input
#    input_data = Input(name='the_input', shape=(None, input_dim))
#
#
#    # Add convolutional layer: 1
#    conv_1d_1 = Conv1D(filters, kernel_size,
#                     strides=1,
#                     padding=conv_border_mode,
#                     activation='relu',
#                     dilation_rate=2,
#                     name='conv1d_1')(input_data)
#    # max pooling
#    conv_1d_1 = MaxPooling1D()(conv_1d_1)
#
#    # Add batch normalization
#    bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(conv_1d_1)
#
#    # Add convolutional layer: 2
#    conv_1d_2 = Conv1D(filters, kernel_size,
#                     strides=1,
#                     padding=conv_border_mode,
#                     activation='relu',
#                     dilation_rate=4,
#                     name='conv1d_2')(bn_cnn_1)
#    # max pooling
#    conv_1d_2 = MaxPooling1D()(conv_1d_2)
#
#    # Add batch normalization
#    bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(conv_1d_2)
#
#
#
#    # dropout here
#    dropout_1 = Dropout(0.8)(bn_cnn_2)
#
#    # Add convolutional layer: 3
#    conv_1d_3 = Conv1D(filters, kernel_size,
#                     strides=1,
#                     padding=conv_border_mode,
#                     activation='relu',
#                     dilation_rate=8,
#                     name='conv1d_3')(dropout_1)
#
#    # max pooling
#    # conv_1d_3 = MaxPooling1D()(conv_1d_3)
#
#    # Add batch normalization
#    bn_cnn_3 = BatchNormalization(name='bn_conv_1d_3')(conv_1d_3)
#
#    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
#    time_dense = TimeDistributed(Dense(output_dim))(bn_cnn_3)
#
#    # Add softmax activation layer
#    y_pred = Activation('softmax', name='softmax')(time_dense)
#    # Specify the model
#    model = Model(inputs=input_data, outputs=y_pred)
#
#    # model.output_length = lambda x: x
#    model.output_length = lambda x: cnn_output_length(cnn_output_length(x, 2, 'valid', 2), 2, 'valid', 2)
#
#    print(model.summary())
#    return model


# extend cnn_rnn_model model
# def final_model():
def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # bidirectional rnn x 2
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-11', dropout=0.1, recurrent_dropout=0.1))(bn_cnn)
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-12', dropout=0.1, recurrent_dropout=0.1))(bidir_rnn)

    # Add batch normalization
    bn_rnn = BatchNormalization()(bidir_rnn)
    #  Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

# def final_model_not_good(input_dim, units, output_dim=29):
#    """ Build a bidirectional recurrent network for speech
#    """
#    # Main acoustic input
#    input_data = Input(name='the_input', shape=(None, input_dim))
#
#    # 2 bidirectional layer
#    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-11'))(input_data)
#    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-12'))(bidir_rnn)
#    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-21'))(input_data)
#    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='gru-22'))(bidir_rnn)
#
#    # time distrebuted dense layer
#    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
#    # Add softmax activation layer
#    y_pred = Activation('softmax', name='softmax')(time_dense)
#    # Specify the model
#    model = Model(inputs=input_data, outputs=y_pred)
#    model.output_length = lambda x: x
#
#
#    print(model.summary())
#    return model
