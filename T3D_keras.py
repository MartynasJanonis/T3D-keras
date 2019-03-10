import keras
from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Activation, Conv3D, Dropout, Concatenate, AveragePooling3D, MaxPooling3D, Dense, Flatten, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.activations import linear, softmax
from keras.applications import densenet
from keras.layers import TimeDistributed

__all__ = ['DenseNet', 'densenet121', 'densenet161']  # with DropOut


def _DenseLayer(prev_layer, growth_rate, bn_size, drop_rate):
    if prev_layer is None:
        # print('No Layer previous to Dense Layers!!')
        return None
    else:
        x = BatchNormalization(momentum=0.1, epsilon=1e-05)(prev_layer)

    x = Activation('relu')(x)
    x = Conv3D(filters=bn_size * growth_rate, kernel_size=1, strides=1, use_bias=False, padding='valid')(x)
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=growth_rate, kernel_size=3, strides=1, use_bias=False, padding='same')(x)
    x = Dropout(drop_rate)(x)
    x = Concatenate()([x, prev_layer])

    return x


def _DenseBlock(prev_layer, num_layers, bn_size, growth_rate, drop_rate):
    x = prev_layer
    for i in range(num_layers):
        layer = _DenseLayer(x, growth_rate, bn_size, drop_rate)
        if layer is None:
            print('Dense Block not created as no previous layers found!!')
            return None
        else:
            x = layer
    return x


def _Transition(prev_layer, num_output_features):

    # print('In _Transition')
    x = BatchNormalization()(prev_layer)
    x = Activation('relu')(x)
    x = Conv3D(filters=num_output_features, kernel_size=1, strides=1, use_bias=False, padding='valid')(x)
    x = AveragePooling3D(pool_size=2, strides=2)(x)
    # print('Completed _Transition')
    return x


def _TTL(prev_layer, third_kernel_size):
    # print('In _TTL')
    b1 = BatchNormalization()(prev_layer)
    b1 = Activation('relu')(b1)
    # b1 = Conv3D(128, kernel_size=(1), strides=1, use_bias=False, padding='same')(b1)
    b1 = Conv3D(128, kernel_size=(1, 1, 1), strides=1, use_bias=False, padding='same', dilation_rate=1)(b1)

    b2 = BatchNormalization()(prev_layer)
    b2 = Activation('relu')(b2)
    b2 = Conv3D(128, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same', dilation_rate=1)(b2)

    b3 = BatchNormalization()(prev_layer)
    b3 = Activation('relu')(b3)
    b3 = Conv3D(128, kernel_size=third_kernel_size, strides=1, use_bias=False, padding='same', dilation_rate=1)(b3)

    x = keras.layers.concatenate([b1, b2, b3], axis=-1)
    # print('completed _TTL')
    return x


def DenseNet3D(input_shape, growth_rate=32, block_config=(6, 12, 24, 16),
               num_init_features=64, bn_size=4, drop_rate=0, num_classes=5):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    #-----------------------------------------------------------------
    inp_2d = (Input(shape=(224,224,3), name='2d_input'))
    batch_densenet = densenet.DenseNet169(include_top=False, input_shape=(224,224,3), input_tensor=inp_2d, weights='imagenet')
    
    for layer in batch_densenet.layers:
        layer.trainable = False

    # Configure the 2D CNN to take batches of pictures
    inp_2d_batch = (Input(shape=input_shape, name='2d_input_batch'))
    batch_densenet = TimeDistributed(batch_densenet)(inp_2d_batch)
    batch_densenet = Model(inputs=inp_2d_batch, outputs=batch_densenet)
    #-----------------------------------------------------------------

    # inp_3d = (Input(shape=input_shape, name='3d_input'))
    t3d = T3D169(include_top=False, input_shape=input_shape)

    #--------------from 2d densenet model-----------------
    x = GlobalAveragePooling3D(name='avg_pool_t3d')(t3d.output)
    y = GlobalAveragePooling3D(name='avg_pool_densnet3d')(batch_densenet.output)

    #-----------------------------------------------------
    x = keras.layers.concatenate([x,y])
    x = Dropout(0.65)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.35)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[inp_2d_batch, t3d.input], outputs=[out])
    # model.summary()

    return model


# The T3D CNN standalone
def T3D(include_top=True, input_shape=None, growth_rate=32, block_config=(6, 12, 24, 16),
               num_init_features=64, bn_size=4, drop_rate=0, num_classes=5):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    # First convolution-----------------------
    inp_3d = (Input(shape=input_shape, name='3d_input'))


    # need to check padding
    x = (Conv3D(num_init_features, kernel_size=(3, 7, 7),
                strides=2, padding='same', use_bias=False))(inp_3d)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # need to check padding
    # -------------------------------------------------------------------------
    # In the paper it lists it as 3 × 3 × 3 max pool, stride 1, but the output
    # shape as 56 x 56 x 16. In reality the paper's max pool settings result
    # in the output shape of 112 x 112 x 16 with padding and 110 x 110 x 14
    # without.
    x = MaxPooling3D(pool_size=3, strides=(1,2,2), padding='same')(x)
    # -------------------------------------------------------------------------

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        # print('Pass', i)
        x = _DenseBlock(x, num_layers=num_layers,
                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

        num_features = num_features + num_layers * growth_rate

        if i != len(block_config) - 1:
            # print('Not Last layer, so adding Temporal Transition Layer')
            third_kernel_size = (4, 3, 3)
            if i == 0:
                third_kernel_size = (6, 3, 3)

            x = _TTL(x, third_kernel_size)
            num_features = 128*3

            x = _Transition(x, num_output_features=num_features)
            num_features = num_features

    # Final batch norm
    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling3D()(x)
        out = Dense(num_classes, activation='softmax')(x)
    else:
        out = x

    model = Model(inputs=[inp_3d], outputs=[out])
    # model.summary()

    return model


def T3D169_DenseNet(input_shape, nb_classes):
    model = DenseNet3D(input_shape, growth_rate=32, block_config=(
        6, 12, 36, 36), num_init_features=64, drop_rate=0.2, num_classes=nb_classes)
    return model

def T3D169(include_top, input_shape, nb_classes=2):
    model = T3D(include_top=include_top, input_shape=input_shape, block_config=(6,12,32,32), num_classes=nb_classes)
    return model

def T3D169_Dropout(input_shape, nb_classes, d_rate=0.2):
    model = T3D(input_shape,block_config=(6,12,32,32), num_classes=nb_classes, drop_rate=d_rate)
    return model
