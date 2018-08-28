
import keras
import tensorflow as tf
from keras import backend, layers, models, regularizers
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121,DenseNet201

def augment_2d(inputs, rotation=0, horizontal_flip=False, vertical_flip=False):
    if inputs.dtype != tf.float32:
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(inputs)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation > 0:
            angle_rad = rotation * 3.141592653589793 / 180.0
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            f = tf.contrib.image.angles_to_projective_transforms(angles,
                                                                 height, width)
            transforms.append(f)

        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., width, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., height, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

    if transforms:
        f = tf.contrib.image.compose_transforms(*transforms)
        inputs = tf.contrib.image.transform(inputs, f, interpolation='BILINEAR')
    return inputs

def add_regularizers_l2(model):
        for layer in model.layers:
            if type(layer)==keras.engine.training.Model:
                add_regularizers_l2(layer)
            elif hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= regularizers.l2(0.01)
    #             print(layer, layer.kernel_regularizer)
def lock_some_layer(model):
    for layer in model.layers:
        if type(layer)==keras.engine.training.Model:
            for l in layer.layers[:115]:
                l.trainable = False
            for l in layer.layers[115:]:
                l.trainable = True

def resnet_layer(inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
    conv = layers.Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x

def input_resize(x):
    return backend.resize_images(x, 3.5, 3.5, "channels_last")

def resnet():
    num_filters_in = 32
    num_res_blocks = int((29 - 2) / 9)
    inputs = layers.Input(shape=(64,64,3))
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                        num_filters=num_filters_in,
                        conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(4):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                                num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
            y = resnet_layer(inputs=y,
                                num_filters=num_filters_in,
                                conv_first=False)
            y = resnet_layer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
            x = layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = layers.BatchNormalization()(x)
    base_model = models.Model(inputs, x)
    return base_model



def get_base_model(model_type, train_type):

    def add_regularizers_l2(model):
        for layer in model.layers:
            if type(layer)==keras.engine.training.Model:
                add_regularizers_l2(layer)
            elif hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer= regularizers.l2(0.01)
    #             print(layer, layer.kernel_regularizer)
    def lock_some_layer(model):
        for layer in model.layers:
            if type(layer)==keras.engine.training.Model:
                for l in layer.layers[:115]:
                    l.trainable = False
                for l in layer.layers[115:]:
                    l.trainable = True
    
    
    backend.clear_session()
    if model_type=="MobileNet":
        base_model = MobileNet(include_top=False, 
                               weights=None,
                               input_tensor=None,
                               pooling=None)
    
    # elif model_type=="DenseNet":
    #     !git clone https://github.com/titu1994/DenseNet.git
    #     import sys
    #     sys.path.append('DenseNet')
    #     import densenet
    #     base_model = densenet.DenseNet((64,64,3), include_top=False, depth=40, nb_dense_block=4,
    #                               growth_rate=32, nb_filter=12, dropout_rate=0.0,
    #                               bottleneck=True, reduction=0.5, weights=None)
    #     base_model = models.Model(base_model.input, base_model.layers[-2].output)
        
    elif model_type=="DenseNet121":
        base_model = DenseNet121(include_top=False,
                                 weights=None,
                                 input_tensor=None)

    elif model_type=="DenseNet201":
        base_model = DenseNet201(include_top=False,
                                 weights=None,
                                 input_tensor=None)
    
    elif model_type=="Xception":
        base_model = Xception(include_top=False, 
                              weights=None,
                              input_tensor=None)

    elif model_type=="ResNet50":
        base_model = ResNet50(include_top=False, 
                                  weights=None,
                                  input_tensor=None)
        
    elif model_type=="VGG16":
        base_model = VGG16(include_top=False, 
                                  weights=None,
                                  input_tensor=None)
        
    elif model_type=="NASNet":
        base_model = NASNetLarge(include_top=False, 
                                  weights=None,
                                  input_tensor=None)

    return base_model