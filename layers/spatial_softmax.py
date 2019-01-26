#coding=utf-8
"""
@file: spatial_softmax.py
@author: gaofei(gaofei09@baidu.com)
@date: 2019/01/13
@version: 0.1
@brief: file contains spatial softmax neural networks
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from keras.utils import conv_utils


class SpatialSoftmax(Layer):
    """ SpatialSoftmax layer for 3D input (e.g.image)

    It calc softmax along spatial dimensions, i.e. width and height

    # Arguments
        data_format: A string

    # Input shape
        5D tensor with shape:
        - If 'data_format' is "channels_last":
            '(batch, steps, rows, cols, channels)'
        - If 'data_format' is "channels_first":
            '(batch, steps, channels, rows, cols)'

    # Output shape
        3D tensor with shape:
        - If 'data_format' is "channels_last":
            '(batch, steps, channels)'

    # Example

    '''python
        # softmax the input images or feature maps
        model = Sequential()
        model.add(SpatialSoftmax)
    '''
    """

    def __init__(self, data_format=None, **kwargs):
        super(SpatialSoftmax, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        """
        inherit from Layer
        """
        if not isinstance(input_shape, list) and len(input_shape) != 2:
            raise ValueError('Inputs of SpatialSoftmax layer should be 2')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if self.data_format == 'channels_first':
            return (shape1[0], shape1[1], shape1[2])
        elif self.data_format == 'channels_last':
            return (shape1[0], shape1[1], shape1[4])
        else:
            raise ValueError('Wrong data format: ', self.data_format)

    def call(self, inputs):
        """
        inherit from Layer
        """
        if len(inputs) != 2:
            raise ValueError('Inputs of SpatialSoftmax layer should be 2')
        '''
        feature, cropping = inputs
        if self.data_format == 'channels_first':
            feature = tf.transpose(feature, [0, 1, 3, 4, 2])
        fea_shape = tf.shape(feature)
        batch, steps, width, height, channels = tf.unstack(fea_shape)
        feature = tf.reshape(feature, [-1, width, height, channels])

        croping_last = tf.shape(cropping)[-1]
        arange = tf.reshape(tf.range(batch * steps), [-1, 1])
        cropping = tf.reshape(cropping, [-1, croping_last])
        bool_mask = tf.reduce_any(tf.not_equal(cropping, 0), axis=-1, keep_dims=True)
        cropping = tf.concat([arange, cropping], axis=1)
        cropping_feature = tf.gather_nd(feature, cropping) * tf.cast(bool_mask, tf.float32)
        return tf.reshape(cropping_feature, [batch, steps, channels])

        shape = K.int_shape(x)
        x = Reshape((shape[1] * shape[2]))(x)
        x = K.softmax(x, 2)
        fp_x = x * self.x_map
        fp_y = x * self.y_map
        x = K.concatenate([fp_x, fp_y], axis=-1)
        return x
        '''
        feature = inputs
        x = tf.contrib.layers.spatial_softmax(feature)
        return x

    def get_config(self):
        """
        inherit from Layer
        """
        config = {'data_format': self.data_format}
        base_config = super(SpatialSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
