#/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.core import Reshape
from keras.layers import LeakyReLU
#from keras.layers import ReLU
from keras.layers import Dropout
from keras import backend as K


class PerceptionRNN:
    def __init__(self, conf_file):
        pass

    def get_output_obstacle_box_heat_map(self):
        return self._model.get_output_at(0)

    def setup_model(self):
        input_vec = []

        #time sequence
        input_k = Input(shape=(1,), name='perception_input_k')
        #input_vec.append(input_k)

        # iterate at time sequence
        input_last_predict_obs_box = Input(shape=(self.feature_net.W, self.feature_net.H, 1), name='perception_input_last_predict_obs_box')
        input_vec.append(input_last_predict_obs_box)

        self.out_W_size = self.feature_net.out_W_size
        self.out_H_size = self.feature_net.out_H_size
        self.feature_channel_num = self.feature_net.feature_channel_num
        input_features = Input(shape=(self.out_W_size, self.out_H_size, self.feature_channel_num), name='perception_input_feature_net')
        input_vec.append(input_features)

        #不单独用子网卷积处理各输入特征,大部分信息较少
        merge_input_layers = [input_last_predict_obs_box, input_features]
        model_combine_input = Concatenate(name='perception_rnn_combine_inputs', axis=-1)(merge_input_layers)

        #model = self.build_conv_block(self.out_W_size, self.out_H_size, self.feature_channel_num, 1, 1, 'model_perception')(input_features)
        channel_size = 1 + self.feature_channel_num
        model = self.build_conv_block(self.out_W_size, self.out_H_size, channel_size, 1, 1, 'model_perception')(model_combine_input)

        model_shape = K.int_shape(model)

        up_model = UpSampling2D(size=(self.feature_net.W//4//model_shape[1], self.feature_net.H//4//model_shape[2]))(model)
        up_model = Conv2D(2, kernel_size=(2, 2), strides=(1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(
                up_model)
        up_model = UpSampling2D(size=(4, 4))(up_model)
        up_model = Dropout(0.5)(up_model)
        up_model = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(
                up_model)

        out_heat_map = Reshape((self.feature_net.W, self.feature_net.H), name='perception_categ_out_obstacle_box_heat_map')(up_model)

        output_vec = []
        #out_heat_map = Activation('sigmoid', name='categ_out_obstacle_box_heat_map')(model)
        output_vec.append(out_heat_map)
        perception_rnn = Model(inputs=input_vec, outputs=output_vec, name='perception_rnn')
        print(perception_rnn.summary())
        self._model = perception_rnn
        return perception_rnn

    def build_conv_block(self, width, height, channel, block_repeat_num1, block_repeat_num2, model_name):
        x = Input(shape=(width, height, channel))
        model = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                       activation='relu')(x)
        model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
        for i in range(block_repeat_num1):
            model = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                           activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
        for i in range(block_repeat_num2):
            model = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                           activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2))(model)
        # model = Flatten()(model)
        # model = Dense(1000, activation='relu')(model)
        ret_model = Model(inputs=x, outputs=model, name=model_name)
        return ret_model
