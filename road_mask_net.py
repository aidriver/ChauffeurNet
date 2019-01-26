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
from keras.layers import ReLU
from keras.layers import Dropout
from keras import backend as K


class RoadMaskNet:
    '''
    predict onroad mask distribution logits
    '''
    def __init__(self, conf_file):
        pass

    def get_output_road_mask(self):
        return self._model.get_output_at(0)

    def setup_model(self):
        self.out_W_size = self.feature_net.out_W_size
        self.out_H_size = self.feature_net.out_H_size
        self.feature_channel_num = self.feature_net.feature_channel_num
        input_features = Input(shape=(self.out_W_size, self.out_H_size, self.feature_channel_num), name='road_mask_input_feature_net')
        model = self.build_conv_block(self.out_W_size, self.out_H_size, self.feature_channel_num, 1, 1, 'model_road_mask')(input_features)

        model_shape = K.int_shape(model)

        up_model = UpSampling2D(size=(self.feature_net.W//4//model_shape[1], self.feature_net.H//4//model_shape[2]))(model)
        up_model = Conv2D(2,kernel_size=(2, 2), strides=(1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(
                up_model)
        up_model = UpSampling2D(size=(4, 4))(up_model)
        up_model = Dropout(0.5)(up_model)
        up_model = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(
                up_model)

        out_road_mask = Reshape((self.feature_net.W, self.feature_net.H), name='bin_out_road_mask')(up_model)
        '''
        #model = Flatten()(model)
        model = Dense(20)(model)
        #model = Activation(LeakyReLU(0.3))(model)
        model = Activation(ReLU())(model)
        '''
        #model = Dropout(0.5)(model)
        '''
        model = Dense(20)(model)
        model = Activation(LeakyReLU(0.3))(model)
        model = Dropout(0.4)(model)
        '''
        #up_model = UpSampling2D(size=(2, 2))(up_model)

        #model = Flatten()(model)

        output_vec = []
        #out_road_mask = Activation('sigmoid',name='bin_out_road_mask')(model)
        output_vec.append(out_road_mask)

        road_mask_net = Model(inputs=input_features, outputs=output_vec, name='road_mask_net')

        self._model = road_mask_net
        return road_mask_net

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
