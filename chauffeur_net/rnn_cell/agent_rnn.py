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


class AgentRNN:
    def __init__(self, conf_file):
        pass

    def get_output_heading(self):
        return self._model.get_output_at(0)

    def get_output_speed(self):
        return self._model.get_output_at(1)

    def get_output_waypoint(self):
        return self._model.get_output_at(2)

    '''
    def get_output_waypoint_x(self):
        return self._model.get_output_at(2)

    def get_output_waypoint_y(self):
        return self._model.get_output_at(3)
    '''

    def get_ouput_agent_box_heat_map(self):
        return self._model.get_output_at(4)

    def setup_model(self):
        input_vec = []

        #fixed,not iterate at time sequence
        input_ego_past_img = Input(shape=(self.feature_net.W, self.feature_net.H, 1), name='input_ego_past_img')
        input_vec.append(input_ego_past_img)

        #time sequence
        input_k = Input(shape=(1,), name='input_k')
        # input_vec.append(input_k)

        # iterate at time sequence
        input_predict_ego_pos_memory = Input(shape=(self.feature_net.W, self.feature_net.H, 1), name='input_predict_ego_pos_memory')
        # input_vec.append(input_predict_ego_pos_memory)

        # iterate at time sequence
        input_last_predict_ego_box = Input(shape=(self.feature_net.W, self.feature_net.H, 1), name='input_last_predict_ego_box')
        # input_vec.append(input_last_predict_ego_box)

        self.out_W_size = self.feature_net.out_W_size
        self.out_H_size = self.feature_net.out_H_size
        self.feature_channel_num = self.feature_net.feature_channel_num
        # fixed,not iterate at time sequence
        input_features = Input(shape=(self.out_W_size, self.out_H_size, self.feature_channel_num))
        input_vec.append(input_features)

        model = self.build_conv_block(self.out_W_size, self.out_H_size, self.feature_channel_num, 1, 1, 'model_agent')(input_features)
        model = Conv2D(1, kernel_size=(1, 1), strides=(1, 1),
                       activation='relu')(model)

        model_shape = K.int_shape(model)

        up_model = UpSampling2D(size=(self.feature_net.W//4//model_shape[1], self.feature_net.H//4//model_shape[2]))(model)
        #up_model = UpSampling2D(size=(2, 2))(model)
        up_model = Conv2D(2,kernel_size=(2, 2), strides=(1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(
                up_model)
        up_model = UpSampling2D(size=(4, 4))(up_model)
        #up_model = UpSampling2D(size=(2, 2))(up_model)
        up_model = Conv2D(2,kernel_size=(2, 2), strides=(1, 1),  activation='relu', padding='same', kernel_initializer='he_normal')(
                up_model)
        up_model = Dropout(0.5)(up_model)
        up_model = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(
                up_model)
        #model = Flatten()(model)
        #model.output_shape
        #kvar_shape = K.variable(value=K.int_shape(model)[:-1])
        #const_shape = K.constant(value=K.int_shape(model)[:-1],shape=K.int_shape(model)[:-1])
        #new_shape = K.shape(K.zeros(K.int_shape(model)[:-1]))
        new_shape = model_shape[1:-1]
        model = Reshape(new_shape)(model)
        #model = Reshape((-1, 8))(model)
        #model = Reshape((self.feature_net.W, self.feature_net.H))(model)
        '''
        #model = Flatten()(model)
        model = Dense(20)(model)
        #model = Activation(LeakyReLU(0.3))(model)
        model = Activation(ReLU())(model)
        '''
        model = Dropout(0.5)(model)
        '''
        model = Dense(20)(model)
        model = Activation(LeakyReLU(0.3))(model)
        model = Dropout(0.4)(model)
        '''
        #up_model = UpSampling2D(size=(2, 2))(up_model)

        model = Flatten()(model)

        output_vec = []
        out_waypoint = Dense(self.feature_net.W * self.feature_net.H, kernel_initializer='normal', activation='softmax',name='categ_out_waypoint')(model)
        #out_waypoint = Dense(self.feature_net.W * self.feature_net.H, kernel_initializer='normal', activation='relu',
        #                     name='categ_out_waypoint')(model)
        #out_waypoint = Dense(1, kernel_initializer='normal', activation='relu',#'softmax',
        #                     name='categ_out_waypoint')(model)  xxx
        output_vec.append(out_waypoint)
        out_heading = Dense(1, activation='sigmoid',name='reg_out_heading')(model)#'relu' may not in bound ,should be in 0-2PI
        output_vec.append(out_heading)
        out_speed = Dense(1, activation='sigmoid',name='reg_out_speed')(model)#'relu' may not in bound ,should be in 0-20
        output_vec.append(out_speed)
        out_heat_map = Reshape((self.feature_net.W, self.feature_net.H), name='categ_out_agent_box_heat_map')(up_model)
        #out_heat_map = Activation('sigmoid',name='categ_out_agent_box_heat_map')(model)
        #out_heat_map = Dense(self.feature_net.W * self.feature_net.H, kernel_initializer='normal',activation='sigmoid', name='categ_out_agent_box_heat_map')(model)
        output_vec.append(out_heat_map)
        out_waypoint_sub_pixel = Dense(2, kernel_initializer='normal', activation='sigmoid',#relu sigmoid must in 0.0-1.0
                             name='waypoint_sub_pixel')(model)
        output_vec.append(out_waypoint_sub_pixel)
        agent_rnn = Model(inputs=input_vec, outputs=output_vec, name='agent_rnn')
        self._model = agent_rnn
        return agent_rnn

    def build_conv_block(self, width, height, channel, block_repeat_num1, block_repeat_num2, model_name):
        x = Input(shape=(width, height, channel))
        model = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                       activation='relu')(x)
        model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
        for i in range(block_repeat_num1):
            model = Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                           activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
        for i in range(block_repeat_num2):
            model = Conv2D(64, (5, 5), activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2))(model)
        # model = Flatten()(model)
        # model = Dense(1000, activation='relu')(model)
        ret_model = Model(inputs=x, outputs=model, name=model_name)
        return ret_model
