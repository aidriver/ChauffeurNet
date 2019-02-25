# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.densenet import DenseNet121
#from keras.applications.nasnet import NASNetMobile
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense
import numpy as np
from keras import backend as K


class FeatureNet:
    def __init__(self, conf_file):
        self.W = 400
        self.H_pos = 400
        self.H_neg = 400
        self.H = self.H_pos + self.H_neg
        self.T_scene = 1.0
        self.T_pose = 8.0
        self.dt = 0.2
        self.N_out_steps = 10  # 25 40
        self.N_past_scene = int(self.T_scene / self.dt)
        self.N_past_pose = int(self.T_pose / self.dt)
        self.resolution_space = 0.2
        self.u0 = self.H_pos
        self.v0 = self.W / 2
        self.max_pertu_theta = 25  # degree
        self.feature_channel_num = 0
        pass

    '''
    def setup_model(self):
        input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

        #model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
        #model = DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
        #                                        input_shape=None, pooling=None, classes=1000)
        model = NASNetMobile(input_shape=None, include_top=True, weights='imagenet',
                                               input_tensor=None, pooling=None, classes=1000)
    '''

    def setup_model(self):
        input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img, input_ego_past_img = self.input_vec[:5]
        input_traffic_light_img_time_seq = self.input_vec[5:5 + self.N_past_scene]
        input_obstacles_past_img_time_seq = self.input_vec[5 + self.N_past_scene:5 + 2 * self.N_past_scene]
        model_roadmap = self.build_conv_block(self.W, self.H, 3, 1, 1, 'model_roadmap')(input_roadmap_img)
        model_speed_limit = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_speed_limit')(input_speed_limit_img)
        model_route = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_route')(input_route_img)
        model_ego_current = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_ego_current')(input_ego_current_img)
        model_ego_past = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_ego_past')(input_ego_past_img)

        # model_traffic_light = [self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_traffic_light_' + str(i))(input_traffic_light_img_time_seq[i]) for i in range(self.N_past_scene)]
        #share_model_traffic_light = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_traffic_light')
        #model_traffic_light = [share_model_traffic_light(input_traffic_light_img_time_seq[i]) for i in
        #                       range(self.N_past_scene)]
        model_traffic_light = Concatenate(name='concat_traffic_lights', axis=-1)(
                [input_traffic_light_img_time_seq[i] for i in
                 range(self.N_past_scene)])
        model_traffic_light = self.build_conv_block(self.W, self.H, self.N_past_scene, 1, 1, 'model_traffic_light')(model_traffic_light)


        # model_obstacles_past = [self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_obstacles_past' + str(i))(input_obstacles_past_img_time_seq[i]) for i in range(self.N_past_scene)]
        #share_model_obstacles_past = self.build_conv_block(self.W, self.H, 1, 1, 1, 'model_obstacles_past')
        #model_obstacles_past = [share_model_obstacles_past(input_obstacles_past_img_time_seq[i]) for i in
        #                        range(self.N_past_scene)]
        model_obstacles_past = Concatenate(name='concat_obstacles_past', axis=-1)([(input_obstacles_past_img_time_seq[i]) for i in
                                range(self.N_past_scene)])
        model_obstacles_past = self.build_conv_block(self.W, self.H, self.N_past_scene, 1, 1, 'model_obstacles_past')(model_obstacles_past)

        merge_layers = [model_roadmap, model_speed_limit, model_route, model_ego_current, model_ego_past]
        #merge_layers.extend(model_traffic_light)
        #merge_layers.extend(model_obstacles_past)
        merge_layers.append(model_traffic_light)
        merge_layers.append(model_obstacles_past)
        model = Concatenate(name='combine_feature_maps', axis=-1)(merge_layers)
        print(K.int_shape(model))
        _, width, height, channel = K.int_shape(model)
        model = self.build_deconv_block(width, height, channel, 1, 1, 'model_feat_deconv')(model)
        model = UpSampling2D(size=(self.W // width, self.H // height))(model)
        output_vec = [model]
        feature_net = Model(inputs=self.input_vec, outputs=output_vec, name='feature_net')
        print(feature_net.summary())
        self._model = feature_net
        return feature_net

    def build_conv_block(self, width, height, channel, block_repeat_num1, block_repeat_num2, model_name):
        x = Input(shape=(width, height, channel))
        unit_size = 16 #32
        model = Conv2D(unit_size, kernel_size=(5, 5), strides=(1, 1), padding='same',
                       activation='relu')(x)
        model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)

        for i in range(block_repeat_num1):
            model = Conv2D(unit_size, kernel_size=(5, 5), strides=(1, 1), padding='same',
                           activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
        unit_size = 32 #64
        for i in range(block_repeat_num2):
            model = Conv2D(unit_size, kernel_size=(5, 5), strides=(1, 1), padding='same',
                           activation='relu')(model)
            model = MaxPooling2D(pool_size=(2, 2))(model)

        ret_model = Model(inputs=x, outputs=model, name=model_name)
        return ret_model

    def build_deconv_block(self, width, height, channel, block_repeat_num1, block_repeat_num2, model_name):
        x = Input(shape=(width, height, channel))
        unit_size = 32  # 64
        for i in range(block_repeat_num2):
            model = Conv2DTranspose(unit_size, (5, 5), strides=(1, 1), padding='same',
                                    activation='relu')(x)

        for i in range(block_repeat_num1):
            model = Conv2DTranspose(unit_size, (5, 5), strides=(1, 1), padding='same',
                                    activation='relu')(model)

        ret_model = Model(inputs=x, outputs=model, name=model_name)
        return ret_model
