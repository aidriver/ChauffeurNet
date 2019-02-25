#/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
sys.path.append("../")
from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.core import Reshape
from keras.optimizers import RMSprop

from keras.layers import Multiply
from keras.layers import Add
from keras.layers import Dot
from keras import backend as K
import tensorflow as tf
import numpy as np
import math
import time

from keras.utils.np_utils import to_categorical
#from keras.utils import multi_gpu_model
from keras.callbacks import ReduceLROnPlateau

from chauffeur_net.feature_net import FeatureNet
from chauffeur_net.agent_rnn import AgentRNN
from chauffeur_net.road_mask_net import RoadMaskNet
from chauffeur_net.perception_rnn import PerceptionRNN

imit_drop_out_ratio = 0.5


class ChauffeurNet:
    def __init__(self, conf_file):
        self.w_imitate = 1.0  # 0.0 random
        self.w_env = 1.0
        self.W = 400
        self.H_pos = 400
        self.H_neg = 400
        self.H = self.H_pos + self.H_neg
        self.T_scene = 1.0  # s
        self.T_pose = 8.0  # s
        self.dt = 0.2  # s
        self.N_out_steps = 10  # 25 40
        self.N_past_scene = int(self.T_scene / self.dt)
        self.N_past_pose = int(self.T_pose / self.dt)
        self.resolution_space = 0.2  # m
        self.u0 = self.H_pos
        self.v0 = self.W / 2
        self.max_pertu_theta = 25  # degree
        self.feature_channel_num = 0

        self._conf = conf_file
        pass

    def setup_model(self):
        input_imit_drop_out_weight = Input(shape=(1,), name='input_imit_drop_out_weight')
        '''
        def imit_mean_absolute_percentage_error_layer(x):
            return imit_mean_absolute_percentage_error(x[0], x[1])

        def imit_mean_absolute_percentage_error(y_true, y_pred):
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
            return K.mean(weight * diff, axis=-1)

        def imit_mean_absolute_error_layer(x):
            return imit_mean_absolute_error(x[0], x[1])

        def imit_mean_absolute_error(y_true, y_pred):
            y_true = K.cast(y_true, 'float32')
            y_pred = K.cast(y_pred, 'float32')
            #K.less has no gradient
            #greater less has no gradient
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true, dtype='float32') * imit_drop_out_ratio),
                            'float32')
            return K.mean(weight * K.abs(y_pred - y_true), axis=-1)

        def imit_categorical_crossentropy_layer(x):
            return imit_categorical_crossentropy(x[0], x[1])

        def imit_categorical_crossentropy(y_true, y_pred):
            # '' 'Expects a binary class matrix instead of a vector of scalar classes.
            # '' '
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            return weight * K.categorical_crossentropy(y_true, y_pred)

        def imit_binary_crossentropy_layer(x):
            return imit_binary_crossentropy(x[0], x[1])

        def imit_binary_crossentropy(y_true, y_pred):
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            return weight * K.binary_crossentropy(y_true, y_pred)

        '''

        def imit_mean_absolute_percentage_error_layer(x):
            return imit_mean_absolute_percentage_error(x[0], x[1])

        def imit_mean_absolute_percentage_error(y_true, y_pred):
            diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
            return K.mean(input_imit_drop_out_weight * diff, axis=-1)

        def imit_mean_absolute_error_layer(x):
            return imit_mean_absolute_error(x[0], x[1])

        def imit_mean_absolute_error(y_true, y_pred):
            y_true = K.cast(y_true, 'float32')
            y_pred = K.cast(y_pred, 'float32')

            return K.mean(input_imit_drop_out_weight * K.abs(y_pred - y_true)
                          ,
                          axis=-1
                          )

        def mean_absolute_error_anytype(y_true, y_pred):
            y_true = K.cast(y_true, 'float32')
            y_pred = K.cast(y_pred, 'float32')

            return K.mean(K.abs(y_pred - y_true)
                          ,
                          axis=-1
                          )

        def zero_loss_layer(y_true, y_pred):
            # return K.cast(K.zeros_like(y_true), 'float32')
            return K.zeros(K.shape(y_true)[0], 'float32')

        def imit_categorical_crossentropy_layer(x):
            return imit_categorical_crossentropy(x[0], x[1])

        def imit_categorical_crossentropy(y_true, y_pred):
            '''Expects a binary class matrix instead of a vector of scalar classes.
            '''
            return input_imit_drop_out_weight * K.categorical_crossentropy(y_true, y_pred)

        def imit_binary_crossentropy_layer_flatten(x):
            return K.sum(imit_binary_crossentropy(x[0], x[1]), axis=K.arange(1, K.ndim(x[0])))

        def imit_binary_crossentropy_layer(x):
            return imit_binary_crossentropy(x[0], x[1])

        def imit_binary_crossentropy(y_true, y_pred):
            return input_imit_drop_out_weight * K.binary_crossentropy(y_true, y_pred)

        # ---------------------------------------------------
        input_roadmap_img = Input(shape=(self.W, self.H, 3), name='input_roadmap_img')

        input_speed_limit_img = Input(shape=(self.W, self.H, 1), name='input_speed_limit_img')

        input_route_img = Input(shape=(self.W, self.H, 1), name='input_route_img')

        input_ego_current_img = Input(shape=(self.W, self.H, 1), name='input_ego_current_img')

        input_ego_past_img = Input(shape=(self.W, self.H, 1), name='input_ego_past_img')

        input_traffic_light_img_time_seq = [Input(shape=(self.W, self.H, 1), name='input_traffic_light_img_' + str(i))
                                            for i in range(self.N_past_scene)]

        input_obstacles_past_img_time_seq = [Input(shape=(self.W, self.H, 1), name='input_obstacles_past_img_' + str(i))
                                             for i in range(self.N_past_scene)]
        # just for calc loss in iteration
        input_road_mask_ground_truth_img = Input(shape=(self.W, self.H, 1), name='input_road_mask_ground_truth_img')
        input_geometry_ground_truth_img = Input(shape=(self.W, self.H, 1), name='input_geometry_ground_truth_img')

        # ---------------------------------------------------
        input_heading_step_ground_truth_for_loss = Input(shape=(self.N_out_steps,),
                                                         name='input_heading_step_ground_truth_for_loss')
        input_speed_step_ground_truth_for_loss = Input(shape=(self.N_out_steps,),
                                                       name='input_speed_step_ground_truth_for_loss')
        # input_categ_out_waypoint_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, self.W * self.H),#one-hot
        #                                               name='input_categ_out_waypoint_step_ground_truth_for_loss')
        # input_categ_out_waypoint_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, ),#integer need to one-hot when calc loss
        #                                               name='input_categ_out_waypoint_step_ground_truth_for_loss')
        # input_categ_out_waypoint_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, 2), dtype='int32',#integer need to one-hot when calc loss
        #                                                name='input_categ_out_waypoint_step_ground_truth_for_loss')
        input_categ_out_waypoint_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, self.W + self.H),
                                                                    # one-hot
                                                                    name='input_categ_out_waypoint_step_ground_truth_for_loss')
        input_categ_out_agent_box_heat_map_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, self.W, self.H),
                                                                              name='input_categ_out_agent_box_heat_map_step_ground_truth_for_loss')
        input_waypoint_sub_pixel_step_ground_truth_for_loss = Input(shape=(self.N_out_steps, 2),
                                                                    name='input_waypoint_sub_pixel_step_ground_truth_for_loss')
        input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss = Input(
                shape=(self.N_out_steps, self.W, self.H),
                name='input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss')

        input_vec = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                     input_ego_past_img]
        input_vec.extend(input_traffic_light_img_time_seq)
        input_vec.extend(input_obstacles_past_img_time_seq)
        input_vec.extend([input_road_mask_ground_truth_img, input_geometry_ground_truth_img,
                          input_heading_step_ground_truth_for_loss, input_speed_step_ground_truth_for_loss,
                          input_categ_out_waypoint_step_ground_truth_for_loss,
                          input_categ_out_agent_box_heat_map_step_ground_truth_for_loss,
                          input_waypoint_sub_pixel_step_ground_truth_for_loss,
                          input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss
                          ])

        input_vec.append(input_imit_drop_out_weight)
        print('input_vec size: ', len(input_vec), input_vec)

        # input_predict_ego_pos_memory_const = Input(shape=(self.W, self.H, 1), name='input_predict_ego_pos_memory_const')
        # input_vec.append(input_predict_ego_pos_memory_const)
        # ---------------------------------------------------

        def tensor_expand(tensor_input, num, axis=0):
            '''
            张量自我复制扩展，将num个tensor_Input串联起来，生成新的张量，
            新的张量的shape=[tensor_Input.shape,num]
            :param tensor_Input:
            :param num:
            :return:
            '''
            tensor_input = tf.expand_dims(tensor_input, axis)
            tensor_output = tensor_input
            for i in range(num - 1):
                tensor_output = tf.concat([tensor_output, tensor_input], axis)
            return tensor_output

        #def slice_tensor(param):
        #    x, row, col = param

        def slice_tensor(x, row, col):
            # cannot do this use uninitialized value
            # row col is tensor batch,not fixed
            # with tf.Session() as sess:
            #    return x[:, row.eval(session=sess), col.eval(session=sess),:]
            return x[row, col, :]

        def set_tensor_index_value(x, row, col, value):
            # ret = tf.Variable(initial_value=x)
            # ret = tf.identity(x)
            # ret = x
            # ret = tf.Variable(tf.zeros(K.shape(x)))
            # new_value = tf.ones(shape=tf.shape(x)[0])
            # new_value = value * tf.ones(shape=(tf.shape(x))[0])
            # new_value = value * x
            # ret[:, row:row + 1, col:col + 1, :].assign(new_value)
            # batch_size = tf.shape(x)[0]
            # new_value = K.zeros_like(x)
            # new_value = np.zeros(K.int_shape(x))
            # new_value[:, row:row + 1, col:col + 1, :] = value
            # ret[:, row:row + 1, col:col + 1, :].assign(value)
            # ret = x
            # ret[:, row:row + 1, col:col + 1, :] = value
            # tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            # 生成一个one_hot张量，长度与tensor_1相同，修改位置为1
            # shape = x.get_shape().as_list()
            '''
            one_hot_row = tf.one_hot(row, self.W, dtype=tf.float32)
            one_hot_col = tf.one_hot(col, self.H, dtype=tf.float32)

            one_hot_rows = tensor_expand(one_hot_row, self.H, 2)
            one_hot_cols = tensor_expand(one_hot_col, self.W, 2)
            one_hot_cols = tf.transpose(one_hot_cols, perm=[1,2]) #should not consider batch_size
            '''
            one_hot_row = K.one_hot(row, self.W)
            one_hot_col = K.one_hot(col, self.H)

            one_hot_rows = tensor_expand(one_hot_row, self.H, 2)
            one_hot_cols = tensor_expand(one_hot_col, self.W, 2)
            one_hot_cols = K.permute_dimensions(one_hot_cols, pattern=[0, 2, 1])  # should not consider batch_size
            # one_hot_cols = K.transpose(one_hot_cols, perm=[1, 2])

            one_hot_matrix = one_hot_rows * one_hot_cols

            # one_hot_matrix = tf.matmul(one_hot_rows, one_hot_cols)
            # 做一个减法运算，将one_hot为一的变为0,再加上新的值，为0的地方没减也没加
            # one_hot_matrix = K.reshape(one_hot_matrix, K.shape(one_hot_matrix) + (1,))
            one_hot_matrix = K.expand_dims(one_hot_matrix, axis=-1)
            # row,col is tensor,should use slice function can't use common index or use eval?
            # with sess.as_default():
            # tf_session = K.get_session()
            # cannot use eval now,train hasn't begun
            # ret = x - x[:, row.eval(session=tf_session), col.eval(session=tf_session), :] * one_hot_matrix + value * one_hot_matrix
            # ret = x - x[:, row, col,:] * one_hot_matrix + value * one_hot_matrix
            # shape = K.shape(x)
            # shape[1:2] = row, col
            # kvar = K.variable(value=(row, col), dtype='int32')
            # shape = shape[0] + kvar + shape[3]
            # new_shape = K.expand_dims(shape[0],row)
            # new_shape = K.expand_dims(new_shape, col)
            # new_shape = K.expand_dims(new_shape, shape[3])
            # shape = np.asarray(K.int_shape(x))
            # shape[1:2] = row, col #row,col is also Tensor
            # K.set_value(new_shape, shape)
            row = K.cast(row, dtype='int32')
            col = K.cast(col, dtype='int32')
            # new_idx = K.stack([shape[0], row, col, shape[3]])
            # new_idx = K.concatenate([K.arange(0,K.shape(x)[0].value), row, col, K.arange(0,K.shape(x)[3].value)])
            # ret = x - K.gather(x, new_idx) * one_hot_matrix + value * one_hot_matrix
            # ret = x - K.gather(x, [None, row, col, None]) * one_hot_matrix + value * one_hot_matrix
            # here is only one dim output,support dtype=('float32','float32')
            slice_pos = K.map_fn(lambda params: slice_tensor(params[0], params[1], params[2]),
                                 (x, row, col), dtype='float32')
            #slice_pos = [batch[row, col, 0] for batch in x]
            # slice_pos = Lambda(slice_tensor)([x,row,col])
            ret = x - slice_pos * one_hot_matrix + value * one_hot_matrix
            return ret

        feature_net = FeatureNet(self._conf)
        feature_net.input_vec = input_vec
        # feature_net.out_W_size = 64  # ?? how to keep size?and deconv?
        # feature_net.out_H_size = 64  # ?? how to keep size?and deconv?
        model_feature_net = feature_net.setup_model()
        features = model_feature_net(input_vec)
        _, feature_net.out_W_size, feature_net.out_H_size, feature_net.feature_channel_num = \
            model_feature_net.get_output_shape_at(0)  # model_feature_net.outputs[0],
        print(feature_net.out_W_size, feature_net.out_H_size, feature_net.feature_channel_num)
        # unrolled
        agent_rnn = AgentRNN(self._conf)
        agent_rnn.feature_net = feature_net
        model_agent_rnn = agent_rnn.setup_model()

        model_agent_rnn_out_reg_out_heading_vec = []
        model_agent_rnn_out_reg_out_speed_vec = []
        model_agent_rnn_out_categ_out_waypoint_vec = []
        model_agent_rnn_out_categ_out_agent_box_heat_map_vec = []
        model_agent_rnn_out_categ_out_waypoint_sub_pixel_vec = []

        model_agent_rnn_out_categ_out_waypoint = None
        model_agent_rnn_out_reg_out_heading = None
        model_agent_rnn_out_reg_out_speed = None
        model_agent_rnn_out_categ_out_agent_box_heat_map = None
        model_agent_rnn_out_categ_out_waypoint_sub_pixel = None

        # batch_size = 1 # ??? K.int_shape(input_speed_limit_img)[0]
        # input_predict_ego_pos_memory = K.zeros(shape=K.shape(input_ego_past_img))
        # input_predict_ego_pos_memory = Lambda(lambda x: x)(K.zeros(shape=K.shape(input_ego_past_img)))

        def zero_layer(x):
            return K.zeros_like(x)

        input_predict_ego_pos_memory = Lambda(zero_layer)(input_ego_past_img)
        # input_predict_ego_pos_memory = Lambda(lambda x: x)(K.zeros_like(input_ego_past_img)) #K.zeros_like not a layer so lambda has no use
        # input_predict_ego_pos_memory = K.zeros(shape=(batch_size, self.W, self.H, 1))
        # input_predict_ego_pos_memory = input_predict_ego_pos_memory_const
        input_last_predict_ego_box = input_ego_current_img
        for k in range(0, self.N_out_steps):
            # print(k)
            # should use Lambda Layer or will wrong in iteration
            #input_k_tensor = K.ones(K.shape(input_ego_past_img)) * k
            # input_k = np.ones((batch_size, 1)) * k
            # input_k_tensor = tf.convert_to_tensor(input_k)
            input_agent_rnn_vec = [input_ego_past_img,  # input_k_tensor,
                                   input_predict_ego_pos_memory,
                                   input_last_predict_ego_box, features]  # [input_ego_past_img, features]  #
            model_agent_rnn_out = model_agent_rnn(input_agent_rnn_vec)

            # must remap sun_model out to layer with name for multi-output&loss keys work normal,
            # or sub model has multi output with same name of submodel name
            model_agent_rnn_out_categ_out_waypoint = Activation('linear', name='categ_out_waypoint_' + str(k))(
                    model_agent_rnn_out[0])
            model_agent_rnn_out_categ_out_waypoint_vec.append(model_agent_rnn_out_categ_out_waypoint)

            model_agent_rnn_out_reg_out_heading = Activation('linear', name='reg_out_heading_' + str(k))(
                    model_agent_rnn_out[1])
            # model_agent_rnn_out_reg_out_heading = model_agent_rnn_out[1]
            model_agent_rnn_out_reg_out_heading_vec.append(model_agent_rnn_out_reg_out_heading)

            model_agent_rnn_out_reg_out_speed = Activation('linear', name='reg_out_speed_' + str(k))(
                    model_agent_rnn_out[2])
            model_agent_rnn_out_reg_out_speed_vec.append(model_agent_rnn_out_reg_out_speed)

            model_agent_rnn_out_categ_out_agent_box_heat_map = Activation('linear',
                                                                          name='categ_out_agent_box_heat_map_' + str(
                                                                                  k))(
                    model_agent_rnn_out[3])
            model_agent_rnn_out_categ_out_agent_box_heat_map_vec.append(
                    model_agent_rnn_out_categ_out_agent_box_heat_map)

            model_agent_rnn_out_categ_out_waypoint_sub_pixel = Activation('linear',
                                                                          name='waypoint_sub_pixel_' + str(k))(
                    model_agent_rnn_out[4])
            model_agent_rnn_out_categ_out_waypoint_sub_pixel_vec.append(
                    model_agent_rnn_out_categ_out_waypoint_sub_pixel)

            # set
            # new_shape = K.concatenate([K.shape(model_agent_rnn_out_categ_out_agent_box_heat_map),
            #                           (1,)])
            new_shape = K.int_shape(model_agent_rnn_out_categ_out_agent_box_heat_map)[1:] + (1,)
            # new_shape = tf.convert_to_tensor(new_shape)
            input_last_predict_ego_box = Reshape(new_shape)(model_agent_rnn_out_categ_out_agent_box_heat_map)

            # add

            # pose_index = K.argmax(model_agent_rnn_out_categ_out_waypoint)
            # pose_index = K.eval(model_agent_rnn_out_categ_out_waypoint) #too slow!!
            # pose_index = K.eval(K.argmax(model_agent_rnn_out_categ_out_waypoint)) #too slow!!
            # r, c = pose_index // self.H, pose_index % self.H
            # r, c = K.cast(model_agent_rnn_out_categ_out_waypoint[0], 'int32'), K.cast(model_agent_rnn_out_categ_out_waypoint[1], 'int32')
            # r, c = model_agent_rnn_out_categ_out_waypoint[0], model_agent_rnn_out_categ_out_waypoint[1]
            # r, c = K.argmax(model_agent_rnn_out_categ_out_waypoint[0]), K.argmax(model_agent_rnn_out_categ_out_waypoint[1])
            r, c = K.argmax(model_agent_rnn_out_categ_out_waypoint[:, 0:self.W]), \
                   K.argmax(model_agent_rnn_out_categ_out_waypoint[:, self.W:])
            # img_array = K.eval(input_predict_ego_pos_memory) #too slow,not allowed when not given data!!
            # img_array[:, r:r+1, c:c+1] = 1.0
            # input_predict_ego_pos_memory = tf.convert_to_tensor(img_array)
            # img_array = np.zeros(K.int_shape(model_agent_rnn_out_categ_out_agent_box_heat_map))
            # img_array[:, r:r + 1, c:c + 1] = 1.0
            # K.set_value(input_predict_ego_pos_memory, img_array)
            set_tensor_index_value_layer = Lambda(set_tensor_index_value,
                                                  arguments={'row': r,
                                                             # note r,c is tensor not fixed value cannot work ,should as batch param?
                                                             'col': c,
                                                             'value': 1.0},
                                                  name='set_tensor_index_value_layer_' + str(k))

            input_predict_ego_pos_memory = set_tensor_index_value_layer(input_predict_ego_pos_memory)
            # new_shape = K.shape(input_predict_ego_pos_memory)[1:] + (1,)
            # input_predict_ego_pos_memory = Reshape(new_shape)(input_predict_ego_pos_memory)
            # input_predict_ego_pos_memory = Add()([agent_rnn_out_location_img, input_predict_ego_pos_memory])

        model_agent_rnn_out_categ_out_waypoint = Activation('linear', name='categ_out_waypoint')(
                model_agent_rnn_out_categ_out_waypoint_vec[-1])

        model_agent_rnn_out_reg_out_heading = Activation('linear', name='reg_out_heading')(
                model_agent_rnn_out_reg_out_heading_vec[-1])

        model_agent_rnn_out_reg_out_speed = Activation('linear', name='reg_out_speed')(
                model_agent_rnn_out_reg_out_speed_vec[-1])

        model_agent_rnn_out_categ_out_agent_box_heat_map = Activation('linear',
                                                                      name='categ_out_agent_box_heat_map')(
                model_agent_rnn_out_categ_out_agent_box_heat_map_vec[-1])

        model_agent_rnn_out_categ_out_waypoint_sub_pixel = Activation('linear', name='waypoint_sub_pixel')(
                model_agent_rnn_out_categ_out_waypoint_sub_pixel_vec[-1])
        model_agent_rnn_out_layers = [
            model_agent_rnn_out_categ_out_waypoint,
            model_agent_rnn_out_reg_out_heading,
            model_agent_rnn_out_reg_out_speed,
            model_agent_rnn_out_categ_out_agent_box_heat_map,
            model_agent_rnn_out_categ_out_waypoint_sub_pixel
        ]

        road_mask_net = RoadMaskNet(self._conf)
        road_mask_net.feature_net = feature_net
        model_road_mask_net = road_mask_net.setup_model()
        model_road_mask_net_out = model_road_mask_net(features)
        # must remap sun_model out to layer with name for multi-output&loss keys work normal,or sub model has multi output with same name of submodel name
        model_road_mask_net_out_bin_out_road_mask = Activation('linear', name='bin_out_road_mask')(
                model_road_mask_net_out)  # model_road_mask_net_out[0]
        model_road_mask_net_out_layers = [model_road_mask_net_out_bin_out_road_mask]

        # unrolled
        perception_rnn = PerceptionRNN(self._conf)
        perception_rnn.feature_net = feature_net
        model_perception_rnn = perception_rnn.setup_model()

        model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec = []
        model_perception_rnn_out_categ_out_obstacle_box_heat_map = None

        input_last_predict_obs_box = input_obstacles_past_img_time_seq[-1]
        for k in range(0, self.N_out_steps):
            # input_k = np.ones(K.int_shape(input_ego_past_img)[0]) * k
            # input_k_tensor = tf.convert_to_tensor(input_k)
            #input_k_tensor = K.ones(K.shape(input_ego_past_img)) * k
            input_perception_rnn = [  # input_k_tensor,
                input_last_predict_obs_box, features]
            model_perception_rnn_out = model_perception_rnn(input_perception_rnn)
            # must remap sun_model out to layer with name for multi-output&loss keys work normal,or sub model has multi output with same name of submodel name
            model_perception_rnn_out_categ_out_obstacle_box_heat_map = Activation('linear',
                                                                                  name='categ_out_obstacle_box_heat_map_' + str(
                                                                                          k))(
                    model_perception_rnn_out)  # model_perception_rnn_out[0]
            model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec.append(
                    model_perception_rnn_out_categ_out_obstacle_box_heat_map)

            # set
            new_shape = K.int_shape(model_perception_rnn_out_categ_out_obstacle_box_heat_map)[1:] + (1,)
            input_last_predict_obs_box = Reshape(new_shape)(model_perception_rnn_out_categ_out_obstacle_box_heat_map)

        model_perception_rnn_out_categ_out_obstacle_box_heat_map = Activation('linear',
                                                                              name='categ_out_obstacle_box_heat_map')(
                model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec[-1])

        model_perception_rnn_out_layers = [model_perception_rnn_out_categ_out_obstacle_box_heat_map]

        final_outputs = []
        final_outputs.extend(model_agent_rnn_out_layers)  # multi-output
        final_outputs.extend(model_road_mask_net_out_layers)  # finally single-ouput
        final_outputs.extend(model_perception_rnn_out_layers)  # finally single-ouput

        # --------loss output can give any value for label,loss not use ground truth,only use predict value----------
        reg_out_heading_loss_vec = []
        reg_out_speed_loss_vec = []
        categ_out_waypoint_loss_vec = []
        categ_out_agent_box_heat_map_loss_vec = []
        waypoint_sub_pixel_loss_vec = []

        collision_loss_vec = []
        on_road_loss_vec = []
        geometry_loss_vec = []
        categ_out_obstacle_box_heat_map_loss_vec = []

        def slice_1d(x, k):
            return x[:, k:k + 1]

        def slice_2d(x, k):
            return x[:, k:k + 1, :]

        def slice_3d(x, k):
            return x[:, k:k + 1, :, :]

        ''''''
        for k in range(0, self.N_out_steps):
            # reg_out_heading_loss = K.reshape(imit_mean_absolute_error(input_heading_step_ground_truth_for_loss[:, k:k+1],
            #                                                model_agent_rnn_out_reg_out_heading_vec[k]),
            #                                 K.shape(model_agent_rnn_out_reg_out_heading_vec[k])[0])
            # input_heading_step_ground_truth_for_loss[:, k:k + 1] wrong not a layer,just a Tensor,cannot as an output layer
            # reg_out_heading_loss = Lambda(lambda x: x)(imit_mean_absolute_error(
            #        Lambda(slice_1d, arguments={'k': k})(input_heading_step_ground_truth_for_loss),
            #                             model_agent_rnn_out_reg_out_heading_vec[k])) #wrong no input layer
            reg_out_heading_loss = Lambda(imit_mean_absolute_error_layer)([
                Lambda(slice_1d, arguments={'k': k})(input_heading_step_ground_truth_for_loss),
                model_agent_rnn_out_reg_out_heading_vec[k]])
            reg_out_heading_loss_vec.append(reg_out_heading_loss)

            reg_out_speed_loss = Lambda(imit_mean_absolute_error_layer)([
                Lambda(slice_1d, arguments={'k': k})(input_speed_step_ground_truth_for_loss),
                model_agent_rnn_out_reg_out_speed_vec[k]])
            reg_out_speed_loss_vec.append(reg_out_speed_loss)

            # categ_out_waypoint_loss = Lambda(imit_categorical_crossentropy_layer)([
            #     Reshape(K.int_shape(model_agent_rnn_out_categ_out_waypoint_vec[k])[1:])
            #             (Lambda(slice_2d, arguments={'k': k})(input_categ_out_waypoint_step_ground_truth_for_loss)
            #                                                                        ),
            #                                                         model_agent_rnn_out_categ_out_waypoint_vec[k]])
            # categ_out_waypoint_gt = Reshape(K.int_shape(model_agent_rnn_out_categ_out_waypoint_vec[k])[1:])
            # (Lambda(slice_2d, arguments={'k': k})(input_categ_out_waypoint_step_ground_truth_for_loss)
            #  )
            # categ_out_waypoint_gt = Lambda(lambda x:x)(K.one_hot(categ_out_waypoint_gt, self.W * self.H))
            # categ_out_waypoint_loss = Lambda(imit_categorical_crossentropy_layer)([categ_out_waypoint_gt,
            #                                                         model_agent_rnn_out_categ_out_waypoint_vec[k]])
            # K.one_hot(model_agent_rnn_out_categ_out_waypoint_vec[k], nb_classes)
            # K.reshape not a layer, no need to nest Lambda so many level? just encode in imit_categorical_crossentropy_layer?
            # categ_out_waypoint_loss = Lambda(imit_categorical_crossentropy_layer)([K.reshape(
            #        Lambda(slice_2d, arguments={'k': k})(input_categ_out_waypoint_step_ground_truth_for_loss),
            #                                                                  K.shape(model_agent_rnn_out_categ_out_waypoint_vec[k])),
            #                                                        model_agent_rnn_out_categ_out_waypoint_vec[k]])
            categ_out_waypoint_gt = Lambda(slice_2d, arguments={'k': k})(
                    input_categ_out_waypoint_step_ground_truth_for_loss)
            # categ_out_waypoint_loss = Lambda(imit_mean_absolute_error_layer)([categ_out_waypoint_gt,
            #                                                        model_agent_rnn_out_categ_out_waypoint_vec[k]])
            categ_out_waypoint_loss = Lambda(imit_categorical_crossentropy_layer)([categ_out_waypoint_gt,
                                                                                   model_agent_rnn_out_categ_out_waypoint_vec[
                                                                                       k]])
            categ_out_waypoint_loss_vec.append(categ_out_waypoint_loss)
            # imit_binary_crossentropy_layer
            categ_out_agent_box_heat_map_loss = Lambda(imit_binary_crossentropy_layer_flatten)([
                Reshape(K.int_shape(model_agent_rnn_out_categ_out_agent_box_heat_map_vec[k])[1:])
                (Lambda(slice_3d, arguments={'k': k})(input_categ_out_agent_box_heat_map_step_ground_truth_for_loss)),
                model_agent_rnn_out_categ_out_agent_box_heat_map_vec[k]])
            categ_out_agent_box_heat_map_loss_vec.append(categ_out_agent_box_heat_map_loss)

            waypoint_sub_pixel_loss = Lambda(imit_mean_absolute_error_layer)([
                Reshape(K.int_shape(model_agent_rnn_out_categ_out_waypoint_sub_pixel_vec[k])[1:])(
                        Lambda(slice_2d, arguments={'k': k})(input_waypoint_sub_pixel_step_ground_truth_for_loss)),
                model_agent_rnn_out_categ_out_waypoint_sub_pixel_vec[k]])
            waypoint_sub_pixel_loss_vec.append(waypoint_sub_pixel_loss)

            flat_agent_heat_map = Flatten()(model_agent_rnn_out_categ_out_agent_box_heat_map_vec[k])

            collision_loss = Dot(1, name='collision_' + str(k))([flat_agent_heat_map,
                                                                 Flatten()(
                                                                         model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec[
                                                                             k])])
            # collision_loss = Dot((1,2),name='collision')([model_agent_rnn_out_categ_out_agent_box_heat_map, model_perception_rnn_out_categ_out_obstacle_box_heat_map])
            # loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
            collision_loss_vec.append(collision_loss)

            onroad = Lambda(lambda x: 1 - x)(input_road_mask_ground_truth_img)
            flat_not_onroad = Flatten()(onroad)
            on_road_loss = Dot(1, name='on_road_' + str(k))([flat_agent_heat_map, flat_not_onroad])
            on_road_loss_vec.append(on_road_loss)

            on_geometry = Lambda(lambda x: 1 - x)(input_geometry_ground_truth_img)
            flat_not_on_geometry = Flatten()(on_geometry)
            geometry_loss = Dot(1, name='geometry_' + str(k))([flat_agent_heat_map, flat_not_on_geometry])
            geometry_loss_vec.append(geometry_loss)
            # imit_binary_crossentropy_layer
            categ_out_obstacle_box_heat_map_loss = Lambda(imit_binary_crossentropy_layer_flatten)([
                Reshape(K.int_shape(model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec[k])[1:])(
                        Lambda(slice_3d, arguments={'k': k})(
                                input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss)),
                model_perception_rnn_out_categ_out_obstacle_box_heat_map_vec[k]])
            categ_out_obstacle_box_heat_map_loss_vec.append(categ_out_obstacle_box_heat_map_loss)

        reg_out_heading_loss = Add(name='add_reg_out_heading_loss')(reg_out_heading_loss_vec)
        reg_out_heading_loss = Reshape((1,), name='reg_out_heading_loss')(reg_out_heading_loss)

        reg_out_speed_loss = Add(name='add_reg_out_speed_loss')(reg_out_speed_loss_vec)
        reg_out_speed_loss = Reshape((1,), name='reg_out_speed_loss')(reg_out_speed_loss)

        categ_out_waypoint_loss = Add(name='categ_out_waypoint_loss')(categ_out_waypoint_loss_vec)

        categ_out_agent_box_heat_map_loss = Add(name='add_categ_out_agent_box_heat_map_loss')(
                categ_out_agent_box_heat_map_loss_vec)
        categ_out_agent_box_heat_map_loss = Reshape((1,), name='categ_out_agent_box_heat_map_loss')(
                categ_out_agent_box_heat_map_loss)

        waypoint_sub_pixel_loss = Add(name='add_waypoint_sub_pixel_loss')(waypoint_sub_pixel_loss_vec)
        waypoint_sub_pixel_loss = Reshape((1,), name='waypoint_sub_pixel_loss')(
                waypoint_sub_pixel_loss)

        collision_loss = Add(name='collision')(collision_loss_vec)
        on_road_loss = Add(name='on_road')(on_road_loss_vec)
        geometry_loss = Add(name='geometry')(geometry_loss_vec)

        categ_out_obstacle_box_heat_map_loss = Add(name='add_categ_out_obstacle_box_heat_map_loss')(
                categ_out_obstacle_box_heat_map_loss_vec)
        categ_out_obstacle_box_heat_map_loss = Reshape((1,), name='categ_out_obstacle_box_heat_map_loss')(
                categ_out_obstacle_box_heat_map_loss)

        final_outputs.append(categ_out_waypoint_loss)
        final_outputs.append(reg_out_heading_loss)
        final_outputs.append(reg_out_speed_loss)
        final_outputs.append(categ_out_agent_box_heat_map_loss)
        final_outputs.append(waypoint_sub_pixel_loss)

        final_outputs.append(collision_loss)
        final_outputs.append(on_road_loss)
        final_outputs.append(geometry_loss)
        final_outputs.append(categ_out_obstacle_box_heat_map_loss)
        print('final_outputs size:', len(final_outputs))

        output_vec = final_outputs
        model = Model(inputs=input_vec, outputs=output_vec)

        rmsprop = RMSprop(  # clipnorm=1.0,#1.0 15.0
                # clipvalue=0.5,#0.5 8.0
                lr=0.01, rho=0.9, epsilon=1e-08, decay=1e-6)
        # key must same with multi-output-layer name. final_ouput merge multi submodel ouputs,so its original layers'name lost,just use model name,and duplicate with multi-ouput submodel
        model.compile(optimizer=rmsprop,
                      # when train must use
                      loss={  # vec shape init numpy.zeros()
                          # 'categ_out_waypoint': imit_categorical_crossentropy,  # 'categorical_crossentropy',
                          # cannot use no gradient function in loss(K.argmax/argmin,eval,round)
                          # 'categ_out_waypoint': 'mae',
                          'categ_out_waypoint': zero_loss_layer,  # no gradient?constant?
                          # 'categ_out_waypoint': mean_absolute_error_anytype,
                          # 'categ_out_waypoint': lambda y_true, y_pred: K.cast(K.zeros_like(y_pred), 'float32'),
                          # 'reg_out_heading': imit_mean_absolute_error,  # calculated loss  # 'mae',
                          'reg_out_heading': 'mae',
                          # 'reg_out_heading': lambda y_true, y_pred: K.zeros_like(y_pred),#0.0
                          # 'reg_out_speed': imit_mean_absolute_error,  # 'mae',
                          'reg_out_speed': 'mae',
                          # 'reg_out_speed': lambda y_true, y_pred: K.zeros_like(y_pred),

                          # 'categ_out_agent_box_heat_map': imit_categorical_crossentropy,#'categorical_crossentropy' #,
                          # 'categ_out_agent_box_heat_map': imit_binary_crossentropy,  # 'binary_crossentropy',
                          'categ_out_agent_box_heat_map': 'mae',
                          # 'categ_out_agent_box_heat_map': lambda y_true, y_pred: K.zeros_like(y_pred),
                          # 'waypoint_sub_pixel': imit_mean_absolute_error,
                          'waypoint_sub_pixel': 'mae',
                          # 'waypoint_sub_pixel': lambda y_true, y_pred: K.zeros_like(y_pred),
                          # 'mae',#note groundth should be pose-[pose]

                          'bin_out_road_mask': 'binary_crossentropy',
                          # no use ,because its weight is 0.0
                          'categ_out_obstacle_box_heat_map': 'mae',  # slow?
                          # 'categ_out_obstacle_box_heat_map': 'categorical_crossentropy',#slow?
                          # gradient is None!!
                          # 'categ_out_obstacle_box_heat_map': lambda y_true, y_pred: K.zeros_like(y_pred),
                          # ------------
                          'categ_out_waypoint_loss': lambda y_true, y_pred: y_pred,
                          'reg_out_heading_loss': lambda y_true, y_pred: y_pred,
                          'reg_out_speed_loss': lambda y_true, y_pred: y_pred,
                          'categ_out_agent_box_heat_map_loss': lambda y_true, y_pred: y_pred,
                          'waypoint_sub_pixel_loss': lambda y_true, y_pred: y_pred,

                          'collision': lambda y_true, y_pred: y_pred,  # the output is loss
                          'on_road': lambda y_true, y_pred: y_pred,
                          'geometry': lambda y_true, y_pred: y_pred,
                          # 'categ_out_obstacle_box_heat_map': 'categorical_crossentropy',
                          # 'categ_out_obstacle_box_heat_map': 'binary_crossentropy',

                          'categ_out_obstacle_box_heat_map_loss': lambda y_true, y_pred: y_pred
                          # not consider iteration k,all other loss should sum over iteration(even perception RNN)

                      },
                      loss_weights={
                          'categ_out_waypoint': 0.0,  # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,
                          'reg_out_heading': 0.0,  # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,
                          'reg_out_speed': 0.0,  # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,
                          'categ_out_agent_box_heat_map': 0.0,
                          # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,
                          'waypoint_sub_pixel': 0.0,  # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,

                          'bin_out_road_mask': self.w_env / (self.W * self.H),
                          # lambda y_true, y_pred, weights: K.ones_like(y_pred) * self.w_env / (self.W * self.H),#self.w_env / (self.W * self.H),

                          'categ_out_obstacle_box_heat_map': 0.0,
                          # lambda y_true, y_pred, weights: K.zeros_like(y_pred),#0.0,
                          # ------------
                          'categ_out_waypoint_loss': self.w_imitate,
                          'reg_out_heading_loss': self.w_imitate,
                          'reg_out_speed_loss': self.w_imitate,
                          'categ_out_agent_box_heat_map_loss': self.w_imitate / (self.W * self.H),
                          'waypoint_sub_pixel_loss': self.w_imitate,

                          'collision': self.w_env / (self.W * self.H),
                          'on_road': self.w_env / (self.W * self.H),
                          'geometry': self.w_env / (self.W * self.H),
                          'categ_out_obstacle_box_heat_map_loss': self.w_env / (self.W * self.H)

                      },
                      metrics=['accuracy'])
        model.summary()
        return model

    def setup_optimizer(self):
        pass


if __name__ == '__main__':
    conf_path = 'conf/chauffeur_net.conf'
    root_input_path = '/home/caros/offline/chauffeur_net'
    root_output_path = '/home/caros/offline/chauffeur_net'
    chauffeur_net = ChauffeurNet(conf_path)
    model = chauffeur_net.setup_model()
    # exit(0)

    lr_cb = ReduceLROnPlateau(
            monitor='val_loss',  # ''val_acc' 'val_loss'
            factor=0.5,  # 0.8 0.5 0.1
            patience=1,  # 1 5 10
            verbose=1,
            mode='auto',  # 'auto' 'max' 'min'
            epsilon=2e-2,  # 5e-2#1e-5 2e-3 valid not use class_weight
            cooldown=0,  # 0
            min_lr=1e-5
    )
    cbks = [lr_cb]

    valid_ratio = 0.5  # 0.2
    train_batch_size = int(1 / (1 - valid_ratio))  # 1 8 512

    input_roadmap_img = np.random.uniform(low=0.0, high=1.0,
                                          size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 3))

    input_speed_limit_img = np.random.uniform(low=0.0, high=1.0,
                                              size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_route_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_current_img = np.random.uniform(low=0.0, high=1.0,
                                              size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_past_img = np.random.uniform(low=0.0, high=1.0,
                                           size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_traffic_light_img_time_seq = [
        np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
        for i in range(chauffeur_net.N_past_scene)]

    input_obstacles_past_img_time_seq = [
        np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
        for i in range(chauffeur_net.N_past_scene)]

    input_road_mask_ground_truth_img = np.random.randint(low=0, high=2,
                                                         size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
    input_geometry_ground_truth_img = np.random.uniform(low=0.0, high=1.0,
                                                        size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
    input_heading_step_ground_truth_for_loss = np.random.uniform(low=-1.0, high=1.0,
                                                                 size=(train_batch_size, chauffeur_net.N_out_steps,))
    input_speed_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0,
                                                               size=(train_batch_size, chauffeur_net.N_out_steps,))
    # input_categ_out_waypoint_step_ground_truth_for_loss = np.random.randint(low=0, high=2, size=(train_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W * chauffeur_net.H))
    # input_categ_out_waypoint_step_ground_truth_for_loss = np.concatenate([np.asarray(np.random.randint(low=0, high=chauffeur_net.W,
    #                                                                                        size=(train_batch_size, chauffeur_net.N_out_steps, 1))),
    #                                                                       np.asarray(np.random.randint(low=0, high=chauffeur_net.H,
    #                                                                                        size=(train_batch_size,
    #                                                                                              chauffeur_net.N_out_steps, 1)))], axis=-1)
    input_categ_out_waypoint_step_ground_truth_for_loss_W = np.asarray(np.random.uniform(low=0, high=chauffeur_net.W,
                                                                                         size=(train_batch_size,
                                                                                               chauffeur_net.N_out_steps,
                                                                                               1)))
    # not neccessary for one-hot?
    input_categ_out_waypoint_step_ground_truth_for_loss_W_pos = np.round(
            input_categ_out_waypoint_step_ground_truth_for_loss_W)
    input_categ_out_waypoint_step_ground_truth_for_loss_W_one_hot = to_categorical(
            input_categ_out_waypoint_step_ground_truth_for_loss_W_pos,
            num_classes=chauffeur_net.W
    )

    input_categ_out_waypoint_step_ground_truth_for_loss_H = np.asarray(np.random.uniform(low=0, high=chauffeur_net.H,
                                                                                         size=(train_batch_size,
                                                                                               chauffeur_net.N_out_steps,
                                                                                               1)))
    # not neccessary for one-hot?
    input_categ_out_waypoint_step_ground_truth_for_loss_H_pos = np.round(
            input_categ_out_waypoint_step_ground_truth_for_loss_H)
    input_categ_out_waypoint_step_ground_truth_for_loss_H_one_hot = to_categorical(
            input_categ_out_waypoint_step_ground_truth_for_loss_H_pos,
            num_classes=chauffeur_net.H
    )

    input_categ_out_waypoint_step_ground_truth_for_loss = np.concatenate(
            [input_categ_out_waypoint_step_ground_truth_for_loss_W_one_hot,
             input_categ_out_waypoint_step_ground_truth_for_loss_H_one_hot
             ], axis=-1)

    input_categ_out_agent_box_heat_map_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W, chauffeur_net.H))
    input_waypoint_sub_pixel_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.N_out_steps, 2))
    input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W, chauffeur_net.H))

    train_inputs = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                    input_ego_past_img]
    train_inputs.extend(input_traffic_light_img_time_seq)
    train_inputs.extend(input_obstacles_past_img_time_seq)
    # need for calculate loss in train phase
    train_inputs.extend([input_road_mask_ground_truth_img, input_geometry_ground_truth_img,
                         input_heading_step_ground_truth_for_loss, input_speed_step_ground_truth_for_loss,
                         input_categ_out_waypoint_step_ground_truth_for_loss,
                         input_categ_out_agent_box_heat_map_step_ground_truth_for_loss,
                         input_waypoint_sub_pixel_step_ground_truth_for_loss,
                         input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss
                         ])
    input_imit_drop_out_weight = (
        np.less(np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,)), imit_drop_out_ratio)).astype(
            np.float32)
    train_inputs.append(input_imit_drop_out_weight)
    print('train_inputs size:', len(train_inputs))

    # model_agent_rnn_out_categ_out_waypoint = np.random.randint(low=0, high=320000, size=(train_batch_size,))
    model_agent_rnn_out_categ_out_waypoint = np.concatenate([
        np.round(np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, 1)) * chauffeur_net.W),
        np.round(np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, 1)) * chauffeur_net.H)
    ], axis=-1)
    model_agent_rnn_out_reg_out_heading = np.random.uniform(low=0.0, high=1.0,
                                                            size=(train_batch_size,))  # 2.0 * math.pi normalized
    model_agent_rnn_out_reg_out_speed = np.random.uniform(low=0.0, high=1.0,
                                                          size=(train_batch_size,))  # 20.0 normalized
    model_agent_rnn_out_categ_out_agent_box_heat_map = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.W, chauffeur_net.H))
    model_agent_rnn_out_categ_out_waypoint_sub_pixel = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, 2))

    model_road_mask_net_out_bin_out_road_mask = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.W, chauffeur_net.H))

    model_perception_rnn_out_categ_out_obstacle_box_heat_map = np.random.uniform(low=0.0, high=1.0, size=(
        train_batch_size, chauffeur_net.W, chauffeur_net.H))

    reg_out_heading_loss = np.zeros((train_batch_size,))
    reg_out_speed_loss = np.zeros((train_batch_size,))
    categ_out_waypoint_loss = np.zeros((train_batch_size,))
    categ_out_agent_box_heat_map_loss = np.zeros((train_batch_size,))
    waypoint_sub_pixel_loss = np.zeros((train_batch_size,))

    collision_loss = np.zeros((train_batch_size,))
    on_road_loss = np.zeros((train_batch_size,))
    geometry_loss = np.zeros((train_batch_size,))

    categ_out_obstacle_box_heat_map_loss = np.zeros((train_batch_size,))

    train_labels = [  # for predict output
        model_agent_rnn_out_categ_out_waypoint,
        model_agent_rnn_out_reg_out_heading,
        model_agent_rnn_out_reg_out_speed,
        model_agent_rnn_out_categ_out_agent_box_heat_map,
        model_agent_rnn_out_categ_out_waypoint_sub_pixel,

        model_road_mask_net_out_bin_out_road_mask,  # only used,other just for loss

        model_perception_rnn_out_categ_out_obstacle_box_heat_map,

        # -----just for loss placeholder, no use, not occupy too more memory----
        categ_out_waypoint_loss,
        reg_out_heading_loss,
        reg_out_speed_loss,
        categ_out_agent_box_heat_map_loss,
        waypoint_sub_pixel_loss,

        collision_loss,
        on_road_loss,
        geometry_loss,
        categ_out_obstacle_box_heat_map_loss
    ]
    print('train_labels size:', len(train_labels))
    ''''''
    print("train......")
    start_time = time.time()
    ''''''
    loss = model.fit(train_inputs, train_labels, epochs=1,
                     # sample_weight=train_sample_weights,
                     # class_weight=self._class_weight,  # label index
                     callbacks=cbks,
                     validation_split=valid_ratio,
                     batch_size=train_batch_size,
                     shuffle=True,  # validation_data=(valid_inputs, valid_labels),
                     verbose=1)
    print("train finished,time cost:", time.time(), time.time() - start_time)

    print("train finished!")
    print("----------------------------------------")
    model_base_name = 'chauffeur_net'
    model_name = model_base_name + '.h5'

    #model.save(model_name)  # creates a HDF5 file 'model_name.h5'

    model_weight_name = model_base_name + '_weights.h5'
    model.save_weights(model_weight_name)

    ''''''
    test_batch_size = 1  # 2

    input_roadmap_img = np.random.uniform(low=0.0, high=1.0,
                                          size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 3))

    input_speed_limit_img = np.random.uniform(low=0.0, high=1.0,
                                              size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_route_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_current_img = np.random.uniform(low=0.0, high=1.0,
                                              size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_past_img = np.random.uniform(low=0.0, high=1.0,
                                           size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))

    input_traffic_light_img_time_seq = [
        np.random.uniform(low=0.0, high=1.0, size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
        for i in range(chauffeur_net.N_past_scene)]

    input_obstacles_past_img_time_seq = [
        np.random.uniform(low=0.0, high=1.0, size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
        for i in range(chauffeur_net.N_past_scene)]

    input_road_mask_ground_truth_img = np.random.randint(low=0, high=2,
                                                         size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
    input_geometry_ground_truth_img = np.random.uniform(low=0.0, high=1.0,
                                                        size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 1))
    input_heading_step_ground_truth_for_loss = np.random.uniform(low=-1.0, high=1.0,
                                                                 size=(test_batch_size, chauffeur_net.N_out_steps))
    input_speed_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0,
                                                               size=(test_batch_size, chauffeur_net.N_out_steps))
    # input_categ_out_waypoint_step_ground_truth_for_loss = np.random.randint(low=0, high=2, size=(test_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W * chauffeur_net.H))
    # input_categ_out_waypoint_step_ground_truth_for_loss = np.concatenate([np.asarray(np.random.randint(low=0, high=chauffeur_net.W,
    #                                                                                        size=(test_batch_size, chauffeur_net.N_out_steps, 1))),
    #                                                                       np.asarray(np.random.randint(low=0, high=chauffeur_net.H,
    #                                                                                        size=(test_batch_size,
    #                                                                                              chauffeur_net.N_out_steps,1)))], axis=-1)
    input_categ_out_waypoint_step_ground_truth_for_loss_W = np.asarray(np.random.uniform(low=0, high=chauffeur_net.W,
                                                                                         size=(test_batch_size,
                                                                                               chauffeur_net.N_out_steps,
                                                                                               1)))
    input_categ_out_waypoint_step_ground_truth_for_loss_W = to_categorical(
            input_categ_out_waypoint_step_ground_truth_for_loss_W,
            num_classes=chauffeur_net.W
    )

    input_categ_out_waypoint_step_ground_truth_for_loss_H = np.asarray(np.random.uniform(low=0, high=chauffeur_net.H,
                                                                                         size=(test_batch_size,
                                                                                               chauffeur_net.N_out_steps,
                                                                                               1)))
    input_categ_out_waypoint_step_ground_truth_for_loss_H = to_categorical(
            input_categ_out_waypoint_step_ground_truth_for_loss_H,
            num_classes=chauffeur_net.H
    )

    input_categ_out_waypoint_step_ground_truth_for_loss = np.concatenate(
            [input_categ_out_waypoint_step_ground_truth_for_loss_W,
             input_categ_out_waypoint_step_ground_truth_for_loss_H
             ], axis=-1)
    input_categ_out_agent_box_heat_map_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        test_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W, chauffeur_net.H))
    input_waypoint_sub_pixel_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        test_batch_size, chauffeur_net.N_out_steps, 2))
    input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss = np.random.uniform(low=0.0, high=1.0, size=(
        test_batch_size, chauffeur_net.N_out_steps, chauffeur_net.W, chauffeur_net.H))

    test_inputs = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                   input_ego_past_img]
    test_inputs.extend(input_traffic_light_img_time_seq)
    test_inputs.extend(input_obstacles_past_img_time_seq)

    # just for loss calc,so can set to zero or random
    test_inputs.extend([input_road_mask_ground_truth_img, input_geometry_ground_truth_img,
                        input_heading_step_ground_truth_for_loss, input_speed_step_ground_truth_for_loss,
                        input_categ_out_waypoint_step_ground_truth_for_loss,
                        input_categ_out_agent_box_heat_map_step_ground_truth_for_loss,
                        input_waypoint_sub_pixel_step_ground_truth_for_loss,
                        input_categ_out_obstacle_box_heat_map_step_ground_truth_for_loss
                        ])
    input_imit_drop_out_weight = (
        np.less(np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,)), imit_drop_out_ratio)).astype(
            np.float32)
    test_inputs.append(input_imit_drop_out_weight)
    print("predict......")
    start_time = time.time()
    out = model.predict(test_inputs)
    print("predict finished,time cost:", time.time(), time.time() - start_time)
    print(out)

    exit(0)

    # Replicates `model` on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    '''
    print("GPU train......")
    n_gpu = 1  # 8
    parallel_model = multi_gpu_model(model, gpus=n_gpu)
    # parallel_model.compile(loss='categorical_crossentropy',
    #                       optimizer='rmsprop')

    # This `fit` call will be distributed on 8 GPUs.
    # Since the batch size is 256, each GPU will process 32 samples.
    parallel_model.fit(train_inputs, train_labels, epochs=1,
                       # sample_weight=train_sample_weights,
                       # class_weight=self._class_weight,  # label index
                       callbacks=cbks,
                       validation_split=valid_ratio,
                       batch_size=train_batch_size * n_gpu,
                       shuffle=True,  # validation_data=(valid_inputs, valid_labels),
                       verbose=1)
    print("GPU train finished!")
    model_base_name = 'chauffeur_net'
    model_name = model_base_name + '_gpu.h5'

    model.save(model_name)  # creates a HDF5 file 'model_name.h5'

    model_weight_name = model_base_name + '_weights_gpu.h5'
    model.save_weights(model_weight_name)
    print("GPU predict......")
    # out = model.predict(test_inputs)
    # print(out)
    '''