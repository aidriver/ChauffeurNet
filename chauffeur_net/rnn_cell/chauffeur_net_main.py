from keras.layers import Input
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers.core import Activation
from keras.optimizers import RMSprop

from keras.layers import Multiply
from keras.layers import Dot
from keras import backend as K
import numpy as np
import math

from chauffeur_net.feature_net import FeatureNet
from chauffeur_net.agent_rnn import AgentRNN
from chauffeur_net.road_mask_net import RoadMaskNet
from chauffeur_net.perception_rnn import PerceptionRNN


class ChauffeurNet:
    def __init__(self, conf_file):
        self.w_imitate = 1.0  # 0.0 random
        self.w_env = 1.0
        self.W = 400
        self.H_pos = 400
        self.H_neg = 400
        self.H = self.H_pos + self.H_neg
        self.T_scene = 1.0 #s
        self.T_pose = 8.0 #s
        self.dt = 0.2 #s
        self.N_out_steps = 10  # 25 40
        self.N_past_scene = int(self.T_scene / self.dt)
        self.N_past_pose = int(self.T_pose / self.dt)
        self.resolution_space = 0.2 #m
        self.u0 = self.H_pos
        self.v0 = self.W / 2
        self.max_pertu_theta = 25  # degree
        self.feature_channel_num = 0

        self._conf = conf_file
        pass

    def setup_model(self):
        input_roadmap_img = Input(shape=(self.W, self.H, 3), name='input_roadmap_img')

        input_speed_limit_img = Input(shape=(self.W, self.H, 1), name='input_speed_limit_img')

        input_route_img = Input(shape=(self.W, self.H, 1), name='input_route_img')

        input_ego_current_img = Input(shape=(self.W, self.H, 1), name='input_ego_current_img')

        input_ego_past_img = Input(shape=(self.W, self.H, 1), name='input_ego_past_img')

        input_traffic_light_img_time_seq = [Input(shape=(self.W, self.H, 1), name='input_traffic_light_img_' + str(i))
                                            for i in range(self.N_past_scene)]

        input_obstacles_past_img_time_seq = [Input(shape=(self.W, self.H, 1), name='input_obstacles_past_img_' + str(i))
                                             for i in range(self.N_past_scene)]

        input_road_mask_ground_truth_img = Input(shape=(self.W, self.H, 1), name='input_road_mask_ground_truth_img')
        input_geometry_ground_truth_img = Input(shape=(self.W, self.H, 1), name='input_geometry_ground_truth_img')

        input_vec = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                     input_ego_past_img, input_road_mask_ground_truth_img, input_geometry_ground_truth_img]
        input_vec.extend(input_traffic_light_img_time_seq)
        input_vec.extend(input_obstacles_past_img_time_seq)

        feature_net = FeatureNet(self._conf)
        feature_net.input_vec = input_vec
        # feature_net.out_W_size = 64  # ?? how to keep size?and deconv?
        # feature_net.out_H_size = 64  # ?? how to keep size?and deconv?
        model_feature_net = feature_net.setup_model()
        features = model_feature_net(input_vec)
        _, feature_net.out_W_size, feature_net.out_H_size, feature_net.feature_channel_num = model_feature_net.get_output_shape_at(
                0)  # model_feature_net.outputs[0],

        agent_rnn = AgentRNN(self._conf)
        agent_rnn.feature_net = feature_net
        model_agent_rnn = agent_rnn.setup_model()

        input_agent_rnn_vec = [input_ego_past_img, features]  # [input_ego_past_img, ,features]
        model_agent_rnn_out = model_agent_rnn(input_agent_rnn_vec)

        # must remap sun_model out to layer with name for multi-output&loss keys work normal,or sub model has multi output with same name of submodel name
        model_agent_rnn_out_categ_out_waypoint = Activation('linear', name='categ_out_waypoint')(model_agent_rnn_out[0])
        model_agent_rnn_out_reg_out_heading = Activation('linear', name='reg_out_heading')(
                model_agent_rnn_out[1])
        model_agent_rnn_out_reg_out_speed = Activation('linear', name='reg_out_speed')(
                model_agent_rnn_out[2])
        model_agent_rnn_out_categ_out_agent_box_heat_map = Activation('linear', name='categ_out_agent_box_heat_map')(
                model_agent_rnn_out[3])
        model_agent_rnn_out_categ_out_waypoint_sub_pixel = Activation('linear', name='waypoint_sub_pixel')(
                model_agent_rnn_out[4])
        model_agent_rnn_out_layers = [model_agent_rnn_out_categ_out_waypoint,
                                      model_agent_rnn_out_reg_out_heading,
                                      model_agent_rnn_out_reg_out_speed,
                                      model_agent_rnn_out_categ_out_agent_box_heat_map,
                                      model_agent_rnn_out_categ_out_waypoint_sub_pixel]

        road_mask_net = RoadMaskNet(self._conf)
        road_mask_net.feature_net = feature_net
        model_road_mask_net = road_mask_net.setup_model()
        model_road_mask_net_out = model_road_mask_net(features)
        # must remap sun_model out to layer with name for multi-output&loss keys work normal,or sub model has multi output with same name of submodel name
        model_road_mask_net_out_bin_out_road_mask = Activation('linear', name='bin_out_road_mask')(
                model_road_mask_net_out)  # model_road_mask_net_out[0]
        model_road_mask_net_out_layers = model_road_mask_net_out_bin_out_road_mask

        perception_rnn = PerceptionRNN(self._conf)
        perception_rnn.feature_net = feature_net
        model_perception_rnn = perception_rnn.setup_model()
        input_perception_rnn = [features]  # [,features]
        model_perception_rnn_out = model_perception_rnn(input_perception_rnn)
        # must remap sun_model out to layer with name for multi-output&loss keys work normal,or sub model has multi output with same name of submodel name
        model_perception_rnn_out_categ_out_obstacle_box_heat_map = Activation('linear',
                                                                              name='categ_out_obstacle_box_heat_map')(
                model_perception_rnn_out)  # model_perception_rnn_out[0]
        model_perception_rnn_out_layers = model_perception_rnn_out_categ_out_obstacle_box_heat_map

        final_outputs = []
        final_outputs.extend(model_agent_rnn_out_layers)  # multi-output
        final_outputs.append(model_road_mask_net_out_layers)  # finally single-ouput
        # final_outputs.extend(model_road_mask_net_out_layers) #finally single-ouput
        final_outputs.append(model_perception_rnn_out_layers)  # finally single-ouput
        # final_outputs.extend(model_perception_rnn_out_layers)  # finally single-ouput
        # combine = Concatenate(name='combine_outputs', axis=-1)(final_outputs)
        # output_vec = [combine]

        #--------loss output can give any value for label,loss not use ground truth,only use predict value----------
        flat_agent_heat_map = Flatten()(model_agent_rnn_out_categ_out_agent_box_heat_map)

        collision_loss = Dot(1, name='collision')([flat_agent_heat_map,
                                                   Flatten()(model_perception_rnn_out_categ_out_obstacle_box_heat_map)])
        # collision_loss = Dot((1,2),name='collision')([model_agent_rnn_out_categ_out_agent_box_heat_map, model_perception_rnn_out_categ_out_obstacle_box_heat_map])
        # loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
        final_outputs.append(collision_loss)

        onroad = Lambda(lambda x: 1 - x)(input_road_mask_ground_truth_img)
        flat_not_onroad = Flatten()(onroad)
        on_road_loss = Dot(1, name='on_road')([flat_agent_heat_map, flat_not_onroad])
        final_outputs.append(on_road_loss)

        on_geometry = Lambda(lambda x: 1 - x)(input_geometry_ground_truth_img)
        flat_not_on_geometry = Flatten()(on_geometry)
        geometry_loss = Dot(1, name='geometry')([flat_agent_heat_map, flat_not_on_geometry])
        final_outputs.append(geometry_loss)

        output_vec = final_outputs
        model = Model(inputs=input_vec, outputs=output_vec)

        imit_drop_out_ratio = 0.5

        def imit_mean_absolute_percentage_error(y_true, y_pred):
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
            return weight * K.mean(diff, axis=-1)

        def imit_mean_absolute_error(y_true, y_pred):
            return K.mean(K.abs(y_pred - y_true), axis=-1)

        def imit_categorical_crossentropy(y_true, y_pred):
            '''Expects a binary class matrix instead of a vector of scalar classes.
            '''
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            return weight * K.categorical_crossentropy(y_pred, y_true)

        def imit_binary_crossentropy(y_true, y_pred):
            weight = K.cast(K.less(K.random_uniform(K.shape(y_true)), K.ones_like(y_true) * imit_drop_out_ratio),
                            'float32')
            return weight * K.binary_crossentropy(y_pred, y_true)

        rmsprop = RMSprop(  # clipnorm=1.0,#1.0 15.0
                # clipvalue=0.5,#0.5 8.0
                lr=0.01, rho=0.9, epsilon=1e-08, decay=1e-6)
        # key must same with multi-output-layer name. final_ouput merge multi submodel ouputs,so its original layers'name lost,just use model name,and duplicate with multi-ouput submodel
        model.compile(optimizer=rmsprop,
                      loss={
                          'reg_out_heading': imit_mean_absolute_error,  # 'mae',
                          'reg_out_speed': imit_mean_absolute_error,  # 'mae',
                          'categ_out_waypoint': imit_categorical_crossentropy,  # 'categorical_crossentropy',
                          # 'categ_out_agent_box_heat_map': imit_categorical_crossentropy,#'categorical_crossentropy' #,
                          'categ_out_agent_box_heat_map': imit_binary_crossentropy,  # 'binary_crossentropy',
                          'waypoint_sub_pixel': imit_mean_absolute_error,
                          # 'mae',#note groundth should be pose-[pose]

                          'collision': lambda y_true, y_pred: y_pred,
                          'on_road': lambda y_true, y_pred: y_pred,
                          'geometry': lambda y_true, y_pred: y_pred,
                          # 'categ_out_obstacle_box_heat_map': 'categorical_crossentropy',
                          'categ_out_obstacle_box_heat_map': 'binary_crossentropy',
                          'bin_out_road_mask': 'binary_crossentropy'#not consider iteration k,all other loss should sum over iteration(even perception RNN)
                      },
                      loss_weights={
                          'reg_out_heading': self.w_imitate,
                          'reg_out_speed': self.w_imitate,
                          'categ_out_waypoint': self.w_imitate,
                          'categ_out_agent_box_heat_map': self.w_imitate / (self.W * self.H),
                          'waypoint_sub_pixel': self.w_imitate,

                          'collision': self.w_env / (self.W * self.H),
                          'on_road': self.w_env / (self.W * self.H),
                          'geometry': self.w_env / (self.W * self.H),
                          'categ_out_obstacle_box_heat_map': self.w_env / (self.W * self.H),
                          'bin_out_road_mask': self.w_env / (self.W * self.H)
                      },
                      metrics=['accuracy'])
        model.summary()
        return model


from keras.callbacks import ReduceLROnPlateau

if __name__ == '__main__':
    conf_path = 'conf/chauffeur_net.conf'
    root_input_path = '/home/caros/offline/chauffeur_net'
    root_output_path = '/home/caros/offline/chauffeur_net'
    chauffeur_net = ChauffeurNet(conf_path)
    model = chauffeur_net.setup_model()
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

    valid_ratio = 0.5#0.2
    train_batch_size = int(1 / (1 - valid_ratio))  # 8

    input_roadmap_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size, chauffeur_net.W, chauffeur_net.H, 3))

    input_speed_limit_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_route_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_current_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_past_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_traffic_light_img_time_seq = [np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
                                        for i in range(chauffeur_net.N_past_scene)]

    input_obstacles_past_img_time_seq = [np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
                                         for i in range(chauffeur_net.N_past_scene)]

    input_road_mask_ground_truth_img = np.random.randint(low=0, high=2, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
    input_geometry_ground_truth_img = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    train_inputs = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                    input_ego_past_img, input_road_mask_ground_truth_img, input_geometry_ground_truth_img]
    train_inputs.extend(input_traffic_light_img_time_seq)
    train_inputs.extend(input_obstacles_past_img_time_seq)

    model_agent_rnn_out_categ_out_waypoint = np.random.randint(low=0, high=320000, size=(train_batch_size,))
    model_agent_rnn_out_reg_out_heading = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,))#2.0 * math.pi normalized
    model_agent_rnn_out_reg_out_speed = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,)) #20.0 normalized
    model_agent_rnn_out_categ_out_agent_box_heat_map = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H))
    model_agent_rnn_out_categ_out_waypoint_sub_pixel = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,))

    model_road_mask_net_out_bin_out_road_mask = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H))

    model_perception_rnn_out_categ_out_obstacle_box_heat_map = np.random.uniform(low=0.0, high=1.0, size=(train_batch_size,chauffeur_net.W, chauffeur_net.H))

    collision_loss = np.zeros((train_batch_size,))
    on_road_loss = np.zeros((train_batch_size,))
    geometry_loss = np.zeros((train_batch_size,))

    train_labels = [model_agent_rnn_out_categ_out_waypoint,
                    model_agent_rnn_out_reg_out_heading,
                    model_agent_rnn_out_reg_out_speed,
                    model_agent_rnn_out_categ_out_agent_box_heat_map,
                    model_agent_rnn_out_categ_out_waypoint_sub_pixel,
                    model_road_mask_net_out_bin_out_road_mask,
                    model_perception_rnn_out_categ_out_obstacle_box_heat_map,
                    collision_loss,
                    on_road_loss,
                    geometry_loss
                    ]
    loss = model.fit(train_inputs, train_labels, epochs=1,
                     # sample_weight=train_sample_weights,
                     # class_weight=self._class_weight,  # label index
                     callbacks=cbks,
                     validation_split=valid_ratio,
                     batch_size=train_batch_size,
                     shuffle=True,  # validation_data=(valid_inputs, valid_labels),
                     verbose=1)

    model_base_name = 'chauffeur_net'
    model_name = model_base_name + '.h5'

    model.save(model_name)# creates a HDF5 file 'model_name.h5'

    model_weight_name = model_base_name + '_weights.h5'
    model.save_weights(model_weight_name)

    test_batch_size = 2

    input_roadmap_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size, chauffeur_net.W, chauffeur_net.H, 3))

    input_speed_limit_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_route_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_current_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_ego_past_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    input_traffic_light_img_time_seq = [np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
                                        for i in range(chauffeur_net.N_past_scene)]

    input_obstacles_past_img_time_seq = [np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
                                         for i in range(chauffeur_net.N_past_scene)]

    input_road_mask_ground_truth_img = np.random.randint(low=0, high=2, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))
    input_geometry_ground_truth_img = np.random.uniform(low=0.0, high=1.0, size=(test_batch_size,chauffeur_net.W, chauffeur_net.H, 1))

    test_inputs = [input_roadmap_img, input_speed_limit_img, input_route_img, input_ego_current_img,
                    input_ego_past_img, input_road_mask_ground_truth_img, input_geometry_ground_truth_img]
    test_inputs.extend(input_traffic_light_img_time_seq)
    test_inputs.extend(input_obstacles_past_img_time_seq)
    out = model.predict(test_inputs)
    print(out)
