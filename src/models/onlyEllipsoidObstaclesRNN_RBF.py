"""
Trajectory prediction neural network that considers only the query robot and other ellipsoidal obstacles (no other agents)
Inherits from dynamicEllipsoidOnstaclesRNN
Uses Radial Basis Function
"""

import tensorflow as tf
from models.dynamicEllipsoidObstaclesRNN import StateEncoder, Decoder
from external.rbflayer import RBFLayer, InitCentersRandom
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers


class DynamicObstaclesEncoder(tfkl.Layer):
    def __init__(self, fc_units = 64, rnn_state_size_bilstm = 64, lambda_ = 0.01):
        super().__init__()
        
        self.lambda_ = lambda_
        # self.rnn_state_size_lstm_obs = 256
        self.fc_units = fc_units
        self.rnn_state_size_bilstm = rnn_state_size_bilstm
        self.concat = tfkl.Concatenate()
        self.fc_obs = tfkl.Dense(self.fc_units//2, activation = 'relu')
        self.rbf_obs = RBFLayer(self.fc_units//2)
        self.bilstm = tfkl.Bidirectional(tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_obs', return_sequences = False, return_state = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_)), merge_mode = 'ave') # Average of forward and backward LSTMs

    def call(self, x):
        obs = []
        for obs_idx in range(x.shape[-1]):
            concat = self.concat([self.fc_obs(x[:, 0:3, obs_idx]),\
                self.rbf_obs(x[:, 3:, obs_idx])])
            obs.append(concat)
        stacked_obs = tf.stack(obs, axis = 1)
        out = self.bilstm(stacked_obs)
        return out


class FullModel(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        self.dt = args.dt
        self.prediction_horizon = args.prediction_horizon
        
        # Define optimizer and loss
        self.loss_object = tfk.losses.MeanSquaredError()
        self.optimizer = tfk.optimizers.Adam()
        self.train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
        self.val_loss = tfk.metrics.MeanSquaredError(name='val_loss')
        
        self.test_prediction_horizons = []
        self.position_L2_errors = []
        self.velocity_L2_errors = []
        if len(args.test_prediction_horizons.split(" ")) > 1:
            for test_prediction_horizon in args.test_prediction_horizons.split(" "):
                self.test_prediction_horizons.append(int(test_prediction_horizon))
                self.position_L2_errors.append(tfk.metrics.RootMeanSquaredError(name='position_L2_error_at_' + test_prediction_horizon))
                self.velocity_L2_errors.append(tfk.metrics.RootMeanSquaredError(name='velocity_L2_error_at_' + test_prediction_horizon))
        
        # Define architecture
        self.state_encoder = StateEncoder(rnn_state_size = args.size_query_agent_state)
        self.obs_encoder = DynamicObstaclesEncoder(fc_units=args.size_obstacles_fc_layer, rnn_state_size_bilstm=args.size_obstacles_bilstm)
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(self.prediction_horizon)
        self.decoder = Decoder(rnn_state_size_lstm_concat = args.size_decoder_lstm, fc_hidden_unit_size = args.size_fc_layer)
                
    def call(self, x):
        x1 = self.state_encoder(x["query_input"])
        x2 = self.obs_encoder( x["obstacles_input"] )
        concat = self.concat([x1, x2])
        repeated = self.repeat(concat)
        out = self.decoder(repeated)
        return out
    
    @tf.function 
    def train_step(self, x):
        with tf.GradientTape() as tape:
            predictions = self(x)
            loss = self.loss_object(x["target"], predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss.update_state(x["target"], predictions)

    @tf.function
    def val_step(self, x):
        predictions = self(x)
        # v_loss = self.loss_object(x["target"], predictions)
        self.val_loss.update_state(x["target"], predictions)
    
    @tf.function
    def testFDE_step(self, x):
        predictions = self(x)

        batch_size = x["target"].shape[0]
        new_target_position= tf.zeros([batch_size, 3])
        new_predicted_position = tf.zeros([batch_size, 3])
        
        for idx in range(len(self.test_prediction_horizons)):
            prediction_horizon = self.test_prediction_horizons[idx]

            if idx == 0:
                time_range = [i for i in range(prediction_horizon)]
            else:
                time_range = [i for i in range(self.test_prediction_horizons[idx-1], prediction_horizon)]

            for time_step in time_range:     
                new_target_position += x["target"][:, time_step, :] * self.dt
                new_predicted_position += predictions[:, time_step, :] * self.dt


            self.position_L2_errors[idx].update_state(new_target_position, new_predicted_position)
            self.velocity_L2_errors[idx].update_state(x["target"][:,prediction_horizon-1,:], predictions[:, prediction_horizon-1, :])
        
        



