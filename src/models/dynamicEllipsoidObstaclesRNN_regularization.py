"""
Trajectory prediction neural network that considers an arbitrary number of quadrotors and constant-speed ellipsoidal obstacles 
"""

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers


class StateEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.regularization_factor = args.regularization_factor
        self.rnn_state_size = args.size_query_agent_state
        
        self.lstm_quad = tfkl.LSTM(self.rnn_state_size, name = 'lstm_state', return_sequences = False,\
            kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor))
        
    def call(self, x):
        out = self.lstm_quad(x)
        return out

class OtherQuadsEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()

        self.regularization_factor = args.regularization_factor
        self.rnn_state_size_lstm_quad = args.size_other_agents_state
        self.rnn_state_size_bilstm = args.size_other_agents_bilstm

        self.lstm_other_quads = tfkl.LSTM(self.rnn_state_size_lstm_quad, name = 'lstm_other_quads', return_sequences = False, \
            kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor))
        self.bilstm = tfkl.Bidirectional(\
            tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_other_quads', return_sequences = False, return_state = False,\
                kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor)), merge_mode = 'ave') # Average of forward and backward LSTMs

        
    def call(self, x):
        other_quads = []
        # for x_quad in x:
        for quad_idx in range(x.shape[-1]):
            other_quads.append(self.lstm_other_quads(x[:, :, :, quad_idx]))
        stacked_other_quads = tf.stack(other_quads, axis = 1)
        out = self.bilstm(stacked_other_quads)
        return out

class DynamicObstaclesEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.regularization_factor = args.regularization_factor
        self.fc_units = args.size_obstacles_fc_layer
        self.rnn_state_size_bilstm = args.size_obstacles_bilstm

        self.fc_obs = tfkl.Dense(self.fc_units, activation = 'relu',\
            kernel_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor))
        self.bilstm = tfkl.Bidirectional(\
            tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_obs', return_sequences = False, return_state = False,\
                kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor)), merge_mode = 'ave') # Average of forward and backward LSTMs

    def call(self, x):
        obs = []
        for obs_idx in range(x.shape[-1]):
            obs.append(self.fc_obs(x[:, :, obs_idx]))
        stacked_obs = tf.stack(obs, axis = 1)
        out = self.bilstm(stacked_obs)
        return out
        

class Decoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.output_dim = 3 # X, Y and Z coordinates of either position or velocity
        self.regularization_factor = args.regularization_factor
    
        self.rnn_state_size_lstm_concat = args.size_decoder_lstm
        self.fc_hidden_unit_size = args.size_fc_layer

        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, name = 'lstm_decoder', return_sequences = True, return_state = False,\
            kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor))
        
        # Achieves better performance but takes a bit longer to train
        # self.lstm_concat = tfkl.Bidirectional( tfkl.LSTM(self.rnn_state_size_lstm_concat, name = 'lstm_decoder', return_sequences = True, return_state = False,\
        #     kernel_regularizer = tfkr.l2(self.regularization_factor), recurrent_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor)) , merge_mode = 'ave')
        
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, name = 'fc_decoder', activation = 'relu',\
            kernel_regularizer = tfkr.l2(self.regularization_factor), bias_regularizer = tfkr.l2(self.regularization_factor), activity_regularizer = tfkr.l2(self.regularization_factor)))
        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim, name = 'fc_out'))
                
    def call(self, x):        
        out = self.lstm_concat(x)
        out = self.fc1(out)
        out = self.fco(out)
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
        self.state_encoder = StateEncoder(args)
        self.other_quads_encoder = OtherQuadsEncoder(args)
        self.obs_encoder = DynamicObstaclesEncoder(args)
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(self.prediction_horizon)
        self.decoder = Decoder(args)
                
    def call(self, x):
        x1 = self.state_encoder(x["query_input"])
        x2 = self.other_quads_encoder( x["others_input"] )
        x3 = self.obs_encoder( x["obstacles_input"] )
        concat = self.concat([x1, x2, x3])
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
        
        



