"""
Trajectory prediction neural network that considers an arbitrary number of quadrotors and constant-speed ellipsoidal obstacles 
Stacks dynamic obstacles and quadrotors together before they get processed by the bidirectional layer which learns their influence on the query robot.
"""

import tensorflow as tf
from models.dynamicEllipsoidObstaclesRNN_regularization import StateEncoder, Decoder
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class CommonEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        if args.size_obstacles_fc_layer != args.size_other_agents_state:
            args.size_obstacles_fc_layer = args.size_other_agents_state
            print(f"{bcolors.FAIL}Size of the obstacles dense layer and the other quadrotors encoder should be the same. Enforcing size_obstacles_fc_layer=size_other_agents_state.{bcolors.ENDC}")
        
        self.reg_factor = 0.01
        self.rnn_state_size_lstm_other_quads = args.size_other_agents_state
        self.fc_state_size_obstacles = args.size_obstacles_fc_layer
        self.rnn_state_size_bilstm = args.size_obstacles_bilstm + args.size_other_agents_bilstm
        
        self.lstm_other_quads = tfkl.LSTM(self.rnn_state_size_lstm_other_quads, name = 'lstm_other_quads', return_sequences = False, kernel_regularizer = tfkr.l2(self.reg_factor), recurrent_regularizer = tfkr.l2(self.reg_factor), bias_regularizer = tfkr.l2(self.reg_factor), activity_regularizer = tfkr.l2(self.reg_factor))
        self.fc_obs = tfkl.Dense(self.fc_state_size_obstacles, activation = 'relu')
        self.bilstm = tfkl.Bidirectional(tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_obs', return_sequences = False, return_state = False, kernel_regularizer = tfkr.l2(self.reg_factor), recurrent_regularizer = tfkr.l2(self.reg_factor), bias_regularizer = tfkr.l2(self.reg_factor), activity_regularizer = tfkr.l2(self.reg_factor)), merge_mode = 'ave') # Average of forward and backward LSTMs
        
        
    def call(self, x):        
        processed_objects = []
        
        for quad_idx in range(x["others_input"].shape[-1]):
            processed_objects.append(self.lstm_other_quads(x["others_input"][:, :, :, quad_idx]))
        
        for obs_idx in range(x["obstacles_input"].shape[-1]):
            processed_objects.append(self.fc_obs(x["obstacles_input"][:, :, obs_idx]))
        
        stacked_obs = tf.stack(processed_objects, axis = 1)
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
        self.state_encoder = StateEncoder(args)
        self.common_encoder = CommonEncoder(args)
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(self.prediction_horizon)
        self.decoder = Decoder(args)
                
    def call(self, x):
        x1 = self.state_encoder(x["query_input"])
        x2 = self.common_encoder( x )
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