"""
Trajectory prediction neural network that considers only the query robot and other ellipsoidal obstacles (no other agents)
Inherits from dynamicEllipsoidOnstaclesRNN
"""

import tensorflow as tf
from models.dynamicEllipsoidObstaclesRNN import StateEncoder, DynamicObstaclesEncoder, Decoder
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

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
        
        



