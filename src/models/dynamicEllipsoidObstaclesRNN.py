"""
varyingNQuadsRNN version compatible with the new argument parser
"""

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

class StateEncoder(tfkl.Layer):
    def __init__(self, rnn_state_size = 256, lambda_ = 0.01):
        super().__init__()
        
        self.lambda_ = lambda_
        self.rnn_state_size = rnn_state_size
        
        self.lstm_quad = tfkl.LSTM(self.rnn_state_size, name = 'lstm_state', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        
    def call(self, x):
        out = self.lstm_quad(x)
        return out

class OtherQuadsEncoder(tfkl.Layer):
    def __init__(self, rnn_state_size_lstm_quad = 256, rnn_state_size_bilstm = 256, lambda_ = 0.01):
        super().__init__()

        self.lambda_ = lambda_
        self.rnn_state_size_lstm_quad = rnn_state_size_lstm_quad
        self.rnn_state_size_bilstm = rnn_state_size_bilstm

        self.lstm_other_quads = tfkl.LSTM(self.rnn_state_size_lstm_quad, name = 'lstm_other_quads', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        self.bilstm = tfkl.Bidirectional(tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_other_quads', return_sequences = False, return_state = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_)), merge_mode = 'ave') # Average of forward and backward LSTMs
        
    def call(self, x):
        other_quads = []
        # for x_quad in x:
        for quad_idx in range(x.shape[-1]):
            other_quads.append(self.lstm_other_quads(x[:, :, :, quad_idx]))
        stacked_other_quads = tf.stack(other_quads, axis = 1)
        out = self.bilstm(stacked_other_quads)
        return out

class DynamicObstaclesEncoder(tfkl.Layer):
    def __init__(self, fc_units = 64, rnn_state_size_bilstm = 64, lambda_ = 0.01):
        super().__init__()
        
        self.lambda_ = lambda_
        # self.rnn_state_size_lstm_obs = 256
        self.fc_units = fc_units
        self.rnn_state_size_bilstm = rnn_state_size_bilstm

        # self.lstm_obs = tfkl.LSTM(self.rnn_state_size_lstm_obs, name = 'lstm_obs', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        self.fc_obs = tfkl.Dense(self.fc_units, activation = 'relu')
        self.bilstm = tfkl.Bidirectional(tfkl.LSTM(self.rnn_state_size_bilstm,  name = 'bilstm_obs', return_sequences = False, return_state = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_)), merge_mode = 'ave') # Average of forward and backward LSTMs

    def call(self, x):
        obs = []
        for obs_idx in range(x.shape[-1]):
            # obs.append(self.lstm_obs(x[:, :, obs_idx]))
            obs.append(self.fc_obs(x[:, :, obs_idx]))
        stacked_obs = tf.stack(obs, axis = 1)
        out = self.bilstm(stacked_obs)
        return out
        

class Decoder(tfkl.Layer):
    def __init__(self, rnn_state_size_lstm_concat = 512, fc_hidden_unit_size = 256):
        super().__init__()
        
        self.output_dim = 3 # X, Y and Z coordinates of either position or velocity
    
        self.rnn_state_size_lstm_concat = rnn_state_size_lstm_concat
        self.fc_hidden_unit_size = fc_hidden_unit_size

        # self.concat = tfkl.Concatenate()
        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, name = 'lstm_decoder', return_sequences = True, return_state = False)
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, name = 'fc_decoder', activation = 'relu'))
        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim, name = 'fc_out'))
                
    def call(self, x):        
        out = self.lstm_concat(x)
        out = self.fc1(out)
        out = self.fco(out)
        return out

class FullModel(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        self.prediction_horizon = args.prediction_horizon
        
        # Define optimizer and loss
        self.loss_object = tfk.losses.MeanSquaredError()
        self.optimizer = tfk.optimizers.Adam()
        self.train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
        self.val_loss = tfk.metrics.MeanSquaredError(name='val_loss')
        
        # Define architecture
        self.state_encoder = StateEncoder(rnn_state_size = args.size_query_agent_state)
        self.other_quads_encoder = OtherQuadsEncoder(rnn_state_size_lstm_quad = args.size_other_agents_state, rnn_state_size_bilstm = args.size_other_agents_bilstm)
        self.obs_encoder = DynamicObstaclesEncoder(fc_units=args.size_obstacles_fc_layer, rnn_state_size_bilstm=args.size_obstacles_bilstm)
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(self.prediction_horizon)
        self.decoder = Decoder(rnn_state_size_lstm_concat = args.size_decoder_lstm, fc_hidden_unit_size = args.size_fc_layer)
                
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

        self.train_loss(x["target"], predictions)

    @tf.function
    def val_step(self, x):
        predictions = self(x)
        v_loss = self.loss_object(x["target"], predictions)

        self.val_loss(x["target"], predictions)


