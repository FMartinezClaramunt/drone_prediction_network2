"""
varyingNQuadsRNN version compatible with the new argument parser
"""

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

class StateEncoder(tfkl.Layer):
    def __init__(self):
        super().__init__()
        
        self.lambda_ = 0.01
        self.rnn_state_size = 256
        
        self.lstm_quad = tfkl.LSTM(self.rnn_state_size, name = 'lstm_state', return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))
        
    def call(self, x):
        x = self.lstm_quad(x)
        return x

class OtherQuadsEncoder(tfkl.Layer):
    def __init__(self):
        super().__init__()

        self.lambda_ = 0.01
        self.rnn_state_size_lstm_quad = 256
        self.rnn_state_size_bilstm = 256

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

class Decoder(tfkl.Layer):
    def __init__(self):
        super().__init__()
        
        self.output_dim = 3 # X, Y and Z coordinates of either position or velocity
    
        self.rnn_state_size_lstm_concat = 512
        self.fc_hidden_unit_size = 256
        self.attention_units = 128

        # self.concat = tfkl.Concatenate()
        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, name = 'lstm_decoder', return_sequences = True, return_state = False)
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, name = 'fc_decoder', activation = 'relu'))
        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim, name = 'fc_out'))
                
    def call(self, x):        
        x = self.lstm_concat(x)
        x = self.fc1(x)
        x = self.fco(x)
        return x

class FullModel(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        prediction_horizon = args.prediction_horizon
        
        # Define optimizer and loss
        self.loss_object = tfk.losses.MeanSquaredError()
        self.optimizer = tfk.optimizers.Adam()
        self.train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
        self.val_loss = tfk.metrics.MeanSquaredError(name='val_loss')
        
        # Define architecture
        self.state_encoder = StateEncoder()
        self.other_quads_encoder = OtherQuadsEncoder()
        self.concat = tfkl.Concatenate()
        self.repeat = tfkl.RepeatVector(prediction_horizon)
        self.decoder = Decoder()
                
    def call(self, x):
        x1 = self.state_encoder(x["query_input"])
        x2 = self.other_quads_encoder( x["others_input"] )
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

        self.train_loss(x["target"], predictions)

    @tf.function
    def val_step(self, x):
        predictions = self(x)
        v_loss = self.loss_object(x["target"], predictions)

        self.val_loss(x["target"], predictions)


