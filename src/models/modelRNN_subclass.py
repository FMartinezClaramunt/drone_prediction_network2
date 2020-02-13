import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

class RNN(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        self.rnn_state_size = 128
        self.rnn_state_size_lstm_quad = 128
        self.rnn_state_size_lstm_concat = 512
        self.rnn_lstm_decoder_quad_size = 128
        self.rnn_lstm_decoder_other_quads_size = 128 
        self.fc_hidden_unit_size = 256
        
        self.truncated_backprop_length = args['past_steps'] # Unused
        self.future_steps = args['future_steps']
        self.output_dim = args['output_features']
        self.input_state_dim = args['input1_features'] # Unused
        self.quad_vector_dim = args['input2_features'] # Unused
        self.lambda_ = 0.01 # Unused

        self.lstm_quad1 = tfkl.LSTM(self.rnn_state_size)
        self.repvec_quad = tfkl.RepeatVector(self.future_steps)
        self.lstm_quad2 = tfkl.LSTM(self.rnn_lstm_decoder_quad_size, return_sequences = True)

        self.lstm_other_quads1 = tfkl.LSTM(self.rnn_state_size_lstm_quad)
        self.repvec_other_quads = tfkl.RepeatVector(self.future_steps)
        self.lstm_other_quads2 = tfkl.LSTM(self.rnn_lstm_decoder_other_quads_size, return_sequences = True)

        self.concat = tfkl.Concatenate()
        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, return_sequences = True)
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, activation = 'relu'))

        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim))
        
    def call(self, x):
        x1 = self.lstm_quad1(x[0])
        x1 = self.repvec_quad(x1)
        x1 = self.lstm_quad2(x1)
        
        x2 = self.lstm_other_quads1(x[1])
        x2 = self.repvec_other_quads(x2)
        x2 = self.lstm_other_quads2(x2)
        
        concat = self.concat([x1, x2])
        
        output = self.lstm_concat(concat)
        output = self.fc1(output)
        
        return self.fco(output)


