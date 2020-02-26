import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers

class StateEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.future_steps = args['future_steps']

        self.rnn_state_size = 128
        self.rnn_lstm_decoder_quad_size = 128
        
        self.lstm_quad1 = tfkl.LSTM(self.rnn_state_size)
        self.repvec_quad = tfkl.RepeatVector(self.future_steps)
        # self.lstm_quad2 = tfkl.LSTM(self.rnn_lstm_decoder_quad_size, return_sequences = False)
        self.lstm_quad2 = tfkl.LSTM(self.rnn_lstm_decoder_quad_size, return_sequences = True)
        
    def call(self, x):
        x = self.lstm_quad1(x)
        x = self.repvec_quad(x)
        return self.lstm_quad2(x)

class OtherQuadsEncoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()

        self.future_steps = args['future_steps']

        self.rnn_state_size_lstm_quad = 128
        self.rnn_lstm_decoder_other_quads_size = 128 

        self.lstm_other_quads1 = tfkl.LSTM(self.rnn_state_size_lstm_quad)
        self.repvec_other_quads = tfkl.RepeatVector(self.future_steps)
        # self.lstm_other_quads2 = tfkl.LSTM(self.rnn_lstm_decoder_other_quads_size, return_sequences = False)
        self.lstm_other_quads2 = tfkl.LSTM(self.rnn_lstm_decoder_other_quads_size, return_sequences = True)
        
    def call(self, x):
        x = self.lstm_other_quads1(x)
        x = self.repvec_other_quads(x)
        return self.lstm_other_quads2(x)

class Decoder(tfkl.Layer):
    def __init__(self, args):
        super().__init__()
        
        self.output_dim = 3 # X, Y and Z coordinates of either position or velocity
    
        self.rnn_state_size_lstm_concat = 512
        self.fc_hidden_unit_size = 256
        self.attention_units = 128

        # self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, return_sequences = True, return_state = True)
        self.lstm_concat = tfkl.LSTM(self.rnn_state_size_lstm_concat, return_sequences = True, return_state = False)
        self.fc1 = tfkl.TimeDistributed(tfkl.Dense(self.fc_hidden_unit_size, activation = 'relu'))
        self.fco = tfkl.TimeDistributed(tfkl.Dense(self.output_dim))
        
        # self.attention = BahdanauAttention(self.attention_units)
        
    def call(self, x): # , hidden, enc_output):
        # context_vector, attention_weights = self.attention(hidden, enc_output)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        x = self.lstm_concat(x)
        x = self.fc1(x)
        return self.fco(x)

class BahdanauAttention(tfkl.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class MultiHeadAttention(tfkl.Layer):
    """ Source: https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2 """

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs
        
class RNN(tfk.Model):
    def __init__(self, args):
        super().__init__()
        
        # Define optimizer and loss
        self.loss_object = tfk.losses.MeanSquaredError()
        self.optimizer = tfk.optimizers.Adam()
        self.train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
        self.val_loss = tfk.metrics.MeanSquaredError(name='val_loss')
        
        # Define architecture
        self.state_encoder = StateEncoder(args)
        self.other_quads_encoder = OtherQuadsEncoder(args)
        self.concat = tfkl.Concatenate()
        self.decoder = Decoder(args)
        
    def call(self, x):
        x1 = self.state_encoder(x[0])
        x2a = self.other_quads_encoder(x[1][0])
        x2b = self.other_quads_encoder(x[1][1])
        x2c = self.other_quads_encoder(x[1][2])
        concat = self.concat([x1, x2a, x2b, x2c])
        return self.decoder(concat)
    
    @tf.function 
    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            predictions = self(X)
            loss = self.loss_object(Y, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(Y, predictions)

    @tf.function
    def val_step(self, X, Y):
        predictions = self(X)
        v_loss = loss_object(Y, predictions)

        self.val_loss(Y, predictions)


