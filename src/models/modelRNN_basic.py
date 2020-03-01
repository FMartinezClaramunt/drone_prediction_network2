import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
tfkb = tfk.backend

def buildEncoderDecoder(args):
    past_steps = args['past_steps']
    future_steps = args['future_steps']
 
    input_features = args['input_features']
    output_features = args['output_features']
    
    model = tfkm.Sequential()
    model.add(tfkl.LSTM(200, input_shape=(past_steps, input_features)))
    model.add(tfkl.RepeatVector(future_steps))
    model.add(tfkl.LSTM(200, activation='relu', return_sequences=True))
    model.add(tfkl.TimeDistributed(tfkl.Dense(100, activation='relu')))
    model.add(tfkl.TimeDistributed(tfkl.Dense(output_features)))

    return model

def buildMultiInputModel_original(args): # As it was implemented by Bruno
    past_steps = args['past_steps']
    future_steps = args['future_steps']

    input_state_dim = 3 # [vx, vy, vz]
    pedestrian_vector_dim = 6
    output_dim = 3

    rnn_state_size = 32
    rnn_state_size_lstm_grid = 256
    rnn_state_size_lstm_quads = 128
    rnn_state_size_lstm_concat = 510
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    input1 = tfkl.GRU(rnn_state_size, input_shape=[past_steps, input_state_dim],
                      name='ped_lstm',return_sequences=True, return_state=True,kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    
    input2 = tfkl.Dense(rnn_state_size_lstm_quads,
                        input_shape=[past_steps, pedestrian_vector_dim], activation='relu')
    ped_lstm = tfkl.GRU(rnn_state_size_lstm_quads,
                        name='other_peds_lstm', return_sequences=True, return_state=True, kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input2)

    concat = tfkl.concatenate([input1, ped_lstm])

    decoder = tfkl.GRU(rnn_state_size_lstm_concat,
                       return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')(concat)
    decoder = tfkl.Dense(fc_hidden_unit_size, activation='relu', name='concat_fc',
                         kernel_regularizer=tfk.regularizers.l2(l=self.lambda_))(decoder)
    predictions = tfkl.Dense(output_dim*future_steps, activation='linear',
                             name='out_fc', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(decoder)


    model = tfkm.Model(inputs = [input1, input2], outputs = predictions)

    return model

def buildMultiInputModel(args): # Modified so that it returns 3D output
    past_steps = args['past_steps']
    future_steps = args['future_steps']

    input_state_dim = args['input1_features']
    pedestrian_vector_dim = args['input2_features']
    output_dim = args['output_features']

    rnn_state_size = 32
    rnn_state_size_lstm_grid = 256
    rnn_state_size_lstm_quads = 128
    rnn_state_size_lstm_concat = 510
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    input1 = tfkl.Input(shape = [past_steps, input_state_dim])
    x1 = tfkl.GRU(rnn_state_size,
                      name='ped_lstm', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input1)

    input2 = tfkl.Input([past_steps, pedestrian_vector_dim])
    x2 = tfkl.GRU(rnn_state_size, return_sequences = True, kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input2)
    x2 = tfkl.TimeDistributed( tfkl.Dense(rnn_state_size_lstm_quads, activation='relu') )(x2)
    x2 = tfkl.GRU(rnn_state_size_lstm_quads, name='other_peds_lstm', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(x2)

    concat = tfkl.concatenate([x1, x2])
    concat = tfkl.RepeatVector(future_steps)(concat)

    decoder = tfkl.GRU(rnn_state_size_lstm_concat,
                       return_sequences=True, recurrent_initializer='glorot_uniform')(concat)
    decoder = tfkl.TimeDistributed(tfkl.Dense(fc_hidden_unit_size, activation='relu', 
                            name='concat_fc', kernel_regularizer=tfk.regularizers.l2(l=lambda_)))(decoder)
    predictions = tfkl.TimeDistributed(tfkl.Dense(output_dim, 
                            activation='linear', name='out_fc', kernel_regularizer=tfk.regularizers.l2(l=lambda_)))(decoder)

    model = tfkm.Model(inputs = [input1, input2], outputs = predictions)

    return model

def buildSharedMultiInputModel(args):
    past_steps = args['past_steps']
    future_steps = args['future_steps']
    
    input_state_dim = 3
    output_dim = 3
    if args['X_type'] == 'vel_pos':
        other_quads_vector_dim = 3
    elif args['X_type'] == 'vel_full':
        other_quads_vector_dim = 6

    rnn_state_size = 32
    rnn_state_size_lstm_quads = 128
    rnn_state_size_lstm_concat = 510
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    # Query agent input
    input1 = tfkl.Input(shape = [past_steps, input_state_dim])
    x1 = tfkl.GRU(rnn_state_size,
                      name='ped_lstm', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input1)


    # Other agents inputs
    input2a = tfkl.Input([past_steps, other_quads_vector_dim])
    input2b = tfkl.Input([past_steps, other_quads_vector_dim])
    input2c = tfkl.Input([past_steps, other_quads_vector_dim])
    
    sharedLSTM1 = tfkl.LSTM(rnn_state_size, name = 'sharedLSTM1', return_sequences = True, kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    sharedDense = tfkl.TimeDistributed( tfkl.Dense(rnn_state_size_lstm_quads, activation='relu') )
    sharedLSTM2 = tfkl.LSTM(rnn_state_size_lstm_quads, name = 'sharedLSTM2', kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    
    x2a = sharedLSTM1(input2a)
    x2a = sharedDense(x2a)
    x2a = sharedLSTM2(x2a)

    x2b = sharedLSTM1(input2b)
    x2b = sharedDense(x2b)
    x2b = sharedLSTM2(x2b)
    
    x2c = sharedLSTM1(input2c)
    x2c = sharedDense(x2c)
    x2c = sharedLSTM2(x2c)

    concat = tfkl.concatenate([x1, x2a, x2b, x2c])
    concat = tfkl.RepeatVector(future_steps)(concat)

    decoder = tfkl.GRU(rnn_state_size_lstm_concat,
                       return_sequences=True, recurrent_initializer='glorot_uniform')(concat)
    decoder = tfkl.TimeDistributed(tfkl.Dense(fc_hidden_unit_size, activation='relu', 
                            name='concat_fc', kernel_regularizer=tfk.regularizers.l2(l=lambda_)))(decoder)
    predictions = tfkl.TimeDistributed(tfkl.Dense(output_dim, 
                            activation='linear', name='out_fc', kernel_regularizer=tfk.regularizers.l2(l=lambda_)))(decoder)

    model = tfkm.Model(inputs = [input1, input2a, input2b, input2c], outputs = predictions)

    return model

def buildAttentionModel_v1(args): # Modified so that it returns 3D output
    past_steps = args['past_steps']
    future_steps = args['future_steps']
    
    input_state_dim = 3
    output_dim = 3
    if args['X_type'] == 'vel_pos':
        other_quads_vector_dim = 3
    elif args['X_type'] == 'vel_full':
        other_quads_vector_dim = 6

    rnn_state_size = 256
    # rnn_state_size_lstm_other_quads = 512
    rnn_state_size_lstm_other_quads = rnn_state_size
    rnn_state_size_decoder = 256
    rnn_state_size_final_decoder = 128
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    # Query agent input
    input_state = tfkl.Input(shape = [past_steps, input_state_dim])
    
    # Other agents inputs
    input_other_quad_1 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])
    input_other_quad_2 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])
    input_other_quad_3 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])

    encoded_state = tfkl.LSTM(rnn_state_size, return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input_state)
    
    sharedLSTM = tfkl.LSTM(rnn_state_size, return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    encoded_other_quad_1 = sharedLSTM(input_other_quad_1)
    encoded_other_quad_2 = sharedLSTM(input_other_quad_2)
    encoded_other_quad_3 = sharedLSTM(input_other_quad_3)
    
    encoded_other_quad_1 = tf.concat([encoded_state, encoded_other_quad_1], axis = 1)
    encoded_other_quad_2 = tf.concat([encoded_state, encoded_other_quad_2], axis = 1)
    encoded_other_quad_3 = tf.concat([encoded_state, encoded_other_quad_3], axis = 1)
    
    stacked_other_quads = tf.stack([encoded_other_quad_1, encoded_other_quad_2, encoded_other_quad_3], axis = 1)

    OtherQuadsEncoder = tfkl.Bidirectional(tfkl.LSTM(rnn_state_size_lstm_other_quads,  return_sequences = True, return_state = False), merge_mode = 'ave') # merge_mode options: {'sum', 'mul', 'concat', 'ave', None}
    
    OtherQuadsDecoder = tfkl.Bidirectional(tfkl.LSTM(rnn_state_size_decoder,  return_sequences = True, return_state = False), merge_mode = 'ave')
    
    other_quads_encoded = OtherQuadsEncoder(stacked_other_quads)

    # There has been no observed performance increase by using attention vs averaging, although more extensive testing would be required to tell for certain    
    # decoded = attention()(other_quads_encoded) # Custom Keras layer implementing Bahdanau attention
    decoded = tfkl.GlobalAveragePooling1D()(other_quads_encoded)

    repeated_decoded = tfkl.RepeatVector(future_steps)(decoded)
    final_decoding = tfkl.LSTM(rnn_state_size_final_decoder, return_sequences=True)(repeated_decoded)
    final_decoding = tfkl.TimeDistributed(tfkl.Dense(100, activation='relu'))(final_decoding)
    predictions = tfkl.TimeDistributed(tfkl.Dense(output_dim))(final_decoding)

    model = tfkm.Model(inputs = [input_state, input_other_quad_1, input_other_quad_2, input_other_quad_3], outputs = predictions)

    return model

def buildAttentionModel_v2(args): # Modified so that it returns 3D output
    past_steps = args['past_steps']
    future_steps = args['future_steps']
    
    input_state_dim = 3
    output_dim = 3
    if args['X_type'] == 'vel_pos':
        other_quads_vector_dim = 3
    elif args['X_type'] == 'vel_full':
        other_quads_vector_dim = 6

    rnn_state_size = 256
    # rnn_state_size_lstm_other_quads = 512
    rnn_state_size_lstm_other_quads = rnn_state_size
    rnn_state_size_decoder = 256
    rnn_state_size_final_decoder = 128
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    # Query agent input
    input_state = tfkl.Input(shape = [past_steps, input_state_dim])
    
    # Other agents inputs
    input_other_quad_1 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])
    input_other_quad_2 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])
    input_other_quad_3 = tfkl.Input(shape = [past_steps, other_quads_vector_dim])

    # LSTM encoders. Both have the same size so that they may be stacked.
    # TODO: Try adding intermediate Dense layers so that the LSTMs may have different sizes.
    encoded_state = tfkl.LSTM(rnn_state_size, return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input_state)
    # encoded_state_rep = tfkl.RepeatVector(3)(encoded_state)
    
    sharedLSTM = tfkl.LSTM(rnn_state_size, return_sequences = False, kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    encoded_other_quad_1 = sharedLSTM(input_other_quad_1)
    encoded_other_quad_2 = sharedLSTM(input_other_quad_2)
    encoded_other_quad_3 = sharedLSTM(input_other_quad_3)
    
    encoded_other_quad_1 = tf.concat([encoded_state, encoded_other_quad_1], axis = 1)
    encoded_other_quad_2 = tf.concat([encoded_state, encoded_other_quad_2], axis = 1)
    encoded_other_quad_3 = tf.concat([encoded_state, encoded_other_quad_3], axis = 1)
    
    stacked_other_quads = tf.stack([encoded_other_quad_1, encoded_other_quad_2, encoded_other_quad_3], axis = 1)

    OtherQuadsEncoder = tfkl.Bidirectional(tfkl.LSTM(rnn_state_size_lstm_other_quads,  return_sequences = True, return_state = False), merge_mode = 'ave') # merge_mode options: {'sum', 'mul', 'concat', 'ave', None}
    
    """other_quads_encoded = OtherQuadsEncoder(stacked_other_quads)
    decoded = tfkl.concatenate([encoded_state, other_quads_encoded])"""
    
    OtherQuadsDecoder = tfkl.Bidirectional(tfkl.LSTM(rnn_state_size_decoder,  return_sequences = True, return_state = False), merge_mode = 'ave')
    # OtherQuadsDecoder = tfkl.TimeDistributed(tfkl.Dense(rnn_state_size_decoder, activation = 'relu'))
    
    # Based on steps described on https://blog.floydhub.com/attention-mechanism/ (Luong Attention)
    other_quads_encoded = OtherQuadsEncoder(stacked_other_quads) # Step 1
    
    # other_quads_decoded = OtherQuadsDecoder(other_quads_encoded) # Step 2
    
    # encoded_state = tf.expand_dims(encoded_state, axis = 1)
    # context_vector = tfkl.Attention(causal = False)([other_quads_encoded, encoded_state]) # Steps 3, 4 and 5
    # decoded = tfkl.concatenate([other_quads_decoded, context_vector]) # Step 6
    # decoded = attention()(other_quads_encoded)
    decoded = tfkl.GlobalAveragePooling1D()(other_quads_encoded)
    
    repeated_decoded = tfkl.RepeatVector(future_steps)(decoded)
    final_decoding = tfkl.LSTM(rnn_state_size_final_decoder, return_sequences=True)(repeated_decoded)
    final_decoding = tfkl.TimeDistributed(tfkl.Dense(100, activation='relu'))(final_decoding)
    predictions = tfkl.TimeDistributed(tfkl.Dense(output_dim))(final_decoding)

    model = tfkm.Model(inputs = [input_state, input_other_quad_1, input_other_quad_2, input_other_quad_3], outputs = predictions)

    return model

class attention(tfkl.Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=tfkb.squeeze(tfkb.tanh(tfkb.dot(x,self.W)+self.b), axis=-1)
        at=tfkb.softmax(et)
        at=tfkb.expand_dims(at,axis=-1)
        output=x*at
        return tfkb.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()