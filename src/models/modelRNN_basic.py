import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers


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
    rnn_state_size_lstm_ped = 128
    rnn_state_size_lstm_concat = 510
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    input1 = tfkl.GRU(rnn_state_size, input_shape=[past_steps, input_state_dim],
                      name='ped_lstm',return_sequences=True, return_state=True,kernel_regularizer=tfk.regularizers.l2(l=lambda_))
    
    input2 = tfkl.Dense(rnn_state_size_lstm_ped,
                        input_shape=[past_steps, pedestrian_vector_dim], activation='relu')
    ped_lstm = tfkl.GRU(rnn_state_size_lstm_ped,
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
    rnn_state_size_lstm_ped = 128
    rnn_state_size_lstm_concat = 510
    fc_hidden_unit_size = 256

    lambda_ = 0.01

    input1 = tfkl.Input(shape = [past_steps, input_state_dim])
    x1 = tfkl.GRU(rnn_state_size,
                      name='ped_lstm', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input1)

    input2 = tfkl.Input([past_steps, pedestrian_vector_dim])
    x2 = tfkl.GRU(rnn_state_size, return_sequences = True, kernel_regularizer=tfk.regularizers.l2(l=lambda_))(input2)
    x2 = tfkl.TimeDistributed( tfkl.Dense(rnn_state_size_lstm_ped, activation='relu') )(x2)
    x2 = tfkl.GRU(rnn_state_size_lstm_ped, name='other_peds_lstm', kernel_regularizer=tfk.regularizers.l2(l=lambda_))(x2)

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

