import tensorflow as tf
import utils.data_handler as dh
import utils.plot_utils as plu
from models.modelRNN_basic import *
import matplotlib.pyplot as plt
import pickle as pkl

tfkc = tf.keras.callbacks
tfkm = tf.keras.models

########## PARAMETER DEFINITION ##########
# model_savename = "./trained_models/basic_single_input_pos"
# model_savename = "./trained_models/basic_single_input_vel"
model_savename = "./trained_models/basic_multi_input_vel"
TRAIN = True
SAVE_VIDEO = False
DISPLAY_ANIMATION = True

EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 32
MULTI_INPUT = True
loss = 'mean_squared_error'
# loss = 'huber_loss'
optimizer = 'adam'

data_folder = './data'

# List of datasets to use for training
train_datasets = []
train_datasets.append('basicNormalInteraction')
train_datasets.append('goalSequence1')
train_datasets.append('goalSequence2')
train_datasets.append('goalSequence3')
train_datasets.append('goalSequence5')
train_datasets.append('goalSequence6')
train_datasets.append('goalSequence7')

# List of datasets to use for validation
validation_datasets = []
validation_datasets.append('goalSequence4')
validation_datasets.append('goalSequence8')

# List of datasets to use for testing
test_datasets = []
test_datasets.append('complexGoalSequence1')
test_datasets.append('basicHighInteraction')

# Creation of variable to store the parameters
if TRAIN:
    args = {
        'X_type': 'full', # Data to be used as input (pos, vel or full)
        'Y_type': 'vel', # Data to be used as target (pos or vel)
        'split': MULTI_INPUT, # Whether to split the input to the network in data for quadrotor 0 and data for the rest of quadrotors
        'past_steps': 10, # Number of past time steps to consider as inputs to the network
        'future_steps': 10, # Number of time steps into the future to predict
        'folder_path': data_folder,
        'train_datasets': train_datasets,
        'validation_datasets': validation_datasets,
        'test_datasets': test_datasets,
        'scale_data': True
    }
    pkl.dump( args, open( model_savename + "_args.pkl", "wb" ) )
else:
    args = pkl.load( open( model_savename + "_args.pkl", "rb" ) )

########## LOAD DATA ##########
X_train, X_val, X_test, Y_train, Y_val, Y_test, args = dh.get_dataset(args)

if args['scale_data']:
    input_scaler = dh.get_scaler(X_train)
    X_train = dh.scale_data(X_train, input_scaler)
    X_val = dh.scale_data(X_val, input_scaler)
    X_test = dh.scale_data(X_test, input_scaler)

    output_scaler = dh.get_scaler(Y_train)    
    Y_train = dh.scale_data(Y_train, output_scaler)
    Y_val = dh.scale_data(Y_val, output_scaler)
    Y_test = dh.scale_data(Y_test, output_scaler)
    
    args.update({'input_scaler': input_scaler, 'output_scaler': output_scaler})

########## TRAINING or MODEL LOADING ##########
if args['split']:
    model = buildMultiInputModel(args)
else:
    model = buildEncoderDecoder(args)

if TRAIN:
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    es = tfkc.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    mc = tfkc.ModelCheckpoint(model_savename + '_ckpt.h5', monitor='val_accuracy',\
                        mode='max', verbose=1, save_best_only=True)
    
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),\
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, mc])

    # Summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show(block = False)
else:
    # Load weights into new model
    model.load_weights(model_savename + "_ckpt.h5")
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

# model.evaluate(X_test, Y_test) # Quantitative evaluation on the test set

scene = plu.plot_scenario(model, args, save = SAVE_VIDEO, display = DISPLAY_ANIMATION)
