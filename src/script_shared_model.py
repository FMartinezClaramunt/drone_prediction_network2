import os
import tensorflow as tf
import utils.data_handler_v2 as dh
import utils.plot_utils_v2 as plu
from models.modelRNN_basic import *
import matplotlib.pyplot as plt
import pickle as pkl

tfkc = tf.keras.callbacks
tfkm = tf.keras.models

########## PARAMETER DEFINITION ##########
model_savename = "./trained_models/shared_layer_model"
TRAIN = True
SAVE_VIDEO = False
DISPLAY_ANIMATION = True

EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 32
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
        'X_type': 'vel_full', # Data to be used as input (pos, vel or full)
        'Y_type': 'vel', # Data to be used as target (pos or vel)
        'past_steps': 10, # Number of past time steps to consider as inputs to the network
        'future_steps': 10, # Number of time steps into the future to predict
        'folder_path': data_folder,
        'train_datasets': train_datasets,
        'validation_datasets': validation_datasets,
        'test_datasets': test_datasets,
        'scale_data': True,
        'kerasAPI': 'functional'
    }
    pkl.dump( args, open( model_savename + "_args.pkl", "wb" ) )
else:
    args = pkl.load( open( model_savename + "_args.pkl", "rb" ) )

########## LOAD DATA ##########
train_data_paths = []
for dataset in args['train_datasets']:
    train_data_paths.append(os.path.join(args['folder_path'], dataset + '.mat'))
    
val_data_paths = []
for dataset in args['validation_datasets']:
    val_data_paths.append(os.path.join(args['folder_path'], dataset + '.mat'))

# test_data_paths = []
# for dataset in args['test_datasets']:
#     test_data_paths.append(os.path.join(args['folder_path'], dataset + '.mat'))
    
X_train, Y_train = dh.prepare_data(train_data_paths, args)
X_val, Y_val = dh.prepare_data(val_data_paths, args)
# X_test, Y_test = dh.prepare_data(test_data_paths, args)


if args['scale_data']:
    input_scaler = dh.get_scaler(X_train)
    X_train = dh.scale_data(X_train, input_scaler)
    X_val = dh.scale_data(X_val, input_scaler)
    # X_test = dh.scale_data(X_test, input_scaler)

    output_scaler = dh.get_scaler(Y_train)    
    Y_train = dh.scale_data(Y_train, output_scaler)
    Y_val = dh.scale_data(Y_val, output_scaler)
    # Y_test = dh.scale_data(Y_test, output_scaler)
    
    args.update({'input_scaler': input_scaler, 'output_scaler': output_scaler})
    
########## TRAINING or MODEL LOADING ##########
model = buildSharedMultiInputModel(args)    

if TRAIN:
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

    es = tfkc.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=PATIENCE)
    mc = tfkc.ModelCheckpoint(model_savename + '_ckpt.h5', monitor='val_accuracy',\
                        mode='max', verbose=1, save_best_only=True)
    
    # X_train = [X_train[0], X_train[1][0], X_train[1][1], X_train[1][2]]
    # X_val = [X_val[0], X_val[1][0], X_val[1][1], X_val[1][2]]
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, mc])

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

