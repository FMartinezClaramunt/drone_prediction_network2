import os
import time
import tensorflow as tf
import utils.data_handler_v2 as dh
import utils.plot_utils_v2 as plu
import matplotlib.pyplot as plt
import pickle as pkl

tfkc = tf.keras.callbacks
tfkm = tf.keras.models

########## PARAMETER DEFINITION ##########
model_savename = "./trained_models/attentionRNN"

TRAIN = True
SAVE_VIDEO = False
DISPLAY_ANIMATION = True

EPOCHS = 15
PATIENCE = 5
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
        'X_type': 'vel_full', # Data to be used as input (pos, vel, full, vel_pos, vel_full)
        'Y_type': 'vel', # Data to be used as target (pos or vel)
        'split': True, # Whether to split the input to the network in data for quadrotor 0 and data for the rest of quadrotors
        'past_steps': 10, # Number of past time steps to consider as inputs to the network
        'future_steps': 10, # Number of time steps into the future to predict
        'folder_path': data_folder,
        'train_datasets': train_datasets,
        'validation_datasets': validation_datasets,
        'test_datasets': test_datasets,
        'scale_data': True,
        'kerasAPI': 'subclass'
    }
    pkl.dump( args, open( model_savename + "_args.pkl", "wb" ) )
else:
    args = pkl.load( open( model_savename + "_args.pkl", "rb" ) )


########## LOAD AND SCALE DATA ##########
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
# Construct model
from models.attentionRNN import RNN
model = RNN(args)

if TRAIN:
    
    """
    ### Keras standard training ###
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    es = tfkc.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=PATIENCE)
    # es = tfkc.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    mc = tfkc.ModelCheckpoint(model_savename + '_ckpt.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only = True)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=32, callbacks=[es, mc])

    # Summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show(block = False)"""
    
    best_val_loss = 999
    counter = 0

    for epoch in range(EPOCHS):
        model.train_loss.reset_states()
        model.val_loss.reset_states()

        for i in range(0, X_train[0].shape[0], BATCH_SIZE):
            X_batch, Y_batch = dh.get_batch(X_train, Y_train, BATCH_SIZE, i)
            model.train_step(X_batch, Y_batch)

        for i in range(0, X_val[0].shape[0], BATCH_SIZE):
            X_batch, Y_batch = dh.get_batch(X_val, Y_val, BATCH_SIZE, i)
            model.train_step(X_batch, Y_batch)

        if model.val_loss.result() < best_val_loss:
            print("Model improved")
            model.save_weights(model_savename + '_ckpt.h5')
            best_model = model
            best_val_loss = model.val_loss.result()
            counter = 0
        else:
            print("Model did not improve")
            counter += 1

        print("Epoch: %d, Train loss: %.4f, Validation loss: %.4f" % (epoch, model.train_loss.result(), model.val_loss.result()))

        if counter >= PATIENCE:
            print("Early stopping")
            break

    model = best_model
    
else:
    X_first = [X_train[0][0:1,:,:], [X_train[1][0][0:1,:,:], X_train[1][1][0:1,:,:], X_train[1][2][0:1,:,:]]]
    model.call(X_first) # To define the sizes of the layers
    # Load weights into the model
    model.load_weights(model_savename + "_ckpt.h5")

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# model.evaluate(X_test, Y_test) # Quantitative evaluation on the test set

scene = plu.plot_scenario(model, args, save = SAVE_VIDEO, display = DISPLAY_ANIMATION, nframes = 2400, figsize=(12,9))
