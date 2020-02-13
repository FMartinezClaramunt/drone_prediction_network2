import tensorflow as tf
import utils.data_handler as dh
import utils.plot_utils as plu
import matplotlib.pyplot as plt
import pickle as pkl

tfk = tf.keras
tfkc = tf.keras.callbacks
tfkm = tf.keras.models

########## PARAMETER DEFINITION ##########
model_savename = "./trained_models/subclass_custom_training"
TRAIN = False
SAVE_VIDEO = False
DISPLAY_ANIMATION = True

EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 64

loss_object = tfk.losses.MeanSquaredError()
optimizer = tfk.optimizers.Adam()
train_loss = tfk.metrics.MeanSquaredError(name='train_loss')
val_loss = tfk.metrics.MeanSquaredError(name='val_loss')

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
        'scale_data': True
    }
    pkl.dump( args, open( model_savename + "_args.pkl", "wb" ) )
else:
    args = pkl.load( open( model_savename + "_args.pkl", "rb" ) )

########## LOAD AND SCALE DATA ##########
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

train_samples = X_train[0].shape[0]
val_samples = X_val[0].shape[0]
test_samples = X_test[0].shape[0]

########## TRAINING or MODEL LOADING ##########
# Construct model
from models.modelRNN_subclass import RNN
model = RNN(args)

if TRAIN:    
    @tf.function 
    def train_step(X, Y):
        with tf.GradientTape() as tape:
            predictions = model(X)
            loss = loss_object(Y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(Y, predictions)
    
    @tf.function
    def val_step(X, Y):
        predictions = model(X)
        v_loss = loss_object(Y, predictions)

        val_loss(Y, predictions)
    
    best_val_loss = 999
    counter = 0

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        val_loss.reset_states()
        
        for i in range(0, train_samples, BATCH_SIZE):
            train_step([X_train[0][i:i+BATCH_SIZE,:,:], X_train[1][i:i+BATCH_SIZE,:,:]], Y_train[i:i+BATCH_SIZE,:,:])
        
        for i in range(0, val_samples, BATCH_SIZE):
            val_step([X_val[0][i:i+BATCH_SIZE,:,:], X_val[1][i:i+BATCH_SIZE,:,:]], Y_val[i:i+BATCH_SIZE,:,:])
            
        if val_loss.result() < best_val_loss:
            print("Model improved")
            model.save_weights(model_savename + '_ckpt.h5')
            best_model = model
            best_val_loss = val_loss.result()
            counter = 0
        else:
            print("Model did not improve")
            counter += 1
        
        print("Epoch: %d, Train loss: %.4f, Validation loss: %.4f" % (epoch, train_loss.result(), val_loss.result()))
        
        if counter >= PATIENCE:
            print("Early stopping")
            break
        
    model = best_model
    
else:
    model.call([X_train[0][0:1,:,:], X_train[1][0:1,:,:]]) # To define the sizes of the layers
    # Load weights into the model
    model.load_weights(model_savename + "_ckpt.h5")

    model.compile(loss = 'mse', optimizer = 'adam')

# model.evaluate(X_test, Y_test) # Quantitative evaluation on the test set

scene = plu.plot_scenario(model, args, save = SAVE_VIDEO, display = DISPLAY_ANIMATION, nframes = 2400, figsize=(12,9))
