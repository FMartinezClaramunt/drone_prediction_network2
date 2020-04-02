import os, sys, time
print(sys.path[0])
import tensorflow as tf
from utils.data_handler_v3 import DataHandler
from utils.plot_utils_v3 import plot_predictions
from utils.model_utils import parse_args, model_selector, parse_input_types, parse_mat_dataset_names

import matplotlib.pyplot as plt
from math import ceil
import pickle as pkl
# from progressbar import progressbar
from tqdm import trange

tfkc = tf.keras.callbacks
tfkm = tf.keras.models
tfku = tf.keras.utils

#### Directory definitions ####
root_dir = os.path.dirname(sys.path[0])
data_dir = os.path.join(root_dir, "data", "")
trained_models_dir = os.path.join(root_dir, "trained_models", "")

#### Model selection ####
model_name = "varyingNQuadsRNN_v2"
model_number = 0

#### Script options ####
TRAIN = True
WARMSTART = False
DISPLAY = True

SUMMARY = False # To include a summary of the results of the model in a csv file
RECORD = False # To store recording of the animation
N_FRAMES = 3E3 # Number of frames to display/record
DT = 25 # Used to compute the FPS of the video
PLOT_GOALS = False # To plot the goals of the quadrotors

#### Datasets selection ####
datasets_training = "basicNormalInteraction\
                    goalSequence1\
                    goalSequence2"
datasets_validation = "goalSequence4"
datasets_test = "complexGoalSequence1"

#### Training parameters ####
MAX_EPOCHS = 15
MAX_STEPS = 1E6
TRAIN_PATIENCE = 5 # Number of epochs before early stopping
BATCH_SIZE = 64 

#### Network architecture ####
input_types = "vel relpos_vel"
past_horizon = 10
prediction_horizon = 10

# Encoder sizes
size_query_agent_state = 256
size_other_agents_state = 256
size_other_agents_bilstm = 256
size_action_encoding = 0

# Decoder sizes
size_decoder_lstm = 512
size_fc_layer = 256

#### Parse args and define relevant paths for the model ####
args = parse_args(locals())
model_dir = os.path.join(trained_models_dir, model_name, str(model_number), "")
parameters_path = os.path.join(model_dir, "model_parameters.pkl")
checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")

data_object = DataHandler(args)
model = model_selector(args)

# Load weights if warmstarting the model or if we are testing it
if args.warmstart or not args.train:
    assert os.path.isfile(checkpoint_path)
    sample_input_batch = data_object.getSampleInputBatch()
    model.call(sample_input_batch)
    model.load_weights(checkpoint_path)

if args.train:
    # TODO: Create a loop with trange for the steps in an epoch. For this, we need to know the number of samples in the training dataset, which we don't know right now
    # steps_per_epoch = 
    step = 1
    validation = len(data_object.datasets_validation) > 0
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in trange(args.max_epochs):
        new_epoch = False
        validation_finished = False
        model.train_loss.reset_states()
        model.val_loss.reset_states()
        start_time = time.time()
        
        while not new_epoch: # for step in trange(steps_per_epoch):
            batch = data_object.getBatch()
            model.train_step(batch)
            step += 1
            new_epoch = batch['new_epoch']
        
        # If we are performing validation
        if validation:
            while not validation_finished:
                batch = data_object.getValidationBatch()
                model.val_step(batch)
                validation_finished = batch['new_epoch']
            
            print("Train loss: %.4e, Validation loss: %.4e, Epoch time: %.1f sec\n" % (model.train_loss.result(), model.val_loss.result(), time.time()-start_time))
            
            if model.val_loss.result() < best_loss:
                print("Validation loss improved, saving new best model")
                best_loss = float(model.val_loss.result())
                best_model = model
                patience_counter = 0
            else:
                print("Validation loss did not improve")
                patience_counter += 1
        
        # If we are not performing validation
        else:
            print("Train loss: %.4e, Epoch time: %.1f sec\n" % (model.train_loss.result(), time.time()-start_time))
            
            if model.train_loss.result() < best_loss:
                print("Training loss improved, saving new best model")
                best_loss = float(model.train_loss.result())
                best_model = model
                patience_counter = 0
            else:
                print("Training loss did not improve")
                patience_counter += 1
        
        best_model.save_weights(checkpoint_path)
        
        if patience_counter >= args.max_patience:
            print("Maximum patience reached, terminating early")
            break
        
        if step > args.max_steps:
            print("Maximum number of training steps reached, terminating early")
            break
    model = best_model

# Set model back to stateless for prediction
if model.stateful:
    for i in range(len(model.layers)):
        model.layers[i].stateful = False

# TODO: Implement evaluation metrics for the models and a function to save the data to a CSV file. It could be useful to store the date in which the model was trained.
# Get summary of the model performance and store it in a CSV file
if args.summary:
    pass

# Plot and/or record animation to evaluate the network's prediction performance 
if args.display or args.record:
    for dataset_idx in range(len(data_object.datasets_testing)):
        plotting_data = data_object.getPlottingData(dataset_idx)
        
        # Path the recording will be stored if this option has been selected
        dataset_name = data_object.datasets_testing[dataset_idx].split(".")[0] # Remove extension from dataset name
        recording_path = os.path.join(model_dir, "Recordings", dataset_name + ".mp4")

        # Create animation
        plot_predictions(plotting_data, args, recording_path)

print("Master script finished")